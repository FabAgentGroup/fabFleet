"""강화학습 배차 - 선형 정책 + REINFORCE (정책경사, §8 D4)

후보 (job, vehicle)의 특징 φ(픽업 거리·차량 부하·bias)로 cost = w·φ를 추정하고,
softmax(-cost/τ) 정책으로 차량을 선택한다. 에피소드 보상(음의 평균 대기)으로
REINFORCE 갱신해 거리·부하 가중을 학습한다. 거리 가중 초기화로 nearest에서 출발한다.

외부 RL 프레임워크 없이 경량·결정적(시드 고정)으로 동작한다. Dispatcher 인터페이스를
그대로 따르므로 시뮬레이터 변경이 필요 없고, 학습 루프가 reset_episode/end_episode를
호출한다.
"""

from __future__ import annotations

import math
import random

from oht_sim.core.job import Job
from oht_sim.core.layout import manhattan
from oht_sim.core.vehicle import Vehicle

_NF = 3  # 특징 차원: [거리, 부하, bias]


class PolicyGradientDispatcher:
    """선형 정책 + REINFORCE 배차"""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        lr: float = 0.05,
        temperature: float = 0.3,
        rng: random.Random | None = None,
        train: bool = True,
    ):
        self.W = grid_width
        self.H = grid_height
        self.lr = lr
        self.tau = temperature
        self.rng = rng or random.Random(0)
        self.train = train
        # cost = w·φ. 거리 가중은 1로 고정(nearest에서 출발), 부하 페널티 w[1]만 학습
        # -> 정책은 "nearest + 학습된 부하 균형"으로 제약되어 발산하지 않음
        self.w: list[float] = [1.0, 0.0, 0.0]
        self._trainable = (1,)  # 부하 가중만 학습
        self.baseline: float | None = None
        self._ret_var: float = 1.0  # advantage 정규화용 분산 추정
        self._count: dict[int, int] = {}
        self._grad: list[float] = [0.0] * _NF

    # ----- 에피소드 경계 -----

    def reset_episode(self) -> None:
        self._count = {}
        self._grad = [0.0] * _NF

    def end_episode(self, reward: float) -> None:
        """에피소드 보상으로 REINFORCE 갱신 (baseline + 분산 정규화 + 클리핑)"""
        if self.baseline is None:
            self.baseline = reward
        adv = reward - self.baseline
        self._ret_var = 0.9 * self._ret_var + 0.1 * adv * adv
        norm_adv = adv / (self._ret_var ** 0.5 + 1e-6)
        for k in self._trainable:
            self.w[k] += self.lr * norm_adv * self._grad[k]
            self.w[k] = min(4.0, max(0.0, self.w[k]))  # 부하 페널티는 0~4로 제한
        self.baseline = 0.9 * self.baseline + 0.1 * reward

    # ----- 정책 -----

    def _features(self, job: Job, v: Vehicle) -> list[float]:
        dist = manhattan(v.pos, job.src.coord) / (self.W + self.H)
        load = self._count.get(v.id, 0) / 10.0
        return [dist, load, 1.0]

    def _cost(self, phi: list[float]) -> float:
        return sum(a * b for a, b in zip(self.w, phi))

    def assign(
        self,
        pending_jobs: list[Job],
        idle_vehicles: list[Vehicle],
    ) -> list[tuple[Job, Vehicle]]:
        out: list[tuple[Job, Vehicle]] = []
        avail = list(idle_vehicles)
        for job in sorted(pending_jobs, key=lambda j: (-j.priority, j.created_time)):
            if not avail:
                break
            phis = [self._features(job, v) for v in avail]
            costs = [self._cost(p) for p in phis]

            if self.train:
                logits = [-c / self.tau for c in costs]
                m = max(logits)
                exps = [math.exp(l - m) for l in logits]
                z = sum(exps)
                probs = [e / z for e in exps]
                idx = self._sample(probs)
                # ∇_w log π(idx) = (-φ_idx + Σ_j p_j φ_j) / τ
                mean_phi = [
                    sum(probs[j] * phis[j][k] for j in range(len(avail)))
                    for k in range(_NF)
                ]
                for k in range(_NF):
                    self._grad[k] += (-phis[idx][k] + mean_phi[k]) / self.tau
            else:
                idx = min(range(len(avail)), key=lambda j: costs[j])

            chosen = avail[idx]
            self._count[chosen.id] = self._count.get(chosen.id, 0) + 1
            out.append((job, chosen))
            avail.pop(idx)
        return out

    def _sample(self, probs: list[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1
