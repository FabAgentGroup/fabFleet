"""강화학습 배차 - 선형 정책 + REINFORCE (정책경사, §8 D4)

후보 (job, vehicle)의 특징 φ(픽업 거리·차량 부하·bias)로 cost = w·φ를 추정하고,
softmax(-cost/τ) 정책으로 차량을 선택한다. 에피소드 보상(음의 평균 대기)으로
REINFORCE 갱신해 거리·부하 가중을 학습한다. 거리 가중 초기화로 nearest에서 출발한다.

D4의 한계(상태 3차원·선형·부하 가중만 학습)를 D6에서 풍부한 상태 + 비선형 MLP로
확장한다(MLPPolicyDispatcher). 둘 다 외부 RL 프레임워크 없이 경량·결정적(시드 고정)으로
동작하고 Dispatcher 인터페이스를 그대로 따르므로 시뮬레이터 변경이 필요 없으며, 학습
루프가 reset_episode/end_episode를 호출한다.
"""

from __future__ import annotations

import math
import random

import numpy as np

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


_MLP_NF = 6  # 특징 차원: [픽업거리, 부하, 큐압력, 작업길이, 경쟁도, bias]


class MLPPolicyDispatcher:
    """풍부한 상태 + 비선형(MLP) 정책 + REINFORCE 배차 (§8 D6)

    후보 (job, vehicle)마다 6차원 특징을 공유 MLP(은닉 1층, tanh)로 점수화하고
    softmax(점수/τ) 정책으로 차량을 선택한다. 에피소드 보상으로 정책경사를 수동
    역전파(numpy)해 전 파라미터를 학습한다. D4 선형 정책과 달리 상태에 큐 압력·작업
    길이·경쟁도를 더해 부하 영역에 따라 거동을 바꾼다.

    외부 RL 프레임워크 없이 결정적(시드 고정)으로 동작하며 Dispatcher 인터페이스를
    그대로 따른다.
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        hidden: int = 8,
        lr: float = 0.02,
        temperature: float = 0.5,
        seed: int = 0,
        train: bool = True,
    ):
        self.W = grid_width
        self.H = grid_height
        self.lr = lr
        self.tau = temperature
        self.train = train
        rng = np.random.default_rng(seed)
        # 작은 가중 초기화로 초기 정책을 거의 균등하게 두고 학습으로 분화
        self.W1 = rng.normal(0.0, 0.1, (hidden, _MLP_NF))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, 0.1, hidden)
        self.b2 = 0.0
        self._np_rng = rng
        self.baseline: float | None = None
        self._ret_var: float = 1.0
        self._count: dict[int, int] = {}
        self._reset_grad()

    def _reset_grad(self) -> None:
        self.gW1 = np.zeros_like(self.W1)
        self.gb1 = np.zeros_like(self.b1)
        self.gW2 = np.zeros_like(self.W2)
        self.gb2 = 0.0

    # ----- 에피소드 경계 -----

    def reset_episode(self) -> None:
        self._count = {}
        self._reset_grad()

    def end_episode(self, reward: float) -> None:
        """에피소드 보상으로 REINFORCE 갱신 (baseline + 분산 정규화 + 가중 클리핑)"""
        if self.baseline is None:
            self.baseline = reward
        adv = reward - self.baseline
        self._ret_var = 0.9 * self._ret_var + 0.1 * adv * adv
        norm_adv = adv / (self._ret_var ** 0.5 + 1e-6)
        self.W1 = np.clip(self.W1 + self.lr * norm_adv * self.gW1, -5.0, 5.0)
        self.b1 = np.clip(self.b1 + self.lr * norm_adv * self.gb1, -5.0, 5.0)
        self.W2 = np.clip(self.W2 + self.lr * norm_adv * self.gW2, -5.0, 5.0)
        self.b2 = float(np.clip(self.b2 + self.lr * norm_adv * self.gb2, -5.0, 5.0))
        self.baseline = 0.9 * self.baseline + 0.1 * reward

    # ----- 정책 -----

    def _features(
        self, job: Job, v: Vehicle, n_pending: int, n_idle: int
    ) -> np.ndarray:
        scale = self.W + self.H
        dist = manhattan(v.pos, job.src.coord) / scale
        load = self._count.get(v.id, 0) / 10.0
        queue_pressure = n_pending / (n_pending + n_idle + 1)
        job_len = manhattan(job.src.coord, job.dst.coord) / scale
        contention = min(1.0, n_idle / (n_pending + 1)) / 1.0
        return np.array([dist, load, queue_pressure, job_len, contention, 1.0])

    def _forward(self, phi: np.ndarray):
        z = self.W1 @ phi + self.b1
        h = np.tanh(z)
        score = float(self.W2 @ h + self.b2)
        return score, h

    def assign(
        self,
        pending_jobs: list[Job],
        idle_vehicles: list[Vehicle],
    ) -> list[tuple[Job, Vehicle]]:
        out: list[tuple[Job, Vehicle]] = []
        avail = list(idle_vehicles)
        n_pending = len(pending_jobs)
        for job in sorted(pending_jobs, key=lambda j: (-j.priority, j.created_time)):
            if not avail:
                break
            phis = [self._features(job, v, n_pending, len(avail)) for v in avail]
            fwd = [self._forward(p) for p in phis]
            scores = [s for s, _ in fwd]

            if self.train:
                logits = [s / self.tau for s in scores]
                m = max(logits)
                exps = [math.exp(l - m) for l in logits]
                z = sum(exps)
                probs = [e / z for e in exps]
                idx = self._sample(probs)
                self._accumulate_grad(idx, probs, phis, fwd)
            else:
                idx = max(range(len(avail)), key=lambda j: scores[j])

            chosen = avail[idx]
            self._count[chosen.id] = self._count.get(chosen.id, 0) + 1
            out.append((job, chosen))
            avail.pop(idx)
        return out

    def _accumulate_grad(self, idx, probs, phis, fwd) -> None:
        """∇_θ log π(idx) 를 후보별 역전파로 누적 (softmax + MLP)"""
        for j, (phi, (_, h)) in enumerate(zip(phis, fwd)):
            # ∂logπ(idx)/∂score_j = (1[j=idx] - p_j)/τ
            dscore = ((1.0 if j == idx else 0.0) - probs[j]) / self.tau
            self.gW2 += dscore * h
            self.gb2 += dscore
            dh = dscore * self.W2 * (1.0 - h * h)  # tanh'
            self.gW1 += np.outer(dh, phi)
            self.gb1 += dh

    def _sample(self, probs: list[float]) -> int:
        r = float(self._np_rng.random())
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1
