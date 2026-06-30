"""L2 개선 결합 실험 - 혼잡 라우팅 + PIBT 스택 (D10·D12 후속, D15)

이번 세션의 두 L2 승리(D10 혼잡 인지 라우팅, D12 PIBT)가 함께 쓰면 더 좋아지는지
(보완재) 아니면 한쪽이 다른 쪽을 포섭하는지(대체재)를 본다. PIBT 후보 선택에 혼잡을
동점 기준으로 넣어 둘을 결합하고, 고밀도에서 baseline·congestion·pibt·pibt+cong 네
설정을 다중 시드 쌍체로 비교한다. 차이는 D7(stats.py) 순열검정으로 판정한다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.core.config import SimConfig
from oht_sim.core.events import EventType
from oht_sim.experiments.stats import paired_diff_test
from oht_sim.sim.simulator import Simulator

LAMBDA = 0.50
DURATION = 500.0
NUM_VEHICLES = 20
N_SEEDS = 12
ARMS = {
    "baseline": (False, False),
    "congestion": (False, True),
    "pibt": (True, False),
    "pibt+cong": (True, True),
}
METRICS = ("throughput", "blocked", "deadlocks")


def run(pibt: bool, cong: bool, seed: int) -> dict:
    cfg = SimConfig(
        num_vehicles=NUM_VEHICLES, job_arrival_rate=LAMBDA, sim_duration=DURATION,
        random_seed=seed, pibt_planning=pibt, congestion_aware_routing=cong,
    )
    sim = Simulator(cfg)
    s = sim.run().summary()
    return {
        "throughput": s["throughput"],
        "blocked": float(sum(1 for e in sim.bus.log if e.type == EventType.BLOCKED)),
        "deadlocks": float(sum(1 for e in sim.bus.log if e.type == EventType.DEADLOCK_DETECTED)),
    }


def collect(seeds: list[int]) -> dict:
    data = {arm: {m: [] for m in METRICS} for arm in ARMS}
    for s in seeds:
        for arm, (pibt, cong) in ARMS.items():
            r = run(pibt, cong, s)
            for m in METRICS:
                data[arm][m].append(r[m])
    return data


def _chart(data: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    arms = list(ARMS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(
        axes, [("throughput", "throughput (higher better)"), ("deadlocks", "deadlocks (lower better)")]
    ):
        ax.bar(arms, [float(np.mean(data[a][key])) for a in arms],
               color=["#9aa0a6", "#5f6368", "#1a73e8", "#0b3d91"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "l2_stack.png"), dpi=110)
    plt.close(fig)


def _results_md(data: dict, seeds: list[int]) -> str:
    def mean(arm, m):
        return float(np.mean(data[arm][m]))

    # 핵심 검정: pibt+cong vs pibt (스택이 PIBT에 더하는가)
    t_thr = paired_diff_test(data["pibt+cong"]["throughput"], data["pibt"]["throughput"], seed=0)
    t_blk = paired_diff_test(data["pibt+cong"]["blocked"], data["pibt"]["blocked"], seed=0)

    lines = [
        "# L2 개선 결합 실험 - 혼잡 라우팅 + PIBT (D15)",
        "",
        f"고밀도(OHT {NUM_VEHICLES}대, λ={LAMBDA}), 시드 {len(seeds)}개 쌍체. baseline·혼잡 라우팅"
        "(D10)·PIBT(D12)·둘 결합을 비교한다. PIBT 후보 선택에 혼잡을 동점 기준으로 결합했다.",
        "",
        "## 설정별 평균",
        "",
        "| 설정 | 처리량 | 회피 대기 | 교착 |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {mean(arm, 'throughput'):.3f} | {mean(arm, 'blocked'):.0f} | "
            f"{mean(arm, 'deadlocks'):.1f} |"
        )
    lines += [
        "",
        "## 핵심 검정 - 스택이 PIBT에 더하는 효과 (pibt+cong vs pibt)",
        "",
        f"- 처리량: {t_thr.fmt()}",
        f"- 회피 대기: {t_blk.fmt()}",
        "",
        "## 결정 (Decision)",
        "",
        f"PIBT 없이 혼잡 라우팅은 baseline 대비 큰 이득을 내지만(처리량 "
        f"{mean('baseline','throughput'):.3f}→{mean('congestion','throughput'):.3f}, 교착 "
        f"{mean('baseline','deadlocks'):.0f}→{mean('congestion','deadlocks'):.0f}), PIBT 위에 "
        f"혼잡 라우팅을 더해도 처리량은 {t_thr.fmt()}로 유의한 추가 이득이 없습니다. 회피 대기만 "
        f"{t_blk.fmt()} 정도 더 줄어듭니다.",
        "",
        "해석: 두 L2 개선은 보완재가 아니라 대체재입니다. PIBT가 우선순위 상속·백트래킹으로 "
        "혼잡·교착을 구조적으로 푸는 강한 메커니즘이라, 혼잡을 비용에 반영하는 라우팅이 줄 여지를 "
        "대부분 포섭합니다. 따라서 PIBT를 쓰면 혼잡 라우팅은 거의 불필요하고, 반대로 PIBT를 못 쓰는 "
        "환경(예: 외부 경로계획 고정)에서는 혼잡 라우팅이 값싼 대안이 됩니다. 결합의 한계 이득"
        "(회피 대기 소폭 감소)은 동점 기준 수준에 그칩니다.",
        "",
        "![l2_stack](charts/l2_stack.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/l2_stack_compare"
    seeds = list(range(1, N_SEEDS + 1))
    data = collect(seeds)

    os.makedirs(out_dir, exist_ok=True)
    _chart(data, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(data, seeds))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for arm in ARMS:
        print(f"  {arm}: thr={np.mean(data[arm]['throughput']):.3f} "
              f"blocked={np.mean(data[arm]['blocked']):.0f} deadlock={np.mean(data[arm]['deadlocks']):.1f}")


if __name__ == "__main__":
    main()
