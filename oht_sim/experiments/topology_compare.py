"""토폴로지 비교 실험 - 자유 격자 vs 방향성 레일 (OHT 충실도, D16)

기존 실험들은 자유 2D 격자(AGV/AMR식)에서 수행됐다. 실제 OHT는 천장 단방향 모노레일
(인터베이 루프 + 인트라베이 베이) 위를 달린다. 같은 작업 부하를 두 토폴로지에서 돌려,
레일 제약이 처리량·리드타임에 미치는 영향을 다중 시드 쌍체로 비교한다. 자유 격자가
이동 비용을 얼마나 과소평가했는지, 즉 도메인 충실도의 차이를 정량화한다.
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

WIDTH, HEIGHT = 14, 10
NUM_STATIONS = 10
NUM_VEHICLES = 10
DURATION = 500.0
LOADS = [0.15, 0.25, 0.35]
N_SEEDS = 10


def run(topology: str, load: float, seed: int) -> dict:
    cfg = SimConfig(
        grid_width=WIDTH, grid_height=HEIGHT, num_stations=NUM_STATIONS,
        num_vehicles=NUM_VEHICLES, job_arrival_rate=load, sim_duration=DURATION,
        random_seed=seed, topology=topology,
    )
    sim = Simulator(cfg)
    s = sim.run().summary()
    return {
        "throughput": s["throughput"],
        "avg_lead": s["avg_lead_time"],
        "deadlocks": float(sum(1 for e in sim.bus.log if e.type == EventType.DEADLOCK_DETECTED)),
    }


def collect(seeds: list[int]) -> dict:
    data: dict = {}
    for load in LOADS:
        data[load] = {
            "grid": {k: [] for k in ("throughput", "avg_lead", "deadlocks")},
            "rail": {k: [] for k in ("throughput", "avg_lead", "deadlocks")},
        }
        for s in seeds:
            for topo in ("grid", "rail"):
                r = run(topo, load, s)
                for k in r:
                    data[load][topo][k].append(r[k])
    return data


def _chart(data: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(
        axes, [("throughput", "throughput (higher better)"), ("avg_lead", "avg lead (lower better)")]
    ):
        ax.plot(LOADS, [float(np.mean(data[l]["grid"][key])) for l in LOADS],
                color="#9aa0a6", marker="o", label="free grid (AGV-like)")
        ax.plot(LOADS, [float(np.mean(data[l]["rail"][key])) for l in LOADS],
                color="#1a73e8", marker="o", label="rail (OHT-like)")
        ax.set_xlabel("arrival rate (lambda)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "topology.png"), dpi=110)
    plt.close(fig)


def _results_md(data: dict, seeds: list[int]) -> str:
    lines = [
        "# 토폴로지 비교 - 자유 격자 vs 방향성 레일 (D16)",
        "",
        f"격자 {WIDTH}x{HEIGHT}, Station {NUM_STATIONS}, OHT {NUM_VEHICLES}대, 시드 {len(seeds)}개 "
        "쌍체. 자유 격자(AGV/AMR식 4방향 자유 이동)와 방향성 레일(OHT 단방향 모노레일: 인터베이 "
        "루프 + 인트라베이 베이)을 부하별로 비교한다. 리드 차이는 쌍체 순열검정으로 판정한다.",
        "",
        "## 부하별 비교 (Δ = rail - grid)",
        "",
        "| 부하 λ | 처리량 grid→rail | 평균 리드 grid→rail (Δ, p) | 교착 grid→rail |",
        "|---|---|---|---|",
    ]
    for l in LOADS:
        g, r = data[l]["grid"], data[l]["rail"]
        gt, rt = float(np.mean(g["throughput"])), float(np.mean(r["throughput"]))
        gl, rl = float(np.mean(g["avg_lead"])), float(np.mean(r["avg_lead"]))
        gd, rd = float(np.mean(g["deadlocks"])), float(np.mean(r["deadlocks"]))
        t = paired_diff_test(r["avg_lead"], g["avg_lead"], seed=0)
        lines.append(
            f"| {l:.2f} | {gt:.3f}→{rt:.3f} | {gl:.1f}→{rl:.1f} "
            f"({t.mean_diff:+.1f}, p={t.p_value:.3f}{'*' if t.significant else ''}) | "
            f"{gd:.0f}→{rd:.0f} |"
        )

    l0 = LOADS[0]
    g0, r0 = data[l0]["grid"], data[l0]["rail"]
    lead_x = np.mean(r0["avg_lead"]) / np.mean(g0["avg_lead"]) if np.mean(g0["avg_lead"]) else 0
    thr_drop = (np.mean(r0["throughput"]) - np.mean(g0["throughput"])) / np.mean(g0["throughput"]) * 100
    lines += [
        "",
        "(* 리드 차이 p<0.05)",
        "",
        "## 결정 (Decision)",
        "",
        f"같은 작업 부하에서 레일 토폴로지는 리드타임을 약 {lead_x:.1f}배로 늘리고 처리량을 "
        f"{thr_drop:+.0f}%(λ={l0}) 떨어뜨립니다. 단방향 레일은 목적지가 진행 방향 뒤면 루프를 "
        "돌아야 해 이동 거리가 크게 늘기 때문입니다. 다만 정면 교착은 줄어듭니다(단방향).",
        "",
        "해석: 자유 격자(AGV/AMR식)는 이동 비용을 크게 과소평가했고, 실제 OHT 모노레일의 단방향 "
        "제약이 성능을 지배함을 보여줍니다. 즉 기존 D1~D15의 절대 수치는 격자 가정에서 낙관적이며, "
        "OHT 충실도를 높이려면 토폴로지가 출발점입니다. 본 레일은 외곽 루프 + 베이 골격의 최소 "
        "구현이라 실제 fab보다 제약이 강합니다. 실 fab은 숏컷·복수 루프·양방향 인터베이로 이를 "
        "완화하므로, 다음 단계는 숏컷·정션 용량·로드포트 blocking을 더해 실제 AMHS에 다가가는 "
        "것입니다. 이 위에서 D10 혼잡 라우팅·D12 PIBT를 다시 평가하면 토폴로지 의존 효과가 "
        "드러날 것입니다.",
        "",
        "![topology](charts/topology.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/topology_compare"
    seeds = list(range(1, N_SEEDS + 1))
    data = collect(seeds)

    os.makedirs(out_dir, exist_ok=True)
    _chart(data, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(data, seeds))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for l in LOADS:
        g, r = data[l]["grid"], data[l]["rail"]
        print(f"  λ={l}: grid thr={np.mean(g['throughput']):.3f} lead={np.mean(g['avg_lead']):.1f} | "
              f"rail thr={np.mean(r['throughput']):.3f} lead={np.mean(r['avg_lead']):.1f}")


if __name__ == "__main__":
    main()
