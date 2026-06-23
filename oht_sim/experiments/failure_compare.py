"""차량 고장이 반송 성능에 미치는 영향 실험 (L1 신뢰성, D8)

OHT가 확률적으로 고장·수리되는 시나리오를 고장 강도(MTBF)별로 돌려, 무고장 대비
처리량·리드타임·완료 작업 수의 저하를 다중 시드 평균·표준편차로 정량화한다. 고장
차량은 정지 장애물로 남아 경로계획(L2)이 우회하고, 미완 작업은 회수되어 재배차된다.

부트스트랩 신뢰구간·유의성은 D7(experiments/stats.py)의 프레임을 그대로 적용할 수
있으며, 여기서는 효과 크기가 커 평균·표준편차로 충분히 드러난다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.core.config import SimConfig
from oht_sim.sim.simulator import Simulator

LAMBDA = 0.30
DURATION = 600.0
NUM_VEHICLES = 6
REPAIR_TIME = 40.0
# (라벨, MTBF) - None이면 무고장
SCENARIOS = [("무고장", None), ("MTBF 300", 300.0), ("MTBF 150", 150.0)]
# 차트용 ASCII 라벨 (matplotlib 한글 글리프 없음)
ASCII_LABEL = {"무고장": "no-fail", "MTBF 300": "MTBF 300", "MTBF 150": "MTBF 150"}


def _run(mtbf: float | None, seed: int) -> dict:
    cfg = SimConfig(
        num_vehicles=NUM_VEHICLES, job_arrival_rate=LAMBDA, sim_duration=DURATION,
        random_seed=seed,
        vehicle_failure=mtbf is not None,
        failure_mtbf=mtbf if mtbf is not None else 1.0,
        repair_time=REPAIR_TIME,
    )
    s = Simulator(cfg).run().summary()
    return {
        "throughput": s["throughput"],
        "avg_lead": s["avg_lead_time"],
        "completed": float(s["completed_jobs"]),
    }


def evaluate(mtbf: float | None, seeds: list[int]) -> dict[str, float]:
    rows = [_run(mtbf, s) for s in seeds]
    out: dict[str, float] = {}
    for k in ("throughput", "avg_lead", "completed"):
        vals = [r[k] for r in rows]
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    return out


def _charts(comp: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    labels = list(comp)
    xticks = [ASCII_LABEL.get(s, s) for s in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(
        axes, [("throughput", "throughput (higher better)"), ("avg_lead", "avg lead (lower better)")]
    ):
        means = [comp[s][key] for s in labels]
        stds = [comp[s][key + "_std"] for s in labels]
        ax.bar(xticks, means, yerr=stds, capsize=6, color="#1a73e8", alpha=0.8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "failure_impact.png"), dpi=110)
    plt.close(fig)


def _results_md(comp: dict, seeds: list[int]) -> str:
    base = comp["무고장"]

    def deg(label: str, key: str) -> str:
        b, v = base[key], comp[label][key]
        return f"{(v - b) / b * 100:+.0f}%" if b else "n/a"

    lines = [
        "# 차량 고장이 반송 성능에 미치는 영향 (D8)",
        "",
        f"부하 λ={LAMBDA}, OHT {NUM_VEHICLES}대, 수리 시간 평균 {REPAIR_TIME:.0f}, "
        f"시드 {len(seeds)}개 평균±표준편차. 고장 차량은 정지 장애물로 남고 미완 작업은 회수·재배차된다.",
        "",
        "## 시나리오별 성능 (평균±표준편차)",
        "",
        "| 시나리오 | 처리량 | 평균 리드 | 완료 작업 | 무고장 대비 처리량 |",
        "|---|---|---|---|---|",
    ]
    for s in comp:
        c = comp[s]
        lines.append(
            f"| {s} | {c['throughput']:.3f}±{c['throughput_std']:.3f} | "
            f"{c['avg_lead']:.1f}±{c['avg_lead_std']:.1f} | "
            f"{c['completed']:.1f} | {deg(s, 'throughput')} |"
        )
    lines += [
        "",
        "## 결정 (Decision)",
        "",
        f"고장이 잦을수록(MTBF 감소) 처리량이 단조 감소하고 리드타임이 증가합니다. "
        f"MTBF 150에서 처리량 {deg('MTBF 150', 'throughput')}, 리드타임 {deg('MTBF 150', 'avg_lead')}. "
        "고장 차량이 트랙을 점유해 우회·재배차 비용이 더해지고, 가용 차량이 줄어 큐가 쌓입니다.",
        "",
        "해석: 신뢰성은 처리량에 1차적으로 직접 영향을 줍니다. 이 시나리오는 가용성 저하가 "
        "성능에 미치는 크기를 보여 주며, 향후 L3 관제가 고장을 이상으로 감지해 유휴 차량 "
        "재배치·우선순위 조정으로 완화하는 폐루프의 평가 베드가 됩니다. 통계적 유의성이 "
        "필요하면 D7(stats.py)의 부트스트랩 CI·쌍체 검정을 그대로 적용할 수 있습니다.",
        "",
        "![failure_impact](charts/failure_impact.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/failure_compare"
    seeds = list(range(1, 11))
    comp = {label: evaluate(mtbf, seeds) for label, mtbf in SCENARIOS}

    os.makedirs(out_dir, exist_ok=True)
    _charts(comp, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(comp, seeds))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for s in comp:
        c = comp[s]
        print(f"  {s}: 처리량={c['throughput']:.3f} 리드={c['avg_lead']:.1f} 완료={c['completed']:.1f}")


if __name__ == "__main__":
    main()
