"""고장 인지 관제의 완화 효과 실험 (L3 캡스톤, D9)

차량 고장(D8)이 있는 시나리오에서 규칙 기반 관제(RuleSupervisor)가 고장 저하를
감지(fleet_degraded)하고 부하 균형·재배치로 완화하는지, 부하 영역별로 본다. 관제가
도울 여지는 배차 슬랙(유휴 차량)에 달려 있으므로 경부하~고부하를 스윕해 무관제 대비
리드타임·처리량 변화를 다중 시드 평균으로 비교한다.

LLM 없이 결정적으로 실행되며, 같은 관찰→감지→개입 루프를 LLM 관제도 공유한다(스냅샷에
가용·고장 정보 포함). 결과를 정직하게 제시한다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.agents.rule_supervisor import RuleSupervisor
from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.sim.simulator import Simulator

LOADS = [0.16, 0.22, 0.30]
DURATION = 600.0
NUM_VEHICLES = 6
FAILURE_MTBF = 150.0
REPAIR_TIME = 40.0
SUPERVISOR_INTERVAL = 30.0


def run_arm(load: float, supervised: bool, seed: int) -> dict:
    """한 부하·한 설정(관제 on/off)·한 시드 실행 → 운영 지표"""
    cfg = SimConfig(
        num_vehicles=NUM_VEHICLES, job_arrival_rate=load, sim_duration=DURATION,
        random_seed=seed,
        vehicle_failure=True, failure_mtbf=FAILURE_MTBF, repair_time=REPAIR_TIME,
    )
    sim = Simulator(cfg)
    mit = 0
    if supervised:
        agent_cfg = AgentConfig(
            supervisor_interval=SUPERVISOR_INTERVAL, availability_threshold=0.75
        )
        sup = RuleSupervisor(agent_cfg)
        sim.attach_supervisor(sup, agent_cfg)
    s = sim.run().summary()
    if supervised:
        mit = sum(1 for e in sup.timeline if e["actions"])
    return {
        "throughput": s["throughput"],
        "avg_lead": s["avg_lead_time"],
        "mitigations": float(mit),
    }


def evaluate(load: float, supervised: bool, seeds: list[int]) -> dict[str, float]:
    rows = [run_arm(load, supervised, s) for s in seeds]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def _chart(results: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(LOADS, [results[(l, False)]["avg_lead"] for l in LOADS],
            color="#9aa0a6", marker="o", label="no-supervision")
    ax.plot(LOADS, [results[(l, True)]["avg_lead"] for l in LOADS],
            color="#1a73e8", marker="o", label="rule-supervision")
    ax.set_xlabel("arrival rate (lambda)")
    ax.set_ylabel("avg lead (lower better)")
    ax.set_title("Failure mitigation vs load (with vehicle failures)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "mitigation_vs_load.png"), dpi=110)
    plt.close(fig)


def _results_md(results: dict, seeds: list[int]) -> str:
    lines = [
        "# 고장 인지 관제의 완화 효과 (D9)",
        "",
        f"OHT {NUM_VEHICLES}대, MTBF {FAILURE_MTBF:.0f}, 수리 {REPAIR_TIME:.0f}, 관제 주기 "
        f"{SUPERVISOR_INTERVAL:.0f}, 시드 {len(seeds)}개 평균. 규칙 관제는 가용 비율이 임계 "
        "미만이면(fleet_degraded) 부하 균형 + 대기 최다 구역 재배치로 완화한다.",
        "",
        "## 부하별 리드타임 (낮을수록 좋음)",
        "",
        "| 부하 λ | 무관제 리드 | 관제 리드 | Δ리드 | 관제 처리량 Δ | 평균 개입 |",
        "|---|---|---|---|---|---|",
    ]
    for l in LOADS:
        u, v = results[(l, False)], results[(l, True)]
        dl = (v["avg_lead"] - u["avg_lead"]) / u["avg_lead"] * 100 if u["avg_lead"] else 0.0
        dt = (v["throughput"] - u["throughput"]) / u["throughput"] * 100 if u["throughput"] else 0.0
        lines.append(
            f"| {l:.2f} | {u['avg_lead']:.1f} | {v['avg_lead']:.1f} | {dl:+.1f}% | "
            f"{dt:+.1f}% | {v['mitigations']:.1f} |"
        )

    light = results[(LOADS[0], False)], results[(LOADS[0], True)]
    dl_light = (light[1]["avg_lead"] - light[0]["avg_lead"]) / light[0]["avg_lead"] * 100
    heavy = results[(LOADS[-1], False)], results[(LOADS[-1], True)]
    dl_heavy = (heavy[1]["avg_lead"] - heavy[0]["avg_lead"]) / heavy[0]["avg_lead"] * 100
    lines += [
        "",
        "## 결정 (Decision)",
        "",
        f"경부하(λ={LOADS[0]})에서 관제가 리드타임 {dl_light:+.1f}%로 완화하지만, "
        f"고부하(λ={LOADS[-1]})에서는 {dl_heavy:+.1f}%로 효과가 사라집니다. 완화 여지는 "
        "배차 슬랙(유휴 차량)에 비례합니다.",
        "",
        "해석: 관제는 잃은 가용 용량을 되돌릴 수 없습니다. 유휴 슬랙이 있는 경부하에서는 "
        "남은 차량을 부하 균형·수요 구역으로 돌려 리드타임을 줄이지만, 용량이 포화된 "
        "고부하에서는 어떤 정책이든 같은 배정이 되어 개입이 발동해도 효과가 0에 수렴합니다. "
        "즉 용량 부족은 관제로 메울 수 없고, 관제의 가치는 슬랙의 운용 효율에 있습니다. "
        "핵심 성과는 L1 고장이 L3 관찰(스냅샷 가용·고장)→감지(fleet_degraded)→개입(액션 "
        "API)으로 닫힌 루프에 연결되었다는 점이며, 통계적 유의성은 D7(stats.py)의 쌍체 "
        "검정을 그대로 적용할 수 있습니다.",
        "",
        "![mitigation_vs_load](charts/mitigation_vs_load.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/failure_supervision_compare"
    seeds = list(range(1, 11))
    results = {
        (l, sup): evaluate(l, sup, seeds) for l in LOADS for sup in (False, True)
    }

    os.makedirs(out_dir, exist_ok=True)
    _chart(results, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(results, seeds))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for l in LOADS:
        u, v = results[(l, False)], results[(l, True)]
        print(f"  λ={l:.2f}: 무관제 리드={u['avg_lead']:.1f} | 관제 리드={v['avg_lead']:.1f} "
              f"개입={v['mitigations']:.1f}")


if __name__ == "__main__":
    main()
