"""인과 평가·액션 가드 실험 (L3 neurosymbolic, D11)

D5는 관제 개입이 평균적으로 지표를 악화시키고, 효과 점수가 인과를 가리지 못함을
보였다. 본 실험은 (1) 드리프트 보정 인과 점수가 단순 전후 효과 점수의 과대 귀속을
바로잡는지, (2) 인과 가드가 해로운 개입을 거부하는지, (3) 그 운영 효과를, 무개입·
순진 개입(가드 off)·인과 가드(가드 on) 세 설정으로 비교한다. LLM 없이 결정적으로
실행되며 처리량 차이는 D7(stats.py)의 쌍체 검정으로 판정한다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.agents.guarded_supervisor import GuardedSupervisor
from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.experiments.stats import paired_diff_test
from oht_sim.sim.simulator import Simulator

LAMBDA = 0.40
DURATION = 500.0
NUM_VEHICLES = 8
QUEUE_THRESHOLD = 5
SUPERVISOR_INTERVAL = 20.0
N_SEEDS = 12


def run(mode: str, seed: int) -> dict:
    """mode: none | naive | guard"""
    sim = Simulator(SimConfig(
        num_vehicles=NUM_VEHICLES, job_arrival_rate=LAMBDA,
        sim_duration=DURATION, random_seed=seed,
    ))
    sup = None
    if mode != "none":
        ac = AgentConfig(supervisor_interval=SUPERVISOR_INTERVAL, queue_threshold=QUEUE_THRESHOLD)
        sup = GuardedSupervisor(ac, guard=(mode == "guard"))
        sim.attach_supervisor(sup, ac)
    s = sim.run().summary()
    out = {"throughput": s["throughput"], "avg_lead": s["avg_lead_time"]}
    if sup is not None:
        st = sup.stats()
        eff = sup.ledger.efficacy_summary()
        n = max(1, eff["total"])
        out.update({
            "acts": float(st["acts"]),
            "vetoes": float(st["vetoes"]),
            "avg_effect": eff["avg_score"],
            "avg_causal": eff["avg_causal"],
            "eff_harm_pct": 100.0 * eff["counts"]["악화"] / n,
            "causal_harm_pct": 100.0 * eff["causal_counts"]["악화"] / n,
        })
    return out


def evaluate(mode: str, seeds: list[int]) -> dict:
    rows = [run(mode, s) for s in seeds]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


def _chart(naive: dict, guard: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 측정 교정: 효과 vs 인과의 악화 비율
    axes[0].bar(["effect score", "causal score"],
                [naive["eff_harm_pct"], naive["causal_harm_pct"]],
                color=["#9aa0a6", "#1a73e8"])
    axes[0].set_ylabel("% interventions labeled harmful")
    axes[0].set_title("Attribution: naive vs drift-adjusted")
    axes[0].grid(True, axis="y", alpha=0.3)
    # 가드: 순진 vs 가드의 개입·거부
    width = 0.35
    xs = range(2)
    axes[1].bar([x - width / 2 for x in xs], [naive["acts"], naive["vetoes"]], width,
                label="naive", color="#9aa0a6")
    axes[1].bar([x + width / 2 for x in xs], [guard["acts"], guard["vetoes"]], width,
                label="guard", color="#1a73e8")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels(["acted", "vetoed"])
    axes[1].set_title("Guard suppresses flagged-harmful actions")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "causal_guard.png"), dpi=110)
    plt.close(fig)


def _results_md(agg: dict, raw: dict, seeds: list[int]) -> str:
    none, naive, guard = agg["none"], agg["naive"], agg["guard"]
    th_naive = paired_diff_test([r["throughput"] for r in raw["naive"]],
                                [r["throughput"] for r in raw["none"]], seed=0)
    th_guard = paired_diff_test([r["throughput"] for r in raw["guard"]],
                                [r["throughput"] for r in raw["none"]], seed=0)
    lines = [
        "# 인과 평가·액션 가드 실험 (D11)",
        "",
        f"OHT {NUM_VEHICLES}대, 부하 λ={LAMBDA}, 큐 임계 {QUEUE_THRESHOLD}, 관제 주기 "
        f"{SUPERVISOR_INTERVAL:.0f}, 시드 {len(seeds)}개. 무개입 vs 순진 개입(가드 off) vs "
        "인과 가드(가드 on). 개입 액션은 대기 최다 구역으로 유휴 차량 재배치.",
        "",
        "## 1) 측정 교정 - 단순 효과 점수 vs 드리프트 보정 인과 점수 (순진 개입)",
        "",
        "| 척도 | 평균 점수 | 악화로 분류된 개입 |",
        "|---|---|---|",
        f"| 단순 효과(전-후) | {naive['avg_effect']:.2f} | {naive['eff_harm_pct']:.0f}% |",
        f"| 인과(반사실 보정) | {naive['avg_causal']:.2f} | {naive['causal_harm_pct']:.0f}% |",
        "",
        "단순 전후 점수는 주변 큐 드리프트를 개입 탓으로 과대 귀속한다. 드리프트를 빼면 "
        "악화로 분류되는 개입 비율이 줄어, 개입이 보이는 것만큼 해롭지는 않음을 드러낸다.",
        "",
        "## 2) 가드 동작 - 해로운 개입 거부",
        "",
        "| 설정 | 평균 개입 | 평균 거부 |",
        "|---|---|---|",
        f"| 순진(가드 off) | {naive['acts']:.1f} | 0.0 |",
        f"| 인과 가드(가드 on) | {guard['acts']:.1f} | {guard['vetoes']:.1f} |",
        "",
        "## 3) 운영 효과 (처리량, 쌍체 검정)",
        "",
        "| 설정 | 처리량 | 평균 리드 | vs 무개입 |",
        "|---|---|---|---|",
        f"| 무개입 | {none['throughput']:.4f} | {none['avg_lead']:.1f} | - |",
        f"| 순진 개입 | {naive['throughput']:.4f} | {naive['avg_lead']:.1f} | {th_naive.fmt()} |",
        f"| 인과 가드 | {guard['throughput']:.4f} | {guard['avg_lead']:.1f} | {th_guard.fmt()} |",
        "",
        "## 결정 (Decision)",
        "",
        f"인과 점수는 순진 효과 점수의 과대 귀속을 바로잡습니다(악화 분류 "
        f"{naive['eff_harm_pct']:.0f}% -> {naive['causal_harm_pct']:.0f}%, 평균 "
        f"{naive['avg_effect']:.1f} -> {naive['avg_causal']:.1f}). 가드는 인과적으로 해로운 "
        f"개입을 거부해 개입을 {naive['acts']:.0f} -> {guard['acts']:.0f}회로 줄입니다. "
        f"처리량 차이는 순진 {th_naive.fmt()}, 가드 {th_guard.fmt()}입니다.",
        "",
        "해석: D5에서 관제 개입이 순손해로 보인 핵심 원인은 효과 점수가 주변 부하 변동을 "
        "개입 탓으로 돌린 데 있습니다. 반사실(개입 직전 추세) 보정으로 인과를 분리하면 개입의 "
        "해악이 과대평가였음이 드러나고, 가드는 그래도 인과 음수인 개입을 차단합니다. 다만 이 "
        "관제 액션(재배치·정책 전환)은 D7이 보였듯 처리량을 거의 못 움직이는 저레버리지라 "
        "운영 차이는 noise 범위입니다. 즉 가드의 가치는 처리량 회복이 아니라 해롭다고 인과 "
        "귀속된 개입의 낭비를 줄이는 데 있습니다. 같은 인과 점수·가드는 LLM 관제 경로에도 "
        "연결되어(reflection 프롬프트에 인과 점수 주입, action_guard로 해로운 액션 거부), 본 "
        "결정적 실험은 그 메커니즘을 키 없이 검증한 것입니다. 더 큰 레버리지(혼잡 라우팅 D10)와 "
        "결합하는 것이 다음 과제입니다.",
        "",
        "![causal_guard](charts/causal_guard.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/causal_guard_compare"
    seeds = list(range(1, N_SEEDS + 1))
    agg, raw = {}, {}
    for mode in ("none", "naive", "guard"):
        agg[mode], raw[mode] = evaluate(mode, seeds)

    os.makedirs(out_dir, exist_ok=True)
    _chart(agg["naive"], agg["guard"], out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(agg, raw, seeds))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    n = agg["naive"]
    print(f"  측정: 효과 악화 {n['eff_harm_pct']:.0f}% -> 인과 악화 {n['causal_harm_pct']:.0f}%")
    print(f"  가드: 개입 {n['acts']:.0f} -> {agg['guard']['acts']:.0f}, 거부 {agg['guard']['vetoes']:.0f}")
    for m in ("none", "naive", "guard"):
        print(f"  {m}: thr={agg[m]['throughput']:.4f} lead={agg[m]['avg_lead']:.1f}")


if __name__ == "__main__":
    main()
