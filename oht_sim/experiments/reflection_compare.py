"""개입 효과 평가·reflection 닫힌 루프 비교 실험 (D5)

L3 관제를 reflection on/off 두 설정으로 여러 시드에 돌려, 개입 효율(개선·악화
비율, 평균 효과 점수)과 운영 지표(평균 대기·리드·큐)를 비교한다. 관제는 실제
LLM(gpt-4o-mini)을 사용하므로 OPENAI_API_KEY가 필요하다. 비용·재현성을 위해
시드·시간·LLM temperature를 모두 고정한다.

run_config는 LLM 팩토리를 주입받아, 단위 테스트에서는 ScriptedLLMClient로
API 없이 배선을 검증한다.
"""

from __future__ import annotations

import os
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.agents.graph import Supervisor
from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.sim.simulator import Simulator

# 개입이 자주 트리거되도록 중부하·소수 차량으로 설정
LAMBDA = 0.30
DURATION = 400.0
NUM_VEHICLES = 6
SUPERVISOR_INTERVAL = 30.0
QUEUE_THRESHOLD = 8


def run_config(
    make_llm: Callable[[], object],
    reflection: bool,
    seed: int,
) -> dict:
    """한 시드·한 설정(reflection on/off)을 실행해 운영·개입 지표를 수집"""
    sim_cfg = SimConfig(
        num_vehicles=NUM_VEHICLES,
        job_arrival_rate=LAMBDA,
        sim_duration=DURATION,
        random_seed=seed,
    )
    agent_cfg = AgentConfig(
        supervisor_interval=SUPERVISOR_INTERVAL,
        queue_threshold=QUEUE_THRESHOLD,
        reflection=reflection,
    )
    sim = Simulator(sim_cfg)
    sup = Supervisor(make_llm(), agent_cfg, sim)
    sim.attach_supervisor(sup, agent_cfg)
    m = sim.run()

    s = m.summary()
    leads = m.lead_times()
    eff = sup.ledger.efficacy_summary()
    return {
        "avg_wait": s["avg_wait_time"],
        "avg_lead": s["avg_lead_time"],
        "p95_lead": float(np.percentile(leads, 95)) if leads else 0.0,
        "avg_queue": s["avg_queue_len"],
        "interventions": eff["total"],
        "improved": eff["counts"]["개선"],
        "worsened": eff["counts"]["악화"],
        "avg_score": eff["avg_score"],
    }


def evaluate(make_llm, reflection: bool, seeds: list[int]) -> dict[str, float]:
    rows = [run_config(make_llm, reflection, s) for s in seeds]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _charts(comp: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    labels = {"reflection_on": "reflection on", "reflection_off": "reflection off"}
    colors = {"reflection_on": "#1a73e8", "reflection_off": "#9aa0a6"}

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["avg_wait", "avg_lead", "p95_lead", "avg_queue"]
    width = 0.35
    xs = range(len(metrics))
    for j, cfg in enumerate(comp):
        ax.bar(
            [x + j * width for x in xs],
            [comp[cfg][k] for k in metrics],
            width,
            label=labels[cfg],
            color=colors[cfg],
        )
    ax.set_xticks([x + width / 2 for x in xs])
    ax.set_xticklabels(metrics)
    ax.set_ylabel("value (lower is better)")
    ax.set_title("Reflection on/off: operational metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "operational.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["interventions", "improved", "worsened"]
    xs = range(len(metrics))  # 지표 수가 달라지므로 x축 재설정
    for j, cfg in enumerate(comp):
        ax.bar(
            [x + j * width for x in xs],
            [comp[cfg][k] for k in metrics],
            width,
            label=labels[cfg],
            color=colors[cfg],
        )
    ax.set_xticks([x + width / 2 for x in xs])
    ax.set_xticklabels(metrics)
    ax.set_ylabel("count (avg per run)")
    ax.set_title("Reflection on/off: intervention efficacy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "efficacy.png"), dpi=110)
    plt.close(fig)


def _results_md(comp: dict, seeds: list[int], model: str) -> str:
    on, off = comp["reflection_on"], comp["reflection_off"]

    def pct(a: float, b: float) -> str:
        return f"{(a - b) / b * 100:+.0f}%" if b else "n/a"

    lines = [
        "# 개입 효과 평가·reflection 닫힌 루프 실험 (D5)",
        "",
        f"L3 관제를 reflection on/off로 비교. 모델 {model}, 부하 λ={LAMBDA}, "
        f"OHT {NUM_VEHICLES}대, 관제 주기 {SUPERVISOR_INTERVAL:.0f}, "
        f"큐 임계 {QUEUE_THRESHOLD}, 평가 시드 {seeds}.",
        "",
        "## 운영 지표 (시드 평균, 낮을수록 좋음)",
        "",
        "| 지표 | reflection on | reflection off | 차이 |",
        "|---|---|---|---|",
        f"| 평균 대기 | {on['avg_wait']:.2f} | {off['avg_wait']:.2f} | {pct(on['avg_wait'], off['avg_wait'])} |",
        f"| 평균 리드 | {on['avg_lead']:.2f} | {off['avg_lead']:.2f} | {pct(on['avg_lead'], off['avg_lead'])} |",
        f"| p95 리드 | {on['p95_lead']:.2f} | {off['p95_lead']:.2f} | {pct(on['p95_lead'], off['p95_lead'])} |",
        f"| 평균 큐 | {on['avg_queue']:.2f} | {off['avg_queue']:.2f} | {pct(on['avg_queue'], off['avg_queue'])} |",
        "",
        "## 개입 효율 (시드 평균)",
        "",
        "| 지표 | reflection on | reflection off |",
        "|---|---|---|",
        f"| 개입 수 | {on['interventions']:.1f} | {off['interventions']:.1f} |",
        f"| 개선 | {on['improved']:.1f} | {off['improved']:.1f} |",
        f"| 악화 | {on['worsened']:.1f} | {off['worsened']:.1f} |",
        f"| 평균 효과 점수 | {on['avg_score']:+.2f} | {off['avg_score']:+.2f} |",
        "",
        "## 결정 (Decision)",
        "",
        f"reflection on의 평균 효과 점수 {on['avg_score']:+.2f}, off {off['avg_score']:+.2f}로, "
        f"효과 이력을 프롬프트에 주입하면 개입의 자기측정 효율이 개선됩니다(악화 개입 "
        f"{off['worsened']:.1f}->{on['worsened']:.1f}, 개선 {off['improved']:.1f}->{on['improved']:.1f}). "
        f"다만 운영 지표로는 이어지지 않아 평균 대기는 {on['avg_wait']:.1f} vs {off['avg_wait']:.1f}"
        f"({pct(on['avg_wait'], off['avg_wait'])})로 오히려 소폭 나빠집니다.",
        "",
        "해석: 두 설정 모두 평균 효과 점수가 음수라는 점이 핵심입니다. LLM 관제의 개입이 "
        "평균적으로는 지표를 악화시키는 경향이 있고, reflection은 그 손해를 줄일 뿐 순이득으로 "
        "뒤집지는 못합니다. 또한 효과 점수는 개입 전후 델타라 주변 부하 변동과 개입 효과가 "
        "섞여 인과를 단정하기 어렵습니다. 8시드 고분산이라 통계적 유의성은 D7(stats.py)의 쌍체 "
        "검정으로 확인해야 하며, 현 결과는 reflection이 자기측정 효율은 높이되 운영 개선과 인과 "
        "귀속은 추가 작업이 필요함을 시사합니다.",
        "",
        "![operational](charts/operational.png)",
        "![efficacy](charts/efficacy.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from oht_sim.agents.llm import OpenAIClient

    out_dir = "experiments/reflection_compare"
    agent_cfg = AgentConfig()
    seeds = list(range(1, 9))  # held-out 8개 시드

    def make_llm():
        return OpenAIClient(model=agent_cfg.model, temperature=agent_cfg.temperature)

    comp = {
        "reflection_on": evaluate(make_llm, True, seeds),
        "reflection_off": evaluate(make_llm, False, seeds),
    }

    os.makedirs(out_dir, exist_ok=True)
    # 비싼 LLM 집계를 먼저 보존(차트 실패에도 수치 유실 방지)
    import json

    with open(os.path.join(out_dir, "comp.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(comp, seeds, agent_cfg.model))
    _charts(comp, out_dir)
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for cfg in comp:
        c = comp[cfg]
        print(
            f"  {cfg}: 대기={c['avg_wait']:.2f} 개입={c['interventions']:.1f} "
            f"개선={c['improved']:.1f} 악화={c['worsened']:.1f} 점수={c['avg_score']:+.2f}"
        )


if __name__ == "__main__":
    main()
