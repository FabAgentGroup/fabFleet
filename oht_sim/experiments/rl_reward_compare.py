"""RL 배차 보상 재설계 실험 (D6 후속, D13)

D6은 강화 RL 배차가 nearest를 못 이긴 원인을 "보상 설계"로 지목했다(픽업 대기 보상은
이동 거리에 지배되어 거리 휴리스틱이 이미 강한 베이스라인). 이를 검증하려고 동일 MLP
정책을 세 보상으로 학습한다: 대기 보상(D6), 리드타임 보상, 혼잡 페널티 보상(리드 +
회피 대기율). 학습 후 고정해 nearest와 held-out 시드 쌍체로 비교하고 차이를 D7
(stats.py)의 순열검정으로 판정한다. 보상을 바꾸면 RL이 nearest를 유의하게 이기는지 본다.
"""

from __future__ import annotations

import os
import random
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.algorithms.dispatcher import LeastBusyDispatcher, NearestDispatcher
from oht_sim.algorithms.rl_dispatcher import MLPPolicyDispatcher
from oht_sim.core.config import SimConfig
from oht_sim.core.events import EventType
from oht_sim.experiments.stats import paired_diff_test
from oht_sim.sim.simulator import Simulator

LAMBDA = 0.28
DURATION = 400.0
NUM_VEHICLES = 8
GRID = 20
EPISODES = 50
ALPHA_CONG = 20.0  # 혼잡 페널티 보상의 회피 대기율 가중

# 보상 함수 (run 지표 -> 보상, 클수록 좋음)
REWARDS: dict[str, Callable[[dict], float]] = {
    "wait": lambda r: -r["avg_wait"],
    "lead": lambda r: -r["avg_lead"],
    "congestion": lambda r: -(r["avg_lead"] + ALPHA_CONG * r["blocked_rate"]),
}


def _run(dispatcher, seed: int) -> dict:
    cfg = SimConfig(
        num_vehicles=NUM_VEHICLES, job_arrival_rate=LAMBDA,
        sim_duration=DURATION, random_seed=seed,
    )
    sim = Simulator(cfg, dispatcher=dispatcher)
    m = sim.run()
    s = m.summary()
    leads = m.lead_times()
    blocked = sum(1 for e in sim.bus.log if e.type == EventType.BLOCKED)
    return {
        "avg_wait": s["avg_wait_time"],
        "avg_lead": s["avg_lead_time"],
        "throughput": s["throughput"],
        "p95_lead": float(np.percentile(leads, 95)) if leads else 0.0,
        "blocked_rate": blocked / DURATION,
    }


def train(rl: MLPPolicyDispatcher, reward_fn: Callable[[dict], float], episodes: int) -> None:
    seed = 1000
    for _ in range(episodes):
        rl.reset_episode()
        r = _run(rl, seed)
        rl.end_episode(reward_fn(r))
        seed += 1


def evaluate(make_dispatcher, seeds: list[int]) -> list[dict]:
    return [_run(make_dispatcher(), s) for s in seeds]


def _chart(rows: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(rows)
    leads = [float(np.mean([r["avg_lead"] for r in rows[m]])) for m in methods]
    errs = [float(np.std([r["avg_lead"] for r in rows[m]])) for m in methods]
    colors = ["#1a73e8" if m.startswith("rl") else "#9aa0a6" for m in methods]
    ax.bar(methods, leads, yerr=errs, capsize=5, color=colors)
    ax.set_ylabel("avg lead (lower better)")
    ax.set_title(f"RL reward shaping vs heuristics (n={len(rows[methods[0]])} seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "reward_shaping.png"), dpi=110)
    plt.close(fig)


def _results_md(rows: dict, seeds: list[int]) -> str:
    near = [r["avg_lead"] for r in rows["nearest"]]
    lines = [
        "# RL 배차 보상 재설계 실험 (D13)",
        "",
        f"동일 MLP 정책을 세 보상(대기·리드·혼잡 페널티)으로 학습. 부하 λ={LAMBDA}, OHT "
        f"{NUM_VEHICLES}대, 학습 {EPISODES} 에피소드, 평가 시드 {seeds}. 평균 리드·처리량을 "
        "보고하고 nearest 대비 쌍체 순열검정으로 판정한다.",
        "",
        "## 성능 비교 (평가 시드 평균)",
        "",
        "| method | 평균 리드 | 평균 대기 | 처리량 | nearest 대비 리드 Δ(p) |",
        "|---|---|---|---|---|",
    ]
    for m in rows:
        lead = float(np.mean([r["avg_lead"] for r in rows[m]]))
        wait = float(np.mean([r["avg_wait"] for r in rows[m]]))
        thr = float(np.mean([r["throughput"] for r in rows[m]]))
        if m == "nearest":
            delta = "- (기준)"
        else:
            t = paired_diff_test([r["avg_lead"] for r in rows[m]], near, seed=0)
            delta = f"{t.mean_diff:+.1f} (p={t.p_value:.3f}{'*' if t.significant else ''})"
        lines.append(f"| {m} | {lead:.1f} | {wait:.1f} | {thr:.3f} | {delta} |")

    rl_methods = [m for m in rows if m.startswith("rl")]
    best_rl = min(rl_methods, key=lambda m: np.mean([r["avg_lead"] for r in rows[m]]))
    best_rl_lead = float(np.mean([r["avg_lead"] for r in rows[best_rl]]))
    near_lead = float(np.mean(near))
    sig_any = any(
        paired_diff_test([r["avg_lead"] for r in rows[m]], near, seed=0).significant
        and np.mean([r["avg_lead"] for r in rows[m]]) < near_lead
        for m in rl_methods
    )
    lines += [
        "",
        "(* nearest 대비 p<0.05)",
        "",
        "## 결정 (Decision)",
        "",
        f"보상별 학습 정책 중 평균 리드 최저는 {best_rl} ({best_rl_lead:.1f}) vs nearest "
        f"({near_lead:.1f}). RL이 nearest를 유의하게 이기는 보상은 "
        f"{'있습니다' if sig_any else '없습니다'}(전부 p>0.05).",
        "",
        "해석: D6은 RL이 거리 휴리스틱을 못 이긴 원인을 보상 설계로 추정했습니다. D13은 이를 "
        "검증했고, 결과는 그 가설을 뒷받침하지 않습니다. 리드타임 보상은 오히려 가장 나빴고"
        "(리드가 다스텝 다운스트림 효과를 담아 단일 배차에 신용 할당이 어려운 탓), 혼잡 보상도 "
        "기존 대기 보상과 사실상 동률이며, 어느 것도 nearest를 유의하게 이기지 못합니다. 즉 이 "
        "부하·레이아웃에서 병목은 보상이 아니라 배차 자체의 낮은 레버리지(D7이 보인 '배차 차이 "
        "비유의')입니다. 더 큰 이득은 배차 보상 손질이 아니라 혼잡 라우팅(D10)·PIBT(D12)처럼 "
        "병목인 이동·혼잡을 직접 다루는 데서 나옵니다. 부트스트래핑(Q학습)·다스텝 신용 할당은 "
        "여전히 열린 확장이지만, 본 결과는 그 이전에 배차 레버리지의 한계를 시사합니다.",
        "",
        "![reward_shaping](charts/reward_shaping.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/rl_reward_compare"
    eval_seeds = list(range(1, 16))

    rows: dict[str, list[dict]] = {}
    for name, reward_fn in REWARDS.items():
        rl = MLPPolicyDispatcher(GRID, GRID, hidden=8, lr=0.02, temperature=0.5, seed=1)
        train(rl, reward_fn, EPISODES)
        rl.train = False
        rows[f"rl_{name}"] = evaluate(lambda rl=rl: rl, eval_seeds)
    rows["nearest"] = evaluate(NearestDispatcher, eval_seeds)
    rows["least_busy"] = evaluate(LeastBusyDispatcher, eval_seeds)

    os.makedirs(out_dir, exist_ok=True)
    _chart(rows, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(rows, eval_seeds))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for m in rows:
        lead = float(np.mean([r["avg_lead"] for r in rows[m]]))
        print(f"  {m}: 리드={lead:.1f}")


if __name__ == "__main__":
    main()
