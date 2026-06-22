"""정책 비교 실험 러너 (백테스트식)

동일 시나리오·시드에서 배차 정책을 다수 부하(λ)로 비교한다. 평균뿐 아니라
표준편차(분산)와 p95 리드타임(꼬리 위험)까지 산출해 results.md와 차트로 남긴다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oht_sim.algorithms.dispatcher import LeastBusyDispatcher, NearestDispatcher
from oht_sim.core.config import SimConfig
from oht_sim.sim.simulator import Simulator

POLICIES = {"nearest": NearestDispatcher, "least_busy": LeastBusyDispatcher}
_METRICS = ["throughput", "avg_lead", "p95_lead", "avg_queue", "max_queue", "deadlocks"]


@dataclass
class RunResult:
    policy: str
    lam: float
    seed: int
    values: dict[str, float]


def _run_one(policy: str, lam: float, seed: int, num_vehicles: int, duration: float) -> dict[str, float]:
    cfg = SimConfig(
        num_vehicles=num_vehicles,
        job_arrival_rate=lam,
        sim_duration=duration,
        random_seed=seed,
    )
    sim = Simulator(cfg, dispatcher=POLICIES[policy]())
    m = sim.run()
    s = m.summary()
    leads = m.lead_times()
    return {
        "throughput": s["throughput"],
        "avg_lead": s["avg_lead_time"],
        "p95_lead": float(np.percentile(leads, 95)) if leads else 0.0,
        "avg_queue": s["avg_queue_len"],
        "max_queue": s["max_queue_len"],
        "deadlocks": float(s["deadlocks"]),
    }


def run(
    lambdas: list[float] | None = None,
    seeds: list[int] | None = None,
    num_vehicles: int = 8,
    duration: float = 500.0,
) -> list[RunResult]:
    lambdas = lambdas or [0.15, 0.20, 0.25, 0.30, 0.35]
    seeds = seeds or [1, 2, 3, 4, 5]
    results: list[RunResult] = []
    for policy in POLICIES:
        for lam in lambdas:
            for seed in seeds:
                vals = _run_one(policy, lam, seed, num_vehicles, duration)
                results.append(RunResult(policy, lam, seed, vals))
    return results


def _agg(results: list[RunResult]) -> dict[tuple[str, float], dict[str, tuple[float, float]]]:
    """정책 x λ 별 지표 평균·표준편차 집계"""
    table: dict[tuple[str, float], dict[str, tuple[float, float]]] = {}
    keys = sorted({(r.policy, r.lam) for r in results})
    for policy, lam in keys:
        rows = [r for r in results if r.policy == policy and r.lam == lam]
        table[(policy, lam)] = {
            m: (
                float(np.mean([r.values[m] for r in rows])),
                float(np.std([r.values[m] for r in rows])),
            )
            for m in _METRICS
        }
    return table


def _charts(table, lambdas, out_dir: str) -> list[str]:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    policies = list(POLICIES)
    specs = [
        ("avg_lead", "Average lead time vs load", "avg lead time", "lead_time.png"),
        ("p95_lead", "Tail risk (p95 lead time) vs load", "p95 lead time", "tail_p95.png"),
        ("throughput", "Throughput vs load", "throughput", "throughput.png"),
    ]
    paths = []
    for metric, title, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for policy in policies:
            means = [table[(policy, lam)][metric][0] for lam in lambdas]
            stds = [table[(policy, lam)][metric][1] for lam in lambdas]
            ax.errorbar(lambdas, means, yerr=stds, marker="o", capsize=3, label=policy)
        ax.set_xlabel("job arrival rate (lambda)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, "charts", fname)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def _results_md(table, lambdas, seeds, num_vehicles, duration) -> str:
    lines = [
        "# 정책 비교 실험 (D1) - 배차 정책",
        "",
        f"설정: OHT {num_vehicles}대, 격자 20x20, 충돌·교착 회피(L2) 적용, "
        f"시뮬레이션 {duration:.0f}, 시드 {seeds}(n={len(seeds)}).",
        "값은 시드 간 평균(괄호는 표준편차).",
        "",
    ]
    for metric, label in [
        ("avg_lead", "평균 리드타임"),
        ("p95_lead", "p95 리드타임 (꼬리 위험)"),
        ("throughput", "처리량"),
        ("max_queue", "최대 큐"),
        ("deadlocks", "교착 감지수"),
    ]:
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| λ | nearest | least_busy | 우세 |")
        lines.append("|---|---|---|---|")
        for lam in lambdas:
            nm, ns = table[("nearest", lam)][metric]
            lm, ls = table[("least_busy", lam)][metric]
            lower_better = metric != "throughput"
            if abs(nm - lm) < 1e-9:
                win = "="
            elif (nm < lm) == lower_better:
                win = "nearest"
            else:
                win = "least_busy"
            lines.append(f"| {lam:.2f} | {nm:.2f} ({ns:.2f}) | {lm:.2f} ({ls:.2f}) | {win} |")
        lines.append("")

    # 결정 서술 (데이터 기반)
    lead_wins = {"nearest": 0, "least_busy": 0}
    for lam in lambdas:
        nm = table[("nearest", lam)]["avg_lead"][0]
        lm = table[("least_busy", lam)]["avg_lead"][0]
        lead_wins["nearest" if nm < lm else "least_busy"] += 1
    overall = "least_busy" if lead_wins["least_busy"] > lead_wins["nearest"] else "nearest"
    lines += [
        "## 결정 (Decision)",
        "",
        f"평균 리드타임 기준 {len(lambdas)}개 부하 중 nearest {lead_wins['nearest']}회 / "
        f"least_busy {lead_wins['least_busy']}회 우세. 종합 우세 정책: **{overall}**.",
        "",
        "트레이드오프: least_busy는 부하 균형으로 특정 차량 쏠림을 줄이나, "
        "유휴 차량이 드문 고부하 영역에서는 선택 여지가 적어 효과가 줄어듭니다. "
        "nearest는 저부하에서 이동 거리를 줄여 유리할 수 있습니다. "
        "꼬리 위험(p95)은 위 표로 비교합니다.",
        "",
        "![lead time](charts/lead_time.png)",
        "![tail p95](charts/tail_p95.png)",
        "![throughput](charts/throughput.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    lambdas = [0.15, 0.20, 0.25, 0.30, 0.35]
    seeds = [1, 2, 3, 4, 5]
    num_vehicles, duration = 8, 500.0
    out_dir = "experiments/policy_compare"

    results = run(lambdas, seeds, num_vehicles, duration)
    table = _agg(results)
    os.makedirs(out_dir, exist_ok=True)
    _charts(table, lambdas, out_dir)
    md = _results_md(table, lambdas, seeds, num_vehicles, duration)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"실험 완료: {out_dir}/results.md + charts/ ({len(results)} runs)")


if __name__ == "__main__":
    main()
