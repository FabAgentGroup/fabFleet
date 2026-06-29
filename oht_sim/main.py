"""진입점 - 시나리오 구성·시뮬레이션 실행·지표 출력"""

from __future__ import annotations

import argparse

from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.sim.simulator import Simulator


def build_config(args: argparse.Namespace) -> SimConfig:
    cfg = SimConfig()
    if args.vehicles is not None:
        cfg.num_vehicles = args.vehicles
    if args.arrival_rate is not None:
        cfg.job_arrival_rate = args.arrival_rate
    if args.duration is not None:
        cfg.sim_duration = args.duration
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.failure:
        cfg.vehicle_failure = True
    if args.congestion:
        cfg.congestion_aware_routing = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="OHT 반송 시뮬레이터 (Layer 1)")
    parser.add_argument("--vehicles", type=int, default=None, help="OHT 수")
    parser.add_argument("--arrival-rate", type=float, default=None, help="작업 도착률 λ")
    parser.add_argument("--duration", type=float, default=None, help="시뮬레이션 시간")
    parser.add_argument("--seed", type=int, default=None, help="난수 시드")
    parser.add_argument("--gif", action="store_true", help="애니메이션 GIF 저장")
    parser.add_argument(
        "--failure", action="store_true", help="OHT 확률적 고장·수리 활성화 (L1 신뢰성)"
    )
    parser.add_argument(
        "--congestion", action="store_true", help="혼잡 인지 동적 라우팅 활성화 (L2)"
    )
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="Layer 3 LLM 관제 활성화 (OPENAI_API_KEY 필요)",
    )
    args = parser.parse_args()

    cfg = build_config(args)
    sim = Simulator(cfg)

    if args.supervise:
        from oht_sim.agents.graph import Supervisor
        from oht_sim.agents.llm import OpenAIClient

        agent_cfg = AgentConfig()
        llm = OpenAIClient(model=agent_cfg.model, temperature=agent_cfg.temperature)
        sim.attach_supervisor(Supervisor(llm, agent_cfg, sim), agent_cfg)

    metrics = sim.run()
    metrics.print_summary()

    if args.supervise and sim.supervisor is not None:
        triggered = [e for e in sim.supervisor.timeline if e["anomalies"]]
        print(f"\n관제 그래프 실행 {len(sim.supervisor.timeline)}회, 개입 트리거 {len(triggered)}회")
        for e in triggered[:5]:
            act = e["response"]["decision"].get("action") if e.get("response") else None
            eff = e.get("effect")
            tail = f" -> 효과 {eff['label']}(점수 {eff['effect_score']:+.1f})" if eff else ""
            print(f"  t={e['time']:.0f} 이상={[a['type'] for a in e['anomalies']]} -> 액션={act}{tail}")

        summary = sim.supervisor.ledger.efficacy_summary()
        if summary["total"]:
            c = summary["counts"]
            print(
                f"\n개입 효과: 평가 {summary['total']}건 "
                f"(개선 {c['개선']}·변화없음 {c['변화없음']}·악화 {c['악화']}), "
                f"평균 점수 {summary['avg_score']:+.2f}"
            )

    if args.gif:
        from oht_sim.viz.visualize import animate

        path = animate(sim)
        print(f"\n애니메이션 저장: {path}")


if __name__ == "__main__":
    main()
