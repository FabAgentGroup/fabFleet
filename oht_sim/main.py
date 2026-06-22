"""진입점 - 시나리오 구성·시뮬레이션 실행·지표 출력"""

from __future__ import annotations

import argparse

from oht_sim.core.config import SimConfig
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
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="OHT 반송 시뮬레이터 (Layer 1)")
    parser.add_argument("--vehicles", type=int, default=None, help="OHT 수")
    parser.add_argument("--arrival-rate", type=float, default=None, help="작업 도착률 λ")
    parser.add_argument("--duration", type=float, default=None, help="시뮬레이션 시간")
    parser.add_argument("--seed", type=int, default=None, help="난수 시드")
    parser.add_argument("--gif", action="store_true", help="애니메이션 GIF 저장")
    args = parser.parse_args()

    cfg = build_config(args)
    sim = Simulator(cfg)
    metrics = sim.run()
    metrics.print_summary()

    if args.gif:
        from oht_sim.viz.visualize import animate

        path = animate(sim)
        print(f"\n애니메이션 저장: {path}")


if __name__ == "__main__":
    main()
