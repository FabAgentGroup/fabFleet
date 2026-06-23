"""예측 기반 사전 배차 테스트 (forecast·hotspot 수요·D2 효과)"""

import statistics

from oht_sim.algorithms.forecast import DemandForecaster
from oht_sim.core.config import SimConfig
from oht_sim.core.layout import zone_of
from oht_sim.sim.simulator import Simulator


# ----- DemandForecaster -----

def test_forecaster_ema_tracks_observations():
    f = DemandForecaster(zones=2, alpha=0.5)
    for _ in range(4):
        f.observe((0, 0))
    f.end_window()
    assert f.ema[(0, 0)] == 2.0  # 0.5*4 + 0.5*0
    assert f.hottest() == (0, 0)


def test_forecaster_empty_has_no_hottest():
    f = DemandForecaster(zones=2)
    f.end_window()
    assert f.hottest() is None


# ----- 핫스팟 수요 -----

def test_hotspot_demand_skews_src_to_hot_zone():
    cfg = SimConfig(
        num_vehicles=4, job_arrival_rate=0.5, sim_duration=200, random_seed=1,
        demand_hotspot=True, demand_zones=2, hotspot_period=10_000, hotspot_weight=0.8,
    )
    sim = Simulator(cfg)
    gen = sim.job_gen
    hot = gen.hot_zone(0.0)
    jobs = [gen.create(50.0) for _ in range(200)]
    in_hot = sum(
        1 for j in jobs
        if zone_of(j.src.coord, cfg.grid_width, cfg.grid_height, cfg.demand_zones) == hot
    )
    assert in_hot / len(jobs) > 0.5  # 출발지가 핫존으로 쏠림


# ----- D2 효과 (중부하 sweet spot) -----

def _wait(predict: bool, seed: int) -> float:
    cfg = SimConfig(
        num_vehicles=8, job_arrival_rate=0.20, sim_duration=600, random_seed=seed,
        demand_hotspot=True, demand_zones=2, hotspot_period=300, hotspot_weight=0.75,
        predictive_dispatch=predict, forecast_interval=25, predict_reposition_k=2,
    )
    return Simulator(cfg).run().summary()["avg_wait_time"]


def test_predictive_reduces_wait_at_sweet_spot():
    seeds = [1, 2, 3]
    base = statistics.mean(_wait(False, s) for s in seeds)
    pred = statistics.mean(_wait(True, s) for s in seeds)
    assert pred < base  # 중부하 핫스팟에서 예측 선제 배치가 대기 단축


def test_predictive_reproducible():
    assert _wait(True, 7) == _wait(True, 7)  # 시드 고정 -> 결정적
