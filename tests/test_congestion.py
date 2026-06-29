"""혼잡 인지 동적 라우팅 테스트 (L2, D10)"""

from oht_sim.algorithms.congestion import CongestionField
from oht_sim.algorithms.mapf import plan_route
from oht_sim.core.config import SimConfig
from oht_sim.core.events import EventType
from oht_sim.core.layout import Grid
from oht_sim.sim.simulator import Simulator


# ----- 혼잡장 -----

def test_field_accumulates_and_weights_blocked_more():
    f = CongestionField(decay=0.6, alpha=2.0, blocked_weight=3.0)
    f.observe(moves={(1, 1): 2}, blocks={(2, 2): 1})
    # 회피 대기 1건(가중 3) > 통행 2건
    assert f.penalty((2, 2)) > f.penalty((1, 1)) > 0


def test_field_decays_over_time():
    f = CongestionField(decay=0.5, alpha=1.0, blocked_weight=1.0, prune_eps=0.0)
    f.observe(moves={(0, 0): 4}, blocks={})
    w1 = f.penalty((0, 0))
    f.observe(moves={}, blocks={})  # 유입 없음 -> 감쇠
    assert 0 < f.penalty((0, 0)) < w1


def test_field_prunes_negligible():
    f = CongestionField(decay=0.1, alpha=1.0, prune_eps=0.05)
    f.observe(moves={(0, 0): 1}, blocks={})
    for _ in range(5):
        f.observe(moves={}, blocks={})
    assert (0, 0) not in f.weights  # 감쇠로 소멸


# ----- 가중 라우팅 -----

def test_plan_route_avoids_congested_cells():
    grid = Grid(5, 1, set())  # 1차원 통로 -> 우회 불가, 비용만 증가
    base = plan_route(grid, (0, 0), (4, 0))
    weighted = plan_route(grid, (0, 0), (4, 0), cost_fn=lambda c: 100.0 if c == (2, 0) else 0.0)
    assert base == weighted  # 우회로 없으면 경로 동일(비용만 다름)


def test_plan_route_detours_around_congestion():
    grid = Grid(3, 3, set())
    # (1,0) 강한 혼잡 -> 같은 길이의 대안 경로로 우회
    cost = lambda c: 50.0 if c == (1, 0) else 0.0
    path = plan_route(grid, (0, 0), (2, 0), cost_fn=cost)
    assert (1, 0) not in path  # 혼잡 셀 회피
    assert path[0] == (0, 0) and path[-1] == (2, 0)


def test_plan_route_static_unchanged_without_cost_fn():
    grid = Grid(6, 6, {(2, 2), (3, 3)})
    a = plan_route(grid, (0, 0), (5, 5))
    b = plan_route(grid, (0, 0), (5, 5), cost_fn=None)
    assert a == b  # cost_fn 없으면 기존 정적 경로와 동일


# ----- 시뮬레이터 통합 -----

def test_congestion_off_by_default():
    sim = Simulator(SimConfig(num_vehicles=4, sim_duration=1))
    assert sim.congestion is None


def test_congestion_reduces_blocked_high_density():
    common = dict(num_vehicles=14, job_arrival_rate=0.5, sim_duration=400, random_seed=3)
    def blocked(cong):
        sim = Simulator(SimConfig(congestion_aware_routing=cong, **common))
        sim.run()
        return sum(1 for e in sim.bus.log if e.type == EventType.BLOCKED)
    assert blocked(True) < blocked(False)  # 혼잡 인지가 회피 대기를 줄임
