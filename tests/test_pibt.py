"""PIBT 이동 계획 테스트 (lifelong MAPF, D12)"""

from oht_sim.algorithms.pibt import PIBTPlanner
from oht_sim.core.config import SimConfig
from oht_sim.core.events import EventType
from oht_sim.core.layout import Grid
from oht_sim.sim.simulator import Simulator


def _step(grid, pos, goal, movable, order):
    nxt = PIBTPlanner(grid, dict(pos), dict(goal), set(movable)).solve(order)
    cells = list(nxt.values())
    assert len(cells) == len(set(cells)), f"정점 충돌 {nxt}"
    for a in nxt:
        for b in nxt:
            if a < b and nxt[a] == pos.get(b) and nxt[b] == pos.get(a):
                raise AssertionError(f"스왑 충돌 {nxt}")
    return nxt


# ----- 플래너 -----

def test_pibt_no_conflict_and_progress():
    g = Grid(5, 3)
    pos = {0: (0, 1), 1: (4, 1)}
    goal = {0: (4, 1), 1: (0, 1)}
    for _ in range(20):
        nxt = _step(g, pos, goal, {0, 1}, [0, 1])
        pos = dict(nxt)
        if pos[0] == goal[0] and pos[1] == goal[1]:
            break
    assert pos[0] == (4, 1) and pos[1] == (0, 1)  # 교차 후 둘 다 도달


def test_pibt_pushes_idle_obstacle():
    g = Grid(3, 3)
    pos = {0: (0, 0), 1: (1, 0)}
    goal = {0: (2, 0), 1: (1, 0)}  # 1은 유휴(goal=현위치) -> 밀림 가능
    nxt = _step(g, pos, goal, {0, 1}, [0])
    assert nxt[0] != (0, 0)  # 0 전진
    assert nxt[1] != (1, 0)  # 유휴 1 비켜섬


def test_pibt_hard_obstacle_not_pushed():
    g = Grid(3, 1)  # 1행 통로, 우회 불가
    pos = {0: (0, 0), 1: (1, 0)}
    goal = {0: (2, 0), 1: (1, 0)}
    nxt = _step(g, pos, goal, {0}, [0])  # 1은 movable 아님(고정 장애물)
    assert nxt[0] == (0, 0)  # 밀 수 없어 대기


def test_pibt_higher_priority_first():
    g = Grid(3, 1)
    pos = {0: (0, 0), 1: (2, 0)}
    goal = {0: (2, 0), 1: (0, 0)}  # 정면 (1행 통로)
    nxt = _step(g, pos, goal, {0, 1}, [0, 1])
    # 고우선 0이 가운데로 전진, 저우선 1은 양보(대기) - 스왑·정점 충돌 없음
    assert nxt[0] == (1, 0) and nxt[1] == (2, 0)


def test_pibt_headon_swap_blocked():
    g = Grid(2, 1)
    pos = {0: (0, 0), 1: (1, 0)}
    goal = {0: (1, 0), 1: (0, 0)}  # 인접 정면 -> 스왑 불가, 둘 다 대기
    nxt = _step(g, pos, goal, {0, 1}, [0, 1])
    assert nxt[0] == (0, 0) and nxt[1] == (1, 0)


# ----- 시뮬레이터 통합 -----

def test_pibt_run_no_overlap_and_completes():
    sim = Simulator(SimConfig(num_vehicles=12, job_arrival_rate=0.5, sim_duration=400,
                              random_seed=3, pibt_planning=True))
    m = sim.run()
    assert m.summary()["completed_jobs"] > 0
    poss = [v.pos for v in sim.vehicles if not v.is_failed]
    assert len(poss) == len(set(poss))  # 최종 위치 충돌 없음


def test_pibt_reduces_deadlock_vs_windowed():
    common = dict(num_vehicles=16, job_arrival_rate=0.5, sim_duration=400, random_seed=3)
    def deadlocks(pibt):
        sim = Simulator(SimConfig(pibt_planning=pibt, **common))
        sim.run()
        return sum(1 for e in sim.bus.log if e.type == EventType.DEADLOCK_DETECTED)
    assert deadlocks(True) < deadlocks(False)  # PIBT가 교착 대폭 감소


def test_pibt_off_by_default():
    assert SimConfig().pibt_planning is False


# ----- 혼잡 결합 (D15) -----

def test_pibt_cost_fn_breaks_ties_toward_low_congestion():
    # (1,0)·(0,1) 둘 다 목표 (1,1)에 같은 거리. (1,0)이 혼잡하면 (0,1) 선택
    g = Grid(2, 2)
    pos = {0: (0, 0)}
    goal = {0: (1, 1)}
    cost = lambda c: 9.0 if c == (1, 0) else 0.0
    nxt = PIBTPlanner(g, pos, goal, {0}, cost_fn=cost).solve([0])
    assert nxt[0] == (0, 1)  # 혼잡한 (1,0) 대신 동거리 (0,1)


def test_pibt_with_congestion_runs():
    sim = Simulator(SimConfig(num_vehicles=14, job_arrival_rate=0.5, sim_duration=300,
                              random_seed=3, pibt_planning=True, congestion_aware_routing=True))
    m = sim.run()
    assert sim.congestion is not None  # 혼잡장 구성됨
    assert m.summary()["completed_jobs"] > 0
