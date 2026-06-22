"""mapf 단위 테스트 - 충돌·교착 회피 검증 (§6.4)"""

import bisect

from oht_sim.algorithms.mapf import ReservationTable, plan_route
from oht_sim.core.config import SimConfig
from oht_sim.core.events import EventType
from oht_sim.core.layout import Grid
from oht_sim.sim.simulator import Simulator


# ----- ReservationTable -----

def test_reservation_cell_free_and_reserve():
    rt = ReservationTable()
    assert rt.cell_free((1, 1), 5, vid=0)
    rt.reserve((1, 1), 5, vid=1)
    assert not rt.cell_free((1, 1), 5, vid=0)
    assert rt.cell_free((1, 1), 5, vid=1)  # 본인 예약은 자유
    assert rt.cell_free((1, 1), 6, vid=0)  # 다른 시각은 자유


def test_reservation_edge_blocks_swap():
    rt = ReservationTable()
    rt.reserve((1, 1), 5, vid=1, frm=(0, 1))  # (0,1)→(1,1)
    # 반대 방향 (1,1)→(0,1) 동시각 이동은 막힘
    assert not rt.edge_free((1, 1), (0, 1), 5, vid=2)
    assert rt.edge_free((1, 1), (0, 1), 6, vid=2)


def test_reservation_prune_before():
    rt = ReservationTable()
    rt.reserve((0, 0), 1, vid=0)
    rt.reserve((0, 0), 9, vid=0)
    rt.prune_before(5)
    assert rt.cell_free((0, 0), 1, vid=1)  # 과거 예약 제거됨
    assert not rt.cell_free((0, 0), 9, vid=1)


# ----- plan_route (A*) -----

def test_plan_route_straight():
    grid = Grid(10, 10)
    path = plan_route(grid, (0, 0), (0, 3))
    assert path[0] == (0, 0) and path[-1] == (0, 3)
    assert len(path) == 4


def test_plan_route_avoids_obstacle():
    # (1,0),(1,1),(1,2) 벽 -> 우회 필요
    grid = Grid(5, 5, blocked={(1, 0), (1, 1), (1, 2)})
    path = plan_route(grid, (0, 0), (2, 0))
    assert path[0] == (0, 0) and path[-1] == (2, 0)
    assert all(c not in grid.blocked for c in path)
    assert len(path) > 3  # 직선(3)보다 길어짐


def test_plan_route_blocked_cells_param():
    grid = Grid(5, 5)
    # 다른 차량이 (1,0),(1,1)에 정지 -> 우회
    path = plan_route(grid, (0, 0), (2, 0), blocked={(1, 0), (1, 1)})
    assert path[-1] == (2, 0)
    assert (1, 0) not in path


def test_plan_route_unreachable_returns_start():
    grid = Grid(5, 5, blocked={(1, 0), (0, 1)})  # (0,0) 고립
    path = plan_route(grid, (0, 0), (4, 4))
    assert path == [(0, 0)]


# ----- 통합: 충돌·교착 회피 -----

def _collisions(sim, cfg) -> tuple[int, int]:
    """이벤트 로그로 틱별 위치를 복원해 (동일셀 충돌, 스왑) 횟수 산출"""
    mt = cfg.move_time_per_cell
    tl = {v.id: [(0.0, sim.initial_positions[v.id])] for v in sim.vehicles}
    for e in sim.bus.log:
        if e.type == EventType.MOVE:
            tl[e.vehicle_id].append((e.time, e.location))
    for vid in tl:
        tl[vid].sort()

    def pos_at(vid, t):
        times = [x[0] for x in tl[vid]]
        return tl[vid][bisect.bisect_right(times, t) - 1][1]

    steps = int(cfg.sim_duration / mt)
    coll = swap = 0
    prev = None
    for k in range(steps + 1):
        cur = {vid: pos_at(vid, k * mt) for vid in tl}
        seen: dict = {}
        for vid, p in cur.items():
            if p in seen:
                coll += 1
            seen[p] = vid
        if prev is not None:
            for a in tl:
                for b in tl:
                    if a < b and prev[a] == cur[b] and prev[b] == cur[a] and prev[a] != cur[a]:
                        swap += 1
        prev = cur
    return coll, swap


def test_no_collision_across_seeds():
    for seed in (1, 7, 42):
        for lam in (0.2, 0.5):
            cfg = SimConfig(
                num_vehicles=8, job_arrival_rate=lam,
                sim_duration=300, random_seed=seed,
            )
            sim = Simulator(cfg)
            sim.run()
            coll, swap = _collisions(sim, cfg)
            assert coll == 0, f"seed={seed} λ={lam} 충돌 {coll}"
            assert swap == 0, f"seed={seed} λ={lam} 스왑 {swap}"


def test_deadlock_detected_and_recovered():
    # 차량을 빽빽이(좁은 영역) 두어 정면 교착을 유발, 그래도 작업이 완료되는지(회복) 확인
    cfg = SimConfig(
        grid_width=6, grid_height=6, num_vehicles=6, num_stations=6,
        job_arrival_rate=0.4, sim_duration=400, random_seed=3,
    )
    sim = Simulator(cfg)
    metrics = sim.run()
    s = metrics.summary()
    coll, swap = _collisions(sim, cfg)
    assert coll == 0 and swap == 0  # 좁은 영역에서도 충돌 없음
    assert s["deadlocks"] > 0  # 교착이 실제로 감지됨
    assert s["completed_jobs"] > 0  # 멈추지 않고 회복하여 작업 완료


def test_collision_avoidance_changes_metrics_vs_l1():
    # L1(충돌 무시) 대비 L2(회피)는 회피 대기가 생기고 처리량이 낮아짐 (현실성 반영)
    base = dict(num_vehicles=8, job_arrival_rate=0.4, sim_duration=400, random_seed=11)
    l1 = Simulator(SimConfig(collision_avoidance=False, **base)).run().summary()
    l2 = Simulator(SimConfig(collision_avoidance=True, **base)).run().summary()
    assert l1["avoidance_waits"] == 0
    assert l2["avoidance_waits"] > 0
    assert l2["throughput"] <= l1["throughput"]
