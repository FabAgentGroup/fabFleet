"""방향성 레일 토폴로지 테스트 (OHT 모노레일, D16)"""

from collections import deque

from oht_sim.core.config import SimConfig
from oht_sim.core.layout import RailGraph, build_rail_graph
from oht_sim.sim.simulator import Simulator


def _bfs(start, succ):
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in succ(u):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


# ----- 레일 그래프 -----

def test_rail_graph_is_strongly_connected():
    g, stations = build_rail_graph(12, 10, num_bays=4)
    nodes = g.track
    radj = {n: [] for n in nodes}
    for u in nodes:
        for v in g.neighbors(u):
            radj[v].append(u)
    s = next(iter(nodes))
    assert _bfs(s, g.neighbors) == nodes  # 전 노드 도달
    assert _bfs(s, lambda u: radj[u]) == nodes  # 전 노드가 도달 -> 강연결


def test_rail_graph_is_directed_one_way():
    g, _ = build_rail_graph(12, 10, num_bays=4)
    bidir = sum(1 for u in g.track for v in g.neighbors(u) if u in g.neighbors(v))
    assert bidir == 0  # 양방향 엣지 없음(완전 단방향)


def test_rail_stations_on_track():
    g, stations = build_rail_graph(12, 10, num_bays=4)
    assert stations and all(g.passable(c) for c in stations)


def test_rail_blocked_excludes_neighbor():
    g, _ = build_rail_graph(8, 8, num_bays=2)
    node = next(u for u in g.track if g.neighbors(u))
    nb = g.neighbors(node)[0]
    g.blocked.add(nb)
    assert nb not in g.neighbors(node) and not g.passable(nb)


# ----- 시뮬레이터 통합 -----

def test_rail_sim_keeps_vehicles_on_track_and_completes():
    sim = Simulator(SimConfig(
        grid_width=14, grid_height=10, num_stations=10, num_vehicles=8,
        job_arrival_rate=0.3, sim_duration=400, random_seed=3, topology="rail",
    ))
    m = sim.run()
    assert isinstance(sim.grid, RailGraph)
    assert all(sim.grid.passable(v.pos) for v in sim.vehicles)  # 전 차량 레일 위
    assert m.summary()["completed_jobs"] > 0


def test_grid_topology_default_unchanged():
    sim = Simulator(SimConfig(sim_duration=1))
    from oht_sim.core.layout import Grid
    assert isinstance(sim.grid, Grid)  # 기본은 자유 격자(회귀 없음)
