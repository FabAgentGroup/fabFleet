"""예약테이블·충돌/교착 회피 (L2)

틱 동기식 우선순위 계획(Prioritized Planning)을 쓴다. 매 틱 모든 이동 차량이
우선순위 순으로 다음 셀을 예약하며, 셀·엣지 예약으로 동일 셀 점유와 정면 충돌
(스왑)을 구조적으로 차단한다.
"""

from __future__ import annotations

import heapq

from oht_sim.core.layout import Coord, Grid, manhattan


class ReservationTable:
    """셀 단위 시공간 예약표 (셀, 시각) + 스왑 방지용 엣지 예약"""

    def __init__(self) -> None:
        self._cells: dict[tuple[Coord, int], int] = {}  # (cell, tick) -> vehicle_id
        self._edges: dict[tuple[Coord, Coord, int], int] = {}  # (a, b, tick) -> vid

    def cell_free(self, cell: Coord, tick: int, vid: int) -> bool:
        owner = self._cells.get((cell, tick))
        return owner is None or owner == vid

    def edge_free(self, a: Coord, b: Coord, tick: int, vid: int) -> bool:
        """a→b 이동(도착 tick)이 반대 방향 b→a 예약과 충돌하지 않는지"""
        owner = self._edges.get((b, a, tick))
        return owner is None or owner == vid

    def reserve(self, cell: Coord, tick: int, vid: int, frm: Coord | None = None) -> None:
        self._cells[(cell, tick)] = vid
        if frm is not None and frm != cell:
            self._edges[(frm, cell, tick)] = vid

    def reserve_path(self, path: list[Coord], start_tick: int, vid: int) -> None:
        """틱별 셀 시퀀스를 점유 예약 (대기는 동일 셀 반복)"""
        for i, cell in enumerate(path):
            frm = path[i - 1] if i > 0 else None
            self.reserve(cell, start_tick + i, vid, frm)

    def prune_before(self, tick: int) -> None:
        """과거 틱 예약 정리"""
        self._cells = {k: v for k, v in self._cells.items() if k[1] >= tick}
        self._edges = {k: v for k, v in self._edges.items() if k[2] >= tick}


def plan_route(
    grid: Grid,
    start: Coord,
    goal: Coord,
    blocked: set[Coord] | None = None,
) -> list[Coord]:
    """A* 최단 경로 (정적 장애물 + blocked 셀 회피, 맨해튼 휴리스틱)

    도달 불가 시 [start] 반환.
    """
    if start == goal:
        return [start]
    blocked = blocked or set()

    open_heap: list[tuple[int, int, Coord]] = [(manhattan(start, goal), 0, start)]
    gbest: dict[Coord, int] = {start: 0}
    came: dict[Coord, Coord] = {}

    while open_heap:
        _, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            break
        for nxt in grid.neighbors(cell):
            if nxt in blocked and nxt != goal:
                continue
            ng = g + 1
            if ng < gbest.get(nxt, 1 << 30):
                gbest[nxt] = ng
                came[nxt] = cell
                heapq.heappush(open_heap, (ng + manhattan(nxt, goal), ng, nxt))

    if goal not in came:
        return [start]

    path: list[Coord] = [goal]
    cur = goal
    while cur != start:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path
