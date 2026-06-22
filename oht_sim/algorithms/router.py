"""Router 인터페이스 + ManhattanRouter(L1) + AStarRouter(L2)"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from oht_sim.core.layout import Coord, Grid

if TYPE_CHECKING:
    from oht_sim.algorithms.mapf import ReservationTable


class Router(Protocol):
    """start→goal 경로 계산 전략 인터페이스"""

    def find_path(
        self,
        grid: Grid,
        start: Coord,
        goal: Coord,
        reservation: "ReservationTable | None" = None,
        start_time: float = 0.0,
    ) -> list[Coord]:
        """start→goal 경로(셀 시퀀스) 반환, reservation은 L2 충돌 회피용"""
        ...


class ManhattanRouter:
    """장애물 없다 가정, 직각(L자) 경로 연결 - L1 베이스라인"""

    def find_path(
        self,
        grid: Grid,
        start: Coord,
        goal: Coord,
        reservation: "ReservationTable | None" = None,
        start_time: float = 0.0,
    ) -> list[Coord]:
        path: list[Coord] = [start]
        x, y = start
        gx, gy = goal
        step_x = 1 if gx > x else -1
        while x != gx:
            x += step_x
            path.append((x, y))
        step_y = 1 if gy > y else -1
        while y != gy:
            y += step_y
            path.append((x, y))
        return path


# TODO(Layer 2, §6.1 - AStarRouter: 장애물·일방통행 반영, 맨해튼 휴리스틱)
