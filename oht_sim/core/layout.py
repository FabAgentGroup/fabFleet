"""Grid·Node/Station 정의 및 로딩"""

from __future__ import annotations

import random
from dataclasses import dataclass

Coord = tuple[int, int]


@dataclass(frozen=True)
class Station:
    """적재·하역 지점"""

    id: int
    coord: Coord


class Grid:
    """FAB을 2D 격자로 추상화한 레이아웃"""

    def __init__(self, width: int, height: int, blocked: set[Coord] | None = None):
        self.width = width
        self.height = height
        self.blocked: set[Coord] = set(blocked or set())

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, c: Coord) -> bool:
        return self.in_bounds(c) and c not in self.blocked

    def neighbors(self, c: Coord) -> list[Coord]:
        """상하좌우 통행 가능 이웃 셀"""
        x, y = c
        cands = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in cands if self.passable(n)]


def manhattan(a: Coord, b: Coord) -> int:
    """맨해튼 거리"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_stations(
    num_stations: int,
    grid: Grid,
    rng: random.Random,
    coords: list[Coord] | None = None,
) -> list[Station]:
    """Station 목록 생성 (coords 미지정 시 통행 가능 셀에서 무작위 샘플)"""
    if coords is not None:
        return [Station(i, c) for i, c in enumerate(coords)]

    passable = [
        (x, y)
        for x in range(grid.width)
        for y in range(grid.height)
        if grid.passable((x, y))
    ]
    picked = rng.sample(passable, k=min(num_stations, len(passable)))
    return [Station(i, c) for i, c in enumerate(picked)]
