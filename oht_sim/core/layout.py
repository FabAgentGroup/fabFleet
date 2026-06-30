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


class RailGraph:
    """방향성 레일 그래프 (OHT 모노레일 추상화, Grid 호환 인터페이스)

    노드는 (x,y) 셀, 엣지는 허용된 단방향 이동이다. 자유 격자와 달리 차량은 레일 위
    허용 방향으로만 이동한다(인터베이 외곽 루프 + 인트라베이 베이). passable·neighbors·
    blocked·width·height를 Grid와 동일하게 제공해 라우터·MAPF·PIBT가 그대로 동작한다.
    """

    def __init__(self, width: int, height: int, adj: dict[Coord, list[Coord]],
                 blocked: set[Coord] | None = None):
        self.width = width
        self.height = height
        self._adj = adj  # node -> 허용 단방향 이웃
        self.track: set[Coord] = set(adj.keys())
        self.blocked: set[Coord] = set(blocked or set())

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, c: Coord) -> bool:
        return c in self.track and c not in self.blocked

    def neighbors(self, c: Coord) -> list[Coord]:
        """허용 단방향 이웃 (차단 셀 제외)"""
        return [n for n in self._adj.get(c, ()) if n not in self.blocked]


def build_rail_graph(
    width: int, height: int, num_bays: int = 4
) -> tuple[RailGraph, list[Coord]]:
    """인터베이 외곽 루프(시계방향 단방향) + 인트라베이 베이(방향 교대) 레일 생성

    외곽은 강연결 사이클, 베이는 좌우 루프를 잇는 단일 차선이다. Station 후보(베이 내부
    셀) 목록을 함께 반환한다. 그래프는 강연결이라 모든 Station이 상호 도달 가능하다.
    """
    adj: dict[Coord, list[Coord]] = {}

    def add(a: Coord, b: Coord) -> None:
        adj.setdefault(a, [])
        if b not in adj[a]:
            adj[a].append(b)
        adj.setdefault(b, [])

    W, H = width, height
    # 외곽 인터베이 루프 (시계방향): 상→ 우↓ 하← 좌↑
    for x in range(W - 1):
        add((x, H - 1), (x + 1, H - 1))
    for y in range(H - 1, 0, -1):
        add((W - 1, y), (W - 1, y - 1))
    for x in range(W - 1, 0, -1):
        add((x, 0), (x - 1, 0))
    for y in range(H - 1):
        add((0, y), (0, y + 1))

    # 인트라베이 베이 (내부 행, 좌우 방향 교대)
    bay_rows: list[int] = []
    if num_bays > 0 and H > 2:
        step = (H - 1) / (num_bays + 1)
        for i in range(1, num_bays + 1):
            b = min(H - 2, max(1, int(round(i * step))))
            if b not in bay_rows:
                bay_rows.append(b)

    stations: list[Coord] = []
    for idx, b in enumerate(bay_rows):
        if idx % 2 == 0:  # 우향 베이 (좌 진입 -> 우 진출)
            for x in range(W - 1):
                add((x, b), (x + 1, b))
        else:  # 좌향 베이 (우 진입 -> 좌 진출)
            for x in range(W - 1, 0, -1):
                add((x, b), (x - 1, b))
        stations.extend((x, b) for x in range(1, W - 1))

    return RailGraph(W, H, adj), stations


def manhattan(a: Coord, b: Coord) -> int:
    """맨해튼 거리"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zone_of(cell: Coord, width: int, height: int, n: int) -> tuple[int, int]:
    """셀이 속한 n x n 구역 좌표"""
    zx = min(n - 1, cell[0] * n // width)
    zy = min(n - 1, cell[1] * n // height)
    return (zx, zy)


def zone_center(zone: tuple[int, int], width: int, height: int, n: int) -> Coord:
    """구역 중심 셀"""
    cx = min(width - 1, int((zone[0] + 0.5) * width / n))
    cy = min(height - 1, int((zone[1] + 0.5) * height / n))
    return (cx, cy)


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
