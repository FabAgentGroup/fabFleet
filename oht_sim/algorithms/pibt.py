"""PIBT - Priority Inheritance with Backtracking (lifelong MAPF, §8 D12)

한 틱에 모든 이동 에이전트의 다음 셀을 우선순위 순으로 결정한다. 고우선 에이전트가
원하는 셀을 저우선 에이전트가 점유하면, 저우선 에이전트가 우선순위를 상속받아 비켜서고
(재귀), 비킬 곳이 없으면 백트래킹한다. 정점·스왑 충돌을 구조적으로 차단하며 잘 정의된
인스턴스에서 교착 없이 동작한다(Okumura et al. 2019). 기존의 임시 우선순위 플래너를
원칙적·확장 가능한 알고리즘으로 대체한다.

유휴 차량(goal=현재 위치)도 밀어낼 수 있는 장애물로 다뤄(priority inheritance), 목표를
막은 차량을 비켜세우는 기존 교착 회복을 일반화한다. 작업 중(적재·하역)·고장 차량은 밀 수
없는 고정 장애물이다. 외부 라이브러리 없이 결정적으로 동작한다.
"""

from __future__ import annotations

from typing import Callable

from oht_sim.core.layout import Coord, Grid, manhattan


class PIBTPlanner:
    """단일 틱 PIBT 플래너 (충돌 없는 다음 위치 산출)

    cost_fn이 주어지면 같은 목표 거리의 후보 중 혼잡이 낮은 셀을 우선해(동점 기준),
    PIBT의 진전 보장을 유지하면서 트래픽을 분산한다(D10 혼잡 라우팅과 결합).
    """

    def __init__(
        self,
        grid: Grid,
        positions: dict[int, Coord],
        goals: dict[int, Coord],
        movable: set[int],
        cost_fn: Callable[[Coord], float] | None = None,
    ):
        self.grid = grid
        self.pos = positions  # 모든 차량 vid -> 현재 셀
        self.goal = goals  # 이동·밀림 가능 차량 vid -> 목표(유휴는 현재 셀)
        self.movable = movable  # 밀거나 이동시킬 수 있는 vid 집합
        self.cost_fn = cost_fn  # 혼잡 페널티(동점 기준) | None
        self.occupied: dict[Coord, int] = {c: vid for vid, c in positions.items()}
        self.next: dict[int, Coord] = {}  # vid -> 확정 다음 셀
        self.reserved: dict[Coord, int] = {}  # 다음 셀 -> 선점 vid

    def solve(self, order: list[int]) -> dict[int, Coord]:
        """우선순위 순서대로 이동 에이전트를 계획하고 다음 위치 맵 반환"""
        for vid in order:
            if vid not in self.next:
                self._pibt(vid, None)
        return self.next

    def _candidates(self, vid: int) -> list[Coord]:
        p = self.pos[vid]
        g = self.goal.get(vid, p)
        cands = self.grid.neighbors(p) + [p]  # 이웃 + 제자리
        if self.cost_fn is None:
            cands.sort(key=lambda c: (manhattan(c, g), c))  # 목표에 가까운 셀 우선
        else:
            # 1차 목표 거리, 2차 혼잡(낮은 셀 우선) - 진전 보장 유지하며 트래픽 분산
            cands.sort(key=lambda c: (manhattan(c, g), self.cost_fn(c), c))
        return cands

    def _pibt(self, vid: int, forbidden: Coord | None) -> bool:
        """vid의 다음 셀을 결정. forbidden(호출자 현재 셀)으로의 진입은 스왑이라 금지."""
        for c in self._candidates(vid):
            if c in self.reserved:
                continue  # 이미 다른 에이전트가 향함
            if forbidden is not None and c == forbidden:
                continue  # 정면 스왑 회피
            self.reserved[c] = vid  # 잠정 선점
            occ = self.occupied.get(c)
            if occ is not None and occ != vid and occ not in self.next:
                # c의 현 점유자가 아직 미계획 -> 우선순위 상속으로 밀어내기
                if occ in self.movable and self._pibt(occ, self.pos[vid]):
                    self.next[vid] = c
                    return True
                # 백트래킹 - 내 잠정 선점일 때만 해제(밀린 차량이 c에 눌러앉았으면 유지)
                if self.reserved.get(c) == vid:
                    del self.reserved[c]
                continue
            # 빈 셀이거나 점유자가 이미 떠나기로 계획됨
            self.next[vid] = c
            return True
        # 진전 가능한 셀 없음 -> 제자리 대기 (현재 셀을 확정 점유)
        self.next[vid] = self.pos[vid]
        self.reserved[self.pos[vid]] = vid
        return False
