"""Dispatcher 인터페이스 + NearestDispatcher(L1) + 정책들"""

from __future__ import annotations

from typing import Protocol

from oht_sim.core.job import Job
from oht_sim.core.layout import manhattan
from oht_sim.core.vehicle import Vehicle


class Dispatcher(Protocol):
    """대기 Job을 유휴 OHT에 할당하는 정책 인터페이스"""

    def assign(
        self,
        pending_jobs: list[Job],
        idle_vehicles: list[Vehicle],
    ) -> list[tuple[Job, Vehicle]]:
        """대기 Job과 유휴 OHT를 받아 (Job, Vehicle) 할당 쌍 목록 반환"""
        ...


class NearestDispatcher:
    """유휴 OHT 중 출발 Station까지 맨해튼 거리 최소인 OHT 배정 - L1 베이스라인"""

    def assign(
        self,
        pending_jobs: list[Job],
        idle_vehicles: list[Vehicle],
    ) -> list[tuple[Job, Vehicle]]:
        assignments: list[tuple[Job, Vehicle]] = []
        available = list(idle_vehicles)
        # 생성 순(우선순위·시각) 작업부터 가까운 차량 배정
        for job in sorted(pending_jobs, key=lambda j: (-j.priority, j.created_time)):
            if not available:
                break
            nearest = min(available, key=lambda v: manhattan(v.pos, job.src.coord))
            assignments.append((job, nearest))
            available.remove(nearest)
        return assignments


class LeastBusyDispatcher:
    """누적 배차가 가장 적은 유휴 OHT부터 배정 - 부하 균형 정책 (L3 개입 대상)

    set_dispatch_policy로 nearest와 교체해 특정 차량 쏠림을 완화한다.
    """

    def __init__(self) -> None:
        self._count: dict[int, int] = {}

    def assign(
        self,
        pending_jobs: list[Job],
        idle_vehicles: list[Vehicle],
    ) -> list[tuple[Job, Vehicle]]:
        assignments: list[tuple[Job, Vehicle]] = []
        available = list(idle_vehicles)
        for job in sorted(pending_jobs, key=lambda j: (-j.priority, j.created_time)):
            if not available:
                break
            # 누적 배차 최소, 동률은 출발지까지 거리로 보조 정렬
            chosen = min(
                available,
                key=lambda v: (self._count.get(v.id, 0), manhattan(v.pos, job.src.coord)),
            )
            self._count[chosen.id] = self._count.get(chosen.id, 0) + 1
            assignments.append((job, chosen))
            available.remove(chosen)
        return assignments

