"""Dispatcher 인터페이스 + NearestDispatcher(L1) + 정책들"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from oht_sim.core.job import Job
    from oht_sim.core.vehicle import Vehicle


class Dispatcher(Protocol):
    """대기 Job을 유휴 OHT에 할당하는 정책 인터페이스"""

    def assign(
        self,
        pending_jobs: list["Job"],
        idle_vehicles: list["Vehicle"],
    ) -> list[tuple["Job", "Vehicle"]]:
        """대기 Job과 유휴 OHT를 받아 (Job, Vehicle) 할당 쌍 목록 반환"""
        ...


# TODO(Layer 1, §5.4 - NearestDispatcher: 맨해튼 거리 최소 OHT 배정)
# TODO(Layer 3, §4.3(2) - set_dispatch_policy로 교체 가능한 정책들: least_busy 등)
