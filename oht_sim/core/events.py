"""구조화 이벤트 스키마·EventBus (L3 인터페이스)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from oht_sim.core.layout import Coord


class EventType(str, Enum):
    """시뮬레이터가 발행하는 사건 종류"""

    JOB_CREATED = "JOB_CREATED"
    JOB_ASSIGNED = "JOB_ASSIGNED"
    PICKUP = "PICKUP"
    DROPOFF = "DROPOFF"
    STATE_CHANGE = "STATE_CHANGE"  # OHT 상태 전이
    MOVE = "MOVE"  # 셀 단위 이동 (시각화·혼잡도)
    BLOCKED = "BLOCKED"  # 충돌 회피 대기 (L2)
    DEADLOCK_DETECTED = "DEADLOCK_DETECTED"  # 교착 감지 (L2)
    ACTION = "ACTION"  # 관제 에이전트 개입 (L3)
    STEP = "STEP"  # 주기적 스냅샷 (큐 길이 등)
    VEHICLE_FAILED = "VEHICLE_FAILED"  # OHT 고장 정지 (L1 신뢰성)
    VEHICLE_REPAIRED = "VEHICLE_REPAIRED"  # OHT 수리 복귀 (L1 신뢰성)


@dataclass
class Event:
    """구조화 운영 이벤트 (L1→L3 관찰 인터페이스의 단위)"""

    time: float
    type: EventType
    job_id: int | None = None
    vehicle_id: int | None = None
    location: Coord | None = None
    payload: dict = field(default_factory=dict)


class EventBus:
    """이벤트 발행·구독·로그 보관"""

    def __init__(self) -> None:
        self._log: list[Event] = []
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        """이벤트 수신 콜백 등록"""
        self._subscribers.append(fn)

    def publish(self, event: Event) -> None:
        """이벤트를 로그에 적재하고 구독자에게 전달"""
        self._log.append(event)
        for fn in self._subscribers:
            fn(event)

    @property
    def log(self) -> list[Event]:
        return self._log
