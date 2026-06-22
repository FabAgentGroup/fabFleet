"""OHT 상태머신·위치·이동 프로세스"""

from __future__ import annotations

from enum import Enum

from oht_sim.core.job import Job
from oht_sim.core.layout import Coord


class VehicleState(str, Enum):
    """OHT 상태"""

    IDLE = "IDLE"
    MOVING_TO_PICKUP = "MOVING_TO_PICKUP"
    LOADING = "LOADING"
    MOVING_TO_DROPOFF = "MOVING_TO_DROPOFF"
    UNLOADING = "UNLOADING"


# non-idle 집합 (가동률 산출용)
BUSY_STATES = frozenset(VehicleState) - {VehicleState.IDLE}


class Vehicle:
    """반송 로봇 (저수준 에이전트) - 상태·위치·현재 작업 보유"""

    def __init__(self, vehicle_id: int, pos: Coord):
        self.id = vehicle_id
        self.pos: Coord = pos
        self.state: VehicleState = VehicleState.IDLE
        self.job: Job | None = None
        self.path: list[Coord] = []
        self.goal: Coord | None = None  # 현재 이동 목표 (L2 동기식 mover)
        self.moving: bool = False  # 틱 동기식 이동 대상 여부

    @property
    def is_idle(self) -> bool:
        return self.state == VehicleState.IDLE

    def __repr__(self) -> str:
        return f"Vehicle(id={self.id}, pos={self.pos}, state={self.state.value})"
