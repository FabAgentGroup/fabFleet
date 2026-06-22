"""Job 정의·JobGenerator 도착 프로세스"""

from __future__ import annotations

import random
from dataclasses import dataclass

from oht_sim.core.layout import Station


@dataclass
class Job:
    """반송 요청 (출발 Station → 도착 Station으로 FOUP 1개 운반)"""

    id: int
    src: Station
    dst: Station
    created_time: float
    priority: int = 0


class JobGenerator:
    """포아송 도착으로 Job을 생성 (출발·도착 Station 무작위, 동일 제외)"""

    def __init__(self, stations: list[Station], arrival_rate: float, rng: random.Random):
        self.stations = stations
        self.arrival_rate = arrival_rate
        self.rng = rng
        self._counter = 0

    def next_interarrival(self) -> float:
        """다음 도착까지 간격 (지수 분포)"""
        return self.rng.expovariate(self.arrival_rate)

    def create(self, now: float) -> Job:
        """현재 시각 기준 새 Job 생성"""
        src, dst = self.rng.sample(self.stations, k=2)
        job = Job(id=self._counter, src=src, dst=dst, created_time=now)
        self._counter += 1
        return job
