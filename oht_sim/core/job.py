"""Job 정의·JobGenerator 도착 프로세스"""

from __future__ import annotations

import random
from dataclasses import dataclass

from oht_sim.core.layout import Station, zone_of


@dataclass
class Job:
    """반송 요청 (출발 Station → 도착 Station으로 FOUP 1개 운반)"""

    id: int
    src: Station
    dst: Station
    created_time: float
    priority: int = 0


class JobGenerator:
    """포아송 도착으로 Job을 생성 (출발·도착 Station 무작위, 동일 제외)

    demand_hotspot이 켜지면 출발 Station이 시간에 따라 이동하는 핫스팟 구역으로
    쏠린다(시간 변동 수요). 핫스팟 구역은 hotspot_period마다 순환한다.
    """

    def __init__(
        self,
        stations: list[Station],
        arrival_rate: float,
        rng: random.Random,
        grid_width: int = 0,
        grid_height: int = 0,
        hotspot: bool = False,
        zones: int = 2,
        hotspot_period: float = 150.0,
        hotspot_weight: float = 0.6,
    ):
        self.stations = stations
        self.arrival_rate = arrival_rate
        self.rng = rng
        self._counter = 0

        self.hotspot = hotspot
        self.hotspot_period = hotspot_period
        self.hotspot_weight = hotspot_weight
        # 구역별 Station 목록 (핫스팟 출발지 샘플용)
        self._by_zone: dict[tuple[int, int], list[Station]] = {}
        if hotspot and grid_width and grid_height:
            for s in stations:
                z = zone_of(s.coord, grid_width, grid_height, zones)
                self._by_zone.setdefault(z, []).append(s)
        self._zone_keys = sorted(self._by_zone)

    def next_interarrival(self) -> float:
        """다음 도착까지 간격 (지수 분포)"""
        return self.rng.expovariate(self.arrival_rate)

    def hot_zone(self, now: float) -> tuple[int, int] | None:
        """현재 시각의 핫스팟 구역 (비활성 시 None)"""
        if not self._zone_keys:
            return None
        idx = int(now // self.hotspot_period) % len(self._zone_keys)
        return self._zone_keys[idx]

    def create(self, now: float) -> Job:
        """현재 시각 기준 새 Job 생성"""
        if self.hotspot and self._zone_keys:
            if self.rng.random() < self.hotspot_weight:
                src = self.rng.choice(self._by_zone[self.hot_zone(now)])
            else:
                src = self.rng.choice(self.stations)
            dst = self.rng.choice([s for s in self.stations if s.id != src.id])
        else:
            # 비-핫스팟은 기존 동작 유지 (재현성 보존)
            src, dst = self.rng.sample(self.stations, k=2)
        job = Job(id=self._counter, src=src, dst=dst, created_time=now)
        self._counter += 1
        return job
