"""관제용 상태 스냅샷 빌더 (이벤트→LLM 입력)"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from oht_sim.core.events import EventType
from oht_sim.core.layout import Coord

if TYPE_CHECKING:
    from oht_sim.core.config import AgentConfig
    from oht_sim.sim.simulator import Simulator


@dataclass
class ZoneStat:
    """구역별 운영 상태"""

    zone: tuple[int, int]
    vehicles: int
    blocked_recent: int  # 최근 충돌 회피 대기
    deadlocks_recent: int


@dataclass
class Snapshot:
    """관제 에이전트 입력 - 집계 운영 지표 + 최근 이상 (자연어 + 구조화 JSON)"""

    time: float
    queue_len: int
    queue_trend: int  # 직전 스냅샷 대비 증감
    vehicle_states: dict[str, int]
    utilization_now: float
    zones: list[ZoneStat]
    recent_deadlocks: list[Coord]
    recent_lead_time: float
    dispatch_policy: str

    def to_struct(self) -> dict:
        return {
            "time": round(self.time, 1),
            "queue_len": self.queue_len,
            "queue_trend": self.queue_trend,
            "vehicle_states": self.vehicle_states,
            "utilization_now": round(self.utilization_now, 3),
            "recent_lead_time": round(self.recent_lead_time, 1),
            "dispatch_policy": self.dispatch_policy,
            "zones": [
                {
                    "zone": list(z.zone),
                    "vehicles": z.vehicles,
                    "blocked_recent": z.blocked_recent,
                    "deadlocks_recent": z.deadlocks_recent,
                }
                for z in self.zones
            ],
            "recent_deadlocks": [list(c) for c in self.recent_deadlocks],
        }

    def to_prompt(self) -> str:
        """LLM 입력용 자연어 요약 + 구조화 JSON"""
        hot = [z for z in self.zones if z.blocked_recent or z.deadlocks_recent]
        hot_txt = (
            ", ".join(
                f"구역{z.zone} 회피대기 {z.blocked_recent}·교착 {z.deadlocks_recent}"
                for z in hot
            )
            or "특이 구역 없음"
        )
        summary = (
            f"시각 {self.time:.0f}, 대기 큐 {self.queue_len}(추세 {self.queue_trend:+d}), "
            f"가동률 {self.utilization_now:.0%}, 최근 리드타임 {self.recent_lead_time:.0f}, "
            f"배차정책 {self.dispatch_policy}. 혼잡 구역: {hot_txt}."
        )
        return summary + "\n구조화 데이터:\n" + json.dumps(
            self.to_struct(), ensure_ascii=False
        )


def _zone_of(cell: Coord, width: int, height: int, n: int) -> tuple[int, int]:
    zx = min(n - 1, cell[0] * n // width)
    zy = min(n - 1, cell[1] * n // height)
    return (zx, zy)


class SnapshotBuilder:
    """시뮬레이터 상태·이벤트를 관제 스냅샷으로 집계"""

    def __init__(self, config: "AgentConfig"):
        self.config = config
        self._prev_queue: int | None = None

    def build(self, sim: "Simulator") -> Snapshot:
        cfg = self.config
        now = sim.env.now
        win_start = now - cfg.recent_window
        n = cfg.num_zones
        w, h = sim.grid.width, sim.grid.height

        recent = [e for e in sim.bus.log if e.time >= win_start]

        # 구역별 집계
        zstat: dict[tuple[int, int], ZoneStat] = {
            (zx, zy): ZoneStat((zx, zy), 0, 0, 0)
            for zx in range(n)
            for zy in range(n)
        }
        for v in sim.vehicles:
            zstat[_zone_of(v.pos, w, h, n)].vehicles += 1
        deadlocks: list[Coord] = []
        for e in recent:
            if e.location is None:
                continue
            z = _zone_of(e.location, w, h, n)
            if e.type == EventType.BLOCKED:
                zstat[z].blocked_recent += 1
            elif e.type == EventType.DEADLOCK_DETECTED:
                zstat[z].deadlocks_recent += 1
                deadlocks.append(e.location)

        # 최근 완료 작업 리드타임
        leads = [
            e.time - sim.metrics.jobs[e.job_id]["created"]
            for e in recent
            if e.type == EventType.DROPOFF
            and e.job_id in sim.metrics.jobs
            and "created" in sim.metrics.jobs[e.job_id]
        ]
        recent_lead = sum(leads) / len(leads) if leads else 0.0

        states = Counter(v.state.value for v in sim.vehicles)
        idle = states.get("IDLE", 0)
        util = (len(sim.vehicles) - idle) / len(sim.vehicles) if sim.vehicles else 0.0

        queue_len = len(sim.pending)
        trend = 0 if self._prev_queue is None else queue_len - self._prev_queue
        self._prev_queue = queue_len

        return Snapshot(
            time=now,
            queue_len=queue_len,
            queue_trend=trend,
            vehicle_states=dict(states),
            utilization_now=util,
            zones=list(zstat.values()),
            recent_deadlocks=deadlocks,
            recent_lead_time=recent_lead,
            dispatch_policy=sim.dispatch_policy_name,
        )
