"""이벤트 로깅·지표 산출"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from oht_sim.core.events import Event, EventBus, EventType
from oht_sim.core.vehicle import BUSY_STATES

if TYPE_CHECKING:
    from oht_sim.core.config import SimConfig

_BUSY_NAMES = {s.value for s in BUSY_STATES}


class MetricsCollector:
    """이벤트 스트림을 구독해 운영 지표를 산출 (이벤트 소싱)"""

    def __init__(self, bus: EventBus, config: "SimConfig"):
        self.bus = bus
        self.config = config
        self.jobs: dict[int, dict[str, float]] = {}
        self.busy: dict[int, float] = {}
        self.queue_series: list[tuple[float, int]] = []
        self.avoidance_waits: int = 0  # 충돌 회피 대기 횟수 (L2)
        self.deadlocks: int = 0  # 교착 감지 횟수 (L2)
        self.interventions: int = 0  # 관제 개입 횟수 (L3)
        self._cur: dict[int, tuple[str, float]] = {}  # vehicle_id -> (state, since)
        self._duration: float = config.sim_duration
        bus.subscribe(self._on_event)

    def _on_event(self, e: Event) -> None:
        t = e.time
        if e.type == EventType.JOB_CREATED:
            self.jobs[e.job_id] = {"created": t}
        elif e.type == EventType.JOB_ASSIGNED:
            self.jobs.setdefault(e.job_id, {})["assigned"] = t
        elif e.type == EventType.DROPOFF:
            self.jobs.setdefault(e.job_id, {})["completed"] = t
        elif e.type == EventType.STATE_CHANGE:
            v = e.vehicle_id
            state, since = self._cur.get(v, ("IDLE", 0.0))
            if state in _BUSY_NAMES:
                self.busy[v] = self.busy.get(v, 0.0) + (t - since)
            self._cur[v] = (e.payload["to"], t)
        elif e.type == EventType.STEP:
            self.queue_series.append((t, e.payload["queue_len"]))
        elif e.type == EventType.BLOCKED:
            self.avoidance_waits += 1
        elif e.type == EventType.DEADLOCK_DETECTED:
            self.deadlocks += 1
        elif e.type == EventType.ACTION:
            self.interventions += 1

    def finalize(self, duration: float) -> None:
        """실행 종료 시 열린 가동 구간을 마감"""
        self._duration = duration
        for v, (state, since) in self._cur.items():
            if state in _BUSY_NAMES:
                self.busy[v] = self.busy.get(v, 0.0) + (duration - since)

    # ----- 지표 -----

    def lead_times(self) -> list[float]:
        """완료 작업의 리드타임 분포 (꼬리 위험·p95 산출용)"""
        return [
            j["completed"] - j["created"]
            for j in self.jobs.values()
            if "completed" in j and "created" in j
        ]

    def summary(self) -> dict[str, float]:
        completed = [j for j in self.jobs.values() if "completed" in j]
        assigned = [j for j in self.jobs.values() if "assigned" in j]

        lead = [j["completed"] - j["created"] for j in completed]
        wait = [j["assigned"] - j["created"] for j in assigned]
        queue_lens = [q for _, q in self.queue_series]

        n = self.config.num_vehicles
        util = sum(self.busy.values()) / (n * self._duration) if n else 0.0

        return {
            "completed_jobs": len(completed),
            "created_jobs": len(self.jobs),
            "throughput": len(completed) / self._duration if self._duration else 0.0,
            "avg_lead_time": _mean(lead),
            "avg_wait_time": _mean(wait),
            "utilization": util,
            "avg_queue_len": _mean(queue_lens),
            "max_queue_len": float(max(queue_lens)) if queue_lens else 0.0,
            "avoidance_waits": self.avoidance_waits,
            "deadlocks": self.deadlocks,
            "interventions": self.interventions,
        }

    def events_dataframe(self) -> pd.DataFrame:
        """원시 이벤트 로그를 DataFrame으로 변환"""
        rows = [
            {
                "time": e.time,
                "type": e.type.value,
                "job_id": e.job_id,
                "vehicle_id": e.vehicle_id,
                "location": e.location,
                **e.payload,
            }
            for e in self.bus.log
        ]
        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        """지표를 표로 출력"""
        s = self.summary()
        labels = {
            "completed_jobs": "완료 작업수",
            "created_jobs": "생성 작업수",
            "throughput": "처리량(완료/시간)",
            "avg_lead_time": "평균 리드타임",
            "avg_wait_time": "평균 대기시간",
            "utilization": "OHT 가동률",
            "avg_queue_len": "평균 큐 길이",
            "max_queue_len": "최대 큐 길이",
            "avoidance_waits": "충돌 회피 대기수",
            "deadlocks": "교착 감지수",
            "interventions": "관제 개입수",
        }
        width = max(len(v) for v in labels.values())
        print("\n=== 시뮬레이션 지표 ===")
        for k, label in labels.items():
            val = s[k]
            shown = f"{val:.3f}" if isinstance(val, float) else str(val)
            print(f"{label:<{width}} : {shown}")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
