"""모니터링 에이전트 (이상 징후 감지)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oht_sim.agents.state import Snapshot
    from oht_sim.core.config import AgentConfig


@dataclass
class Anomaly:
    """감지된 이상 징후"""

    type: str
    severity: str  # low | medium | high
    location: tuple[int, int] | None
    evidence: str


class Monitor:
    """규칙/통계 임계치 기반 이상 1차 감지 (LLM 미사용, 트리거 역할)

    가벼운 규칙으로 이상을 거른 뒤에만 진단·대응(LLM)으로 넘겨 호출 비용을 줄인다.
    """

    def __init__(self, config: "AgentConfig"):
        self.config = config

    def detect(self, snapshot: "Snapshot") -> list[Anomaly]:
        cfg = self.config
        anomalies: list[Anomaly] = []

        if snapshot.queue_len >= cfg.queue_threshold:
            sev = "high" if snapshot.queue_trend > 0 else "medium"
            anomalies.append(
                Anomaly(
                    type="queue_backlog",
                    severity=sev,
                    location=None,
                    evidence=f"대기 큐 {snapshot.queue_len}(추세 {snapshot.queue_trend:+d})",
                )
            )

        if snapshot.failed_vehicles > 0 and snapshot.availability < cfg.availability_threshold:
            anomalies.append(
                Anomaly(
                    type="fleet_degraded",
                    severity="high" if snapshot.availability < 0.5 else "medium",
                    location=None,
                    evidence=f"가용 {snapshot.availability:.0%}(고장 {snapshot.failed_vehicles}대)",
                )
            )

        for z in snapshot.zones:
            if z.deadlocks_recent >= cfg.zone_deadlock_threshold:
                anomalies.append(
                    Anomaly(
                        type="zone_deadlock",
                        severity="high",
                        location=z.zone,
                        evidence=f"구역{z.zone} 최근 교착 {z.deadlocks_recent}",
                    )
                )
            elif z.blocked_recent >= cfg.zone_block_threshold:
                anomalies.append(
                    Anomaly(
                        type="zone_congestion",
                        severity="medium",
                        location=z.zone,
                        evidence=f"구역{z.zone} 최근 회피대기 {z.blocked_recent}",
                    )
                )

        return anomalies
