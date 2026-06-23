"""규칙 기반 관제 (LLM 없는 결정적 컨트롤러)

LLM Supervisor와 동일한 관찰→감지→개입 루프를 따르되, 진단·대응을 규칙으로 대체해
API 없이 결정적으로 동작한다. 특히 고장으로 가용 차량이 줄면(fleet_degraded) 남은
차량을 부하 균형(least_busy)으로 돌리고 대기가 가장 많은 구역으로 유휴 OHT를 재배치해
완화한다. 키 없이 실험(D9)에서 고장 완화 효과를 정량화하는 데 쓰며, sim.attach_supervisor로
주입하면 supervisor_interval마다 step이 호출된다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oht_sim.agents.monitor import Monitor
from oht_sim.agents.state import SnapshotBuilder
from oht_sim.core.layout import zone_of

if TYPE_CHECKING:
    from oht_sim.core.config import AgentConfig
    from oht_sim.sim.simulator import Simulator


class RuleSupervisor:
    """규칙 기반 감지·완화 컨트롤러 (고장 인지 포함)"""

    def __init__(self, agent_config: "AgentConfig"):
        self.config = agent_config
        self.builder = SnapshotBuilder(agent_config)
        self.monitor = Monitor(agent_config)
        self.timeline: list[dict] = []

    def _busiest_zone(self, sim: "Simulator") -> tuple[int, int]:
        """대기 작업이 가장 많은 구역 (없으면 (0,0))"""
        n = self.config.num_zones
        counts: dict[tuple[int, int], int] = {}
        for job in sim.pending:
            z = zone_of(job.src.coord, sim.grid.width, sim.grid.height, n)
            counts[z] = counts.get(z, 0) + 1
        if not counts:
            return (0, 0)
        return max(counts, key=lambda z: counts[z])

    def step(self, sim: "Simulator") -> dict:
        """스냅샷을 만들어 이상을 감지하고 규칙으로 완화 액션을 적용"""
        snapshot = self.builder.build(sim)
        anomalies = self.monitor.detect(snapshot)
        types = {a.type for a in anomalies}
        actions: list[str] = []

        if "fleet_degraded" in types:
            # 줄어든 가용 차량을 균등 분산하고 대기 최다 구역으로 유휴 OHT 재배치
            if sim.dispatch_policy_name != "least_busy":
                sim.set_dispatch_policy("least_busy")
                actions.append("set_dispatch_policy:least_busy")
            sim.rebalance_idle_vehicles(self._busiest_zone(sim))
            actions.append("rebalance_idle_vehicles")
        elif "queue_backlog" in types:
            if sim.dispatch_policy_name != "least_busy":
                sim.set_dispatch_policy("least_busy")
                actions.append("set_dispatch_policy:least_busy")

        entry = {
            "time": snapshot.time,
            "availability": snapshot.availability,
            "failed_vehicles": snapshot.failed_vehicles,
            "anomalies": sorted(types),
            "actions": actions,
        }
        self.timeline.append(entry)
        return entry
