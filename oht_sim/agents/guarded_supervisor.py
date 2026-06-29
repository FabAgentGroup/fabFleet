"""인과 가드 관제 (LLM 없는 결정적 컨트롤러, §8 D11)

D5는 관제 개입이 평균적으로 지표를 악화시키는 경향(평균 효과 음수)을 보였고, 효과
점수가 인과를 가리지 못함을 드러냈다. 이 컨트롤러는 (1) 개입을 인과 점수(드리프트
보정)로 평가하고, (2) 학습된 인과 통계로 해로운 액션을 거부하는 가드(neurosymbolic)를
적용해 "순손해 개입"을 차단한다. guard=False면 이상마다 무조건 개입하는 순진한 관제,
guard=True면 인과 가드 관제다.

attach_supervisor로 주입하면 supervisor_interval마다 step이 호출된다. 결정적으로
동작해 키 없이 D11에서 가드 효과를 정량화한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oht_sim.agents.monitor import Monitor
from oht_sim.agents.reflection import InterventionLedger
from oht_sim.agents.state import SnapshotBuilder
from oht_sim.core.layout import zone_of

if TYPE_CHECKING:
    from oht_sim.core.config import AgentConfig
    from oht_sim.sim.simulator import Simulator

_CANDIDATE = "rebalance_idle_vehicles"  # 동역학 영향이 큰(따라서 해로울 수도 있는) 액션


class GuardedSupervisor:
    """이상 감지 시 유휴 차량 재배치로 개입하되, 인과 가드로 해로운 개입을 거부"""

    def __init__(self, agent_config: "AgentConfig", guard: bool = True):
        self.config = agent_config
        self.guard = guard
        self.builder = SnapshotBuilder(agent_config)
        self.monitor = Monitor(agent_config)
        self.ledger = InterventionLedger(agent_config)
        self.timeline: list[dict] = []

    def _busiest_zone(self, sim: "Simulator") -> tuple[int, int]:
        n = self.config.num_zones
        counts: dict[tuple[int, int], int] = {}
        for job in sim.pending:
            z = zone_of(job.src.coord, sim.grid.width, sim.grid.height, n)
            counts[z] = counts.get(z, 0) + 1
        return max(counts, key=lambda z: counts[z]) if counts else (0, 0)

    def step(self, sim: "Simulator") -> dict:
        snapshot = self.builder.build(sim)
        self.ledger.evaluate_pending(snapshot)  # 직전 개입 인과 평가(가드 학습)

        anomalies = self.monitor.detect(snapshot)
        types = {a.type for a in anomalies}
        action, params, vetoed = "none", {}, False

        if types:
            if self.guard and not self.ledger.should_act(_CANDIDATE):
                vetoed = True  # 인과 가드가 해로운 액션 거부
            else:
                zone = self._busiest_zone(sim)
                sim.rebalance_idle_vehicles(zone)
                action, params = _CANDIDATE, {"zone": list(zone)}

        idx = len(self.timeline)
        entry = {
            "time": snapshot.time,
            "anomalies": sorted(types),
            "action": action,
            "vetoed": vetoed,
            "queue": snapshot.queue_len,
        }
        self.timeline.append(entry)
        if action != "none":
            self.ledger.record(
                snapshot.time,
                {"action": action, "params": params},
                {"target_zone": None},
                snapshot,
                idx,
            )
        return entry

    def stats(self) -> dict:
        """개입·거부 집계 + 인과 효율"""
        acts = sum(1 for e in self.timeline if e["action"] != "none")
        vetoes = sum(1 for e in self.timeline if e["vetoed"])
        eff = self.ledger.efficacy_summary()
        return {
            "acts": acts,
            "vetoes": vetoes,
            "evaluated": eff["total"],
            "avg_causal": eff["avg_causal"],
            "causal_harmful": eff["causal_counts"]["악화"],
        }
