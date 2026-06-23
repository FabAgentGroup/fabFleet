"""대응 에이전트 (화이트리스트 액션 실행)"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oht_sim.agents.llm import LLMClient
    from oht_sim.agents.monitor import Anomaly
    from oht_sim.agents.state import Snapshot
    from oht_sim.sim.simulator import Simulator

# 안전한 화이트리스트 액션 (§4.3 (2))
WHITELIST = {
    "none",
    "set_dispatch_policy",
    "block_segment",
    "reprioritize_job",
    "rebalance_idle_vehicles",
}

_SYSTEM = (
    "당신은 반도체 FAB OHT 반송 관제의 대응 결정자입니다. "
    "진단을 바탕으로 아래 화이트리스트 액션 중 하나만 선택합니다. "
    "반드시 JSON으로만 답합니다: "
    '{"action": str, "params": object, "rationale": str}. '
    "action 후보와 params: "
    'set_dispatch_policy{"name":"nearest|least_busy"}, '
    'block_segment{"cells":[[x,y],...],"duration":number}, '
    'reprioritize_job{"job_id":int,"priority":int}, '
    'rebalance_idle_vehicles{"zone":[zx,zy]}, '
    'none{}. 개입이 불필요하면 none을 선택합니다.'
)


@dataclass
class ActionRecord:
    """실행된 개입 기록 (개입 전/후 비교·감사용)"""

    time: float
    action: str
    params: dict
    applied: bool
    error: str | None = None


class ActionExecutor:
    """LLM 결정을 화이트리스트로 검증해 시뮬레이터에 안전하게 적용"""

    def __init__(self, sim: "Simulator"):
        self.sim = sim
        self.log: list[ActionRecord] = []

    def execute(self, decision: dict) -> ActionRecord:
        action = decision.get("action", "none")
        params = decision.get("params") or {}
        rec = ActionRecord(self.sim.env.now, action, params, applied=False)

        if action not in WHITELIST:
            rec.error = f"비화이트리스트 액션 거부: {action}"
            self.log.append(rec)
            return rec
        try:
            if action == "none":
                pass
            elif action == "set_dispatch_policy":
                self.sim.set_dispatch_policy(params["name"])
            elif action == "block_segment":
                cells = [tuple(c) for c in params["cells"]]
                self.sim.block_segment(cells, float(params.get("duration", 30.0)))
            elif action == "reprioritize_job":
                self.sim.reprioritize_job(int(params["job_id"]), int(params["priority"]))
            elif action == "rebalance_idle_vehicles":
                self.sim.rebalance_idle_vehicles(tuple(params["zone"]))
            rec.applied = action != "none"
        except (KeyError, ValueError, TypeError) as ex:
            rec.error = f"파라미터 오류: {ex}"
        self.log.append(rec)
        return rec


class Responder:
    """진단 기반으로 화이트리스트 액션을 선택·실행"""

    def __init__(self, llm: "LLMClient", executor: ActionExecutor):
        self.llm = llm
        self.executor = executor

    def respond(
        self,
        snapshot: "Snapshot",
        anomalies: list["Anomaly"],
        diagnosis: dict,
        reflection: str = "",
    ) -> dict:
        user = (
            "운영 스냅샷:\n"
            + snapshot.to_prompt()
            + "\n\n진단:\n"
            + json.dumps(diagnosis, ensure_ascii=False)
        )
        if reflection:
            user += "\n\n" + reflection
        decision = self.llm.complete_json("responder", _SYSTEM, user)
        record = self.executor.execute(decision)
        return {"decision": decision, "applied": record.applied, "error": record.error}
