"""LangGraph 오케스트레이션 (monitor→diagnose→respond)"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, StateGraph

from oht_sim.agents.diagnoser import Diagnoser
from oht_sim.agents.monitor import Monitor
from oht_sim.agents.responder import ActionExecutor, Responder
from oht_sim.agents.state import SnapshotBuilder

if TYPE_CHECKING:
    from oht_sim.agents.llm import LLMClient
    from oht_sim.agents.rag import Retriever
    from oht_sim.core.config import AgentConfig
    from oht_sim.sim.simulator import Simulator


class SupervisorState(TypedDict, total=False):
    """관제 그래프 상태"""

    snapshot: object
    anomalies: list
    diagnosis: dict
    response: dict


def build_graph(monitor: Monitor, diagnoser: Diagnoser, responder: Responder):
    """monitor→(이상 있음?)→diagnose→respond 조건부 그래프 컴파일"""

    def monitor_node(state: SupervisorState) -> dict:
        return {"anomalies": monitor.detect(state["snapshot"])}

    def diagnose_node(state: SupervisorState) -> dict:
        return {"diagnosis": diagnoser.diagnose(state["snapshot"], state["anomalies"])}

    def respond_node(state: SupervisorState) -> dict:
        return {
            "response": responder.respond(
                state["snapshot"], state["anomalies"], state["diagnosis"]
            )
        }

    def route(state: SupervisorState) -> str:
        return "diagnose" if state.get("anomalies") else END

    g = StateGraph(SupervisorState)
    g.add_node("monitor", monitor_node)
    g.add_node("diagnose", diagnose_node)
    g.add_node("respond", respond_node)
    g.set_entry_point("monitor")
    g.add_conditional_edges("monitor", route, {"diagnose": "diagnose", END: END})
    g.add_edge("diagnose", "respond")
    g.add_edge("respond", END)
    return g.compile()


class Supervisor:
    """관제 에이전트 묶음 - 스냅샷 빌더·그래프·액션 실행·추론 타임라인"""

    def __init__(
        self,
        llm: "LLMClient",
        agent_config: "AgentConfig",
        sim: "Simulator",
        retriever: "Retriever | None" = None,
    ):
        self.builder = SnapshotBuilder(agent_config)
        self.monitor = Monitor(agent_config)
        self.diagnoser = Diagnoser(llm, retriever=retriever)
        self.executor = ActionExecutor(sim)
        self.responder = Responder(llm, self.executor)
        self.graph = build_graph(self.monitor, self.diagnoser, self.responder)
        self.llm = llm
        self.timeline: list[dict] = []

    def step(self, sim: "Simulator") -> dict:
        """스냅샷을 만들어 그래프를 실행하고 추론·개입 기록을 남김"""
        snapshot = self.builder.build(sim)
        result = self.graph.invoke({"snapshot": snapshot})
        entry = {
            "time": snapshot.time,
            "summary": snapshot.to_prompt().splitlines()[0],
            "anomalies": [a.__dict__ for a in result.get("anomalies", [])],
            "diagnosis": result.get("diagnosis"),
            "response": result.get("response"),
        }
        self.timeline.append(entry)
        return entry
