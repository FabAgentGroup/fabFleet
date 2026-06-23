"""진단 에이전트 (원인 추론·RAG 선택)"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oht_sim.agents.llm import LLMClient
    from oht_sim.agents.monitor import Anomaly
    from oht_sim.agents.rag import Retriever
    from oht_sim.agents.state import Snapshot

_SYSTEM = (
    "당신은 반도체 FAB OHT 반송 시스템의 관제 진단 전문가입니다. "
    "주어진 운영 스냅샷과 감지된 이상으로부터 근본 원인을 추론합니다. "
    "참고 사례가 주어지면 근거로 활용하고 cited_sources에 사례 id를 남깁니다. "
    "반드시 다음 키를 가진 JSON으로만 답합니다: "
    '{"root_cause": str, "reasoning": str, "target_zone": [int,int]|null, '
    '"confidence": float, "cited_sources": [str]}. '
    "confidence는 0~1, target_zone은 원인 구역(없으면 null), cited_sources는 근거 사례 id 목록."
)


class Diagnoser:
    """감지된 이상의 근본 원인을 LLM으로 추론 (선택적 RAG로 사례 근거 보강)"""

    def __init__(self, llm: "LLMClient", retriever: "Retriever | None" = None, top_k: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k

    def _query(self, anomalies: list["Anomaly"]) -> str:
        return " ".join(f"{a.type} {a.evidence}" for a in anomalies)

    def diagnose(self, snapshot: "Snapshot", anomalies: list["Anomaly"]) -> dict:
        user = (
            "운영 스냅샷:\n"
            + snapshot.to_prompt()
            + "\n\n감지된 이상:\n"
            + json.dumps(
                [
                    {"type": a.type, "severity": a.severity, "location": a.location, "evidence": a.evidence}
                    for a in anomalies
                ],
                ensure_ascii=False,
            )
        )
        if self.retriever is not None:
            hits = self.retriever.search(self._query(anomalies), k=self.top_k)
            if hits:
                cases = "\n\n".join(f"[{d.id}] {d.title}\n{d.text}" for d, _ in hits)
                user += "\n\n참고 사례(과거 장애·운영 규칙):\n" + cases
        return self.llm.complete_json("diagnoser", _SYSTEM, user)
