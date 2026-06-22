"""진단 에이전트 (원인 추론·RAG 선택)"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oht_sim.agents.llm import LLMClient
    from oht_sim.agents.monitor import Anomaly
    from oht_sim.agents.state import Snapshot

_SYSTEM = (
    "당신은 반도체 FAB OHT 반송 시스템의 관제 진단 전문가입니다. "
    "주어진 운영 스냅샷과 감지된 이상으로부터 근본 원인을 추론합니다. "
    "반드시 다음 키를 가진 JSON으로만 답합니다: "
    '{"root_cause": str, "reasoning": str, "target_zone": [int,int]|null, "confidence": float}. '
    "confidence는 0~1, target_zone은 원인 구역(없으면 null)."
)


class Diagnoser:
    """감지된 이상의 근본 원인을 LLM으로 추론 (RAG는 §8 심화로 분리)"""

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

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
        return self.llm.complete_json("diagnoser", _SYSTEM, user)
