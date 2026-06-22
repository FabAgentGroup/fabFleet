"""LLM 클라이언트 - OpenAI 구조화 출력·호출 로그 + 테스트용 스크립트 클라이언트"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMCall:
    """LLM 호출 기록 (재현성·감사용)"""

    agent: str
    system: str
    user: str
    response: dict


class LLMClient(Protocol):
    """구조화(JSON) 응답 LLM 인터페이스"""

    calls: list[LLMCall]

    def complete_json(self, agent: str, system: str, user: str) -> dict:
        """system·user 프롬프트로 JSON 응답을 받아 파싱해 반환"""
        ...


class OpenAIClient:
    """OpenAI Chat Completions 기반 구조화 출력 클라이언트"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.calls: list[LLMCall] = []

    def complete_json(self, agent: str, system: str, user: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"_parse_error": content}
        self.calls.append(LLMCall(agent, system, user, data))
        return data


@dataclass
class ScriptedLLMClient:
    """테스트·오프라인 데모용 - agent별 미리 정한 응답 반환 (API 미사용)"""

    responses: dict[str, dict] = field(default_factory=dict)
    calls: list[LLMCall] = field(default_factory=list)

    def complete_json(self, agent: str, system: str, user: str) -> dict:
        data = self.responses.get(agent, {})
        self.calls.append(LLMCall(agent, system, user, data))
        return data
