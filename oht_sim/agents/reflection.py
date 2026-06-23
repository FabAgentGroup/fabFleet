"""개입 효과 평가 + reflection 원장 (L3 닫힌 루프)

관제 에이전트가 실행한 개입을 적용 시점의 운영 지표(before)와 함께 기록하고,
다음 스냅샷(after)과 비교해 효과 점수·라벨을 산출한다. 평가 결과는 reflection
요약으로 묶여 다음 진단·대응 프롬프트에 주입되며, 악화한 접근은 지양하고 효과적
접근은 강화하도록 유도한다. 이로써 "개입하고 그 효과를 지표로 검증"하는 닫힌
루프가 실제로 닫힌다.

LLM 없이 결정적으로 동작하며 효과 점수·임계는 모두 config로 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oht_sim.agents.state import Snapshot
    from oht_sim.core.config import AgentConfig


def _zone_congestion(snapshot: "Snapshot", zone: tuple[int, int] | None) -> float:
    """대상 구역의 최근 혼잡도(회피대기 + 교착) - 구역 미지정 시 전 구역 합"""
    total = 0.0
    for z in snapshot.zones:
        if zone is None or tuple(z.zone) == tuple(zone):
            total += z.blocked_recent + z.deadlocks_recent
    return total


@dataclass
class InterventionOutcome:
    """개입 한 건의 전/후 비교 기록"""

    time: float
    action: str
    params: dict
    target_zone: tuple[int, int] | None
    before: dict[str, float]
    timeline_idx: int
    after: dict[str, float] | None = None
    effect_score: float | None = None
    label: str = "대기"  # 대기 | 개선 | 변화없음 | 악화

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 1),
            "action": self.action,
            "target_zone": list(self.target_zone) if self.target_zone else None,
            "before": self.before,
            "after": self.after,
            "effect_score": (
                round(self.effect_score, 2) if self.effect_score is not None else None
            ),
            "label": self.label,
        }


class InterventionLedger:
    """개입 전/후 지표를 비교해 효과를 평가하고 reflection 요약을 생성

    효과 점수는 (전-후) 가중합으로, 양수면 개선(큐·리드타임·구역 혼잡 감소)을 뜻한다.
    none(개입 없음) 결정은 기록하지 않는다.
    """

    def __init__(self, config: "AgentConfig"):
        self.config = config
        self.outcomes: list[InterventionOutcome] = []
        self._pending: InterventionOutcome | None = None

    @staticmethod
    def _metrics(snapshot: "Snapshot", zone: tuple[int, int] | None) -> dict[str, float]:
        return {
            "queue": float(snapshot.queue_len),
            "lead_time": float(snapshot.recent_lead_time),
            "zone_congestion": _zone_congestion(snapshot, zone),
        }

    def evaluate_pending(self, snapshot: "Snapshot") -> InterventionOutcome | None:
        """직전 개입을 현재 스냅샷(after)으로 평가해 점수·라벨 확정"""
        if self._pending is None:
            return None
        cfg = self.config
        o = self._pending
        o.after = self._metrics(snapshot, o.target_zone)
        d_queue = o.before["queue"] - o.after["queue"]
        d_lead = o.before["lead_time"] - o.after["lead_time"]
        d_cong = o.before["zone_congestion"] - o.after["zone_congestion"]
        o.effect_score = (
            cfg.effect_w_queue * d_queue
            + cfg.effect_w_lead * d_lead
            + cfg.effect_w_congestion * d_cong
        )
        if o.effect_score > cfg.effect_eps:
            o.label = "개선"
        elif o.effect_score < -cfg.effect_eps:
            o.label = "악화"
        else:
            o.label = "변화없음"
        self.outcomes.append(o)
        self._pending = None
        return o

    def record(
        self,
        time: float,
        decision: dict,
        diagnosis: dict,
        snapshot: "Snapshot",
        timeline_idx: int,
    ) -> None:
        """적용된 개입의 before 지표를 기록해 다음 평가를 예약 (none은 무시)"""
        action = decision.get("action", "none")
        if action == "none":
            return
        tz = diagnosis.get("target_zone") if isinstance(diagnosis, dict) else None
        zone = tuple(tz) if isinstance(tz, (list, tuple)) and len(tz) == 2 else None
        self._pending = InterventionOutcome(
            time=time,
            action=action,
            params=decision.get("params") or {},
            target_zone=zone,
            before=self._metrics(snapshot, zone),
            timeline_idx=timeline_idx,
        )

    def reflection_text(self) -> str:
        """최근 평가된 개입 이력을 진단·대응에 주입할 자연어 요약으로 변환"""
        recent = self.outcomes[-self.config.reflection_window :]
        if not recent:
            return ""
        lines = []
        for o in recent:
            zone = f" 구역{list(o.target_zone)}" if o.target_zone else ""
            lines.append(
                f"t={o.time:.0f} {o.action}{zone} "
                f"큐 {o.before['queue']:.0f}->{o.after['queue']:.0f} "
                f"{o.label}(점수 {o.effect_score:+.1f})"
            )
        return (
            "최근 개입 효과 이력(악화한 접근은 지양, 개선한 접근은 우선 고려):\n"
            + "\n".join(lines)
        )

    def efficacy_summary(self) -> dict:
        """개입 효율 집계 (개선·악화·변화없음 건수와 평균 점수)"""
        scored = [o for o in self.outcomes if o.effect_score is not None]
        counts = {"개선": 0, "변화없음": 0, "악화": 0}
        for o in scored:
            counts[o.label] = counts.get(o.label, 0) + 1
        avg = sum(o.effect_score for o in scored) / len(scored) if scored else 0.0
        return {"total": len(scored), "counts": counts, "avg_score": round(avg, 2)}
