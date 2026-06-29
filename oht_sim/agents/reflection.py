"""개입 효과 평가 + reflection 원장 + 인과 가드 (L3 닫힌 루프)

관제 에이전트가 실행한 개입을 적용 시점의 운영 지표(before)와 함께 기록하고,
다음 스냅샷(after)과 비교해 효과 점수·라벨을 산출한다. 평가 결과는 reflection
요약으로 묶여 다음 진단·대응 프롬프트에 주입된다.

단순 전후 효과 점수(before-after)는 주변 부하 변동과 개입 효과가 섞여 인과를 가리지
못한다(D5 한계). 이를 보정하려고 개입 직전 추세(queue_trend)를 "개입 안 했을 때"의
반사실 투영으로 삼아, 인과 점수 = 반사실 투영 - 실측 후값으로 드리프트를 제거한다.
액션 유형별 인과 점수를 누적해, 충분히 탐색한 뒤 평균 인과가 음수인 액션은 거부하는
가드(should_act)를 제공한다(neurosymbolic: 학습된 통계로 LLM·규칙 결정을 감싼다).

LLM 없이 결정적으로 동작하며 점수·임계는 모두 config로 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """개입 한 건의 전/후 비교 기록 (단순 효과 + 드리프트 보정 인과)"""

    time: float
    action: str
    params: dict
    target_zone: tuple[int, int] | None
    before: dict[str, float]
    timeline_idx: int
    before_trend: float = 0.0  # 개입 직전 큐 추세 (반사실 드리프트)
    after: dict[str, float] | None = None
    effect_score: float | None = None
    label: str = "대기"  # 대기 | 개선 | 변화없음 | 악화
    causal_score: float | None = None  # 반사실 보정 인과 효과
    causal_label: str = "대기"

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
            "causal_score": (
                round(self.causal_score, 2) if self.causal_score is not None else None
            ),
            "causal_label": self.causal_label,
        }


class InterventionLedger:
    """개입 전/후 지표로 효과·인과를 평가하고 reflection 요약·액션 가드를 제공

    효과 점수는 (전-후) 가중합. 인과 점수는 개입 직전 추세로 반사실(개입 안 했을 때)을
    투영해 드리프트를 뺀 값으로, 주변 부하 변동과 개입 효과를 분리한다. none(개입 없음)
    결정은 기록하지 않는다.
    """

    def __init__(self, config: "AgentConfig"):
        self.config = config
        self.outcomes: list[InterventionOutcome] = []
        self._pending: InterventionOutcome | None = None
        self._action_causal: dict[str, list[float]] = {}  # 액션별 인과 점수 누적(가드)

    @staticmethod
    def _metrics(snapshot: "Snapshot", zone: tuple[int, int] | None) -> dict[str, float]:
        return {
            "queue": float(snapshot.queue_len),
            "lead_time": float(snapshot.recent_lead_time),
            "zone_congestion": _zone_congestion(snapshot, zone),
        }

    def _label(self, score: float) -> str:
        if score > self.config.effect_eps:
            return "개선"
        if score < -self.config.effect_eps:
            return "악화"
        return "변화없음"

    def evaluate_pending(self, snapshot: "Snapshot") -> InterventionOutcome | None:
        """직전 개입을 현재 스냅샷(after)으로 평가해 효과·인과 점수를 확정"""
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
        o.label = self._label(o.effect_score)

        # 반사실: 개입 안 했으면 큐는 추세대로 흘렀을 것 -> 인과 = 투영 - 실측
        counterfactual_queue = o.before["queue"] + o.before_trend
        causal_queue = counterfactual_queue - o.after["queue"]
        o.causal_score = causal_queue + cfg.causal_w_congestion * d_cong
        o.causal_label = self._label(o.causal_score)

        self._action_causal.setdefault(o.action, []).append(o.causal_score)
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
        """적용된 개입의 before 지표·추세를 기록해 다음 평가를 예약 (none은 무시)"""
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
            before_trend=float(snapshot.queue_trend),
        )

    def should_act(self, action: str) -> bool:
        """액션 가드 - 탐색 후 평균 인과 점수가 임계 미만이면 거부 (해로운 개입 차단)"""
        hist = self._action_causal.get(action, [])
        if len(hist) < self.config.guard_explore_n:
            return True  # 탐색 구간은 허용
        return (sum(hist) / len(hist)) >= self.config.guard_threshold

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
                f"{o.causal_label}(인과 {o.causal_score:+.1f}, 효과 {o.effect_score:+.1f})"
            )
        return (
            "최근 개입 효과 이력(인과 점수 기준, 인과 음수 접근은 지양):\n"
            + "\n".join(lines)
        )

    def efficacy_summary(self) -> dict:
        """개입 효율 집계 (효과·인과 라벨 건수와 평균 점수)"""
        scored = [o for o in self.outcomes if o.effect_score is not None]
        counts = {"개선": 0, "변화없음": 0, "악화": 0}
        causal_counts = {"개선": 0, "변화없음": 0, "악화": 0}
        for o in scored:
            counts[o.label] = counts.get(o.label, 0) + 1
            causal_counts[o.causal_label] = causal_counts.get(o.causal_label, 0) + 1
        avg = sum(o.effect_score for o in scored) / len(scored) if scored else 0.0
        avg_causal = (
            sum(o.causal_score for o in scored) / len(scored) if scored else 0.0
        )
        return {
            "total": len(scored),
            "counts": counts,
            "avg_score": round(avg, 2),
            "causal_counts": causal_counts,
            "avg_causal": round(avg_causal, 2),
        }
