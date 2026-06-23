"""agents 단위 테스트 - 모킹 LLM으로 그래프 흐름·파싱·액션 호출 검증 (§7.5)"""

from oht_sim.agents.graph import Supervisor
from oht_sim.agents.llm import ScriptedLLMClient
from oht_sim.agents.monitor import Monitor
from oht_sim.agents.reflection import InterventionLedger
from oht_sim.agents.responder import ActionExecutor
from oht_sim.agents.state import Snapshot, SnapshotBuilder, ZoneStat
from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.sim.simulator import Simulator


def _scripted() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        responses={
            "diagnoser": {
                "root_cause": "구역 과부하",
                "reasoning": "배차 쏠림으로 특정 구역 혼잡",
                "target_zone": [0, 0],
                "confidence": 0.8,
            },
            "responder": {
                "action": "set_dispatch_policy",
                "params": {"name": "least_busy"},
                "rationale": "부하 균형으로 쏠림 완화",
            },
        }
    )


def _snapshot(queue_len: int, blocked: int = 0, deadlocks: int = 0) -> Snapshot:
    return Snapshot(
        time=100.0,
        queue_len=queue_len,
        queue_trend=queue_len,
        vehicle_states={"IDLE": 1},
        utilization_now=0.5,
        zones=[ZoneStat((0, 0), 3, blocked, deadlocks)],
        recent_deadlocks=[],
        recent_lead_time=50.0,
        dispatch_policy="nearest",
    )


# ----- monitor (규칙) -----

def test_monitor_flags_queue_backlog():
    mon = Monitor(AgentConfig(queue_threshold=10))
    anomalies = mon.detect(_snapshot(queue_len=20))
    assert any(a.type == "queue_backlog" for a in anomalies)


def test_monitor_flags_zone_deadlock():
    mon = Monitor(AgentConfig(queue_threshold=999, zone_deadlock_threshold=1))
    anomalies = mon.detect(_snapshot(queue_len=0, deadlocks=2))
    assert any(a.type == "zone_deadlock" and a.location == (0, 0) for a in anomalies)


def test_monitor_quiet_when_normal():
    mon = Monitor(AgentConfig(queue_threshold=999, zone_block_threshold=999, zone_deadlock_threshold=999))
    assert mon.detect(_snapshot(queue_len=3)) == []


# ----- 액션 실행 (화이트리스트) -----

def test_executor_rejects_non_whitelist():
    sim = Simulator(SimConfig(sim_duration=1))
    ex = ActionExecutor(sim)
    rec = ex.execute({"action": "delete_everything", "params": {}})
    assert not rec.applied and rec.error is not None


def test_set_dispatch_policy_changes_policy():
    sim = Simulator(SimConfig(sim_duration=1))
    assert sim.dispatch_policy_name == "nearest"
    sim.set_dispatch_policy("least_busy")
    assert sim.dispatch_policy_name == "least_busy"


def test_block_segment_adds_and_schedules_unblock():
    sim = Simulator(SimConfig(sim_duration=1))
    sim.block_segment([(5, 5)], duration=10)
    assert (5, 5) in sim.grid.blocked


# ----- 그래프 흐름 (모킹 LLM) -----

def _run_supervised(agent_cfg, sim_cfg):
    llm = _scripted()
    sim = Simulator(sim_cfg)
    sup = Supervisor(llm, agent_cfg, sim)
    sim.attach_supervisor(sup, agent_cfg)
    sim.run()
    return llm, sup, sim


def test_graph_triggers_diagnose_respond_on_anomaly():
    # 낮은 임계로 이상을 강제 -> diagnose·respond 실행, 액션 적용
    agent_cfg = AgentConfig(
        supervisor_interval=20.0, queue_threshold=1, model="mock"
    )
    sim_cfg = SimConfig(num_vehicles=2, job_arrival_rate=0.5, sim_duration=120)
    llm, sup, sim = _run_supervised(agent_cfg, sim_cfg)

    # LLM이 진단·대응에 호출됨
    agents_called = {c.agent for c in llm.calls}
    assert "diagnoser" in agents_called and "responder" in agents_called
    # 화이트리스트 액션이 실제 적용됨 (정책 교체)
    assert sim.dispatch_policy_name == "least_busy"
    assert any(r.applied for r in sup.executor.log)
    # 추론 타임라인 기록
    assert any(e["anomalies"] for e in sup.timeline)


def test_graph_skips_llm_when_no_anomaly():
    # 높은 임계 -> 이상 없음 -> LLM 미호출 (비용·재현성)
    agent_cfg = AgentConfig(
        supervisor_interval=20.0,
        queue_threshold=10_000,
        zone_block_threshold=10_000,
        zone_deadlock_threshold=10_000,
    )
    sim_cfg = SimConfig(num_vehicles=6, job_arrival_rate=0.1, sim_duration=120)
    llm, sup, sim = _run_supervised(agent_cfg, sim_cfg)

    assert llm.calls == []  # 이상 트리거 없으면 LLM 호출 없음
    assert sim.dispatch_policy_name == "nearest"  # 개입 없음


# ----- 개입 효과 평가 + reflection 닫힌 루프 -----

def _decision(action="set_dispatch_policy"):
    return {"action": action, "params": {"name": "least_busy"}, "rationale": "x"}


def test_ledger_scores_improvement():
    # 큐 20 -> 12 로 감소하면 개선
    ledger = InterventionLedger(AgentConfig())
    ledger.record(100.0, _decision(), {"target_zone": None}, _snapshot(queue_len=20), 0)
    o = ledger.evaluate_pending(_snapshot(queue_len=12))
    assert o is not None and o.label == "개선" and o.effect_score > 0


def test_ledger_scores_worsening():
    # 큐 12 -> 20 으로 증가하면 악화
    ledger = InterventionLedger(AgentConfig())
    ledger.record(100.0, _decision(), {"target_zone": None}, _snapshot(queue_len=12), 0)
    o = ledger.evaluate_pending(_snapshot(queue_len=20))
    assert o is not None and o.label == "악화" and o.effect_score < 0


def test_ledger_ignores_none_action():
    # none(개입 없음)은 평가 대상이 아님
    ledger = InterventionLedger(AgentConfig())
    ledger.record(100.0, {"action": "none"}, {}, _snapshot(queue_len=20), 0)
    assert ledger.evaluate_pending(_snapshot(queue_len=10)) is None


def test_ledger_uses_target_zone_congestion():
    # 대상 구역 혼잡 감소도 점수에 반영
    ledger = InterventionLedger(AgentConfig())
    ledger.record(100.0, _decision("block_segment"), {"target_zone": [0, 0]}, _snapshot(queue_len=5, blocked=9), 0)
    o = ledger.evaluate_pending(_snapshot(queue_len=5, blocked=1))
    assert o.before["zone_congestion"] == 9 and o.after["zone_congestion"] == 1
    assert o.label == "개선"


def test_reflection_text_injected_into_prompts():
    # 한 번 개입 후 다음 스텝의 진단·대응 프롬프트에 효과 이력이 들어감
    agent_cfg = AgentConfig(
        supervisor_interval=20.0, queue_threshold=1, model="mock", reflection=True
    )
    sim_cfg = SimConfig(num_vehicles=2, job_arrival_rate=0.5, sim_duration=160)
    llm, sup, sim = _run_supervised(agent_cfg, sim_cfg)

    # 평가된 개입이 생기고, 이후 호출 프롬프트에 reflection 머리말이 포함됨
    assert sup.ledger.outcomes
    assert any("최근 개입 효과 이력" in c.user for c in llm.calls)


def test_reflection_off_keeps_prompts_clean():
    agent_cfg = AgentConfig(
        supervisor_interval=20.0, queue_threshold=1, model="mock", reflection=False
    )
    sim_cfg = SimConfig(num_vehicles=2, job_arrival_rate=0.5, sim_duration=160)
    llm, sup, sim = _run_supervised(agent_cfg, sim_cfg)

    assert sup.ledger.outcomes == []
    assert all("최근 개입 효과 이력" not in c.user for c in llm.calls)


def test_timeline_effect_backfilled():
    # 개입 항목에 다음 스텝에서 효과가 소급 기록됨
    agent_cfg = AgentConfig(
        supervisor_interval=20.0, queue_threshold=1, model="mock", reflection=True
    )
    sim_cfg = SimConfig(num_vehicles=2, job_arrival_rate=0.5, sim_duration=160)
    llm, sup, sim = _run_supervised(agent_cfg, sim_cfg)

    assert any(e.get("effect") for e in sup.timeline)
    summary = sup.ledger.efficacy_summary()
    assert summary["total"] >= 1


# ----- 스냅샷 빌더 -----

def test_snapshot_builder_produces_fields():
    sim = Simulator(SimConfig(num_vehicles=4, job_arrival_rate=0.3, sim_duration=100))
    sim.run()
    snap = SnapshotBuilder(AgentConfig(num_zones=2)).build(sim)
    assert snap.queue_len >= 0
    assert len(snap.zones) == 4  # 2x2 구역
    assert "구조화 데이터" in snap.to_prompt()
