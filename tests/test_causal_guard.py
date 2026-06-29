"""인과 평가·액션 가드 테스트 (L3 neurosymbolic, D11)"""

from oht_sim.agents.guarded_supervisor import GuardedSupervisor
from oht_sim.agents.reflection import InterventionLedger
from oht_sim.agents.state import Snapshot, ZoneStat
from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.sim.simulator import Simulator


def _snap(queue: int, trend: int, blocked: int = 0) -> Snapshot:
    return Snapshot(
        time=100.0, queue_len=queue, queue_trend=trend,
        vehicle_states={"IDLE": 1}, utilization_now=0.5,
        zones=[ZoneStat((0, 0), 3, blocked, 0)], recent_deadlocks=[],
        recent_lead_time=50.0, dispatch_policy="nearest",
    )


def _decision(action="rebalance_idle_vehicles"):
    return {"action": action, "params": {}}


# ----- 인과 점수: 드리프트 보정 -----

def test_causal_credits_action_for_resisting_drift():
    # 큐 20, 추세 +6 -> 개입 안 했으면 26 예상. 실측 22면 인과는 양수(드리프트 저항)
    led = InterventionLedger(AgentConfig())
    led.record(100.0, _decision(), {"target_zone": None}, _snap(20, 6), 0)
    o = led.evaluate_pending(_snap(22, 0))
    # 단순 효과 = 20-22 = -2 (악화로 보임), 인과 = (20+6)-22 = +4 (실은 드리프트 저항)
    assert o.effect_score < 0 < o.causal_score


def test_causal_penalizes_action_worse_than_drift():
    # 큐 20, 추세 +2 -> 예상 22. 실측 28이면 인과 음수(개입이 더 악화)
    led = InterventionLedger(AgentConfig())
    led.record(100.0, _decision(), {"target_zone": None}, _snap(20, 2), 0)
    o = led.evaluate_pending(_snap(28, 0))
    assert o.causal_score < 0 and o.causal_label == "악화"


# ----- 가드 -----

def test_guard_explores_then_vetoes_harmful_action():
    cfg = AgentConfig(guard_explore_n=2, guard_threshold=0.0)
    led = InterventionLedger(cfg)
    assert led.should_act("rebalance_idle_vehicles")  # 이력 없음 -> 탐색 허용
    # 해로운 인과 점수 2건 누적
    for _ in range(2):
        led.record(0.0, _decision(), {"target_zone": None}, _snap(20, 0), 0)
        led.evaluate_pending(_snap(30, 0))  # 인과 = 20-30 = -10
    assert not led.should_act("rebalance_idle_vehicles")  # 탐색 후 음수 -> 거부


def test_guard_allows_helpful_action():
    cfg = AgentConfig(guard_explore_n=2, guard_threshold=0.0)
    led = InterventionLedger(cfg)
    for _ in range(2):
        led.record(0.0, _decision(), {"target_zone": None}, _snap(20, 8), 0)
        led.evaluate_pending(_snap(18, 0))  # 인과 = (20+8)-18 = +10
    assert led.should_act("rebalance_idle_vehicles")  # 인과 양수 -> 허용


# ----- 가드 관제 통합 -----

def test_guarded_supervisor_vetoes_more_than_naive():
    common = dict(num_vehicles=8, job_arrival_rate=0.4, sim_duration=400, random_seed=3)
    def run(guard):
        sim = Simulator(SimConfig(**common))
        ac = AgentConfig(supervisor_interval=20.0, queue_threshold=5)
        sup = GuardedSupervisor(ac, guard=guard)
        sim.attach_supervisor(sup, ac)
        sim.run()
        return sup.stats()
    naive, guarded = run(False), run(True)
    assert naive["vetoes"] == 0
    assert guarded["vetoes"] > 0  # 가드가 일부 개입 거부
    assert guarded["acts"] < naive["acts"]  # 개입 수 감소


def test_action_executor_guard_vetoes():
    # 가드가 특정 액션을 거부하면 적용되지 않고 거부 사유가 남음
    from oht_sim.agents.responder import ActionExecutor

    sim = Simulator(SimConfig(sim_duration=1))
    ex = ActionExecutor(sim, guard=lambda a: a != "set_dispatch_policy")
    rec = ex.execute({"action": "set_dispatch_policy", "params": {"name": "least_busy"}})
    assert not rec.applied and "가드 거부" in rec.error
    assert sim.dispatch_policy_name == "nearest"  # 미적용


def test_llm_supervisor_guards_learned_harmful_action():
    # LLM 관제(스크립트)에 action_guard 적용 -> 해로운 액션 학습 후 거부 발생
    from oht_sim.agents.graph import Supervisor
    from oht_sim.agents.llm import ScriptedLLMClient

    llm = ScriptedLLMClient(responses={
        "diagnoser": {"root_cause": "x", "reasoning": "x", "target_zone": None, "confidence": 0.5},
        "responder": {"action": "rebalance_idle_vehicles", "params": {"zone": [0, 0]}, "rationale": "x"},
    })
    sim = Simulator(SimConfig(num_vehicles=8, job_arrival_rate=0.4, sim_duration=500, random_seed=3))
    ac = AgentConfig(supervisor_interval=20.0, queue_threshold=5, model="mock", action_guard=True)
    sup = Supervisor(llm, ac, sim)
    sim.attach_supervisor(sup, ac)
    sim.run()
    assert any(r.error and "가드 거부" in r.error for r in sup.executor.log)  # 가드 발동


def test_efficacy_summary_has_causal_keys():
    led = InterventionLedger(AgentConfig())
    led.record(0.0, _decision(), {"target_zone": None}, _snap(20, 0), 0)
    led.evaluate_pending(_snap(15, 0))
    s = led.efficacy_summary()
    assert "avg_causal" in s and "causal_counts" in s and s["total"] == 1
