"""정책 비교 실험 러너 스모크 테스트"""

from oht_sim.agents.llm import ScriptedLLMClient
from oht_sim.core.config import SimConfig
from oht_sim.experiments.policy_compare import run, _agg
from oht_sim.experiments.reflection_compare import run_config
from oht_sim.sim.simulator import Simulator


def test_lead_times_distribution():
    m = Simulator(SimConfig(num_vehicles=4, job_arrival_rate=0.3, sim_duration=200)).run()
    leads = m.lead_times()
    assert len(leads) == m.summary()["completed_jobs"]
    assert all(t >= 0 for t in leads)


def test_run_returns_results_for_each_combo():
    results = run(lambdas=[0.2, 0.3], seeds=[1, 2], num_vehicles=4, duration=120)
    # 정책 2종 x λ 2개 x 시드 2개 = 8
    assert len(results) == 2 * 2 * 2
    metrics = {"throughput", "avg_lead", "p95_lead", "avg_queue", "max_queue", "deadlocks"}
    assert all(metrics <= set(r.values) for r in results)


def test_aggregate_reproducible():
    args = dict(lambdas=[0.25], seeds=[1, 2], num_vehicles=4, duration=120)
    a = _agg(run(**args))
    b = _agg(run(**args))
    assert a == b  # 시드 고정 -> 결정적


# ----- D5 reflection 비교 (스크립트 LLM으로 API 없이 배선 검증) -----

def _scripted_factory():
    return lambda: ScriptedLLMClient(
        responses={
            "diagnoser": {"root_cause": "구역 과부하", "reasoning": "x", "target_zone": [0, 0], "confidence": 0.8},
            "responder": {"action": "set_dispatch_policy", "params": {"name": "least_busy"}, "rationale": "x"},
        }
    )


def test_reflection_compare_runs_both_configs():
    keys = {"avg_wait", "avg_lead", "p95_lead", "avg_queue", "interventions", "improved", "worsened", "avg_score"}
    on = run_config(_scripted_factory(), reflection=True, seed=1)
    off = run_config(_scripted_factory(), reflection=False, seed=1)
    assert keys <= set(on) and keys <= set(off)
    # 효과 측정은 두 설정 모두 수행(동일 기준 A/B) - 차이는 프롬프트 주입 여부
    assert on["interventions"] >= 1
    assert off["interventions"] >= 1


def test_reflection_compare_reproducible():
    a = run_config(_scripted_factory(), reflection=True, seed=3)
    b = run_config(_scripted_factory(), reflection=True, seed=3)
    assert a == b  # 시드·스크립트 고정 -> 결정적


# ----- D7 유의성 비교 하니스 -----

def test_significance_collect_pairs_by_seed():
    from oht_sim.algorithms.dispatcher import LeastBusyDispatcher, NearestDispatcher
    from oht_sim.experiments.significance_compare import collect

    data = collect({"nearest": NearestDispatcher, "least_busy": LeastBusyDispatcher}, seeds=[1, 2, 3])
    assert set(data) == {"nearest", "least_busy"}
    assert all(len(v) == 3 for v in data.values())  # 시드별 쌍체
    assert all(w >= 0 for v in data.values() for w in v)


# ----- D9 고장 인지 관제 하니스 -----

def test_failure_supervision_arms_run():
    from oht_sim.experiments.failure_supervision_compare import run_arm

    base = run_arm(0.16, supervised=False, seed=3)
    sup = run_arm(0.16, supervised=True, seed=3)
    assert base["mitigations"] == 0.0  # 무관제는 개입 없음
    assert sup["mitigations"] >= 1.0  # 관제는 고장 저하에 개입
    assert base["throughput"] >= 0 and sup["avg_lead"] >= 0


# ----- D13 RL 보상 재설계 하니스 -----

def test_rl_reward_train_changes_policy():
    from oht_sim.algorithms.rl_dispatcher import MLPPolicyDispatcher
    from oht_sim.experiments.rl_reward_compare import REWARDS, train

    rl = MLPPolicyDispatcher(20, 20, hidden=8, lr=0.02, temperature=0.5, seed=1)
    before = rl.W2.copy()
    train(rl, REWARDS["lead"], episodes=4)  # 리드 보상으로 학습
    import numpy as np
    assert not np.array_equal(rl.W2, before)  # 보상 신호로 정책 변화


def test_rl_reward_run_has_metrics():
    from oht_sim.algorithms.dispatcher import NearestDispatcher
    from oht_sim.experiments.rl_reward_compare import _run

    r = _run(NearestDispatcher(), seed=1)
    assert {"avg_wait", "avg_lead", "throughput", "blocked_rate"} <= set(r)
    assert r["blocked_rate"] >= 0
