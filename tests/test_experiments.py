"""정책 비교 실험 러너 스모크 테스트"""

from oht_sim.core.config import SimConfig
from oht_sim.experiments.policy_compare import run, _agg
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
