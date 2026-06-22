"""simulator 단위 테스트"""

from oht_sim.core.config import SimConfig
from oht_sim.sim.simulator import Simulator


def _run(**overrides):
    cfg = SimConfig(sim_duration=500.0, **overrides)
    return Simulator(cfg).run()


def test_runs_and_produces_metrics():
    m = _run(num_vehicles=5, job_arrival_rate=0.2)
    s = m.summary()
    assert s["created_jobs"] > 0
    assert s["completed_jobs"] > 0
    assert s["completed_jobs"] <= s["created_jobs"]
    assert 0.0 <= s["utilization"] <= 1.0


def test_no_event_after_duration():
    cfg = SimConfig(sim_duration=300.0)
    sim = Simulator(cfg)
    sim.run()
    assert all(e.time <= cfg.sim_duration for e in sim.bus.log)


def test_more_vehicles_reduce_lead_time():
    # 동일 시드·부하에서 OHT를 늘리면 리드타임이 감소 (§5.9)
    few = _run(num_vehicles=2, job_arrival_rate=0.5).summary()
    many = _run(num_vehicles=10, job_arrival_rate=0.5).summary()
    assert many["avg_lead_time"] < few["avg_lead_time"]
    assert many["avg_wait_time"] < few["avg_wait_time"]


def test_reproducible_with_same_seed():
    a = _run(num_vehicles=4, job_arrival_rate=0.3, random_seed=7).summary()
    b = _run(num_vehicles=4, job_arrival_rate=0.3, random_seed=7).summary()
    assert a == b
