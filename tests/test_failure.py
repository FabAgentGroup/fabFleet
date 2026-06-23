"""차량 고장·수리 테스트 (L1 신뢰성, D8)"""

from oht_sim.core.config import SimConfig
from oht_sim.core.events import EventType
from oht_sim.core.job import Job
from oht_sim.core.vehicle import BUSY_STATES, VehicleState
from oht_sim.sim.simulator import Simulator


def _sim(**kw) -> Simulator:
    base = dict(num_vehicles=3, job_arrival_rate=0.3, sim_duration=1)
    base.update(kw)
    return Simulator(SimConfig(**base))


def test_failed_state_not_busy_nor_idle():
    assert VehicleState.FAILED not in BUSY_STATES  # 가동률에서 제외
    assert VehicleState.IDLE not in BUSY_STATES


def test_fail_vehicle_requeues_job_and_sets_failed():
    sim = _sim()
    v = sim.vehicles[0]
    job = Job(id=999, src=sim.stations[0], dst=sim.stations[1], created_time=0.0)
    v.job = job
    sim._fail_vehicle(v)
    assert v.is_failed
    assert v.job is None
    assert job in sim.pending  # 미완 작업 회수
    assert any(e.type == EventType.VEHICLE_FAILED for e in sim.bus.log)


def test_failed_vehicle_excluded_from_dispatch():
    sim = _sim(num_vehicles=1)
    v = sim.vehicles[0]
    sim._fail_vehicle(v)
    job = Job(id=1, src=sim.stations[0], dst=sim.stations[1], created_time=0.0)
    sim.pending.append(job)
    sim._try_dispatch()
    assert job in sim.pending  # 고장 차량에는 배차되지 않음


def test_failure_off_emits_no_failure_events():
    sim = Simulator(SimConfig(num_vehicles=4, job_arrival_rate=0.3, sim_duration=300, random_seed=1))
    sim.run()
    assert not any(e.type == EventType.VEHICLE_FAILED for e in sim.bus.log)


def test_failures_and_repairs_occur_and_sim_completes():
    sim = Simulator(SimConfig(
        num_vehicles=6, job_arrival_rate=0.3, sim_duration=600,
        vehicle_failure=True, failure_mtbf=120.0, repair_time=40.0, random_seed=7,
    ))
    m = sim.run()
    fails = [e for e in sim.bus.log if e.type == EventType.VEHICLE_FAILED]
    reps = [e for e in sim.bus.log if e.type == EventType.VEHICLE_REPAIRED]
    assert len(fails) > 0 and len(reps) > 0
    assert m.summary()["completed_jobs"] > 0  # 고장 중에도 작업 완료


def test_failure_degrades_throughput_same_seed():
    common = dict(num_vehicles=5, job_arrival_rate=0.3, sim_duration=500, random_seed=7)
    base = Simulator(SimConfig(**common)).run().summary()
    fail = Simulator(SimConfig(
        vehicle_failure=True, failure_mtbf=100.0, repair_time=40.0, **common
    )).run().summary()
    assert fail["throughput"] <= base["throughput"]  # 고장은 처리량을 떨어뜨림
    assert fail["completed_jobs"] <= base["completed_jobs"]
