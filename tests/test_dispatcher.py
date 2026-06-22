"""dispatcher 단위 테스트"""

from oht_sim.algorithms.dispatcher import NearestDispatcher
from oht_sim.core.job import Job
from oht_sim.core.layout import Station
from oht_sim.core.vehicle import Vehicle


def _job(jid: int, src: Station, dst: Station) -> Job:
    return Job(id=jid, src=src, dst=dst, created_time=0.0)


def test_assigns_nearest_vehicle():
    s0 = Station(0, (0, 0))
    s1 = Station(1, (9, 9))
    job = _job(0, s0, s1)
    near = Vehicle(0, (1, 0))  # src까지 거리 1
    far = Vehicle(1, (8, 8))  # src까지 거리 16
    pairs = NearestDispatcher().assign([job], [near, far])
    assert pairs == [(job, near)]


def test_no_idle_vehicle_yields_no_assignment():
    s0, s1 = Station(0, (0, 0)), Station(1, (5, 5))
    pairs = NearestDispatcher().assign([_job(0, s0, s1)], [])
    assert pairs == []


def test_each_vehicle_assigned_at_most_once():
    s0, s1 = Station(0, (0, 0)), Station(1, (5, 5))
    jobs = [_job(0, s0, s1), _job(1, s0, s1)]
    v = Vehicle(0, (0, 0))
    pairs = NearestDispatcher().assign(jobs, [v])
    assert len(pairs) == 1
    assert pairs[0][1] is v


def test_priority_job_assigned_first():
    s0, s1 = Station(0, (0, 0)), Station(1, (5, 5))
    low = _job(0, s0, s1)
    high = Job(id=1, src=s0, dst=s1, created_time=10.0, priority=5)
    v = Vehicle(0, (0, 0))
    pairs = NearestDispatcher().assign([low, high], [v])
    assert pairs[0][0] is high
