"""L3 고장 인지 캡스톤 테스트 (P5/D9) - 스냅샷 가용 정보·감지·규칙 완화"""

from oht_sim.agents.monitor import Monitor
from oht_sim.agents.rule_supervisor import RuleSupervisor
from oht_sim.agents.state import Snapshot, SnapshotBuilder, ZoneStat
from oht_sim.core.config import AgentConfig, SimConfig
from oht_sim.core.vehicle import VehicleState
from oht_sim.sim.simulator import Simulator


def _snapshot(failed: int, availability: float, queue_len: int = 0) -> Snapshot:
    return Snapshot(
        time=100.0, queue_len=queue_len, queue_trend=0,
        vehicle_states={"IDLE": 1}, utilization_now=0.5,
        zones=[ZoneStat((0, 0), 3, 0, 0)], recent_deadlocks=[],
        recent_lead_time=50.0, dispatch_policy="nearest",
        failed_vehicles=failed, availability=availability,
    )


# ----- 스냅샷 가용 정보 (관찰 빈틈 메우기) -----

def test_snapshot_reports_availability():
    sim = Simulator(SimConfig(num_vehicles=4, sim_duration=1))
    sim.vehicles[0].state = VehicleState.FAILED
    snap = SnapshotBuilder(AgentConfig(num_zones=2)).build(sim)
    assert snap.failed_vehicles == 1
    assert snap.availability == 0.75
    assert "availability" in snap.to_struct()


def test_snapshot_full_availability_when_no_failure():
    sim = Simulator(SimConfig(num_vehicles=4, sim_duration=1))
    snap = SnapshotBuilder(AgentConfig()).build(sim)
    assert snap.failed_vehicles == 0 and snap.availability == 1.0


# ----- 감지 (fleet_degraded) -----

def test_monitor_flags_fleet_degraded():
    mon = Monitor(AgentConfig(availability_threshold=0.75))
    a = next(a for a in mon.detect(_snapshot(failed=3, availability=0.4)) if a.type == "fleet_degraded")
    assert a.severity == "high"  # 가용 40% < 0.5 -> high
    a2 = next(a for a in mon.detect(_snapshot(failed=1, availability=0.6)) if a.type == "fleet_degraded")
    assert a2.severity == "medium"  # 0.5 <= 가용 < 0.75 -> medium


def test_monitor_quiet_when_availability_ok():
    mon = Monitor(AgentConfig(availability_threshold=0.75))
    anomalies = mon.detect(_snapshot(failed=1, availability=0.9))
    assert not any(a.type == "fleet_degraded" for a in anomalies)


def test_monitor_no_degraded_without_failures():
    mon = Monitor(AgentConfig(availability_threshold=0.99))
    anomalies = mon.detect(_snapshot(failed=0, availability=1.0))
    assert not any(a.type == "fleet_degraded" for a in anomalies)


# ----- 규칙 관제 완화 -----

def test_rule_supervisor_mitigates_on_degraded():
    # 고장으로 가용 저하 -> least_busy 전환 등 개입 발생
    sim = Simulator(SimConfig(
        num_vehicles=6, job_arrival_rate=0.3, sim_duration=600,
        vehicle_failure=True, failure_mtbf=120.0, repair_time=40.0, random_seed=7,
    ))
    agent_cfg = AgentConfig(supervisor_interval=30.0, availability_threshold=0.75)
    sup = RuleSupervisor(agent_cfg)
    sim.attach_supervisor(sup, agent_cfg)
    sim.run()
    assert any("fleet_degraded" in e["anomalies"] for e in sup.timeline)
    assert any(e["actions"] for e in sup.timeline)  # 완화 액션 적용


def test_rule_supervisor_busiest_zone():
    sim = Simulator(SimConfig(num_vehicles=2, sim_duration=1))
    sup = RuleSupervisor(AgentConfig(num_zones=2))
    z = sup._busiest_zone(sim)  # 대기 없음 -> 기본 (0,0)
    assert z == (0, 0)
