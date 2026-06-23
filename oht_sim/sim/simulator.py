"""SimPy env 구성·프로세스 조립·메인 루프"""

from __future__ import annotations

import random
from collections.abc import Generator

import simpy

from oht_sim.algorithms.dispatcher import Dispatcher, NearestDispatcher
from oht_sim.algorithms.mapf import ReservationTable, plan_route
from oht_sim.algorithms.router import AStarRouter, ManhattanRouter, Router
from oht_sim.core.config import SimConfig
from oht_sim.core.events import Event, EventBus, EventType
from oht_sim.core.job import Job, JobGenerator
from oht_sim.core.layout import (
    Coord,
    Grid,
    build_stations,
    manhattan,
    zone_center,
    zone_of,
)
from oht_sim.core.vehicle import Vehicle, VehicleState
from oht_sim.sim.metrics import MetricsCollector


class Simulator:
    """OHT 반송 이산사건 시뮬레이터 (L2 - 충돌·교착 회피)

    이동은 틱 동기식 우선순위 계획으로 처리한다. 작업 프로세스는 목표를 지정하고
    도착 이벤트를 기다리며, 중앙 mover가 매 틱 모든 이동 차량을 충돌 없이 전진시킨다.
    collision_avoidance=False면 L1 동작(충돌 무시 직진)으로 되돌린다.
    """

    def __init__(
        self,
        config: SimConfig,
        router: Router | None = None,
        dispatcher: Dispatcher | None = None,
    ):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.env = simpy.Environment()
        self.bus = EventBus()

        self.grid = Grid(
            config.grid_width, config.grid_height, set(config.blocked_cells)
        )
        self.stations = build_stations(
            config.num_stations, self.grid, self.rng, config.station_coords
        )
        self.vehicles = self._spawn_vehicles(config.num_vehicles)
        self.initial_positions = {v.id: v.pos for v in self.vehicles}
        self.pending: list[Job] = []

        default_router = (
            AStarRouter() if config.collision_avoidance else ManhattanRouter()
        )
        self.router = router or default_router
        self.dispatcher = dispatcher or NearestDispatcher()
        self.dispatch_policy_name = "nearest"
        self.job_gen = JobGenerator(
            self.stations,
            config.job_arrival_rate,
            self.rng,
            grid_width=config.grid_width,
            grid_height=config.grid_height,
            hotspot=config.demand_hotspot,
            zones=config.demand_zones,
            hotspot_period=config.hotspot_period,
            hotspot_weight=config.hotspot_weight,
        )
        self.metrics = MetricsCollector(self.bus, config)

        self.reservation = ReservationTable()
        self._stuck: dict[int, int] = {v.id: 0 for v in self.vehicles}
        self._arrival: dict[int, simpy.Event] = {}

        # Layer 3 관제 (선택) - attach_supervisor로 주입
        self.supervisor = None
        self.agent_config = None

        # 예측 기반 사전 배차 (선택, §8)
        self.forecaster = None
        if config.predictive_dispatch:
            from oht_sim.algorithms.forecast import DemandForecaster

            self.forecaster = DemandForecaster(config.demand_zones, config.forecast_alpha)

    # ----- 구성 -----

    def _spawn_vehicles(self, n: int) -> list[Vehicle]:
        passable = [
            (x, y)
            for x in range(self.grid.width)
            for y in range(self.grid.height)
            if self.grid.passable((x, y))
        ]
        spots = self.rng.sample(passable, k=min(n, len(passable)))
        return [Vehicle(i, spots[i]) for i in range(n)]

    def _publish(
        self,
        type_: EventType,
        job_id: int | None = None,
        vehicle_id: int | None = None,
        location: Coord | None = None,
        payload: dict | None = None,
    ) -> None:
        self.bus.publish(
            Event(
                time=self.env.now,
                type=type_,
                job_id=job_id,
                vehicle_id=vehicle_id,
                location=location,
                payload=payload or {},
            )
        )

    def _set_state(self, v: Vehicle, state: VehicleState) -> None:
        old = v.state
        v.state = state
        self._publish(
            EventType.STATE_CHANGE,
            vehicle_id=v.id,
            location=v.pos,
            payload={"from": old.value, "to": state.value},
        )

    def _tick(self) -> int:
        return round(self.env.now / self.config.move_time_per_cell)

    # ----- 작업 프로세스 -----

    def _job_source(self) -> Generator:
        while True:
            yield self.env.timeout(self.job_gen.next_interarrival())
            job = self.job_gen.create(self.env.now)
            self.pending.append(job)
            if self.forecaster is not None:
                self.forecaster.observe(
                    zone_of(
                        job.src.coord,
                        self.grid.width,
                        self.grid.height,
                        self.config.demand_zones,
                    )
                )
            self._publish(
                EventType.JOB_CREATED,
                job_id=job.id,
                location=job.src.coord,
                payload={"dst": job.dst.coord},
            )
            self._try_dispatch()

    def _try_dispatch(self) -> None:
        if not self.pending:
            return
        idle = [v for v in self.vehicles if v.is_idle and not v.moving]
        if not idle:
            return
        for job, vehicle in self.dispatcher.assign(self.pending, idle):
            self.pending.remove(job)
            vehicle.job = job
            self._publish(
                EventType.JOB_ASSIGNED, job_id=job.id, vehicle_id=vehicle.id
            )
            self._set_state(vehicle, VehicleState.MOVING_TO_PICKUP)
            vehicle.proc = self.env.process(self._run_job(vehicle, job))

    def _run_job(self, v: Vehicle, job: Job) -> Generator:
        try:
            yield from self._goto(v, job.src.coord)
            self._set_state(v, VehicleState.LOADING)
            yield self.env.timeout(self.config.load_time)
            self._publish(
                EventType.PICKUP, job_id=job.id, vehicle_id=v.id, location=job.src.coord
            )

            self._set_state(v, VehicleState.MOVING_TO_DROPOFF)
            yield from self._goto(v, job.dst.coord)
            self._set_state(v, VehicleState.UNLOADING)
            yield self.env.timeout(self.config.unload_time)
            self._publish(
                EventType.DROPOFF, job_id=job.id, vehicle_id=v.id, location=job.dst.coord
            )

            v.job = None
            v.proc = None
            self._set_state(v, VehicleState.IDLE)
            self._try_dispatch()
        except simpy.Interrupt:
            # 고장으로 중단 - 정리(작업 회수·이동 상태)는 _fail_vehicle가 수행
            return

    def _goto(self, v: Vehicle, goal: Coord) -> Generator:
        """목표 도착까지 대기 (이동은 mover가 충돌 없이 수행)"""
        if v.pos == goal:
            return
        if not self.config.collision_avoidance:
            yield from self._move_simple(v, goal)
            return
        v.goal = goal
        v.moving = True
        ev = self.env.event()
        self._arrival[v.id] = ev
        yield ev

    def _move_simple(self, v: Vehicle, goal: Coord) -> Generator:
        """L1 동작 - 충돌 무시 직진 (비교용)"""
        path = self.router.find_path(self.grid, v.pos, goal)
        v.path = path
        for cell in path[1:]:
            yield self.env.timeout(self.config.move_time_per_cell)
            v.pos = cell
            self._publish(EventType.MOVE, vehicle_id=v.id, location=cell)
        v.path = []

    # ----- 틱 동기식 mover (우선순위 계획) -----

    def _mover(self) -> Generator:
        while True:
            yield self.env.timeout(self.config.move_time_per_cell)
            self._advance_tick()

    def _stationary_cells(self, exclude: Vehicle) -> set[Coord]:
        return {u.pos for u in self.vehicles if u is not exclude and not u.moving}

    def _advance_tick(self) -> None:
        movers = [v for v in self.vehicles if v.moving]
        if not movers:
            return

        ntick = self._tick() + 1
        res = self.reservation
        res.prune_before(self._tick())
        # 틱 시작 시점의 모든 차량 점유 셀 (이 셀로의 진입을 금지해 충돌·스왑 차단)
        occupied_now = {v.pos for v in self.vehicles}
        # 정지 차량은 다음 틱에도 제자리 점유
        for u in self.vehicles:
            if not u.moving:
                res.reserve(u.pos, ntick, u.id)

        # 우선순위: 오래 대기한 차량 먼저 (기아 방지), 동률은 id
        movers.sort(key=lambda v: (-self._stuck[v.id], v.id))

        for v in movers:
            nxt = self._desired_next(v, ntick, occupied_now)
            if nxt is None:
                res.reserve(v.pos, ntick, v.id)
                self._stuck[v.id] += 1
                self._publish(EventType.BLOCKED, vehicle_id=v.id, location=v.pos)
                if self._stuck[v.id] == self.config.deadlock_threshold:
                    self._publish(
                        EventType.DEADLOCK_DETECTED,
                        vehicle_id=v.id,
                        location=v.pos,
                        payload={"goal": v.goal},
                    )
                    self._recover_deadlock(v)
            else:
                res.reserve(nxt, ntick, v.id, frm=v.pos)
                v.pos = nxt
                self._stuck[v.id] = 0
                self._publish(EventType.MOVE, vehicle_id=v.id, location=nxt)
                if v.pos == v.goal:
                    v.moving = False
                    v.goal = None
                    v.path = []
                    ev = self._arrival.pop(v.id, None)
                    if ev is not None and not ev.triggered:
                        ev.succeed()

    def _desired_next(
        self, v: Vehicle, ntick: int, occupied_now: set[Coord]
    ) -> Coord | None:
        """다음 셀 결정

        정지 차량 회피 A* 방향을 우선하되, 틱 시작 시 점유된 셀(occupied_now)과 다음
        틱 예약 셀은 진입하지 않는다. 진전 가능한 셀이 없으면 대기(None).
        """
        blocked = self._stationary_cells(exclude=v)
        route = plan_route(self.grid, v.pos, v.goal, blocked)
        preferred = route[1] if len(route) >= 2 else None

        candidates: list[Coord] = []
        if preferred is not None:
            candidates.append(preferred)
        for n in sorted(self.grid.neighbors(v.pos), key=lambda c: manhattan(c, v.goal)):
            if n not in candidates:
                candidates.append(n)

        cur_dist = manhattan(v.pos, v.goal)
        free = [
            c
            for c in candidates
            if c not in occupied_now and self.reservation.cell_free(c, ntick, v.id)
        ]
        # 1차: 목표 방향(선호 셀) 또는 거리 비증가 셀로 전진
        for c in free:
            if c == preferred or manhattan(c, v.goal) <= cur_dist:
                return c
        # 2차: 교착 임계 초과 시 거리 증가라도 비켜서 정면 교착 해소 (escape)
        if self._stuck[v.id] >= self.config.deadlock_threshold and free:
            return free[0]
        return None

    def _recover_deadlock(self, requester: Vehicle) -> None:
        """goal을 점유한 유휴 차량을 빈 이웃으로 비켜 교착 회복 (rebalance 입문)

        비켜설 차량은 goal을 비울 한 칸 이동을 mover에 위임한다. 이동 완료 시
        mover가 moving 플래그를 내린다(별도 대기 프로세스 불필요).
        """
        goal = requester.goal
        ntick = self._tick() + 1
        for u in self.vehicles:
            if u is requester or u.moving or u.pos != goal:
                continue
            if u.state != VehicleState.IDLE:
                continue
            blocked = self._stationary_cells(exclude=u)
            for n in sorted(self.grid.neighbors(u.pos)):
                if n in blocked or not self.reservation.cell_free(n, ntick, u.id):
                    continue
                u.goal = n
                u.moving = True  # mover가 다음 틱부터 비켜세움
                break

    def _hot_zone_target(self, hot: tuple[int, int]) -> Coord:
        """핫존 내 Station 중심(실제 픽업 위치)으로 선제 배치 타깃 산출"""
        n = self.config.demand_zones
        in_hot = [
            s.coord
            for s in self.stations
            if zone_of(s.coord, self.grid.width, self.grid.height, n) == hot
        ]
        if not in_hot:
            return zone_center(hot, self.grid.width, self.grid.height, n)
        cx = round(sum(c[0] for c in in_hot) / len(in_hot))
        cy = round(sum(c[1] for c in in_hot) / len(in_hot))
        return (min(self.grid.width - 1, cx), min(self.grid.height - 1, cy))

    def _predictive_dispatch_proc(self) -> Generator:
        """예측 수요 최고 구역의 Station 중심으로 유휴 OHT 일부를 선제 재배치 (§8)

        슬랙(유휴 수 > 대기 큐)이 있을 때만 재배치해 dispatch 굶주림을 피한다.
        """
        n = self.config.demand_zones
        while True:
            yield self.env.timeout(self.config.forecast_interval)
            self.forecaster.end_window()
            hot = self.forecaster.hottest()
            if hot is None:
                continue
            target = self._hot_zone_target(hot)
            slack = sum(1 for v in self.vehicles if v.is_idle and not v.moving) - len(self.pending)
            budget = min(self.config.predict_reposition_k, max(0, slack))
            if budget <= 0:
                continue
            idle_far = [
                v
                for v in self.vehicles
                if v.is_idle
                and not v.moving
                and zone_of(v.pos, self.grid.width, self.grid.height, n) != hot
            ]
            idle_far.sort(key=lambda v: -manhattan(v.pos, target))
            moved = idle_far[:budget]
            for v in moved:
                v.goal = target
                v.moving = True
            if moved:
                self._publish(
                    EventType.ACTION,
                    location=target,
                    payload={"action": "predictive_reposition", "zone": list(hot), "moved": len(moved)},
                )

    def _snapshot(self) -> Generator:
        while True:
            yield self.env.timeout(self.config.snapshot_interval)
            idle = sum(1 for v in self.vehicles if v.is_idle)
            self._publish(
                EventType.STEP,
                payload={
                    "queue_len": len(self.pending),
                    "idle_vehicles": idle,
                    "busy_vehicles": len(self.vehicles) - idle,
                },
            )

    # ----- Layer 3 제어 액션 API (화이트리스트, §4.3 (2)) -----

    def set_dispatch_policy(self, name: str) -> None:
        """배차 정책 교체 (nearest | least_busy)"""
        from oht_sim.algorithms.dispatcher import (
            LeastBusyDispatcher,
            NearestDispatcher,
        )

        policies = {"nearest": NearestDispatcher, "least_busy": LeastBusyDispatcher}
        if name not in policies:
            raise ValueError(f"미지원 배차 정책: {name}")
        if name == self.dispatch_policy_name:
            return  # 동일 정책이면 무시 (상태 초기화 방지, 멱등)
        self.dispatcher = policies[name]()
        self.dispatch_policy_name = name
        self._publish(EventType.ACTION, payload={"action": "set_dispatch_policy", "name": name})
        self._try_dispatch()

    def block_segment(self, cells: list[Coord], duration: float) -> None:
        """특정 구간을 일시 통제(우회 유도) 후 자동 해제"""
        added = [c for c in cells if c not in self.grid.blocked]
        self.grid.blocked.update(added)
        self._publish(
            EventType.ACTION,
            location=added[0] if added else None,
            payload={"action": "block_segment", "cells": [list(c) for c in added], "duration": duration},
        )
        self.env.process(self._unblock_after(added, duration))

    def _unblock_after(self, cells: list[Coord], duration: float) -> Generator:
        yield self.env.timeout(duration)
        for c in cells:
            self.grid.blocked.discard(c)

    def reprioritize_job(self, job_id: int, priority: int) -> None:
        """대기 중인 작업의 우선순위 조정"""
        for j in self.pending:
            if j.id == job_id:
                j.priority = priority
                self._publish(
                    EventType.ACTION,
                    job_id=job_id,
                    payload={"action": "reprioritize_job", "priority": priority},
                )
                self._try_dispatch()
                return
        raise ValueError(f"대기 큐에 없는 작업: {job_id}")

    def rebalance_idle_vehicles(self, zone: tuple[int, int]) -> None:
        """유휴 OHT 일부를 지정 구역 중심으로 재배치"""
        n = self.agent_config.num_zones if self.agent_config else 2
        cx = int((zone[0] + 0.5) * self.grid.width / n)
        cy = int((zone[1] + 0.5) * self.grid.height / n)
        center = (
            min(self.grid.width - 1, cx),
            min(self.grid.height - 1, cy),
        )
        moved = 0
        for v in self.vehicles:
            if v.is_idle and not v.moving and moved < 2:
                v.goal = center
                v.moving = True
                moved += 1
        self._publish(
            EventType.ACTION,
            location=center,
            payload={"action": "rebalance_idle_vehicles", "zone": list(zone), "moved": moved},
        )

    # ----- 관제(supervisor) -----

    def attach_supervisor(self, supervisor, agent_config) -> None:
        """Layer 3 관제 에이전트 주입"""
        self.supervisor = supervisor
        self.agent_config = agent_config

    def _supervise(self) -> Generator:
        while True:
            yield self.env.timeout(self.agent_config.supervisor_interval)
            self.supervisor.step(self)

    # ----- 차량 고장·수리 (L1 신뢰성, §8 D8) -----

    def _failure_proc(self) -> Generator:
        """가동 차량 중 하나를 확률적으로 고장 처리 (지수 분포 고장 간격)

        전체 고장률 = 차량수 / MTBF. 고장 간격을 지수분포로 뽑아 가동 차량을 무작위
        선택해 정지시킨다. 가동 차량이 없으면 다음 간격까지 대기한다.
        """
        cfg = self.config
        while True:
            rate = len(self.vehicles) / cfg.failure_mtbf
            yield self.env.timeout(self.rng.expovariate(rate))
            operational = [v for v in self.vehicles if not v.is_failed]
            if not operational:
                continue
            self._fail_vehicle(self.rng.choice(operational))

    def _fail_vehicle(self, v: Vehicle) -> None:
        """차량을 고장 정지시키고 진행 작업을 회수한 뒤 수리를 예약"""
        if v.job is not None:
            # 미완 작업을 대기 큐로 회수해 다른 차량이 재수행
            self.pending.append(v.job)
            v.job = None
        # 이동·예약 상태 해제 (정지 장애물로 남아 다른 차량은 우회)
        v.moving = False
        v.goal = None
        v.path = []
        # 도착 이벤트는 제거만(트리거하지 않음) - 인터럽트로 _goto 대기를 깬다
        self._arrival.pop(v.id, None)
        self._stuck[v.id] = 0
        proc, v.proc = v.proc, None
        self._set_state(v, VehicleState.FAILED)
        self._publish(EventType.VEHICLE_FAILED, vehicle_id=v.id, location=v.pos)
        if proc is not None and proc.is_alive:
            proc.interrupt()
        self.env.process(self._repair_proc(v))

    def _repair_proc(self, v: Vehicle) -> Generator:
        """수리 시간 경과 후 차량을 가동(IDLE)으로 복귀시키고 배차 재개"""
        yield self.env.timeout(self.rng.expovariate(1.0 / self.config.repair_time))
        self._set_state(v, VehicleState.IDLE)
        self._publish(EventType.VEHICLE_REPAIRED, vehicle_id=v.id, location=v.pos)
        self._try_dispatch()

    # ----- 실행 -----

    def run(self) -> MetricsCollector:
        """시뮬레이션을 sim_duration까지 실행하고 지표 수집기 반환"""
        self.env.process(self._job_source())
        self.env.process(self._snapshot())
        if self.config.collision_avoidance:
            self.env.process(self._mover())
        if self.supervisor is not None:
            self.env.process(self._supervise())
        if self.forecaster is not None:
            self.env.process(self._predictive_dispatch_proc())
        if self.config.vehicle_failure:
            self.env.process(self._failure_proc())
        self.env.run(until=self.config.sim_duration)
        self.metrics.finalize(self.config.sim_duration)
        return self.metrics
