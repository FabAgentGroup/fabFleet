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
from oht_sim.core.layout import Coord, Grid, build_stations, manhattan
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
        self.job_gen = JobGenerator(
            self.stations, config.job_arrival_rate, self.rng
        )
        self.metrics = MetricsCollector(self.bus, config)

        self.reservation = ReservationTable()
        self._stuck: dict[int, int] = {v.id: 0 for v in self.vehicles}
        self._arrival: dict[int, simpy.Event] = {}

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
            self.env.process(self._run_job(vehicle, job))

    def _run_job(self, v: Vehicle, job: Job) -> Generator:
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
        self._set_state(v, VehicleState.IDLE)
        self._try_dispatch()

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

    # ----- 실행 -----

    def run(self) -> MetricsCollector:
        """시뮬레이션을 sim_duration까지 실행하고 지표 수집기 반환"""
        self.env.process(self._job_source())
        self.env.process(self._snapshot())
        if self.config.collision_avoidance:
            self.env.process(self._mover())
        self.env.run(until=self.config.sim_duration)
        self.metrics.finalize(self.config.sim_duration)
        return self.metrics
