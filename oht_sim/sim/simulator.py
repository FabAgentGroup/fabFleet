"""SimPy env 구성·프로세스 조립·메인 루프"""

from __future__ import annotations

import random
from collections.abc import Generator

import simpy

from oht_sim.algorithms.dispatcher import Dispatcher, NearestDispatcher
from oht_sim.algorithms.router import ManhattanRouter, Router
from oht_sim.core.config import SimConfig
from oht_sim.core.events import Event, EventBus, EventType
from oht_sim.core.job import Job, JobGenerator
from oht_sim.core.layout import Coord, Grid, build_stations
from oht_sim.core.vehicle import Vehicle, VehicleState
from oht_sim.sim.metrics import MetricsCollector


class Simulator:
    """OHT 반송 이산사건 시뮬레이터 (L1 - 충돌 무시)"""

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

        self.router = router or ManhattanRouter()
        self.dispatcher = dispatcher or NearestDispatcher()
        self.job_gen = JobGenerator(
            self.stations, config.job_arrival_rate, self.rng
        )
        self.metrics = MetricsCollector(self.bus, config)

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

    # ----- 프로세스 -----

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
        idle = [v for v in self.vehicles if v.is_idle]
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
        yield from self._move(v, job.src.coord)
        self._set_state(v, VehicleState.LOADING)
        yield self.env.timeout(self.config.load_time)
        self._publish(
            EventType.PICKUP, job_id=job.id, vehicle_id=v.id, location=job.src.coord
        )

        self._set_state(v, VehicleState.MOVING_TO_DROPOFF)
        yield from self._move(v, job.dst.coord)
        self._set_state(v, VehicleState.UNLOADING)
        yield self.env.timeout(self.config.unload_time)
        self._publish(
            EventType.DROPOFF, job_id=job.id, vehicle_id=v.id, location=job.dst.coord
        )

        v.job = None
        self._set_state(v, VehicleState.IDLE)
        self._try_dispatch()

    def _move(self, v: Vehicle, goal: Coord) -> Generator:
        path = self.router.find_path(self.grid, v.pos, goal)
        v.path = path
        for cell in path[1:]:
            yield self.env.timeout(self.config.move_time_per_cell)
            v.pos = cell
            self._publish(EventType.MOVE, vehicle_id=v.id, location=cell)
        v.path = []

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
        self.env.run(until=self.config.sim_duration)
        self.metrics.finalize(self.config.sim_duration)
        return self.metrics
