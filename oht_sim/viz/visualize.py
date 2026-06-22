"""Matplotlib 애니메이션·관제 이벤트 오버레이"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # 헤드리스 렌더링

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from oht_sim.core.events import EventType
from oht_sim.core.vehicle import VehicleState

if TYPE_CHECKING:
    from oht_sim.sim.simulator import Simulator

# 상태별 색상
_STATE_COLOR = {
    VehicleState.IDLE.value: "#9aa0a6",
    VehicleState.MOVING_TO_PICKUP.value: "#1a73e8",
    VehicleState.LOADING.value: "#f9ab00",
    VehicleState.MOVING_TO_DROPOFF.value: "#34a853",
    VehicleState.UNLOADING.value: "#ea4335",
}


def _step_tracks(sim: "Simulator") -> tuple[dict, dict]:
    """이벤트 로그에서 차량별 (시각→위치)·(시각→상태) 계단 함수 구성"""
    pos = {v.id: [(0.0, sim.initial_positions[v.id])] for v in sim.vehicles}
    state = {v.id: [(0.0, VehicleState.IDLE.value)] for v in sim.vehicles}
    for e in sim.bus.log:
        if e.type == EventType.MOVE:
            pos[e.vehicle_id].append((e.time, e.location))
        elif e.type == EventType.STATE_CHANGE:
            state[e.vehicle_id].append((e.time, e.payload["to"]))
    return pos, state


def _sample(track: list[tuple[float, object]], t: float):
    times = [x[0] for x in track]
    return track[bisect.bisect_right(times, t) - 1][1]


def animate(
    sim: "Simulator",
    out_path: str = "outputs/oht_sim.gif",
    render_until: float | None = None,
    fps: int = 10,
) -> str:
    """반송 애니메이션을 렌더링하고 GIF로 저장, 저장 경로 반환"""
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cfg = sim.config
    dt = cfg.move_time_per_cell
    horizon = render_until if render_until is not None else min(cfg.sim_duration, 200.0)
    frames = [i * dt for i in range(int(horizon / dt) + 1)]

    pos, state = _step_tracks(sim)

    # 완료 작업수 시계열 (오버레이 텍스트용)
    done_times = sorted(e.time for e in sim.bus.log if e.type == EventType.DROPOFF)

    fig, ax = plt.subplots(figsize=(7, 7))

    def draw(frame_idx: int):
        ax.clear()
        t = frames[frame_idx]
        ax.set_xlim(-1, cfg.grid_width)
        ax.set_ylim(-1, cfg.grid_height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Station
        sx = [s.coord[0] for s in sim.stations]
        sy = [s.coord[1] for s in sim.stations]
        ax.scatter(sx, sy, marker="s", s=120, c="#dadce0", edgecolors="#5f6368", zorder=1)

        # OHT
        for v in sim.vehicles:
            x, y = _sample(pos[v.id], t)
            st = _sample(state[v.id], t)
            ax.scatter(x, y, s=160, c=_STATE_COLOR.get(st, "#000"), zorder=3)
            ax.text(x, y, str(v.id), color="white", ha="center", va="center", fontsize=8, zorder=4)

        completed = bisect.bisect_right(done_times, t)
        # matplotlib 기본 폰트에 한글 글리프 부재로 오버레이는 ASCII 사용
        ax.set_title(
            f"t={t:.0f}   done={completed}   queue={_queue_at(sim, t)}", fontsize=11
        )

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def _queue_at(sim: "Simulator", t: float) -> int:
    steps = [
        (e.time, e.payload["queue_len"])
        for e in sim.bus.log
        if e.type == EventType.STEP
    ]
    if not steps:
        return 0
    times = [x[0] for x in steps]
    idx = bisect.bisect_right(times, t) - 1
    return steps[idx][1] if idx >= 0 else 0
