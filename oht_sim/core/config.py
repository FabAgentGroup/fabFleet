"""시나리오 파라미터 dataclass"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """시뮬레이션 시나리오 파라미터 (모든 값 주입, 매직넘버 금지)"""

    # 레이아웃
    grid_width: int = 20
    grid_height: int = 20
    num_stations: int = 10

    # 차량·작업
    num_vehicles: int = 5
    job_arrival_rate: float = 0.2  # 포아송 도착률 λ (단위시간당 평균 작업수)

    # 소요시간
    move_time_per_cell: float = 1.0
    load_time: float = 3.0
    unload_time: float = 3.0

    # 충돌·교착 회피 (L2)
    collision_avoidance: bool = True  # False면 L1 동작(충돌 무시)으로 비교
    mapf_window: int = 8  # 윈도우 협력 A* 계획 지평(틱)
    deadlock_threshold: int = 5  # 연속 대기 횟수 초과 시 교착으로 판정

    # 실행
    sim_duration: float = 1000.0
    snapshot_interval: float = 10.0  # STEP 이벤트·큐 길이 샘플링 주기
    random_seed: int = 42

    # 선택 주입 (미지정 시 시드 기반 생성)
    station_coords: list[tuple[int, int]] | None = None
    blocked_cells: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Layer 3 관제 에이전트 파라미터 (모든 임계치 주입)"""

    # 스냅샷·감시 주기
    supervisor_interval: float = 30.0  # 스냅샷·관제 그래프 실행 주기
    recent_window: float = 60.0  # 최근 이벤트 집계 윈도우
    num_zones: int = 2  # 격자 분할(축당) - num_zones x num_zones 구역

    # 이상 감지 임계치 (규칙 1차 필터)
    queue_threshold: int = 15  # 대기 큐 길이 이상 임계
    zone_block_threshold: int = 8  # 구역별 최근 회피 대기 이상 임계
    zone_deadlock_threshold: int = 1  # 구역별 최근 교착 이상 임계

    # LLM
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
