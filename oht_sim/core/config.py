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

    # 차량 고장·수리 (L1 신뢰성, §8 D8)
    vehicle_failure: bool = False  # True면 OHT가 확률적으로 고장·수리
    failure_mtbf: float = 300.0  # 차량당 평균 고장 간격(MTBF), 작을수록 잦은 고장
    repair_time: float = 40.0  # 평균 수리 소요(지수분포)

    # 충돌·교착 회피 (L2)
    collision_avoidance: bool = True  # False면 L1 동작(충돌 무시)으로 비교
    mapf_window: int = 8  # 윈도우 협력 A* 계획 지평(틱)
    deadlock_threshold: int = 5  # 연속 대기 횟수 초과 시 교착으로 판정

    # 혼잡 인지 동적 라우팅 (L2, DLWC 계열, §8 D10)
    congestion_aware_routing: bool = False  # True면 혼잡 가중을 라우팅 비용에 반영
    congestion_window: float = 10.0  # 혼잡장 갱신 주기·집계 윈도
    congestion_decay: float = 0.6  # 혼잡 EMA 감쇠
    congestion_alpha: float = 3.0  # 혼잡 페널티 배율
    congestion_blocked_weight: float = 3.0  # 회피 대기 1건의 통행 대비 가중

    # 시간 변동 핫스팟 수요 (§8 예측 배차용 시나리오)
    demand_hotspot: bool = False  # True면 출발 구역이 시간에 따라 쏠림
    demand_zones: int = 2  # 수요 구역 분할(축당)
    hotspot_period: float = 150.0  # 핫스팟 구역 전환 주기
    hotspot_weight: float = 0.6  # 출발지를 핫스팟 구역에서 뽑을 확률

    # 예측 기반 사전 배차 (§8, D2)
    predictive_dispatch: bool = False  # 예측 선제 재배치 활성화
    forecast_interval: float = 30.0  # 예측 갱신·선제 배치 주기
    forecast_alpha: float = 0.4  # 구역 수요 EMA 평활 계수
    predict_reposition_k: int = 2  # 주기당 선제 이동 유휴 OHT 수

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
    availability_threshold: float = 0.75  # 가용 차량 비율 하한 (미만이면 고장 저하 이상)

    # LLM
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # 개입 효과 평가·reflection 닫힌 루프 (L3)
    reflection: bool = True  # 직전 개입 효과를 다음 진단·대응에 반영
    reflection_window: int = 3  # 프롬프트에 주입할 최근 개입 이력 수
    effect_eps: float = 0.5  # 개선/악화 판정 점수 임계 (절댓값 이하면 변화없음)
    effect_w_queue: float = 1.0  # 효과 점수 - 대기 큐 감소 가중
    effect_w_lead: float = 0.1  # 효과 점수 - 리드타임 감소 가중
    effect_w_congestion: float = 1.0  # 효과 점수 - 대상 구역 혼잡 감소 가중
