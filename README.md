# FabFleet

반도체 FAB의 OHT 기반 자동물류반송시스템(AMHS)을 **이산사건 시뮬레이션**으로 구현하고,
그 위에 **LLM 멀티에이전트 관제 레이어**를 얹어 에이전트가 멀티로봇 운영을 감시·진단·개입하는
시스템입니다.

두 종류의 에이전트가 공존합니다. 물리적으로 움직이는 **OHT 에이전트(저수준 제어 대상)** 와,
이를 관제하는 **LLM 에이전트(고수준 제어 주체)** 입니다. 멀티에이전트를 제어 대상·제어 주체
양쪽에서 다룹니다.

## 핵심 특징

- **3-Layer 아키텍처** - 시뮬레이션(SimPy) · 로보틱스(A\* + MAPF) · 멀티에이전트 관제(LangGraph)
- **두 층위의 멀티에이전트** - 저수준(움직이는 OHT) + 고수준(관제 LLM)
- **닫힌 개입 루프** - 관제 에이전트가 관찰만 하지 않고 시뮬레이터에 개입하고, 그 효과를 지표로 검증
- **느슨한 레이어 결합** - EventBus(관찰) + 화이트리스트 액션 API(개입)로만 레이어 간 통신
- **전략 패턴** - Dispatcher·Router를 인터페이스로 두어 레이어 상승 시 알고리즘만 교체
- **정량 근거** - 핵심 의사결정마다 ablation·벤치마크·차트 (`experiments/*/results.md`)

## 아키텍처

```
[ Layer 3 ] LLM 멀티에이전트 관제      
            모니터링 → 진단(RAG) → 대응제안
                  ↕ (관찰 / 개입)
[ Layer 2 ] 경로탐색 + 충돌·교착 회피   ← 로보틱스 (필수)
            A* + MAPF
                  ↕
[ Layer 1 ] SimPy OHT 반송 시뮬레이터   ← 토대 (시뮬레이션)
```

### 레이어 간 인터페이스 (프로젝트의 척추)

```
시뮬레이터 ──(구조화 이벤트 / 스냅샷)──▶ 관제 에이전트(monitor → diagnose → respond)
     ▲                                                    │
     └──────────(제어 액션 API: 정책변경 / 구간통제 / 우선순위)─┘
```

- **관찰 (L1 → L3)** - 시뮬레이터가 모든 사건을 구조화 이벤트로 EventBus에 발행, 상태 스냅샷으로 집계해 LLM 입력 구성
- **개입 (L3 → L1)** - 관제 에이전트는 화이트리스트 액션만 실행 (`set_dispatch_policy`, `block_segment`, `reprioritize_job`, `rebalance_idle_vehicles`)



## 기술 스택

| 영역 | 선택 |
|:-:|:--|
| 언어 | Python 3.11+ |
| 시뮬레이션 엔진 | SimPy 4.x (이산사건 시뮬레이션) |
| 수치·분석 | NumPy, pandas |
| 시각화 | Matplotlib (animation, GIF 저장) |
| 경로계획 | 자체 구현 (A\*, MAPF) |
| 에이전트 오케스트레이션 | LangGraph |
| LLM | OpenAI API (gpt-4o-mini) |
| 테스트 | pytest |

L1·L2는 GPU·외부 API가 불필요합니다. L3에서만 LLM API 키가 필요하며 환경변수로 주입합니다.

## 디렉토리 구조

```
oht_sim/
├── core/          # config, layout, vehicle, job, events(EventBus)
├── algorithms/    # router(Manhattan/A*), dispatcher, mapf(예약테이블)
├── sim/           # simulator(SimPy 조립), metrics
├── agents/        # Layer 3: state, monitor, diagnoser, responder, graph
├── viz/           # Matplotlib 애니메이션 + 관제 오버레이
├── experiments/   # 정책 비교 실험 러너 (D1~Dn)
└── main.py        # 진입점
tests/             # 핵심 로직 단위 테스트
```

## 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m oht_sim.main      # 시뮬레이션 실행 + 지표 출력
pytest                      # 단위 테스트
```

> Layer 3 관제(`python -m oht_sim.main --supervise`) 실행 시 `OPENAI_API_KEY` 환경변수가 필요합니다.
