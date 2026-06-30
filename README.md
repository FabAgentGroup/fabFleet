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
- **전략 패턴** - Dispatcher·Router를 인터페이스로 두어 알고리즘만 교체
- **정량 근거** - 핵심 의사결정마다 벤치마크·차트 (`experiments/*/results.md`)

## 아키텍처

```
[ Layer 3 ] LLM 멀티에이전트 관제
            모니터링 → 진단(RAG) → 대응제안
                  ↕ (관찰 / 개입)
[ Layer 2 ] 경로탐색 + 충돌·교착 회피
            A* + MAPF
                  ↕
[ Layer 1 ] SimPy OHT 반송 시뮬레이터
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
| 검색(RAG) | 자체 구현 하이브리드 (BM25 + TF-IDF + RRF) |
| 테스트 | pytest |

L1·L2는 GPU·외부 API가 불필요합니다. L3 관제에서만 LLM API 키가 필요하며 환경변수로 주입합니다.

## 디렉토리 구조

```
oht_sim/
├── core/          # config, layout, vehicle, job, events(EventBus)
├── algorithms/    # router(A*), dispatcher(nearest/least_busy), mapf(예약테이블),
│                  #   forecast(수요 예측), rl_dispatcher(강화학습), congestion(혼잡장), pibt(PIBT 플래너)
├── sim/           # simulator(SimPy 조립), metrics(이벤트 소싱 지표)
├── agents/        # llm, state, monitor, diagnoser, responder, graph,
│                  #   rag(하이브리드 검색), knowledge/(지식 코퍼스), reflection(인과 평가·가드),
│                  #   rule_supervisor·guarded_supervisor(결정적 관제)
├── viz/           # Matplotlib 애니메이션 + 관제 오버레이
├── experiments/   # 비교 실험 러너 (policy/predictive/retrieval/rl)
└── main.py        # 진입점
experiments/       # 실험 결과·차트 (Dn: results.md + charts)
tests/             # 단위 테스트
```

## 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m oht_sim.main               # 시뮬레이션 실행 + 지표 출력
python -m oht_sim.main --gif         # 애니메이션 GIF 저장
python -m oht_sim.main --supervise   # LLM 관제 활성화 (OPENAI_API_KEY 필요)
python -m oht_sim.main --failure     # OHT 확률적 고장·수리 활성화 (L1 신뢰성)
python -m oht_sim.main --congestion  # 혼잡 인지 동적 라우팅 활성화 (L2)
python -m oht_sim.main --pibt        # PIBT 이동 계획 활성화 (L2)
pytest                               # 단위 테스트
```

> Layer 3 관제(`--supervise`) 실행 시 `OPENAI_API_KEY` 환경변수가 필요합니다(`.env.example` 참고).
> 키는 저장소에 저장하지 않습니다(`.env`는 `.gitignore` 처리).

## 실험 (정량 비교)

핵심 기술 결정은 동일 시나리오·시드에서 비교하고, 평균뿐 아니라 분산·p95(꼬리 위험)까지
봅니다. 각 실험의 수치표·차트는 `experiments/<name>/results.md`에 있습니다.

| 실험 | 내용 | 요약 결과 | 실행 |
|---|---|---|---|
| D1 정책 비교 | nearest vs least_busy x 부하 | 5시드 평균 nearest 우세(단일 시드 노이즈와 상반) | `python -m oht_sim.experiments.policy_compare` |
| D2 예측 배차 | 핫스팟 수요에 OHT 선제 배치 | 중부하(λ=0.20) 평균 대기 약 -69% | `python -m oht_sim.experiments.predictive_compare` |
| D3 RAG 검색 | BM25 / TF-IDF / 하이브리드 hit-rate | 어휘 코퍼스에서 동률(0.87), 시맨틱은 후속 | `python -m oht_sim.experiments.retrieval_compare` |
| D4 강화학습 배차 | 선형 정책 + REINFORCE | 학습 후 nearest 대비 대기 약 -10%(고분산) | `python -m oht_sim.experiments.rl_compare` |
| D6 강화 RL 배차 | 풍부한 상태 6차원 + 비선형 MLP 정책 | 선형 RL과 평균 동률, p95 꼬리만 소폭 개선(병목은 보상 설계) | `python -m oht_sim.experiments.rl_v2_compare` |
| D7 유의성 검정 | 30시드 쌍체 + 부트스트랩 CI·순열검정 | 배차 차이 모두 유의하지 않음(D4 -10%는 시드 노이즈 내) | `python -m oht_sim.experiments.significance_compare` |
| D8 차량 고장 영향 | 확률적 고장·수리(MTBF 스윕) | MTBF 150서 처리량 -31%·리드 +38%(단조 저하) | `python -m oht_sim.experiments.failure_compare` |
| D9 고장 인지 관제 | 고장 감지→완화(부하 스윕) | 경부하 리드 -3.1%, 고부하 효과 0(슬랙 의존) | `python -m oht_sim.experiments.failure_supervision_compare` |
| D10 혼잡 인지 라우팅 | 정적 A* vs 혼잡 가중(DLWC) | 회피 -64%·교착 -84%(유의), 처리량 동률 | `python -m oht_sim.experiments.congestion_compare` |
| D11 인과 평가·가드 | 단순 효과 vs 인과 + 액션 가드 | 악화 귀속 70%→52% 교정, 가드 개입 22→5 | `python -m oht_sim.experiments.causal_guard_compare` |
| D12 PIBT 이동 계획 | 임시 플래너 vs PIBT(차량 수 스윕) | 처리량 +10~16%(유의)·교착 -96~100%, 밀집서 이득↑ | `python -m oht_sim.experiments.pibt_compare` |
