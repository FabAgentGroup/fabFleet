"""RAG 검색·진단 통합 테스트 (D3)"""

from oht_sim.agents.diagnoser import Diagnoser
from oht_sim.agents.llm import ScriptedLLMClient
from oht_sim.agents.monitor import Anomaly
from oht_sim.agents.rag import Retriever, default_corpus_dir, load_corpus, tokenize
from oht_sim.agents.state import Snapshot, ZoneStat


def _retriever() -> Retriever:
    return Retriever(load_corpus(default_corpus_dir()))


def _snapshot() -> Snapshot:
    return Snapshot(
        time=100.0, queue_len=20, queue_trend=5, vehicle_states={"IDLE": 2},
        utilization_now=0.7, zones=[ZoneStat((0, 0), 3, 9, 2)],
        recent_deadlocks=[(1, 1)], recent_lead_time=80.0, dispatch_policy="nearest",
    )


# ----- 리트리버 -----

def test_tokenize_keeps_korean_and_alnum():
    toks = tokenize("OHT 교착 INC-001")
    assert "oht" in toks and "교착" in toks and "001" in toks


def test_corpus_loads_without_readme():
    docs = load_corpus(default_corpus_dir())
    ids = {d.id for d in docs}
    assert "INC-001-zone-deadlock" in ids
    assert not any("readme" in d.id.lower() for d in docs)


def test_search_finds_relevant_case():
    r = _retriever()
    top = r.search("교차로 정면 교착 반복", k=1, method="hybrid")
    assert top and top[0][0].id == "INC-001-zone-deadlock"


def test_methods_all_return_results():
    r = _retriever()
    for method in ("bm25", "tfidf", "hybrid", "lsa", "hybrid_lsa"):
        assert r.search("배차 쏠림 가동률 편차", k=3, method=method)


def test_lsa_latent_space_shapes():
    r = _retriever()
    assert r._doc_latent.shape[0] == len(r.docs)  # 문서당 잠재 벡터
    assert r._Vk.shape[0] == len(r.vocab)  # 용어 x rank 투영


def test_lsa_deterministic():
    r1, r2 = _retriever(), _retriever()
    a = [d.id for d, _ in r1.search("정면 교착", k=3, method="lsa")]
    b = [d.id for d, _ in r2.search("정면 교착", k=3, method="lsa")]
    assert a == b  # 시드·SVD 결정적


def test_hybrid_lsa_finds_relevant_case():
    r = _retriever()
    top = r.search("교차로 정면 교착 반복", k=1, method="hybrid_lsa")
    assert top and top[0][0].id == "INC-001-zone-deadlock"


# ----- diagnoser RAG 통합 -----

def test_diagnoser_injects_retrieved_cases_into_prompt():
    llm = ScriptedLLMClient(responses={
        "diagnoser": {"root_cause": "교착", "reasoning": "사례 근거", "target_zone": [0, 0],
                      "confidence": 0.8, "cited_sources": ["INC-001-zone-deadlock"]}
    })
    diag = Diagnoser(llm, retriever=_retriever())
    anomalies = [Anomaly("zone_deadlock", "high", (0, 0), "구역(0,0) 최근 교착 2")]
    out = diag.diagnose(_snapshot(), anomalies)
    # 검색 사례가 프롬프트에 주입됨
    user = llm.calls[-1].user
    assert "참고 사례" in user
    assert "INC-001-zone-deadlock" in user
    assert out["cited_sources"] == ["INC-001-zone-deadlock"]


def test_diagnoser_without_retriever_has_no_cases():
    llm = ScriptedLLMClient(responses={"diagnoser": {"root_cause": "x"}})
    diag = Diagnoser(llm)  # retriever 미주입 -> 기존 동작
    diag.diagnose(_snapshot(), [Anomaly("queue_backlog", "high", None, "큐 20")])
    assert "참고 사례" not in llm.calls[-1].user
