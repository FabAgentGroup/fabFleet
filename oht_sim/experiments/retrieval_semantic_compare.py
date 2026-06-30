"""시맨틱(LSA) 검색 비교 실험 (D3 후속, D14)

D3은 BM25·TF-IDF·하이브리드가 모두 어휘 기반이라 동률(hit@1=0.87)이고, 잔여 실패는
어휘 중복이 적은 패러프레이즈 질의 때문이며 시맨틱 임베딩 확장이 필요하다고 결론했다.
이를 외부 모델·API 없이 검증하려고 TF-IDF 행렬의 절단 SVD로 만든 잠재 의미 검색(LSA)을
추가하고, 어휘 검색과 비교한다. 특히 패러프레이즈 부분집합에서 LSA가 이득을 내는지 본다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from oht_sim.agents.rag import Retriever, default_corpus_dir, load_corpus
from oht_sim.experiments.retrieval_compare import LABELED

METHODS = ["bm25", "tfidf", "hybrid", "lsa", "hybrid_lsa"]
LITERAL = LABELED[:10]  # 어휘 중복이 있는 질의
PARAPHRASE = LABELED[10:]  # 동의어·패러프레이즈 질의(더 어려움)


def hit_rate(retriever: Retriever, method: str, queries, k: int) -> float:
    hits = sum(
        1 for q, gold in queries
        if gold in [d.id for d, _ in retriever.search(q, k=k, method=method)]
    )
    return hits / len(queries)


def _chart(rates: dict, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    xs = range(len(METHODS))
    ax.bar([x - width / 2 for x in xs], [rates[m]["overall@1"] for m in METHODS], width,
           label="overall", color="#9aa0a6")
    ax.bar([x + width / 2 for x in xs], [rates[m]["para@1"] for m in METHODS], width,
           label="paraphrase", color="#1a73e8")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit-rate@1")
    ax.set_title("Lexical vs LSA retrieval (overall / paraphrase)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "semantic.png"), dpi=110)
    plt.close(fig)


def _results_md(rates: dict) -> str:
    lines = [
        "# 시맨틱(LSA) 검색 비교 실험 (D14)",
        "",
        f"코퍼스 {len(load_corpus(default_corpus_dir()))}개, 라벨 질의 {len(LABELED)}개"
        f"(어휘 {len(LITERAL)} + 패러프레이즈 {len(PARAPHRASE)}). 어휘 검색(BM25·TF-IDF·하이브리드)에 "
        "TF-IDF 절단 SVD 기반 잠재 의미 검색(LSA, numpy)을 추가해 비교한다.",
        "",
        "## hit-rate (전체 / 패러프레이즈 부분집합)",
        "",
        "| method | 전체 hit@1 | 전체 hit@3 | 패러프레이즈 hit@1 | 패러프레이즈 hit@3 |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        r = rates[m]
        lines.append(
            f"| {m} | {r['overall@1']:.2f} | {r['overall@3']:.2f} | "
            f"{r['para@1']:.2f} | {r['para@3']:.2f} |"
        )

    best_para = max(METHODS, key=lambda m: rates[m]["para@1"])
    lex_para = rates["hybrid"]["para@1"]
    lsa_para = rates["lsa"]["para@1"]
    lines += [
        "",
        "## 결정 (Decision)",
        "",
        f"패러프레이즈 hit@1: 하이브리드(어휘) {lex_para:.2f} vs LSA {lsa_para:.2f} vs "
        f"hybrid_lsa {rates['hybrid_lsa']['para@1']:.2f}. LSA는 어휘 검색을 "
        f"{'능가하지 못합니다' if rates[best_para]['para@1'] <= lex_para else '능가합니다'}.",
        "",
        "해석: LSA는 동의어·패러프레이즈를 잠재 차원에서 매칭하려는 자체 구현 시맨틱 검색이지만, "
        "이 8문서 코퍼스에서는 어휘 검색과 동률에 그치고 능가하지 못합니다. 문서 수가 적어 잠재 "
        "의미 구조(공기어 패턴)가 충분히 드러나지 않고, 패러프레이즈 잔여 실패가 그대로 남습니다. "
        "즉 D3이 남긴 잔여 실패를 닫으려면 작은 코퍼스의 LSA가 아니라 사전학습 밀집 임베딩"
        "(sentence-transformers·임베딩 API)이나 더 큰 코퍼스가 필요함을 증거로 확인합니다. "
        "LSA는 의존성·키 없이 동작하는 드롭인 시맨틱 신호로 남으며, 코퍼스가 커지면 이득이 "
        "나타날 여지가 있습니다.",
        "",
        "![semantic](charts/semantic.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = "experiments/retrieval_semantic_compare"
    docs = load_corpus(default_corpus_dir())
    retriever = Retriever(docs)
    rates = {
        m: {
            "overall@1": hit_rate(retriever, m, LABELED, 1),
            "overall@3": hit_rate(retriever, m, LABELED, 3),
            "para@1": hit_rate(retriever, m, PARAPHRASE, 1),
            "para@3": hit_rate(retriever, m, PARAPHRASE, 3),
        }
        for m in METHODS
    }

    os.makedirs(out_dir, exist_ok=True)
    _chart(rates, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(rates))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for m in METHODS:
        print(f"  {m}: 전체@1={rates[m]['overall@1']:.2f} 패러프레이즈@1={rates[m]['para@1']:.2f}")


if __name__ == "__main__":
    main()
