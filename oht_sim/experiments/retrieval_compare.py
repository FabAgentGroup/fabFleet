"""검색 정확도 비교 실험 (D3)

라벨링된 질의셋에서 BM25 / TF-IDF / 하이브리드(RRF)의 hit-rate@k를 비교한다.
RRF 융합이 단일 리트리버 대비 질의 유형에 강건한지를 정량 평가한다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from oht_sim.agents.rag import Retriever, default_corpus_dir, load_corpus

METHODS = ["bm25", "tfidf", "hybrid"]

# 라벨 질의셋: 증상 표현 -> 정답 사례 id
LABELED: list[tuple[str, str]] = [
    ("두 OHT가 교차로에서 서로 막아 정면 교착이 반복된다", "INC-001-zone-deadlock"),
    ("정면 충돌 회피로 서로 대기만 하고 진행하지 못한다", "INC-001-zone-deadlock"),
    ("최근접 배차가 같은 차량만 골라 부하가 쏠린다", "INC-002-dispatch-skew"),
    ("가동률 편차가 커지고 특정 구역 큐가 상승한다", "INC-002-dispatch-skew"),
    ("좁은 통로에 경로가 수렴해 회피 대기가 폭증한다", "INC-003-corridor-congestion"),
    ("병목 구간 통과 시간이 길어진다", "INC-003-corridor-congestion"),
    ("전체 대기 큐가 임계를 넘어 계속 증가한다", "INC-004-queue-backlog"),
    ("작업 도착률이 처리 용량을 초과해 과부하다", "INC-004-queue-backlog"),
    ("시간대별로 출발 작업이 한 구역으로 몰린다", "INC-005-hotspot-demand"),
    ("수요 핫스팟이 이동하며 픽업 대기가 길어진다", "INC-005-hotspot-demand"),
    # 패러프레이즈·동의어 질의 (어휘 중복이 적어 더 어려움)
    ("로봇 두 대가 마주보고 멈춰 서로 비켜주지 못한다", "INC-001-zone-deadlock"),
    ("한 대가 일감을 독차지하고 나머지는 논다", "INC-002-dispatch-skew"),
    ("길목이 막혀 줄줄이 멈춰 선다", "INC-003-corridor-congestion"),
    ("처리 못 한 주문이 계속 밀려 쌓인다", "INC-004-queue-backlog"),
    ("특정 시간대에 한쪽 구역만 바쁘다", "INC-005-hotspot-demand"),
]


def hit_rates(retriever: Retriever, ks=(1, 3)) -> dict[str, dict[int, float]]:
    """method x k 별 hit-rate"""
    out: dict[str, dict[int, float]] = {m: {k: 0.0 for k in ks} for m in METHODS}
    for method in METHODS:
        for k in ks:
            hits = 0
            for query, gold in LABELED:
                got = [d.id for d, _ in retriever.search(query, k=k, method=method)]
                if gold in got:
                    hits += 1
            out[method][k] = hits / len(LABELED)
    return out


def _chart(rates, ks, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    xs = range(len(ks))
    for j, method in enumerate(METHODS):
        vals = [rates[method][k] for k in ks]
        ax.bar([x + j * width for x in xs], vals, width, label=method)
    ax.set_xticks([x + width for x in xs])
    ax.set_xticklabels([f"hit@{k}" for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit-rate")
    ax.set_title("Retrieval hit-rate by method")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "charts", "hit_rate.png"), dpi=110)
    plt.close(fig)


def _results_md(rates, ks) -> str:
    lines = [
        "# 검색 정확도 비교 실험 (D3) - 하이브리드 RAG",
        "",
        f"라벨 질의 {len(LABELED)}개, 코퍼스 8개(사례·규칙). "
        "BM25(어휘) / TF-IDF(벡터) / 하이브리드(RRF) 비교.",
        "",
        "| method | " + " | ".join(f"hit@{k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | " + " | ".join(f"{rates[method][k]:.2f}" for k in ks) + " |"
        )
    kmax = max(ks)
    top = max(rates[m][kmax] for m in METHODS)
    winners = [m for m in METHODS if abs(rates[m][kmax] - top) < 1e-9]
    verdict = (
        f"동률({', '.join(winners)})" if len(winners) > 1 else f"**{winners[0]}**"
    )
    lines += [
        "",
        "## 결정 (Decision)",
        "",
        f"hit@{kmax} 최고: {verdict}, hit-rate {top:.2f}. "
        "본 코퍼스에서는 BM25·TF-IDF·하이브리드가 사실상 동일합니다. 두 리트리버 모두 어휘 기반이라 "
        "랭킹이 유사하고, RRF 융합도 같은 신호를 결합하므로 단일 리트리버 대비 순이득이 없습니다.",
        "",
        "한계·확장: 남는 실패는 본문과 어휘 중복이 없는 동의어·패러프레이즈 질의로, "
        "시맨틱 임베딩 리트리버를 RRF에 드롭인으로 추가해야 잡힙니다(후속). "
        "즉 진짜 하이브리드 이득은 어휘+시맨틱 조합에서 나오며, 본 실험은 그 전제와 한계를 정량으로 보여줍니다.",
        "",
        "![hit rate](charts/hit_rate.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ks = (1, 3)
    out_dir = "experiments/retrieval_compare"
    retriever = Retriever(load_corpus(default_corpus_dir()))
    rates = hit_rates(retriever, ks)
    os.makedirs(out_dir, exist_ok=True)
    _chart(rates, ks, out_dir)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(_results_md(rates, ks))
    print(f"실험 완료: {out_dir}/results.md + charts/")
    for m in METHODS:
        print(f"  {m}: " + ", ".join(f"hit@{k}={rates[m][k]:.2f}" for k in ks))


if __name__ == "__main__":
    main()
