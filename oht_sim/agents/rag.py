"""하이브리드 검색 리트리버 (BM25 + TF-IDF + RRF)

외부 임베딩 모델·API 없이 결정적으로 동작한다. 어휘 검색(BM25)과 벡터 검색
(TF-IDF 코사인)을 RRF로 융합한다. 시맨틱 임베딩은 드롭인 확장 지점이다.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[0-9a-z가-힣]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Doc:
    """지식 문서 (사례·규칙)"""

    id: str
    title: str
    text: str
    tokens: list[str]


def load_corpus(directory: str) -> list[Doc]:
    """디렉토리의 마크다운을 문서로 로드 (README는 제외)"""
    docs: list[Doc] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".md") or name.upper() == "README.MD":
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), name)
        docs.append(Doc(id=name[:-3], title=title, text=text, tokens=tokenize(text)))
    return docs


def default_corpus_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "knowledge")


class Retriever:
    """BM25·TF-IDF 인덱스 + RRF 융합 검색"""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.docs = docs
        self.k1, self.b, self.rrf_k = k1, b, rrf_k
        self._build()

    def _build(self) -> None:
        n = len(self.docs)
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        for d in self.docs:
            tf = Counter(d.tokens)
            self.tf.append(tf)
            self.dl.append(len(d.tokens))
            self.df.update(tf.keys())
        self.avgdl = (sum(self.dl) / n) if n else 0.0
        # BM25 idf, TF-IDF idf (smoothed)
        self.bm25_idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()
        }
        self.tfidf_idf = {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in self.df.items()}
        self._doc_norm = []
        for tf in self.tf:
            vec = {t: (1 + math.log(c)) * self.tfidf_idf[t] for t, c in tf.items()}
            self._doc_norm.append(math.sqrt(sum(w * w for w in vec.values())) or 1.0)

    def _bm25_scores(self, q: list[str]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = self.bm25_idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s += idf * freq * (self.k1 + 1) / denom
            if s > 0:
                scores[i] = s
        return scores

    def _tfidf_scores(self, q: list[str]) -> dict[int, float]:
        qtf = Counter(q)
        qvec = {t: (1 + math.log(c)) * self.tfidf_idf.get(t, 0.0) for t, c in qtf.items()}
        qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
        scores: dict[int, float] = {}
        for i, tf in enumerate(self.tf):
            dot = 0.0
            for t, qw in qvec.items():
                if t in tf:
                    dot += qw * (1 + math.log(tf[t])) * self.tfidf_idf[t]
            if dot > 0:
                scores[i] = dot / (qnorm * self._doc_norm[i])
        return scores

    @staticmethod
    def _rank(scores: dict[int, float]) -> list[int]:
        return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    def search(self, query: str, k: int = 3, method: str = "hybrid") -> list[tuple[Doc, float]]:
        """질의로 top-k 문서를 검색 (method: hybrid | bm25 | tfidf)"""
        q = tokenize(query)
        if method == "bm25":
            ranked = [(i, s) for i, s in sorted(self._bm25_scores(q).items(), key=lambda kv: -kv[1])]
        elif method == "tfidf":
            ranked = [(i, s) for i, s in sorted(self._tfidf_scores(q).items(), key=lambda kv: -kv[1])]
        else:
            bm = self._rank(self._bm25_scores(q))
            tf = self._rank(self._tfidf_scores(q))
            rrf: dict[int, float] = {}
            for lst in (bm, tf):
                for rank, i in enumerate(lst):
                    rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            ranked = sorted(rrf.items(), key=lambda kv: -kv[1])
        return [(self.docs[i], score) for i, score in ranked[:k]]
