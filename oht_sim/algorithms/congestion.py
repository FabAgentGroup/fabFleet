"""혼잡 인지 동적 라우팅용 혼잡장 (L2, Dynamic Link Weight Control 계열)

정적 최단경로(A*)는 특정 셀·링크를 과부하시켜 혼잡·교착을 유발한다. 최근 통행
밀도(MOVE)와 회피 대기(BLOCKED)를 셀별로 EMA로 누적해 혼잡 가중치를 만들고, 이를
라우팅 비용에 더해 트래픽을 분산한다. FAB AMHS의 동적 링크 가중 제어(DLWC)·혼잡
인지 라우팅 연구의 경량 결정적 구현이다.

회피 대기는 실제 혼잡 신호이므로 통행보다 큰 가중을 둔다. 모든 파라미터는 config
주입이며 시드와 무관하게 결정적으로 동작한다.
"""

from __future__ import annotations

from oht_sim.core.layout import Coord


class CongestionField:
    """셀별 혼잡 가중치 (최근 통행·회피의 EMA)"""

    def __init__(
        self,
        decay: float = 0.6,
        alpha: float = 3.0,
        blocked_weight: float = 3.0,
        prune_eps: float = 0.05,
    ):
        self.decay = decay  # EMA 감쇠 (1에 가까울수록 오래 기억)
        self.alpha = alpha  # 라우팅 비용에 더하는 혼잡 페널티 배율
        self.blocked_weight = blocked_weight  # 회피 대기 1건의 통행 대비 가중
        self.prune_eps = prune_eps
        self.weights: dict[Coord, float] = {}

    def observe(self, moves: dict[Coord, int], blocks: dict[Coord, int]) -> None:
        """최근 윈도의 셀별 통행·회피 수로 혼잡장을 EMA 갱신"""
        keys = set(self.weights) | set(moves) | set(blocks)
        nw: dict[Coord, float] = {}
        for c in keys:
            inflow = moves.get(c, 0) + self.blocked_weight * blocks.get(c, 0)
            v = self.decay * self.weights.get(c, 0.0) + inflow
            if v > self.prune_eps:
                nw[c] = v
        self.weights = nw

    def penalty(self, cell: Coord) -> float:
        """라우팅 비용에 더할 셀 진입 페널티 (>=0)"""
        return self.alpha * self.weights.get(cell, 0.0)

    def hottest(self, k: int = 5) -> list[tuple[Coord, float]]:
        """가장 혼잡한 셀 상위 k (디버깅·관제용)"""
        return sorted(self.weights.items(), key=lambda kv: -kv[1])[:k]
