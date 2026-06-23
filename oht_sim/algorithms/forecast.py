"""구역별 도착 수요 예측 (§8 예측 기반 사전 배차)"""

from __future__ import annotations


class DemandForecaster:
    """구역별 Job 도착 수요를 지수이동평균(EMA)으로 예측

    매 윈도우의 구역별 도착 수를 EMA로 누적해 다음 윈도우 수요를 추정한다.
    저비용·결정적이라 재현성이 보장된다.
    """

    def __init__(self, zones: int, alpha: float = 0.4):
        self.zones = zones
        self.alpha = alpha
        self._keys = [(zx, zy) for zx in range(zones) for zy in range(zones)]
        self.ema: dict[tuple[int, int], float] = {z: 0.0 for z in self._keys}
        self._window: dict[tuple[int, int], int] = {z: 0 for z in self._keys}

    def observe(self, zone: tuple[int, int]) -> None:
        """현재 윈도우에 구역 도착 1건 반영"""
        if zone in self._window:
            self._window[zone] += 1

    def end_window(self) -> None:
        """윈도우 종료 - 도착 수를 EMA에 접고 카운트 초기화"""
        for z in self._keys:
            self.ema[z] = self.alpha * self._window[z] + (1 - self.alpha) * self.ema[z]
            self._window[z] = 0

    def predict(self) -> dict[tuple[int, int], float]:
        """구역별 예상 수요(EMA) 반환"""
        return dict(self.ema)

    def hottest(self) -> tuple[int, int] | None:
        """예상 수요 최고 구역 (수요 전무 시 None)"""
        if not any(self.ema.values()):
            return None
        return max(self._keys, key=lambda z: self.ema[z])
