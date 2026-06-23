"""평가 신뢰성 통계 - 부트스트랩 신뢰구간 + 쌍체 순열 유의성 검정 (§8 D7)

다중 시드 실험 결과의 평균에 부트스트랩 신뢰구간을 붙이고, 같은 시드에서 측정한
두 방법의 차이가 우연인지 쌍체 부호뒤집기 순열검정으로 판정한다. scipy 없이 numpy로만
구현하며, 모든 난수에 시드를 고정해 재현성을 보장한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanCI:
    """평균과 부트스트랩 신뢰구간"""

    mean: float
    ci_low: float
    ci_high: float
    n: int

    def fmt(self) -> str:
        return f"{self.mean:.2f} [{self.ci_low:.2f}, {self.ci_high:.2f}]"


@dataclass
class PairedTest:
    """쌍체 차이(a-b) 유의성 검정 결과"""

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def fmt(self) -> str:
        mark = "유의" if self.significant else "유의하지 않음"
        return (
            f"Δ={self.mean_diff:+.2f} [{self.ci_low:+.2f}, {self.ci_high:+.2f}], "
            f"p={self.p_value:.3f} ({mark})"
        )


def mean_ci(
    samples: list[float],
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> MeanCI:
    """평균의 부트스트랩 백분위 신뢰구간 (재표본 평균 분포의 분위수)"""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n == 0:
        return MeanCI(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return MeanCI(float(arr.mean()), lo, hi, n)


def paired_diff_test(
    a: list[float],
    b: list[float],
    alpha: float = 0.05,
    n_perm: int = 10000,
    n_boot: int = 2000,
    seed: int = 0,
) -> PairedTest:
    """같은 시드 쌍 (a_i, b_i)의 차이가 유의한지 부호뒤집기 순열검정

    귀무가설: 차이 d_i = a_i - b_i 의 부호는 무작위(중앙 0). 각 d_i 부호를 무작위로
    뒤집어 평균 분포를 만들고, 관측 평균보다 극단인 비율을 양측 p값으로 본다.
    차이의 부트스트랩 신뢰구간도 함께 제공한다.
    """
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    if len(da) != len(db):
        raise ValueError("쌍체 검정은 같은 길이의 표본이 필요")
    d = da - db
    n = len(d)
    if n == 0:
        return PairedTest(0.0, 0.0, 0.0, 1.0, 0, alpha)
    obs = float(d.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * d).mean(axis=1)
    # 양측: 관측 평균 절댓값 이상이 나온 비율 (+1 보정으로 0 방지)
    p = float((np.sum(np.abs(perm_means) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return PairedTest(obs, lo, hi, p, n, alpha)
