"""평가 신뢰성 통계 테스트 (D7) - 부트스트랩 CI · 쌍체 순열검정"""

import numpy as np
import pytest

from oht_sim.experiments.stats import mean_ci, paired_diff_test


def test_mean_ci_brackets_mean():
    s = [10.0, 12.0, 11.0, 9.0, 13.0, 10.5]
    ci = mean_ci(s, seed=0)
    assert ci.ci_low <= ci.mean <= ci.ci_high
    assert ci.mean == pytest.approx(np.mean(s))
    assert ci.n == len(s)


def test_mean_ci_reproducible():
    s = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mean_ci(s, seed=7) == mean_ci(s, seed=7)


def test_mean_ci_empty():
    ci = mean_ci([], seed=0)
    assert ci.n == 0 and ci.mean == 0.0


def test_paired_detects_clear_difference():
    # b가 항상 a보다 5 큼 -> 차이 일관적 -> 유의
    a = [float(i) for i in range(20)]
    b = [x + 5.0 for x in a]
    t = paired_diff_test(a, b, seed=0)
    assert t.mean_diff == pytest.approx(-5.0)
    assert t.significant and t.p_value < 0.05
    assert t.ci_high < 0  # 차이 구간이 0을 포함하지 않음


def test_paired_nonsignificant_when_noise():
    # 부호가 뒤섞인 작은 차이 -> 유의하지 않음
    a = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1]
    b = [10.1, 10.9, 9.2, 10.4, 9.7, 10.0, 10.0, 9.9]
    t = paired_diff_test(a, b, seed=0)
    assert not t.significant
    assert t.ci_low < 0 < t.ci_high  # 0을 포함

def test_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_diff_test([1.0, 2.0], [1.0], seed=0)


def test_paired_reproducible():
    a = [1.0, 3.0, 2.0, 5.0, 4.0]
    b = [2.0, 2.0, 3.0, 4.0, 5.0]
    t1 = paired_diff_test(a, b, seed=3)
    t2 = paired_diff_test(a, b, seed=3)
    assert t1.p_value == t2.p_value and t1.ci_low == t2.ci_low
