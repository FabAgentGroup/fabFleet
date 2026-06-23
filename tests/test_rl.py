"""강화학습 배차 테스트 (D4 선형 · D6 MLP)"""

import random

import numpy as np

from oht_sim.algorithms.rl_dispatcher import (
    MLPPolicyDispatcher,
    PolicyGradientDispatcher,
)
from oht_sim.core.job import Job
from oht_sim.core.layout import Station
from oht_sim.core.vehicle import Vehicle


def _job(jid: int, src: Station) -> Job:
    return Job(id=jid, src=src, dst=Station(99, (19, 19)), created_time=0.0)


def test_eval_mode_picks_nearest_when_no_load_weight():
    d = PolicyGradientDispatcher(20, 20, train=False)
    d.w = [1.0, 0.0, 0.0]  # 부하 가중 0 -> 거리만 -> nearest
    job = _job(0, Station(0, (0, 0)))
    near = Vehicle(0, (1, 0))
    far = Vehicle(1, (10, 10))
    pairs = d.assign([job], [near, far])
    assert pairs == [(job, near)]


def test_assign_each_vehicle_once():
    d = PolicyGradientDispatcher(20, 20, train=False)
    s = Station(0, (5, 5))
    jobs = [_job(0, s), _job(1, s)]
    v = Vehicle(0, (5, 5))
    pairs = d.assign(jobs, [v])
    assert len(pairs) == 1  # 차량 1대 -> 1건만


def test_end_episode_updates_load_weight_within_bounds():
    d = PolicyGradientDispatcher(20, 20, lr=0.5, rng=random.Random(0))
    s = Station(0, (0, 0))
    # 학습 모드로 몇 번 배차해 grad 누적
    for _ in range(5):
        d.reset_episode()
        d.assign([_job(0, s), _job(1, s)], [Vehicle(0, (1, 1)), Vehicle(1, (9, 9))])
        d.end_episode(reward=-10.0)
    assert 0.0 <= d.w[1] <= 4.0  # 부하 가중은 [0,4]로 클리핑
    assert d.w[0] == 1.0 and d.w[2] == 0.0  # 거리·bias는 고정


def test_reset_episode_clears_counts():
    d = PolicyGradientDispatcher(20, 20)
    d.assign([_job(0, Station(0, (0, 0)))], [Vehicle(0, (0, 0))])
    assert d._count
    d.reset_episode()
    assert d._count == {}


def test_training_changes_weight_from_zero():
    from oht_sim.experiments.rl_compare import train

    d = PolicyGradientDispatcher(20, 20, lr=0.1, temperature=0.25, rng=random.Random(1))
    rewards, w_load = train(d, episodes=6, seeds_per_ep=1)
    assert len(rewards) == 6
    assert w_load[-1] != 0.0  # 학습으로 부하 가중이 0에서 변함


# ----- D6 MLP 정책 -----

def _mlp_jobs(s: Station):
    return [_job(0, s), _job(1, s)]


def test_mlp_assigns_each_vehicle_once():
    d = MLPPolicyDispatcher(20, 20, train=False)
    s = Station(0, (5, 5))
    pairs = d.assign(_mlp_jobs(s), [Vehicle(0, (5, 5))])
    assert len(pairs) == 1  # 차량 1대 -> 1건


def test_mlp_features_are_six_dim():
    d = MLPPolicyDispatcher(20, 20)
    phi = d._features(_job(0, Station(0, (0, 0))), Vehicle(0, (1, 1)), n_pending=3, n_idle=2)
    assert phi.shape == (6,) and phi[-1] == 1.0  # bias 항


def test_mlp_eval_is_deterministic():
    s = Station(0, (0, 0))
    jobs, vs = _mlp_jobs(s), [Vehicle(0, (1, 1)), Vehicle(1, (9, 9))]
    a = MLPPolicyDispatcher(20, 20, seed=7, train=False).assign(jobs, vs)
    b = MLPPolicyDispatcher(20, 20, seed=7, train=False).assign(jobs, vs)
    assert [(j.id, v.id) for j, v in a] == [(j.id, v.id) for j, v in b]


def test_mlp_end_episode_updates_weights_within_bounds():
    d = MLPPolicyDispatcher(20, 20, lr=0.5, seed=0)
    s = Station(0, (0, 0))
    before = d.W1.copy()
    # 보상이 변해야 advantage(보상-baseline)가 0이 아니어서 갱신이 발생
    for i in range(5):
        d.reset_episode()
        d.assign(_mlp_jobs(s), [Vehicle(0, (1, 1)), Vehicle(1, (9, 9))])
        d.end_episode(reward=-10.0 - i)
    assert not np.array_equal(d.W1, before)  # 학습으로 가중 변화
    assert np.all(np.abs(d.W1) <= 5.0)  # [-5,5] 클리핑


def test_mlp_reset_clears_grad_and_counts():
    d = MLPPolicyDispatcher(20, 20)
    d.assign([_job(0, Station(0, (0, 0)))], [Vehicle(0, (0, 0))])
    d.reset_episode()
    assert d._count == {} and np.all(d.gW1 == 0.0)


def test_mlp_training_changes_policy():
    from oht_sim.experiments.rl_v2_compare import train

    d = MLPPolicyDispatcher(20, 20, hidden=8, lr=0.02, temperature=0.5, seed=1)
    before = d.W2.copy()
    curve = train(d, episodes=4, seeds_per_ep=1)
    assert len(curve) == 4
    assert not np.array_equal(d.W2, before)  # 학습으로 정책 변화
