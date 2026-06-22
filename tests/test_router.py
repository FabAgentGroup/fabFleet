"""router 단위 테스트"""

from oht_sim.algorithms.router import ManhattanRouter
from oht_sim.core.layout import Grid, manhattan


def test_manhattan_path_connects_start_to_goal():
    grid = Grid(10, 10)
    router = ManhattanRouter()
    path = router.find_path(grid, (0, 0), (3, 2))
    assert path[0] == (0, 0)
    assert path[-1] == (3, 2)


def test_manhattan_path_length_matches_distance():
    grid = Grid(10, 10)
    router = ManhattanRouter()
    start, goal = (1, 1), (5, 4)
    path = router.find_path(grid, start, goal)
    # 셀 시퀀스 길이는 거리 + 1 (시작 셀 포함)
    assert len(path) == manhattan(start, goal) + 1


def test_manhattan_steps_are_adjacent():
    grid = Grid(10, 10)
    router = ManhattanRouter()
    path = router.find_path(grid, (2, 7), (6, 1))
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1


def test_same_start_goal_is_single_cell():
    grid = Grid(5, 5)
    path = ManhattanRouter().find_path(grid, (2, 2), (2, 2))
    assert path == [(2, 2)]
