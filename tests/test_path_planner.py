"""Unit tests for the A* path planner in src.path_planner.

Compatible with pytest when installed, and executable directly via Python.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union
import numpy as np

# Ensure workspace root is in sys.path when invoked directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pytest
except ImportError:
    # Graceful shim allowing direct execution without pytest installed
    class _PytestFallback:
        class raises:
            def __init__(self, expected_exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> None:
                self.expected_exception = expected_exception

            def __enter__(self) -> _PytestFallback.raises:
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected_exception} was not raised.")
                return issubclass(exc_type, self.expected_exception)

    pytest = _PytestFallback()  # type: ignore[assignment]

from src.environment import OccupancyGrid
from src.path_planner import AStarPlanner


def assert_valid_path(
    grid: OccupancyGrid,
    path: Optional[List[Tuple[int, int]]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    expected_length: Optional[int] = None,
) -> None:
    """Reusable assertion helper verifying all path correctness properties."""
    assert path is not None, f"Expected a valid path from {start} to {goal}, but got None."
    assert len(path) > 0, "Returned path must not be empty."
    assert path[0] == start, f"Path start {path[0]} does not match requested start {start}."
    assert path[-1] == goal, f"Path goal {path[-1]} does not match requested goal {goal}."

    for i, (r, c) in enumerate(path):
        assert grid.is_in_bounds(r, c), f"Path waypoint ({r}, {c}) at index {i} is out of bounds."
        assert grid.is_free(r, c), f"Path waypoint ({r}, {c}) at index {i} is in an obstacle cell."

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
        assert manhattan_dist == 1, (
            f"Consecutive waypoints {path[i]} and {path[i + 1]} at index {i} do not have "
            f"Manhattan distance 1 (got {manhattan_dist}). Non-cardinal or diagonal step taken."
        )

    if expected_length is not None:
        assert len(path) == expected_length, (
            f"Path length {len(path)} does not match expected optimal length {expected_length}."
        )


def test_straight_unobstructed_path() -> None:
    """Test 1: Straight horizontal and vertical paths in free space."""
    grid = OccupancyGrid.from_dimensions(height=5, width=6, initial_value=0)
    planner = AStarPlanner(grid)

    # Horizontal path: (1, 0) to (1, 4) -> 5 cells
    path_h = planner.plan((1, 0), (1, 4))
    assert_valid_path(grid, path_h, start=(1, 0), goal=(1, 4), expected_length=5)

    # Vertical path: (0, 2) to (4, 2) -> 5 cells
    path_v = planner.plan((0, 2), (4, 2))
    assert_valid_path(grid, path_v, start=(0, 2), goal=(4, 2), expected_length=5)


def test_path_around_obstacle() -> None:
    """Test 2: Path routing around a solid vertical obstacle wall."""
    # 5x5 grid with vertical obstacle wall in column 2 from row 1 to row 3
    # Leaving row 0 and row 4 open
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    grid.set_cell(1, 2, 1)
    grid.set_cell(2, 2, 1)
    grid.set_cell(3, 2, 1)

    planner = AStarPlanner(grid)
    start = (2, 0)
    goal = (2, 4)
    path = planner.plan(start, goal)

    # Expected optimal path length:
    # (2,0) -> (1,0) -> (0,0) -> (0,1) -> (0,2) -> (0,3) -> (0,4) -> (1,4) -> (2,4) = 9 cells
    assert_valid_path(grid, path, start=start, goal=goal, expected_length=9)


def test_grid_multiple_possible_routes() -> None:
    """Test 3: Symmetrical obstacle with multiple equally short routes."""
    # 5x5 grid with central obstacle block at (2, 2)
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    grid.set_cell(2, 2, 1)

    planner = AStarPlanner(grid)
    start = (0, 2)
    goal = (4, 2)
    path = planner.plan(start, goal)

    # Both going left around (col 1) or right around (col 3) yield cost = 6 steps -> 7 cells
    assert_valid_path(grid, path, start=start, goal=goal, expected_length=7)


def test_completely_blocked_goal() -> None:
    """Test 4: Goal enclosed by obstacles is completely disconnected."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    # Enclose cell (2, 2) on all 4 cardinal sides
    grid.set_cell(1, 2, 1)  # Up
    grid.set_cell(3, 2, 1)  # Down
    grid.set_cell(2, 1, 1)  # Left
    grid.set_cell(2, 3, 1)  # Right

    planner = AStarPlanner(grid)
    path = planner.plan((0, 0), (2, 2))
    assert path is None, "Expected None for an unreachable walled-in goal."


def test_occupied_start_cell() -> None:
    """Test 5: Requesting an occupied start cell raises ValueError."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    grid.set_cell(1, 1, 1)
    planner = AStarPlanner(grid)

    with pytest.raises(ValueError):
        planner.plan((1, 1), (3, 3))


def test_occupied_goal_cell() -> None:
    """Test 6: Requesting an occupied goal cell raises ValueError."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    grid.set_cell(3, 3, 1)
    planner = AStarPlanner(grid)

    with pytest.raises(ValueError):
        planner.plan((0, 0), (3, 3))


def test_out_of_bounds_start() -> None:
    """Test 7: Out-of-bounds start coordinates raise ValueError."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    planner = AStarPlanner(grid)

    with pytest.raises(ValueError):
        planner.plan((-1, 0), (2, 2))
    with pytest.raises(ValueError):
        planner.plan((4, 0), (2, 2))
    with pytest.raises(ValueError):
        planner.plan((0, -1), (2, 2))
    with pytest.raises(ValueError):
        planner.plan((0, 4), (2, 2))


def test_out_of_bounds_goal() -> None:
    """Test 8: Out-of-bounds goal coordinates raise ValueError."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    planner = AStarPlanner(grid)

    with pytest.raises(ValueError):
        planner.plan((0, 0), (-1, 2))
    with pytest.raises(ValueError):
        planner.plan((0, 0), (4, 2))
    with pytest.raises(ValueError):
        planner.plan((0, 0), (2, -1))
    with pytest.raises(ValueError):
        planner.plan((0, 0), (2, 4))


def test_start_equals_goal() -> None:
    """Test 9: When start equals goal, path of length 1 containing start is returned."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    planner = AStarPlanner(grid)

    path = planner.plan((2, 2), (2, 2))
    assert path == [(2, 2)]
    assert_valid_path(grid, path, start=(2, 2), goal=(2, 2), expected_length=1)


def test_malformed_coordinate_inputs() -> None:
    """Test 10: Non-sequence, invalid length, boolean, and non-integer coordinates raise TypeError or ValueError."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    planner = AStarPlanner(grid)

    # Invalid planner initialization
    with pytest.raises(TypeError):
        AStarPlanner(np.zeros((4, 4)))  # type: ignore[arg-type]

    # Non-sequence coordinate
    with pytest.raises(TypeError):
        planner.plan(0, (1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        planner.plan((1, 1), "goal")  # type: ignore[arg-type]

    # Wrong length
    with pytest.raises(ValueError):
        planner.plan((1, 2, 3), (1, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        planner.plan((1,), (1, 1))  # type: ignore[arg-type]

    # Float coordinates
    with pytest.raises(TypeError):
        planner.plan((1.5, 2), (1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        planner.plan((1, 1), (1, 2.5))  # type: ignore[arg-type]

    # Boolean coordinates
    with pytest.raises(TypeError):
        planner.plan((True, 1), (2, 2))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        planner.plan((1, 1), (False, 2))  # type: ignore[arg-type]


def test_narrow_corridor() -> None:
    """Test 11: Navigation through a narrow, winding serpentine 1-cell corridor."""
    # 5x5 labyrinth where only a winding corridor exists:
    # 0 0 0 1 1
    # 1 1 0 1 1
    # 1 1 0 0 0
    # 1 1 1 1 0
    # 1 1 1 1 0
    raw = np.array([
        [0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
    ], dtype=int)
    grid = OccupancyGrid(raw)
    planner = AStarPlanner(grid)

    start = (0, 0)
    goal = (4, 4)
    path = planner.plan(start, goal)

    # Unique corridor path:
    # (0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,3)->(2,4)->(3,4)->(4,4) = 9 cells
    expected_path = [
        (0, 0), (0, 1), (0, 2),
        (1, 2),
        (2, 2), (2, 3), (2, 4),
        (3, 4), (4, 4)
    ]
    assert path == expected_path
    assert_valid_path(grid, path, start=start, goal=goal, expected_length=9)


def test_grid_no_diagonal_shortcut() -> None:
    """Test 12: Diagonal corner-cutting is forbidden under four-connected motion."""
    # 2x2 checkerboard pattern:
    # [0, 1]
    # [1, 0]
    # In 8-connected motion, (0,0) could jump directly to (1,1).
    # In strict 4-connected motion, both cardinal paths (0,1) and (1,0) are blocked.
    # Therefore, (1,1) must be completely unreachable from (0,0).
    raw = np.array([
        [0, 1],
        [1, 0],
    ], dtype=int)
    grid = OccupancyGrid(raw)
    planner = AStarPlanner(grid)

    path = planner.plan((0, 0), (1, 1))
    assert path is None, "Diagonal corner cutting must not occur; (1,1) must be unreachable."


def run_all_tests() -> None:
    """Direct execution runner for environments where pytest CLI is not installed."""
    tests = [
        test_straight_unobstructed_path,
        test_path_around_obstacle,
        test_grid_multiple_possible_routes,
        test_completely_blocked_goal,
        test_occupied_start_cell,
        test_occupied_goal_cell,
        test_out_of_bounds_start,
        test_out_of_bounds_goal,
        test_start_equals_goal,
        test_malformed_coordinate_inputs,
        test_narrow_corridor,
        test_grid_no_diagonal_shortcut,
    ]
    print(f"Executing {len(tests)} test functions in test_path_planner...")
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  [PASS] {t.__name__}")
    print(f"All {passed}/{len(tests)} tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
