"""Unit tests for the CollisionChecker in src.collision_checker.

Compatible with pytest when installed, and executable directly via Python.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Type, Union
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

        @staticmethod
        def approx(val: float, rel: float = 1e-6, abs: float = 1e-12) -> Any:
            class _Approx:
                def __init__(self, target: float) -> None:
                    self.target = target

                def __eq__(self, other: Any) -> bool:
                    return math.isclose(self.target, float(other), rel_tol=rel, abs_tol=abs)

                def __repr__(self) -> str:
                    return f"approx({self.target})"

            return _Approx(val)

    pytest = _PytestFallback()  # type: ignore[assignment]

from src.collision_checker import CollisionChecker
from src.environment import OccupancyGrid
from src.robot import Robot


def test_robot_centered_in_free_region() -> None:
    """Test 1: Robot centered in a large free region is collision-free."""
    grid = OccupancyGrid.from_dimensions(height=6, width=6, initial_value=0)
    checker = CollisionChecker(grid, robot_radius=0.3)

    assert checker.is_pose_collision_free(3.0, 3.0) is True
    assert checker.is_pose_collision_free((3.0, 3.0)) is True
    assert checker.is_pose_collision_free((3.0, 3.0, 0.0)) is True

    robot = Robot(x=3.0, y=3.0, theta=0.5)
    assert checker.is_pose_collision_free(robot) is True


def test_robot_footprint_intersects_occupied_cell() -> None:
    """Test 2: Robot whose footprint intersects an occupied cell returns False."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    # Place obstacle at row 2, col 2 -> span [2.0, 3.0] x [2.0, 3.0]
    grid.set_cell(2, 2, 1)
    checker = CollisionChecker(grid, robot_radius=0.4)

    # Robot center at (1.8, 2.5): distance to obstacle edge is 0.2 < 0.4
    assert checker.is_pose_collision_free(1.8, 2.5) is False

    # Robot center inside obstacle at (2.5, 2.5)
    assert checker.is_pose_collision_free(2.5, 2.5) is False


def test_footprint_tangent_to_occupied_cell_boundary() -> None:
    """Test 3: Footprint tangent to an obstacle boundary is treated conservatively as collision."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    # Obstacle at row 2, col 2 -> [2.0, 3.0] x [2.0, 3.0]
    grid.set_cell(2, 2, 1)
    radius = 0.3
    checker = CollisionChecker(grid, robot_radius=radius)

    # Tangent to left edge (x=2.0) with center at x = 2.0 - 0.3 = 1.7, y = 2.5
    # Closest point is (2.0, 2.5), distance is exactly 0.3 == radius
    assert checker.is_pose_collision_free(1.7, 2.5) is False

    # Tangent to bottom edge (y=2.0) with center at x = 2.5, y = 2.0 - 0.3 = 1.7
    assert checker.is_pose_collision_free(2.5, 1.7) is False


def test_footprint_does_not_intersect_nearby_obstacle() -> None:
    """Test 4: Footprint close to but strictly outside an obstacle is collision-free."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    # Obstacle at row 2, col 2 -> [2.0, 3.0] x [2.0, 3.0]
    grid.set_cell(2, 2, 1)
    radius = 0.3
    checker = CollisionChecker(grid, robot_radius=radius)

    # Distance to left edge is 2.0 - 1.68 = 0.32 > 0.30
    assert checker.is_pose_collision_free(1.68, 2.5) is True

    # Distance to bottom edge is 2.0 - 1.68 = 0.32 > 0.30
    assert checker.is_pose_collision_free(2.5, 1.68) is True


def test_footprint_extending_beyond_map_boundaries() -> None:
    """Test 5: Footprint extending outside any outer map boundary is not collision-free."""
    # 5x5 grid with resolution 1.0, origin (0, 0) -> [0.0, 5.0] x [0.0, 5.0]
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    radius = 0.25
    checker = CollisionChecker(grid, robot_radius=radius)

    # Left boundary: center at (0.2, 2.5) -> left edge at 0.2 - 0.25 = -0.05 < 0.0
    assert checker.is_pose_collision_free(0.2, 2.5) is False

    # Right boundary: center at (4.8, 2.5) -> right edge at 4.8 + 0.25 = 5.05 > 5.0
    assert checker.is_pose_collision_free(4.8, 2.5) is False

    # Bottom boundary: center at (2.5, 0.2) -> bottom edge at 0.2 - 0.25 = -0.05 < 0.0
    assert checker.is_pose_collision_free(2.5, 0.2) is False

    # Top boundary: center at (2.5, 4.8) -> top edge at 4.8 + 0.25 = 5.05 > 5.0
    assert checker.is_pose_collision_free(2.5, 4.8) is False

    # Entirely outside map
    assert checker.is_pose_collision_free(-1.0, 2.5) is False
    assert checker.is_pose_collision_free(2.5, 6.0) is False

    # Safe margin inside all boundaries: center at (0.3, 2.5) -> left edge at 0.05 >= 0.0
    assert checker.is_pose_collision_free(0.3, 2.5) is True


def test_invalid_and_non_finite_pose_coordinates() -> None:
    """Test 6: Non-numeric and non-finite pose coordinates raise appropriate exceptions."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    checker = CollisionChecker(grid, robot_radius=0.2)

    with pytest.raises(ValueError):
        checker.is_pose_collision_free(float("nan"), 2.0)
    with pytest.raises(ValueError):
        checker.is_pose_collision_free(2.0, float("inf"))
    with pytest.raises(ValueError):
        checker.is_pose_collision_free((float("-inf"), 2.0))

    with pytest.raises(TypeError):
        checker.is_pose_collision_free("2.0", 2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        checker.is_pose_collision_free(2.0, [2.0])  # type: ignore[arg-type]


def test_boolean_coordinates_and_invalid_radius() -> None:
    """Test 7: Booleans rejected as numbers; invalid radius values rejected."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)

    # Boolean coordinate inputs rejected
    checker = CollisionChecker(grid, robot_radius=0.2)
    with pytest.raises(TypeError):
        checker.is_pose_collision_free(True, 2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        checker.is_pose_collision_free(2.0, False)  # type: ignore[arg-type]

    # Invalid radius in constructor
    with pytest.raises(TypeError):
        CollisionChecker(grid, robot_radius=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CollisionChecker(grid, robot_radius="0.2")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CollisionChecker(grid, robot_radius=0.0)
    with pytest.raises(ValueError):
        CollisionChecker(grid, robot_radius=-0.5)
    with pytest.raises(ValueError):
        CollisionChecker(grid, robot_radius=float("nan"))
    with pytest.raises(ValueError):
        CollisionChecker(grid, robot_radius=float("inf"))

    # Invalid grid instance
    with pytest.raises(TypeError):
        CollisionChecker(np.zeros((4, 4)), robot_radius=0.2)  # type: ignore[arg-type]


def test_non_unit_grid_resolution() -> None:
    """Test 8: Geometric collision detection with non-unit cell resolution (res = 0.5)."""
    # 4x4 grid with resolution 0.5 -> world extent [0.0, 2.0] x [0.0, 2.0]
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0, resolution=0.5)
    # Obstacle at row 1, col 1 -> span [0.5, 1.0] x [0.5, 1.0]
    grid.set_cell(1, 1, 1)
    radius = 0.15
    checker = CollisionChecker(grid, robot_radius=radius)

    # Robot at (0.2, 0.75): distance to obstacle left edge (x=0.5) is 0.5 - 0.2 = 0.3 > 0.15
    assert checker.is_pose_collision_free(0.2, 0.75) is True

    # Robot at (0.4, 0.75): distance to obstacle left edge is 0.5 - 0.4 = 0.1 < 0.15 -> collision
    assert checker.is_pose_collision_free(0.4, 0.75) is False


def test_nonzero_map_origin() -> None:
    """Test 9: Geometric collision detection with nonzero map origin (10.0, 20.0)."""
    grid = OccupancyGrid.from_dimensions(
        height=4, width=4, initial_value=0, resolution=1.0, origin=(10.0, 20.0)
    )
    # World extent: [10.0, 14.0] x [20.0, 24.0]
    # Obstacle at row 1, col 1 -> span [11.0, 12.0] x [21.0, 22.0]
    grid.set_cell(1, 1, 1)
    radius = 0.25
    checker = CollisionChecker(grid, robot_radius=radius)

    # Valid free pose
    assert checker.is_pose_collision_free(10.5, 20.5) is True

    # Collision with obstacle
    assert checker.is_pose_collision_free(11.5, 20.9) is False

    # Outside map below origin
    assert checker.is_pose_collision_free(9.9, 20.5) is False
    assert checker.is_pose_collision_free(10.5, 19.9) is False


def test_collision_near_middle_of_obstacle_edge() -> None:
    """Test 10: Detection of collision near the middle of an obstacle edge without reaching corners."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    # Obstacle at row 2, col 2 -> span [2.0, 3.0] x [2.0, 3.0]
    grid.set_cell(2, 2, 1)
    radius = 0.3
    checker = CollisionChecker(grid, robot_radius=radius)

    # Center circle at (1.85, 2.5): y is exactly at the midpoint of the vertical edge (2.0 to 3.0).
    # Corner (2.0, 2.0) is distance sqrt((1.85-2.0)^2 + (2.5-2.0)^2) = sqrt(0.0225 + 0.25) = 0.522 > 0.30
    # Corner (2.0, 3.0) is distance sqrt((1.85-2.0)^2 + (2.5-3.0)^2) = 0.522 > 0.30
    # Neither corner is inside the footprint!
    # But the edge midpoint (2.0, 2.5) is at distance |2.0 - 1.85| = 0.15 < 0.30 -> COLLISION!
    assert checker.is_pose_collision_free(1.85, 2.5) is False


def test_checker_does_not_mutate_occupancy_grid() -> None:
    """Test 11: Collision queries do not mutate or modify the OccupancyGrid."""
    grid = OccupancyGrid.from_dimensions(height=4, width=4, initial_value=0)
    grid.set_cell(1, 1, 1)
    grid_before = grid.grid

    checker = CollisionChecker(grid, robot_radius=0.2)
    _ = checker.is_pose_collision_free(1.0, 1.0)
    _ = checker.is_pose_collision_free(0.5, 0.5)
    _ = checker.is_pose_collision_free(-5.0, 10.0)

    grid_after = grid.grid
    assert np.array_equal(grid_before, grid_after)


def test_motion_free_movement_open_area() -> None:
    """Test 12: Motion across an open area is collision-free."""
    grid = OccupancyGrid.from_dimensions(height=6, width=6, initial_value=0)
    checker = CollisionChecker(grid, robot_radius=0.3)
    free, col_pose = checker.check_motion((1.0, 1.0, 0.0), (4.0, 4.0, 1.5))
    assert free is True
    assert col_pose is None
    assert checker.is_motion_collision_free((1.0, 1.0, 0.0), (4.0, 4.0, 1.5)) is True

    # Test with Robot instance objects
    r1 = Robot(x=1.0, y=1.0, theta=0.0)
    r2 = Robot(x=2.0, y=2.0, theta=0.5)
    assert checker.is_motion_collision_free(r1, r2) is True


def test_motion_approaching_obstacle() -> None:
    """Test 13: Motion approaching an obstacle stops safely before margin, collides when entering margin."""
    grid = OccupancyGrid.from_dimensions(height=6, width=6, initial_value=0)
    # Obstacle at row 3, col 3 -> [3.0, 4.0] x [3.0, 4.0]
    grid.set_cell(3, 3, 1)
    checker = CollisionChecker(grid, robot_radius=0.4)

    # Moving along y=3.5 from x=1.0 towards obstacle at x=3.0
    # Stop before margin at x=2.5: safe (distance 3.0 - 2.5 = 0.5 > 0.4)
    free_stop, col_stop = checker.check_motion((1.0, 3.5), (2.5, 3.5))
    assert free_stop is True
    assert col_stop is None

    # Continuing into margin to x=2.8: collides (distance 3.0 - 2.8 = 0.2 < 0.4)
    collided, col_pose = checker.check_motion((1.0, 3.5), (2.8, 3.5))
    assert collided is False
    assert col_pose is not None
    assert col_pose[0] >= 2.5
    assert col_pose[0] <= 2.8


def test_motion_tunneling_segment_crosses_obstacle() -> None:
    """Test 14: Motion segment crossing an obstacle where endpoints are free but intermediate path collides."""
    grid = OccupancyGrid.from_dimensions(height=6, width=6, initial_value=0)
    # Thin obstacle or single cell at row 3, col 3: [3.0, 4.0] x [3.0, 4.0]
    grid.set_cell(3, 3, 1)
    checker = CollisionChecker(grid, robot_radius=0.2)

    # Start at (2.0, 3.5) and end at (5.0, 3.5)
    # Both endpoints are completely collision-free
    assert checker.is_pose_collision_free(2.0, 3.5) is True
    assert checker.is_pose_collision_free(5.0, 3.5) is True

    # But the motion straight through intersects the obstacle at [3.0, 4.0]
    free, col_pose = checker.check_motion((2.0, 3.5), (5.0, 3.5))
    assert free is False
    assert col_pose is not None
    # First colliding pose should be near the boundary of the obstacle (around x ~ 2.8)
    assert 2.0 <= col_pose[0] <= 4.2
    assert checker.is_motion_collision_free((2.0, 3.5), (5.0, 3.5)) is False


def test_motion_pure_rotation_and_curved_orientation() -> None:
    """Test 15: Pure rotation in place and motion with rotating orientation."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    checker = CollisionChecker(grid, robot_radius=0.3)

    # Pure rotation in free space
    free, col_pose = checker.check_motion((2.5, 2.5, 0.0), (2.5, 2.5, math.pi))
    assert free is True
    assert col_pose is None

    # Rotating while translating
    free, col_pose = checker.check_motion((1.0, 1.0, 0.0), (3.0, 3.0, math.pi / 2))
    assert free is True
    assert col_pose is None

    # Pure rotation while in collision (touching obstacle)
    grid.set_cell(2, 2, 1)
    # Center at (1.8, 2.5) touches obstacle at x=2.0 (distance 0.2 < 0.3)
    free_rot, col_rot = checker.check_motion((1.8, 2.5, 0.0), (1.8, 2.5, 1.0))
    assert free_rot is False
    assert col_rot == (1.8, 2.5, 0.0)


def test_motion_map_boundary_handling() -> None:
    """Test 16: Motion crossing outer map boundaries."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    checker = CollisionChecker(grid, robot_radius=0.3)

    # Motion starting inside and ending outside right boundary
    free, col_pose = checker.check_motion((4.0, 2.5), (5.5, 2.5))
    assert free is False
    assert col_pose is not None
    # Robot leaves free area when center + 0.3 > 5.0 (x > 4.7)
    assert col_pose[0] <= 5.5

    # Motion starting outside boundary
    free_out, col_out = checker.check_motion((-1.0, 2.5), (2.0, 2.5))
    assert free_out is False
    assert col_out == (-1.0, 2.5, 0.0)


def test_motion_invalid_inputs() -> None:
    """Test 17: check_motion and is_motion_collision_free invalid inputs."""
    grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=0)
    checker = CollisionChecker(grid, robot_radius=0.3)

    with pytest.raises(ValueError):
        checker.check_motion((float("nan"), 1.0), (2.0, 2.0))
    with pytest.raises(ValueError):
        checker.check_motion((1.0, 1.0), (float("inf"), 2.0))
    with pytest.raises(TypeError):
        checker.check_motion("bad", (2.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        checker.check_motion((1.0, 1.0), 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        checker.check_motion((1.0,), (2.0, 2.0))

    # Invalid max_substep_size
    with pytest.raises(ValueError):
        checker.check_motion((1.0, 1.0), (2.0, 2.0), max_substep_size=0.0)
    with pytest.raises(ValueError):
        checker.check_motion((1.0, 1.0), (2.0, 2.0), max_substep_size=-0.5)
    with pytest.raises(ValueError):
        checker.check_motion((1.0, 1.0), (2.0, 2.0), max_substep_size=float("nan"))
    with pytest.raises(TypeError):
        checker.check_motion((1.0, 1.0), (2.0, 2.0), max_substep_size=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        checker.check_motion((1.0, 1.0), (2.0, 2.0), max_substep_size="0.1")  # type: ignore[arg-type]


def run_all_tests() -> None:
    """Direct execution runner for environments where pytest CLI is not installed."""
    tests = [
        test_robot_centered_in_free_region,
        test_robot_footprint_intersects_occupied_cell,
        test_footprint_tangent_to_occupied_cell_boundary,
        test_footprint_does_not_intersect_nearby_obstacle,
        test_footprint_extending_beyond_map_boundaries,
        test_invalid_and_non_finite_pose_coordinates,
        test_boolean_coordinates_and_invalid_radius,
        test_non_unit_grid_resolution,
        test_nonzero_map_origin,
        test_collision_near_middle_of_obstacle_edge,
        test_checker_does_not_mutate_occupancy_grid,
        test_motion_free_movement_open_area,
        test_motion_approaching_obstacle,
        test_motion_tunneling_segment_crosses_obstacle,
        test_motion_pure_rotation_and_curved_orientation,
        test_motion_map_boundary_handling,
        test_motion_invalid_inputs,
    ]
    print(f"Executing {len(tests)} test functions in test_collision_checker...")
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  [PASS] {t.__name__}")
    print(f"All {passed}/{len(tests)} tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
