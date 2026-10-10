"""Unit tests for NavigationSimulator (Phase 2F).

Tests cover:
- Successful goal-reaching navigation on an open grid.
- Start-at-goal edge case (immediate success with zero steps).
- Unreachable goal (wall-separated start and goal → NO_PATH).
- Collision termination (narrow corridor forcing collision).
- Timeout termination (max_steps limit).
- Waypoint progression tracking.
- Invalid input validation (types, bounds, collision at start/goal).
- SimulationResult structure and metrics.
"""

import math
import pytest
import numpy as np

from src.environment import OccupancyGrid
from src.simulator import NavigationSimulator, SimulationResult, SimulationStatus


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def open_5x5_grid():
    """5x5 fully open grid with resolution=1.0, origin=(0,0)."""
    return OccupancyGrid.from_dimensions(5, 5, initial_value=0, resolution=1.0)


@pytest.fixture
def corridor_grid():
    """7x7 grid with a clear L-shaped corridor from (0,0) to (6,6).

    Layout (0=free, 1=obstacle):
    Row 0: 0 0 0 0 0 0 0
    Row 1: 0 1 1 1 1 1 0
    Row 2: 0 1 0 0 0 0 0
    Row 3: 0 1 0 1 1 1 0
    Row 4: 0 1 0 0 0 0 0
    Row 5: 0 1 1 1 1 0 0
    Row 6: 0 0 0 0 0 0 0
    """
    data = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    return OccupancyGrid(data, resolution=1.0)


@pytest.fixture
def walled_grid():
    """5x5 grid with a wall separating left and right halves.

    Layout:
    Row 0: 0 0 1 0 0
    Row 1: 0 0 1 0 0
    Row 2: 0 0 1 0 0
    Row 3: 0 0 1 0 0
    Row 4: 0 0 1 0 0
    """
    data = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    return OccupancyGrid(data, resolution=1.0)


@pytest.fixture
def sim_open(open_5x5_grid):
    """Simulator on the open 5x5 grid with small robot radius."""
    return NavigationSimulator(
        open_5x5_grid,
        robot_radius=0.1,
        dt=0.1,
        max_steps=2000,
        goal_tolerance=0.3,
        linear_speed=1.0,
        angular_gain=3.0,
        angular_speed_limit=2.0,
    )


@pytest.fixture
def sim_corridor(corridor_grid):
    """Simulator on the corridor grid."""
    return NavigationSimulator(
        corridor_grid,
        robot_radius=0.1,
        dt=0.1,
        max_steps=5000,
        goal_tolerance=0.3,
        linear_speed=1.0,
        angular_gain=3.0,
        angular_speed_limit=2.0,
    )


@pytest.fixture
def sim_walled(walled_grid):
    """Simulator on the walled grid."""
    return NavigationSimulator(
        walled_grid,
        robot_radius=0.1,
        dt=0.1,
        max_steps=2000,
        goal_tolerance=0.3,
    )


# ─── Test: Successful Navigation ─────────────────────────────────────────────


class TestSuccessfulNavigation:
    """Tests for runs that should reach the goal successfully."""

    def test_straight_line_open_grid(self, sim_open):
        """Robot navigates from bottom-left to top-right on open grid."""
        result = sim_open.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps > 0
        assert len(result.trajectory) > 1
        assert result.path_length > 0.0
        assert result.planned_path_grid is not None
        assert result.planned_path_world is not None
        assert result.message.startswith("Goal reached")

    def test_short_distance(self, sim_open):
        """Robot navigates a very short distance."""
        result = sim_open.run(start=(2.5, 2.5), goal=(3.5, 2.5))
        assert result.status == SimulationStatus.SUCCESS

    def test_off_center_goal(self, sim_open):
        """Robot successfully reaches an off-center continuous goal coordinate."""
        result = sim_open.run(start=(0.5, 0.5), goal=(4.2, 4.2))
        assert result.status == SimulationStatus.SUCCESS
        last = result.trajectory[-1]
        dist = math.hypot(last.x - 4.2, last.y - 4.2)
        assert dist <= sim_open.goal_tolerance

    def test_corridor_navigation(self, sim_corridor):
        """Robot navigates through L-shaped corridor."""
        result = sim_corridor.run(start=(0.5, 0.5), goal=(6.5, 6.5))
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps > 0
        assert len(result.planned_path_grid) > 2  # Non-trivial path

    def test_result_trajectory_starts_near_start(self, sim_open):
        """First trajectory point should be near the start position."""
        start = (1.5, 1.5)
        result = sim_open.run(start=start, goal=(3.5, 3.5))
        assert result.status == SimulationStatus.SUCCESS
        first = result.trajectory[0]
        assert abs(first.x - start[0]) < 0.01
        assert abs(first.y - start[1]) < 0.01

    def test_result_trajectory_ends_near_goal(self, sim_open):
        """Last trajectory point should be near the goal position."""
        goal = (3.5, 3.5)
        result = sim_open.run(start=(0.5, 0.5), goal=goal)
        assert result.status == SimulationStatus.SUCCESS
        last = result.trajectory[-1]
        dist = math.hypot(last.x - goal[0], last.y - goal[1])
        assert dist <= sim_open._goal_tolerance + 0.5  # Allow some margin


# ─── Test: Start == Goal Edge Case ───────────────────────────────────────────


class TestStartEqualsGoal:
    """Tests for the trivial case where start is already at or near goal."""

    def test_exact_same_position(self, sim_open):
        """Start and goal at exactly the same point."""
        result = sim_open.run(start=(2.5, 2.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps == 0
        assert len(result.trajectory) == 1
        assert result.path_length == 0.0
        assert result.planned_path_grid is None
        assert "already within" in result.message.lower()

    def test_within_tolerance(self, sim_open):
        """Start within goal_tolerance of goal — immediate success."""
        result = sim_open.run(start=(2.5, 2.5), goal=(2.6, 2.5))
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps == 0


# ─── Test: Unreachable Goal (NO_PATH) ────────────────────────────────────────


class TestUnreachableGoal:
    """Tests for goals that A* cannot reach."""

    def test_wall_separates_start_and_goal(self, sim_walled):
        """A vertical wall separates start from goal — NO_PATH."""
        result = sim_walled.run(start=(0.5, 2.5), goal=(4.5, 2.5))
        assert result.status == SimulationStatus.NO_PATH
        assert result.steps == 0
        assert len(result.trajectory) == 0
        assert result.planned_path_grid is None
        assert result.path_length == 0.0


# ─── Test: Timeout ───────────────────────────────────────────────────────────


class TestTimeout:
    """Tests for simulations that exceed the maximum step limit."""

    def test_max_steps_exceeded(self, open_5x5_grid):
        """With extremely low max_steps, the robot cannot reach a distant goal."""
        sim = NavigationSimulator(
            open_5x5_grid,
            robot_radius=0.1,
            dt=0.1,
            max_steps=3,        # Very low limit
            goal_tolerance=0.1,  # Very tight tolerance
            linear_speed=0.1,   # Very slow
        )
        result = sim.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        assert result.status == SimulationStatus.TIMEOUT
        assert result.steps == 3
        assert "timed out" in result.message.lower()


# ─── Test: SimulationResult Structure ────────────────────────────────────────


class TestSimulationResult:
    """Tests verifying the structure and correctness of SimulationResult."""

    def test_result_fields_present(self, sim_open):
        """All required fields should be populated on a successful run."""
        result = sim_open.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        assert isinstance(result.status, SimulationStatus)
        assert isinstance(result.steps, int)
        assert isinstance(result.trajectory, list)
        assert isinstance(result.path_length, float)
        assert isinstance(result.message, str)
        assert result.start_world == (0.5, 0.5)
        assert result.goal_world == (4.5, 4.5)

    def test_trajectory_entries_are_robot_states(self, sim_open):
        """Every trajectory entry must be a RobotState instance."""
        from src.robot import RobotState
        result = sim_open.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        for entry in result.trajectory:
            assert isinstance(entry, RobotState)

    def test_planned_paths_consistent(self, sim_open):
        """Grid and world path lists should have the same length."""
        result = sim_open.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        if result.planned_path_grid is not None:
            assert len(result.planned_path_grid) == len(result.planned_path_world)


# ─── Test: Input Validation ──────────────────────────────────────────────────


class TestInputValidation:
    """Tests for constructor and run() input validation."""

    def test_invalid_grid_type(self):
        """Constructor must reject non-OccupancyGrid types."""
        with pytest.raises(TypeError, match="OccupancyGrid"):
            NavigationSimulator(grid="not_a_grid")

    def test_invalid_dt_zero(self, open_5x5_grid):
        """dt must be strictly positive."""
        with pytest.raises(ValueError, match="positive"):
            NavigationSimulator(open_5x5_grid, dt=0.0)

    def test_invalid_dt_negative(self, open_5x5_grid):
        """dt must be positive — negative rejected."""
        with pytest.raises(ValueError, match="positive"):
            NavigationSimulator(open_5x5_grid, dt=-0.1)

    def test_invalid_max_steps_zero(self, open_5x5_grid):
        """max_steps must be >= 1."""
        with pytest.raises(ValueError, match="positive integer"):
            NavigationSimulator(open_5x5_grid, max_steps=0)

    def test_invalid_max_steps_bool(self, open_5x5_grid):
        """max_steps must not be a boolean."""
        with pytest.raises(ValueError):
            NavigationSimulator(open_5x5_grid, max_steps=True)

    def test_invalid_max_substep_size(self, open_5x5_grid):
        """max_substep_size must be positive float or None."""
        with pytest.raises(ValueError, match="positive"):
            NavigationSimulator(open_5x5_grid, max_substep_size=0.0)
        with pytest.raises(ValueError, match="positive"):
            NavigationSimulator(open_5x5_grid, max_substep_size=-0.1)
        with pytest.raises(TypeError, match="numeric"):
            NavigationSimulator(open_5x5_grid, max_substep_size=True)
        with pytest.raises(TypeError, match="numeric"):
            NavigationSimulator(open_5x5_grid, max_substep_size="invalid")

    def test_run_start_not_tuple(self, sim_open):
        """run() must reject non-tuple start."""
        with pytest.raises(TypeError, match="tuple"):
            sim_open.run(start="bad", goal=(4.5, 4.5))

    def test_run_goal_wrong_length(self, sim_open):
        """run() must reject goal with wrong number of elements."""
        with pytest.raises(ValueError, match="2 elements"):
            sim_open.run(start=(0.5, 0.5), goal=(4.5, 4.5, 0.0))

    def test_run_start_in_collision(self):
        """run() should raise if start is inside an obstacle."""
        grid = OccupancyGrid(
            [[1, 0], [0, 0]],
            resolution=1.0,
        )
        sim = NavigationSimulator(grid, robot_radius=0.1)
        with pytest.raises(ValueError, match="collision"):
            sim.run(start=(0.5, 0.5), goal=(1.5, 1.5))

    def test_run_nan_coordinate(self, sim_open):
        """run() must reject NaN coordinates."""
        with pytest.raises(ValueError, match="finite"):
            sim_open.run(start=(float("nan"), 0.5), goal=(4.5, 4.5))



# ─── Test: Waypoint Progression ──────────────────────────────────────────────


class TestWaypointProgression:
    """Tests verifying waypoint advancement and sequence tracking."""

    def test_waypoint_progression_in_order(self, sim_open):
        """Robot visits intermediate waypoints in monotonic sequence."""
        result = sim_open.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        assert result.status == SimulationStatus.SUCCESS
        assert result.planned_path_world is not None
        assert len(result.planned_path_world) > 2

        # Find the time index where the robot is closest to each waypoint
        closest_step_indices = []
        for wx, wy in result.planned_path_world:
            dists = [math.hypot(s.x - wx, s.y - wy) for s in result.trajectory]
            best_idx = int(np.argmin(dists))
            closest_step_indices.append(best_idx)

        # Verify step indices are non-decreasing (advancing forward along the path)
        for i in range(1, len(closest_step_indices)):
            assert closest_step_indices[i] >= closest_step_indices[i - 1], (
                f"Waypoint {i} reached at step {closest_step_indices[i]} before "
                f"waypoint {i-1} at step {closest_step_indices[i-1]}."
            )

    def test_custom_waypoint_tolerance(self, open_5x5_grid):
        """Simulator accepts and respects custom waypoint_tolerance."""
        sim = NavigationSimulator(
            open_5x5_grid,
            robot_radius=0.1,
            dt=0.1,
            max_steps=2000,
            goal_tolerance=0.25,
            waypoint_tolerance=0.5,
        )
        assert sim.waypoint_tolerance == 0.5
        result = sim.run(start=(0.5, 0.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.SUCCESS


# ─── Test: Collision Handling ────────────────────────────────────────────────


class TestCollisionHandling:
    """Tests verifying obstacle collision detection during simulation and at endpoints."""

    def test_collision_during_motion(self):
        """A larger robot moving fast around a corner collides with an obstacle."""
        grid = OccupancyGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            resolution=1.0,
        )
        # With larger footprint radius and fast speed, cutting corner collides with (1, 1)
        sim = NavigationSimulator(
            grid,
            robot_radius=0.35,
            dt=0.1,
            linear_speed=2.0,
            angular_gain=1.0,
            max_steps=1000,
        )
        result = sim.run(start=(0.5, 0.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.COLLISION
        assert result.steps > 0
        assert len(result.trajectory) == result.steps + 1
        assert "collision" in result.message.lower()
        # Verify final state is indeed colliding
        last_state = result.trajectory[-1]
        assert not sim.collision_checker.is_pose_collision_free(last_state.x, last_state.y)

    def test_goal_in_collision_raises(self):
        """run() raises ValueError if goal is placed on an obstacle."""
        grid = OccupancyGrid(
            [[0, 1], [0, 0]],
            resolution=1.0,
        )
        sim = NavigationSimulator(grid, robot_radius=0.1)
        with pytest.raises(ValueError, match="collision"):
            sim.run(start=(0.5, 1.5), goal=(1.5, 0.5))

    def test_start_out_of_bounds_raises(self, open_5x5_grid):
        """run() raises ValueError if start is outside grid boundaries."""
        sim = NavigationSimulator(open_5x5_grid, robot_radius=0.1)
        with pytest.raises(ValueError, match="collision or outside grid bounds"):
            sim.run(start=(-1.0, 0.5), goal=(2.5, 2.5))

    def test_goal_out_of_bounds_raises(self, open_5x5_grid):
        """run() raises ValueError if goal is outside grid boundaries."""
        sim = NavigationSimulator(open_5x5_grid, robot_radius=0.1)
        with pytest.raises(ValueError, match="collision or outside grid bounds"):
            sim.run(start=(0.5, 0.5), goal=(10.0, 2.5))

    def test_mid_motion_collision_sets_robot_to_colliding_pose(self):
        """When motion collides mid-step, robot's recorded pose is the colliding pose."""
        grid = OccupancyGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            resolution=1.0,
        )
        sim = NavigationSimulator(
            grid,
            robot_radius=0.35,
            dt=0.1,
            linear_speed=2.0,
            angular_gain=1.0,
            max_steps=1000,
        )
        result = sim.run(start=(0.5, 0.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.COLLISION
        # Final recorded pose in trajectory must be colliding
        last_state = result.trajectory[-1]
        assert not sim.collision_checker.is_pose_collision_free(last_state.x, last_state.y)
        # Robot's final state matches the colliding trajectory state
        assert sim.robot.state.x == last_state.x
        assert sim.robot.state.y == last_state.y


# ─── Test: Simulator Properties & Accessors ──────────────────────────────────


class TestSimulatorProperties:
    """Tests for property accessors on NavigationSimulator."""

    def test_all_properties_accessible(self, open_5x5_grid):
        """Verify all public property accessors return correct values."""
        sim = NavigationSimulator(
            open_5x5_grid,
            robot_radius=0.15,
            dt=0.05,
            max_steps=500,
            goal_tolerance=0.2,
            waypoint_tolerance=0.3,
            linear_speed=0.8,
            angular_gain=2.5,
            angular_speed_limit=1.5,
        )
        assert sim.grid is open_5x5_grid
        assert sim.robot_radius == 0.15
        assert sim.dt == 0.05
        assert sim.max_steps == 500
        assert sim.goal_tolerance == 0.2
        assert sim.waypoint_tolerance == 0.3
        assert sim.planner is not None
        assert sim.controller is not None
        assert sim.collision_checker is not None
        assert sim.controller.linear_speed == 0.8
        assert sim.controller.angular_gain == 2.5
        assert sim.controller.angular_speed_limit == 1.5
        assert sim.collision_checker.robot_radius == 0.15
        assert sim.max_substep_size is None

        sim_custom = NavigationSimulator(open_5x5_grid, max_substep_size=0.08)
        assert sim_custom.max_substep_size == 0.08


# ─── Test: SimulationStatus Enum ─────────────────────────────────────────────


class TestSimulationStatus:
    """Tests for the SimulationStatus enum values."""

    def test_all_status_values(self):
        """All expected status values should exist."""
        assert SimulationStatus.SUCCESS.value == "SUCCESS"
        assert SimulationStatus.COLLISION.value == "COLLISION"
        assert SimulationStatus.TIMEOUT.value == "TIMEOUT"
        assert SimulationStatus.NO_PATH.value == "NO_PATH"
        assert SimulationStatus.RUNNING.value == "RUNNING"
