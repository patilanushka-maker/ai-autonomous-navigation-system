"""Comprehensive unit and integration test suite for the Pure Pursuit controller.

Tests verify:
1. Straight-path tracking.
2. Path tracking requiring sharp and gradual turns.
3. Heading normalization and curvature direction (left/right, wrapping [-pi, pi)).
4. Angular speed saturation at configured limits.
5. Degenerate paths (single point, start == goal, goal within tolerance).
6. Goal completion and zero-velocity halting.
7. Parameter validation (numeric types, positive ranges, finite bounds, invalid paths).
8. Simulator integration with controller selection and execution.
9. Collision checking remains active during motion with Pure Pursuit.
10. Backward compatibility preserving HeadingProportionalController as default.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import List, Tuple
import pytest
import numpy as np

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.controller import HeadingProportionalController, PurePursuitController
from src.environment import OccupancyGrid
from src.robot import Robot, RobotState
from src.simulator import NavigationSimulator, SimulationResult, SimulationStatus


# ==============================================================================
# 1. Straight-Path Tracking Tests
# ==============================================================================

class TestStraightPathTracking:
    """Tests evaluating linear trajectory tracking along straight line segments."""

    def test_straight_horizontal_path_zero_heading_error(self) -> None:
        """Robot on line facing directly along path maintains zero omega and cruising speed."""
        path = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        ctrl = PurePursuitController(
            lookahead_distance=1.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            goal_tolerance=0.25,
            path=path,
        )
        pose = (2.0, 0.0, 0.0)
        v, omega = ctrl.compute_command(pose)

        assert math.isclose(omega, 0.0, abs_tol=1e-6)
        assert math.isclose(v, 0.5, rel_tol=1e-3)
        assert ctrl.is_goal_reached(pose) is False

    def test_straight_path_lateral_offset_correction(self) -> None:
        """Robot displaced laterally generates steering curvature back toward the line."""
        path = [(0.0, 0.0), (10.0, 0.0)]
        ctrl = PurePursuitController(lookahead_distance=1.0, linear_speed=0.5, path=path)

        # Robot at y = -0.3 (below path) facing theta = 0 (right). Lookahead point is at y = 0.
        # Target angle > 0 -> heading_error > 0 -> omega > 0 (turns left toward the path).
        pose_below = (2.0, -0.3, 0.0)
        v_below, omega_below = ctrl.compute_command(pose_below)
        assert omega_below > 0.0
        assert v_below > 0.0

        # Robot at y = +0.3 (above path) facing theta = 0.
        # Target angle < 0 -> heading_error < 0 -> omega < 0 (turns right toward the path).
        pose_above = (2.0, 0.3, 0.0)
        v_above, omega_above = ctrl.compute_command(pose_above)
        assert omega_above < 0.0
        assert v_above > 0.0

        # Symmetry: magnitudes should match
        assert math.isclose(abs(omega_below), abs(omega_above), rel_tol=1e-4)

    def test_lookahead_point_advances_forward(self) -> None:
        """Lookahead point advances forward along path as robot advances."""
        path = [(float(x), 0.0) for x in range(11)]
        ctrl = PurePursuitController(lookahead_distance=1.5, path=path)

        pt1 = ctrl.find_lookahead_point((1.0, 0.0, 0.0))
        pt2 = ctrl.find_lookahead_point((4.0, 0.0, 0.0))

        assert math.isclose(pt1[0], 2.5, rel_tol=1e-3)
        assert math.isclose(pt2[0], 5.5, rel_tol=1e-3)
        assert pt2[0] > pt1[0]


# ==============================================================================
# 2. Turn Path Tracking Tests
# ==============================================================================

class TestTurnPathTracking:
    """Tests evaluating tracking through path corners and curvature generation."""

    def test_corner_path_demands_curvature(self) -> None:
        """A path with a 90-degree corner commands appropriate rotational velocity."""
        path = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)]
        ctrl = PurePursuitController(lookahead_distance=1.0, linear_speed=0.5, path=path)

        # Robot approaching corner at (1.5, 0.0), theta = 0
        pose = (1.5, 0.0, 0.0)
        v, omega = ctrl.compute_command(pose)

        # Lookahead point should be on the vertical leg (2.0, y > 0)
        la_point = ctrl.current_target_point
        assert la_point is not None
        assert math.isclose(la_point[0], 2.0, abs_tol=1e-3)
        assert la_point[1] > 0.0

        # Steering command must turn left (omega > 0)
        assert omega > 0.0
        assert v > 0.0

    def test_curvature_equation_matches_unicycle_model(self) -> None:
        """Verify omega = v * (2 * sin(alpha) / D) numerically."""
        path = [(0.0, 0.0), (0.0, 5.0)]
        ld = 1.0
        linear_speed = 0.4
        ctrl = PurePursuitController(
            lookahead_distance=ld,
            linear_speed=linear_speed,
            goal_tolerance=0.1,
            path=path,
        )

        # Robot at (0.0, 0.0) with theta = -pi/6 (-30 deg)
        # Lookahead point is at (0.0, 1.0) along +y axis (target_angle = pi/2)
        # heading_error alpha = pi/2 - (-pi/6) = 2pi/3 (120 deg)
        # Since alpha > pi/2, in-place rotation is triggered (v=0).
        # Let's test with theta = pi/4 (45 deg): target_angle = 90 deg -> alpha = 45 deg = pi/4
        pose = (0.0, 0.0, math.pi / 4)
        v, omega = ctrl.compute_command(pose)

        alpha = math.pi / 4  # 45 deg
        d = 1.0
        expected_curvature = 2.0 * math.sin(alpha) / d
        expected_v = linear_speed * math.cos(alpha)  # goal is far (5m), no slowdown
        expected_omega = expected_v * expected_curvature

        assert math.isclose(v, expected_v, rel_tol=1e-3)
        assert math.isclose(omega, expected_omega, rel_tol=1e-3)


# ==============================================================================
# 3. Heading Normalization & Curvature Direction
# ==============================================================================

class TestHeadingNormalizationAndCurvatureDirection:
    """Tests evaluating angle wrapping and signed turning direction."""

    def test_curvature_sign_convention(self) -> None:
        """Positive heading error -> positive omega (CCW); negative -> negative omega (CW)."""
        path_ccw = [(0.0, 0.0), (0.0, 5.0)]  # Target in +y direction
        ctrl_ccw = PurePursuitController(lookahead_distance=1.0, path=path_ccw)
        # Robot facing +x (theta=0), target at +y (pi/2) -> alpha = +pi/2
        pose = (0.0, 0.0, 0.0)
        _, omega_ccw = ctrl_ccw.compute_command(pose)
        assert omega_ccw > 0.0

        path_cw = [(0.0, 0.0), (0.0, -5.0)]  # Target in -y direction
        ctrl_cw = PurePursuitController(lookahead_distance=1.0, path=path_cw)
        # Robot facing +x (theta=0), target at -y (-pi/2) -> alpha = -pi/2
        _, omega_cw = ctrl_cw.compute_command(pose)
        assert omega_cw < 0.0

    def test_heading_wrap_around_plus_minus_pi(self) -> None:
        """Angle difference crossing +/- pi boundary wraps properly into [-pi, pi)."""
        # Robot facing near -pi (e.g. -3.1 rad) and target heading is near +pi (e.g. +3.1 rad)
        # Difference without normalization is 6.2 rad, normalized error is -0.083 rad.
        path = [(-5.0, 0.0), (-10.0, 0.0)]
        ctrl = PurePursuitController(lookahead_distance=1.0, path=path)

        pose = (-5.0, 0.0, 3.1)  # facing almost -x
        v, omega = ctrl.compute_command(pose)
        # Desired heading to (-6.0, 0.0) is pi (or -pi). Error = pi - 3.1 = +0.0415
        assert omega > 0.0
        assert math.isfinite(omega)


# ==============================================================================
# 4. Angular-Speed Saturation Tests
# ==============================================================================

class TestAngularSpeedSaturation:
    """Tests ensuring commanded angular speed strictly respects limits."""

    def test_omega_clamped_to_configured_angular_limit(self) -> None:
        """Sharp angle or high curvature commands do not exceed angular_speed_limit."""
        limit = 0.8
        path = [(0.0, 0.0), (0.0, 2.0)]
        ctrl = PurePursuitController(
            lookahead_distance=0.3,
            linear_speed=1.0,
            angular_speed_limit=limit,
            path=path,
        )

        # Robot facing perpendicular (+x) to path (+y)
        pose = (0.0, 0.0, 0.0)
        v, omega = ctrl.compute_command(pose)
        assert abs(omega) <= limit + 1e-9
        assert math.isclose(abs(omega), limit, rel_tol=1e-3)

    def test_heading_error_threshold_suppresses_linear_motion(self) -> None:
        """When robot faces substantially away (|alpha| >= threshold), v is 0.0."""
        path = [(0.0, 0.0), (5.0, 0.0)]
        ctrl = PurePursuitController(
            heading_error_threshold=math.pi / 2,
            path=path,
        )

        # Facing opposite direction (theta = pi, target at theta = 0)
        pose = (0.0, 0.0, math.pi)
        v, omega = ctrl.compute_command(pose)
        assert v == 0.0
        assert omega != 0.0  # In-place turning active


# ==============================================================================
# 5. Degenerate Paths & Goal Completion Tests
# ==============================================================================

class TestDegeneratePathsAndGoalCompletion:
    """Tests covering zero-length, single-point paths, and goal arrival."""

    def test_single_point_path_within_tolerance(self) -> None:
        """Single-point path where robot is already at goal halts with (0.0, 0.0)."""
        goal = (3.0, 4.0)
        ctrl = PurePursuitController(goal_tolerance=0.25, path=[goal])

        pose = (3.0, 4.0, 0.0)
        v, omega = ctrl.compute_command(pose)
        assert v == 0.0
        assert omega == 0.0
        assert ctrl.is_goal_reached(pose) is True

    def test_single_point_path_outside_tolerance(self) -> None:
        """Single-point path where robot is outside tolerance commands approach."""
        goal = (3.0, 4.0)
        ctrl = PurePursuitController(goal_tolerance=0.25, path=[goal])

        pose = (0.0, 0.0, 0.0)
        v, omega = ctrl.compute_command(pose)
        assert ctrl.is_goal_reached(pose) is False
        assert (v > 0.0 or omega != 0.0)

    def test_goal_arrival_halts_completely(self) -> None:
        """When robot pose is within goal_tolerance of final waypoint, commands are (0.0, 0.0)."""
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        ctrl = PurePursuitController(goal_tolerance=0.3, path=path)

        # Robot at (1.85, 0.0) -> dist to (2.0, 0.0) is 0.15 <= 0.3
        pose = (1.85, 0.0, 0.0)
        v, omega = ctrl.compute_command(pose)
        assert v == 0.0
        assert omega == 0.0
        assert ctrl.is_goal_reached(pose) is True

    def test_is_goal_reached_respects_custom_tolerance(self) -> None:
        """is_goal_reached returns True only when distance <= goal_tolerance."""
        path = [(0.0, 0.0), (5.0, 5.0)]
        ctrl = PurePursuitController(goal_tolerance=0.5, path=path)

        assert ctrl.is_goal_reached((4.6, 5.0, 0.0)) is True   # dist = 0.4 <= 0.5
        assert ctrl.is_goal_reached((4.0, 5.0, 0.0)) is False  # dist = 1.0 > 0.5


# ==============================================================================
# 6. Configuration & Parameter Validation Tests
# ==============================================================================

class TestParameterValidation:
    """Tests ensuring robust validation of constructor inputs and path assignments."""

    def test_lookahead_distance_validation(self) -> None:
        """lookahead_distance must be finite numeric > 0."""
        with pytest.raises(ValueError, match="strictly positive"):
            PurePursuitController(lookahead_distance=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            PurePursuitController(lookahead_distance=-0.5)
        with pytest.raises(TypeError, match="numeric"):
            PurePursuitController(lookahead_distance="0.5")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite"):
            PurePursuitController(lookahead_distance=float("nan"))

    def test_speed_limits_validation(self) -> None:
        """linear_speed and angular_speed_limit must be strictly positive."""
        with pytest.raises(ValueError, match="strictly positive"):
            PurePursuitController(linear_speed=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            PurePursuitController(angular_speed_limit=-1.0)

    def test_goal_tolerance_validation(self) -> None:
        """goal_tolerance must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            PurePursuitController(goal_tolerance=-0.1)

    def test_heading_threshold_validation(self) -> None:
        """heading_error_threshold must be strictly positive and <= pi."""
        with pytest.raises(ValueError, match="strictly positive"):
            PurePursuitController(heading_error_threshold=0.0)
        with pytest.raises(ValueError, match="cannot exceed pi"):
            PurePursuitController(heading_error_threshold=4.0)

    def test_empty_or_invalid_path_rejection(self) -> None:
        """set_path rejects empty sequences, malformed coordinates, and non-numeric items."""
        ctrl = PurePursuitController()
        with pytest.raises(ValueError, match="at least one point"):
            ctrl.set_path([])
        with pytest.raises(TypeError, match="sequence"):
            ctrl.set_path("not a path")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exactly 2 elements"):
            ctrl.set_path([(1.0, 2.0, 3.0)])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="numeric"):
            ctrl.set_path([(1.0, "two")])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="finite"):
            ctrl.set_path([(1.0, float("nan"))])


# ==============================================================================
# 7. Integration with NavigationSimulator Tests
# ==============================================================================

class TestSimulatorIntegration:
    """Tests verifying clean integration of Pure Pursuit into NavigationSimulator."""

    def test_simulator_accepts_pure_pursuit_by_string(self) -> None:
        """NavigationSimulator accepts controller='pure_pursuit' string."""
        grid = OccupancyGrid.from_dimensions(10, 10)
        sim = NavigationSimulator(grid, controller="pure_pursuit", lookahead_distance=0.6)

        assert sim.controller_type == "pure_pursuit"
        assert sim.controller_name == "PurePursuitController"
        assert isinstance(sim.controller, PurePursuitController)
        assert sim.lookahead_distance == 0.6

    def test_simulator_accepts_pure_pursuit_instance(self) -> None:
        """NavigationSimulator accepts an instantiated PurePursuitController."""
        grid = OccupancyGrid.from_dimensions(10, 10)
        custom_ctrl = PurePursuitController(lookahead_distance=0.75, linear_speed=0.4)
        sim = NavigationSimulator(grid, controller=custom_ctrl)

        assert sim.controller is custom_ctrl
        assert sim.controller_type == "pure_pursuit"
        assert sim.lookahead_distance == 0.75

    def test_open_grid_navigation_succeeds(self) -> None:
        """Pure Pursuit completes open-grid navigation with SUCCESS."""
        grid = OccupancyGrid.from_dimensions(10, 10)
        sim = NavigationSimulator(grid, controller="pure_pursuit", lookahead_distance=0.5)

        start = (1.5, 1.5)
        goal = (8.5, 8.5)
        res = sim.run(start, goal)

        assert res.status == SimulationStatus.SUCCESS
        assert res.steps > 0
        assert res.controller_name == "PurePursuitController"
        # Final pose must be within goal_tolerance of goal
        final_state = res.trajectory[-1]
        dist_to_goal = math.hypot(goal[0] - final_state.x, goal[1] - final_state.y)
        assert dist_to_goal <= sim.goal_tolerance

    def test_obstacle_avoidance_succeeds(self) -> None:
        """Pure Pursuit navigates successfully around an obstacle."""
        # 8x8 grid with obstacle blocking direct middle route
        data = np.zeros((8, 8), dtype=np.int8)
        data[2:6, 3:5] = 1  # 4x2 obstacle block in center
        grid = OccupancyGrid(data)

        sim = NavigationSimulator(
            grid,
            robot_radius=0.15,
            controller="pure_pursuit",
            lookahead_distance=0.5,
        )

        res = sim.run((1.5, 1.5), (6.5, 6.5))
        assert res.status == SimulationStatus.SUCCESS
        assert res.controller_name == "PurePursuitController"


# ==============================================================================
# 8. Continuous Collision Checking Remains Active
# ==============================================================================

class TestCollisionCheckingActiveWithPurePursuit:
    """Tests verifying collision detection operates correctly under Pure Pursuit."""

    def test_collision_detected_during_motion_under_pure_pursuit(self) -> None:
        """When robot footprint cuts too close to an obstacle, COLLISION terminates run."""
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
            robot_radius=0.45,
            dt=0.1,
            linear_speed=2.0,
            controller="pure_pursuit",
            lookahead_distance=1.0,
            max_steps=1000,
        )
        result = sim.run(start=(0.5, 0.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.COLLISION
        assert result.steps > 0
        assert "collision" in result.message.lower()
        # Verify terminal state is colliding
        last_state = result.trajectory[-1]
        assert not sim.collision_checker.is_pose_collision_free(last_state.x, last_state.y)


# ==============================================================================
# 9. Backward Compatibility Tests
# ==============================================================================

class TestBackwardCompatibility:
    """Tests ensuring existing default controller behavior is fully preserved."""

    def test_default_controller_is_heading_proportional(self) -> None:
        """Default NavigationSimulator instantiation remains HeadingProportionalController."""
        grid = OccupancyGrid.from_dimensions(5, 5)
        sim = NavigationSimulator(grid)

        assert sim.controller_type == "heading"
        assert sim.controller_name == "HeadingProportionalController"
        assert isinstance(sim.controller, HeadingProportionalController)
        assert sim.lookahead_distance is None

    def test_simulation_result_contains_controller_name_field(self) -> None:
        """SimulationResult defaults controller_name to HeadingProportionalController."""
        res = SimulationResult(
            status=SimulationStatus.SUCCESS,
            steps=10,
            trajectory=[],
            planned_path_grid=None,
            planned_path_world=None,
            start_world=(0.0, 0.0),
            goal_world=(1.0, 1.0),
            path_length=1.0,
            message="Test",
        )
        assert res.controller_name == "HeadingProportionalController"
