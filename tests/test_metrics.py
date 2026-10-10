"""Unit tests for the performance metrics and evaluation module (Phase 2H).

Tests cover:
- Distance calculation accuracy (traveled distance and planned path length).
- Metric extraction for successful navigation runs.
- Metric extraction for NO_PATH, COLLISION, and TIMEOUT outcomes.
- Start-equals-goal and zero-distance edge cases.
- Path efficiency definition, defense, and division-by-zero avoidance.
- Safe handling of empty or missing trajectories and paths.
- Multi-run evaluation aggregation, rates, and mean distance calculations.
- Input validation, type checking, and serialization.
"""

import math
import pytest

from src.environment import OccupancyGrid
from src.robot import RobotState
from src.simulator import NavigationSimulator, SimulationResult, SimulationStatus
from src.metrics import (
    EvaluationSummary,
    NavigationMetrics,
    compute_actual_to_planned_ratio,
    compute_final_goal_error,
    compute_metrics,
    compute_path_efficiency,
    compute_planned_path_length,
    compute_traveled_distance,
    evaluate_multiple_runs,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def open_grid():
    """5x5 fully open grid with resolution=1.0, origin=(0.0, 0.0)."""
    return OccupancyGrid.from_dimensions(5, 5, initial_value=0, resolution=1.0)


@pytest.fixture
def walled_grid():
    """5x5 grid with a vertical wall blocking left from right."""
    data = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    return OccupancyGrid(data, resolution=1.0)


@pytest.fixture
def successful_run_result(open_grid):
    """A real successful simulation run on open grid."""
    sim = NavigationSimulator(open_grid, robot_radius=0.1, dt=0.1, goal_tolerance=0.25)
    return sim.run(start=(0.5, 0.5), goal=(4.5, 4.5))


# ─── Tests: Distance Calculations ────────────────────────────────────────────


class TestDistanceCalculations:
    """Tests for trajectory distance and path length calculation functions."""

    def test_traveled_distance_empty_or_single(self):
        """Empty sequence or single state yields exactly 0.0 meters."""
        assert compute_traveled_distance(None) == 0.0
        assert compute_traveled_distance([]) == 0.0
        assert compute_traveled_distance([RobotState(0, 0, 0, 0, 0)]) == 0.0

    def test_traveled_distance_accurate_robot_states(self):
        """Calculates exact Euclidean sum across RobotState sequence."""
        # Right 3m, then Up 4m -> total distance = 7m
        states = [
            RobotState(0.0, 0.0, 0.0, 0.0, 0.0),
            RobotState(3.0, 0.0, 0.0, 0.0, 0.0),
            RobotState(3.0, 4.0, 0.0, 0.0, 0.0),
        ]
        dist = compute_traveled_distance(states)
        assert math.isclose(dist, 7.0, abs_tol=1e-6)

    def test_traveled_distance_tuple_points(self):
        """Supports (x, y) coordinate tuples in trajectory sequence."""
        points = [(0.0, 0.0), (3.0, 4.0)]  # 3-4-5 triangle
        dist = compute_traveled_distance(points)
        assert math.isclose(dist, 5.0, abs_tol=1e-6)

    def test_planned_path_length_none_and_single(self):
        """None path returns None; single waypoint path returns 0.0."""
        assert compute_planned_path_length(None) is None
        assert compute_planned_path_length([]) == 0.0
        assert compute_planned_path_length([(2.0, 2.0)]) == 0.0

    def test_planned_path_length_accurate(self):
        """Calculates exact Euclidean sum across planned waypoints."""
        waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]
        length = compute_planned_path_length(waypoints)
        assert length is not None
        assert math.isclose(length, 3.0, abs_tol=1e-6)


# ─── Tests: Path Efficiency Formulation ──────────────────────────────────────


class TestPathEfficiency:
    """Tests verifying path efficiency formula and safe behavior."""

    def test_efficiency_successful_run(self):
        """Efficiency is planned_length / traveled_distance for successful runs."""
        eff = compute_path_efficiency(planned_length=10.0, traveled_distance=12.5, success=True)
        assert eff is not None
        assert math.isclose(eff, 0.8, abs_tol=1e-6)

    def test_efficiency_failed_run_is_none(self):
        """Unsuccessful runs return None to prevent misleading efficiency comparisons."""
        assert compute_path_efficiency(planned_length=10.0, traveled_distance=5.0, success=False) is None

    def test_efficiency_zero_traveled_is_none(self):
        """Zero traveled distance returns None (avoids division by zero)."""
        assert compute_path_efficiency(planned_length=10.0, traveled_distance=0.0, success=True) is None

    def test_efficiency_none_or_zero_planned_is_none(self):
        """Missing or zero planned path returns None."""
        assert compute_path_efficiency(planned_length=None, traveled_distance=10.0, success=True) is None
        assert compute_path_efficiency(planned_length=0.0, traveled_distance=10.0, success=True) is None


# ─── Tests: Actual-to-Planned Ratio & Goal Error ─────────────────────────────


class TestDistanceRatioAndGoalError:
    """Tests verifying actual-to-planned distance ratio and final goal error."""

    def test_actual_to_planned_ratio_success(self):
        """Ratio is traveled_distance / planned_length for successful runs."""
        ratio = compute_actual_to_planned_ratio(
            traveled_distance=7.678, planned_length=8.0, success=True
        )
        assert ratio is not None
        assert math.isclose(ratio, 7.678 / 8.0, abs_tol=1e-6)

    def test_actual_to_planned_ratio_failed_run_is_none(self):
        """Unsuccessful runs return None to avoid misleading comparisons for partial paths."""
        assert (
            compute_actual_to_planned_ratio(
                traveled_distance=5.0, planned_length=10.0, success=False
            )
            is None
        )

    def test_actual_to_planned_ratio_none_or_zero_planned_is_none(self):
        """Missing or non-positive planned path returns None."""
        assert (
            compute_actual_to_planned_ratio(
                traveled_distance=5.0, planned_length=None, success=True
            )
            is None
        )
        assert (
            compute_actual_to_planned_ratio(
                traveled_distance=5.0, planned_length=0.0, success=True
            )
            is None
        )

    def test_actual_to_planned_ratio_negative_or_nan(self):
        """Negative or non-finite inputs return None safely."""
        assert (
            compute_actual_to_planned_ratio(
                traveled_distance=-1.0, planned_length=10.0, success=True
            )
            is None
        )
        assert (
            compute_actual_to_planned_ratio(
                traveled_distance=float("nan"), planned_length=10.0, success=True
            )
            is None
        )

    def test_final_goal_error_accurate(self):
        """Computes Euclidean distance between final pose and goal."""
        trajectory = [(0.0, 0.0), (3.0, 4.0)]
        error = compute_final_goal_error(trajectory, goal_world=(3.0, 4.0))
        assert error is not None
        assert math.isclose(error, 0.0, abs_tol=1e-6)

        error_offset = compute_final_goal_error(trajectory, goal_world=(3.0, 5.0))
        assert error_offset is not None
        assert math.isclose(error_offset, 1.0, abs_tol=1e-6)

    def test_final_goal_error_empty_or_missing_is_none(self):
        """Empty trajectory or missing/invalid goal returns None safely."""
        assert compute_final_goal_error([], goal_world=(4.5, 4.5)) is None
        assert compute_final_goal_error(None, goal_world=(4.5, 4.5)) is None
        assert compute_final_goal_error([(1.0, 1.0)], goal_world=None) is None


# ─── Tests: Single-Run Metric Extraction ─────────────────────────────────────


class TestSingleRunMetrics:
    """Tests for extracting NavigationMetrics from real simulation outcomes."""

    def test_successful_run_metrics(self, successful_run_result):
        """Extracts complete metrics from a successful navigation run, validating sample values."""
        m = compute_metrics(successful_run_result)

        assert m.status == SimulationStatus.SUCCESS
        assert m.success is True
        assert m.is_collision is False
        assert m.is_timeout is False
        assert m.is_no_path is False
        assert m.traveled_distance > 0.0
        assert m.planned_path_length is not None
        assert m.planned_path_length > 0.0
        assert m.path_efficiency is not None
        assert m.actual_to_planned_ratio is not None
        assert m.final_goal_error is not None
        assert m.final_goal_error <= 0.25  # within goal_tolerance
        assert m.steps > 0
        assert m.planning_duration is None  # Not recorded by simulator
        assert m.straight_line_distance > 0.0

        # Sample run validation:
        # Planned path: 8 discrete 1m segments -> 8.000m
        # Traveled: ~7.678m due to corner smoothing at (4.5, 0.5) and halting at goal_tolerance (0.25m)
        # Efficiency: 8.000 / 7.678 -> ~1.042
        # Actual/Planned Ratio: 7.678 / 8.000 -> ~0.960
        # Final goal error: ~0.238m
        assert math.isclose(m.planned_path_length, 8.0, abs_tol=1e-4)
        assert math.isclose(m.traveled_distance, 7.678, abs_tol=1e-3)
        assert math.isclose(m.path_efficiency, 1.042, abs_tol=1e-3)
        assert math.isclose(m.actual_to_planned_ratio, 0.960, abs_tol=1e-3)
        assert math.isclose(m.final_goal_error, 0.238, abs_tol=1e-3)

        # Dictionary export check
        d = m.as_dict()
        assert d["status"] == "SUCCESS"
        assert d["success"] is True
        assert d["traveled_distance"] == m.traveled_distance
        assert d["actual_to_planned_ratio"] == m.actual_to_planned_ratio
        assert d["final_goal_error"] == m.final_goal_error

    def test_no_path_run_metrics(self, walled_grid):
        """Extracts metrics from a run that fails due to disconnected goal."""
        sim = NavigationSimulator(walled_grid, robot_radius=0.1)
        result = sim.run(start=(0.5, 2.5), goal=(4.5, 2.5))
        assert result.status == SimulationStatus.NO_PATH

        m = compute_metrics(result)
        assert m.status == SimulationStatus.NO_PATH
        assert m.success is False
        assert m.is_no_path is True
        assert m.is_collision is False
        assert m.is_timeout is False
        assert m.traveled_distance == 0.0
        assert m.planned_path_length is None
        assert m.path_efficiency is None
        assert m.actual_to_planned_ratio is None
        assert m.final_goal_error is None

    def test_collision_run_metrics(self):
        """Extracts metrics from a run terminating in collision."""
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

        m = compute_metrics(result)
        assert m.status == SimulationStatus.COLLISION
        assert m.success is False
        assert m.is_collision is True
        assert m.is_timeout is False
        assert m.is_no_path is False
        assert m.traveled_distance > 0.0
        assert m.path_efficiency is None  # Failed run -> None
        assert m.actual_to_planned_ratio is None
        assert m.final_goal_error is not None

    def test_timeout_run_metrics(self, open_grid):
        """Extracts metrics from a run terminating in timeout."""
        sim = NavigationSimulator(open_grid, robot_radius=0.1, dt=0.1, max_steps=2, linear_speed=0.1)
        result = sim.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        assert result.status == SimulationStatus.TIMEOUT

        m = compute_metrics(result)
        assert m.status == SimulationStatus.TIMEOUT
        assert m.success is False
        assert m.is_timeout is True
        assert m.is_collision is False
        assert m.is_no_path is False
        assert m.steps == 2
        assert m.path_efficiency is None
        assert m.actual_to_planned_ratio is None
        assert m.final_goal_error is not None

    def test_start_equals_goal_zero_distance(self, open_grid):
        """Start-at-goal edge case yields 0.0 distance without division by zero."""
        sim = NavigationSimulator(open_grid, robot_radius=0.1)
        result = sim.run(start=(2.5, 2.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.SUCCESS

        m = compute_metrics(result)
        assert m.status == SimulationStatus.SUCCESS
        assert m.success is True
        assert m.steps == 0
        assert m.traveled_distance == 0.0
        assert m.planned_path_length is None
        assert m.path_efficiency is None
        assert m.actual_to_planned_ratio is None
        assert m.final_goal_error is not None
        assert math.isclose(m.final_goal_error, 0.0, abs_tol=1e-6)
        assert m.straight_line_distance == 0.0

    def test_custom_planning_duration(self, successful_run_result):
        """Allows caller to attach external planning duration if tracked."""
        m = compute_metrics(successful_run_result, planning_duration=0.0125)
        assert m.planning_duration == 0.0125


# ─── Tests: Multi-Run Evaluation Helper ──────────────────────────────────────


class TestMultiRunEvaluation:
    """Tests for multi-run summary aggregation and rate calculations."""

    def test_empty_results_sequence(self):
        """Empty sequence produces zero totals and None means without error."""
        summary = evaluate_multiple_runs([])
        assert summary.total_runs == 0
        assert summary.success_count == 0
        assert summary.success_rate == 0.0
        assert summary.collision_count == 0
        assert summary.collision_rate == 0.0
        assert summary.timeout_count == 0
        assert summary.timeout_rate == 0.0
        assert summary.no_path_count == 0
        assert summary.no_path_rate == 0.0
        assert summary.mean_traveled_distance_successful is None
        assert summary.mean_steps_successful is None
        assert summary.mean_path_efficiency_successful is None

    def test_mixed_outcomes_aggregation(self):
        """Aggregates multiple runs with different outcomes and checks rates."""
        # Create synthetic metrics
        m_success1 = NavigationMetrics(
            status=SimulationStatus.SUCCESS,
            success=True,
            traveled_distance=10.0,
            planned_path_length=10.0,
            steps=100,
            planning_duration=None,
            path_efficiency=1.0,
            actual_to_planned_ratio=1.0,
            final_goal_error=0.10,
            is_collision=False,
            is_timeout=False,
            is_no_path=False,
            straight_line_distance=8.0,
        )
        m_success2 = NavigationMetrics(
            status=SimulationStatus.SUCCESS,
            success=True,
            traveled_distance=14.0,
            planned_path_length=12.0,
            steps=140,
            planning_duration=None,
            path_efficiency=0.857,
            actual_to_planned_ratio=14.0 / 12.0,
            final_goal_error=0.20,
            is_collision=False,
            is_timeout=False,
            is_no_path=False,
            straight_line_distance=10.0,
        )
        m_collision = NavigationMetrics(
            status=SimulationStatus.COLLISION,
            success=False,
            traveled_distance=3.0,
            planned_path_length=10.0,
            steps=30,
            planning_duration=None,
            path_efficiency=None,
            actual_to_planned_ratio=None,
            final_goal_error=5.0,
            is_collision=True,
            is_timeout=False,
            is_no_path=False,
            straight_line_distance=8.0,
        )
        m_timeout = NavigationMetrics(
            status=SimulationStatus.TIMEOUT,
            success=False,
            traveled_distance=8.0,
            planned_path_length=15.0,
            steps=2000,
            planning_duration=None,
            path_efficiency=None,
            actual_to_planned_ratio=None,
            final_goal_error=4.0,
            is_collision=False,
            is_timeout=True,
            is_no_path=False,
            straight_line_distance=12.0,
        )
        m_no_path = NavigationMetrics(
            status=SimulationStatus.NO_PATH,
            success=False,
            traveled_distance=0.0,
            planned_path_length=None,
            steps=0,
            planning_duration=None,
            path_efficiency=None,
            actual_to_planned_ratio=None,
            final_goal_error=None,
            is_collision=False,
            is_timeout=False,
            is_no_path=True,
            straight_line_distance=5.0,
        )

        runs = [m_success1, m_success2, m_collision, m_timeout, m_no_path]
        summary = evaluate_multiple_runs(runs)

        assert summary.total_runs == 5
        assert summary.success_count == 2
        assert math.isclose(summary.success_rate, 2 / 5, abs_tol=1e-6)
        assert summary.collision_count == 1
        assert math.isclose(summary.collision_rate, 1 / 5, abs_tol=1e-6)
        assert summary.timeout_count == 1
        assert math.isclose(summary.timeout_rate, 1 / 5, abs_tol=1e-6)
        assert summary.no_path_count == 1
        assert math.isclose(summary.no_path_rate, 1 / 5, abs_tol=1e-6)

        # Successful averages: (10.0 + 14.0) / 2 = 12.0m
        assert summary.mean_traveled_distance_successful is not None
        assert math.isclose(summary.mean_traveled_distance_successful, 12.0, abs_tol=1e-6)

        # Mean goal error across successful runs: (0.10 + 0.20) / 2 = 0.15m
        assert summary.mean_final_goal_error_successful is not None
        assert math.isclose(summary.mean_final_goal_error_successful, 0.15, abs_tol=1e-6)

        # Mean actual-to-planned ratio across successful runs: (1.0 + 14.0/12.0) / 2 = 1.0833
        assert summary.mean_actual_to_planned_ratio_successful is not None
        assert math.isclose(
            summary.mean_actual_to_planned_ratio_successful,
            (1.0 + (14.0 / 12.0)) / 2,
            abs_tol=1e-6,
        )

        # Mean steps: (100 + 140) / 2 = 120.0
        assert summary.mean_steps_successful is not None
        assert math.isclose(summary.mean_steps_successful, 120.0, abs_tol=1e-6)

    def test_no_path_not_counted_as_collision(self):
        """NO_PATH outcomes are never counted as collisions."""
        m_no_path = NavigationMetrics(
            status=SimulationStatus.NO_PATH,
            success=False,
            traveled_distance=0.0,
            planned_path_length=None,
            steps=0,
            planning_duration=None,
            path_efficiency=None,
            is_collision=False,
            is_timeout=False,
            is_no_path=True,
            straight_line_distance=5.0,
        )
        summary = evaluate_multiple_runs([m_no_path])
        assert summary.total_runs == 1
        assert summary.no_path_count == 1
        assert summary.collision_count == 0
        assert summary.collision_rate == 0.0

    def test_no_successful_runs_mean_is_none(self):
        """When no runs succeed, mean_traveled_distance_successful is None."""
        m_collision = NavigationMetrics(
            status=SimulationStatus.COLLISION,
            success=False,
            traveled_distance=4.0,
            planned_path_length=8.0,
            steps=40,
            planning_duration=None,
            path_efficiency=None,
            is_collision=True,
            is_timeout=False,
            is_no_path=False,
            straight_line_distance=5.0,
        )
        summary = evaluate_multiple_runs([m_collision])
        assert summary.total_runs == 1
        assert summary.success_count == 0
        assert summary.mean_traveled_distance_successful is None

    def test_summary_table_string(self):
        """summary_table formats valid human-readable report."""
        summary = evaluate_multiple_runs([])
        table = summary.summary_table()
        assert "NAVIGATION EVALUATION SUMMARY" in table
        assert "Total Runs Evaluated : 0" in table


# ─── Tests: Input Validation ─────────────────────────────────────────────────


class TestInputValidation:
    """Tests verifying input parameter validation and error handling."""

    def test_compute_metrics_rejects_non_result(self):
        """Rejects non-SimulationResult inputs."""
        with pytest.raises(TypeError, match="SimulationResult"):
            compute_metrics("invalid_result")

    def test_evaluate_multiple_runs_rejects_non_sequence(self):
        """Rejects non-sequence inputs."""
        with pytest.raises(TypeError, match="sequence"):
            evaluate_multiple_runs("not_a_list")

    def test_evaluate_multiple_runs_rejects_invalid_item(self):
        """Rejects sequences containing invalid item types."""
        with pytest.raises(TypeError, match="SimulationResult or NavigationMetrics"):
            evaluate_multiple_runs([123])
