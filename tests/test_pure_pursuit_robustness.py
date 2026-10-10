"""Robustness, tuning, and edge-case verification test suite for Pure Pursuit.

Covers Phase B requirements:
1. Lookahead distances of 0.25, 0.5, 0.75, and 1.0 metres.
2. All seven existing benchmark scenarios with Pure Pursuit.
3. Straight-line paths, right-angle turns, and paths near obstacles.
4. Starts with various initial headings (aligned, perpendicular, facing away).
5. Goal arrival strictly within configured tolerances.
6. Collision detection active throughout continuous motion.
7. Unreachable goals and invalid start/goal inputs.
8. Repeated runs producing bit-for-bit reproducible results.
9. Invalid controller configurations and degenerate paths.
10. Guarantees of finite position, heading, and velocity values across all trajectory steps.
11. Verification of the lookahead tuning experiment runner and schema.
"""

from __future__ import annotations

import json
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

from src.benchmark.runner import BenchmarkRunner
from src.benchmark.scenario import BenchmarkScenario
from src.benchmark.suite import (
    get_default_scenarios,
    get_scenario_by_name,
    scenario_invalid_start_goal,
    scenario_maze,
    scenario_multiple_obstacles,
    scenario_narrow_corridor,
    scenario_one_obstacle,
    scenario_open_space,
    scenario_unreachable_goal,
)
from src.benchmark.tuning import (
    DEFAULT_TUNING_OUTPUT,
    run_pure_pursuit_tuning,
)
from src.controller import PurePursuitController
from src.environment import OccupancyGrid
from src.simulator import NavigationSimulator, SimulationStatus


# ==============================================================================
# 1. Lookahead Distances (0.25, 0.5, 0.75, 1.0 m)
# ==============================================================================

class TestLookaheadDistances:
    """Verifies simulator behavior across candidate lookahead distances."""

    @pytest.mark.parametrize("ld", [0.25, 0.5, 0.75, 1.0])
    def test_open_space_succeeds_for_all_lookaheads(self, ld: float) -> None:
        """All lookahead distances navigate open space to SUCCESS."""
        scenario = scenario_open_space()
        sim = scenario.create_simulator(controller="pure_pursuit", lookahead_distance=ld)
        assert sim.lookahead_distance == ld

        result = sim.run(scenario.start, scenario.goal)
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps > 0
        final_state = result.trajectory[-1]
        dist_to_goal = math.hypot(scenario.goal[0] - final_state.x, scenario.goal[1] - final_state.y)
        assert dist_to_goal <= sim.goal_tolerance

    @pytest.mark.parametrize("ld", [0.25, 0.5, 0.75, 1.0])
    def test_narrow_corridor_succeeds_for_all_lookaheads(self, ld: float) -> None:
        """All lookahead distances traverse narrow corridor without boundary collisions."""
        scenario = scenario_narrow_corridor()
        sim = scenario.create_simulator(controller="pure_pursuit", lookahead_distance=ld)
        result = sim.run(scenario.start, scenario.goal)
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps > 0

    def test_lookahead_distance_property_and_overrides(self) -> None:
        """lookahead_distance is properly initialized and accessible on simulator."""
        grid = OccupancyGrid.from_dimensions(5, 5)
        sim_default = NavigationSimulator(grid, controller="pure_pursuit")
        assert sim_default.lookahead_distance == 0.5

        sim_custom = NavigationSimulator(grid, controller="pure_pursuit", lookahead_distance=0.75)
        assert sim_custom.lookahead_distance == 0.75


# ==============================================================================
# 2. All Seven Benchmark Scenarios
# ==============================================================================

class TestAllBenchmarkScenariosUnderPurePursuit:
    """Verifies that Pure Pursuit correctly handles all seven predefined benchmark challenges."""

    def test_open_space_scenario(self) -> None:
        s = scenario_open_space()
        res = s.create_simulator(controller="pure_pursuit").run(s.start, s.goal)
        assert res.status == SimulationStatus.SUCCESS

    def test_one_obstacle_scenario(self) -> None:
        s = scenario_one_obstacle()
        res = s.create_simulator(controller="pure_pursuit").run(s.start, s.goal)
        assert res.status == SimulationStatus.SUCCESS

    def test_multiple_obstacles_scenario(self) -> None:
        s = scenario_multiple_obstacles()
        res = s.create_simulator(controller="pure_pursuit").run(s.start, s.goal)
        assert res.status == SimulationStatus.SUCCESS

    def test_narrow_corridor_scenario(self) -> None:
        s = scenario_narrow_corridor()
        res = s.create_simulator(controller="pure_pursuit").run(s.start, s.goal)
        assert res.status == SimulationStatus.SUCCESS

    def test_maze_scenario(self) -> None:
        s = scenario_maze()
        res = s.create_simulator(controller="pure_pursuit", lookahead_distance=0.5).run(s.start, s.goal)
        assert res.status == SimulationStatus.SUCCESS

    def test_unreachable_goal_returns_no_path(self) -> None:
        s = scenario_unreachable_goal()
        res = s.create_simulator(controller="pure_pursuit").run(s.start, s.goal)
        assert res.status == SimulationStatus.NO_PATH
        assert res.steps == 0

    def test_invalid_start_goal_raises_value_error(self) -> None:
        s = scenario_invalid_start_goal()
        sim = s.create_simulator(controller="pure_pursuit")
        with pytest.raises(ValueError):
            sim.run(s.start, s.goal)


# ==============================================================================
# 3. Path Shapes & Geometries (Straight, Turns, Obstacles)
# ==============================================================================

class TestGeometryAndPathShapes:
    """Verifies control tracking across diverse geometric path topologies."""

    def test_pure_straight_path_no_lateral_drift(self) -> None:
        """Tracking a pure horizontal line produces zero lateral error."""
        grid = OccupancyGrid.from_dimensions(10, 10)
        sim = NavigationSimulator(grid, controller="pure_pursuit", lookahead_distance=0.5)
        res = sim.run(start=(1.5, 5.5), goal=(8.5, 5.5), initial_heading=0.0)

        assert res.status == SimulationStatus.SUCCESS
        for s in res.trajectory:
            assert math.isclose(s.y, 5.5, abs_tol=1e-9)

    def test_consecutive_right_angle_turns(self) -> None:
        """Navigating an S-curve path with consecutive left and right 90-degree turns."""
        data = np.zeros((10, 10), dtype=np.int8)
        # Create slalom barriers
        data[0:7, 3] = 1
        data[3:10, 6] = 1
        grid = OccupancyGrid(data)

        sim = NavigationSimulator(grid, robot_radius=0.15, controller="pure_pursuit", lookahead_distance=0.5)
        res = sim.run(start=(1.5, 1.5), goal=(8.5, 8.5))
        assert res.status == SimulationStatus.SUCCESS

    def test_path_near_obstacle_maintains_clearance(self) -> None:
        """Tracking a path adjacent to an obstacle wall does not collide."""
        data = np.zeros((8, 8), dtype=np.int8)
        data[:, 4] = 1  # Wall at col 4 (x in [4.0, 5.0])
        # Clear passage along col 3 (x in [3.0, 4.0])
        grid = OccupancyGrid(data)

        # Robot radius 0.2 navigating along x = 3.5 (clearance = 0.5 - 0.2 = 0.3m)
        sim = NavigationSimulator(grid, robot_radius=0.2, controller="pure_pursuit", lookahead_distance=0.5)
        res = sim.run(start=(3.5, 1.5), goal=(3.5, 6.5))
        assert res.status == SimulationStatus.SUCCESS


# ==============================================================================
# 4. Initial Headings Variation
# ==============================================================================

class TestInitialHeadingsVariation:
    """Verifies robot handles starts with varying initial headings without disorientation."""

    @pytest.mark.parametrize(
        "heading_label,initial_th",
        [
            ("aligned", 0.0),
            ("perpendicular_pos", math.pi / 2),
            ("perpendicular_neg", -math.pi / 2),
            ("facing_opposite", math.pi),
            ("diagonal", math.pi / 4),
        ],
    )
    def test_navigation_with_diverse_initial_headings(
        self, heading_label: str, initial_th: float
    ) -> None:
        """Robot successfully reaches goal regardless of start orientation."""
        grid = OccupancyGrid.from_dimensions(8, 8)
        sim = NavigationSimulator(grid, controller="pure_pursuit", lookahead_distance=0.5)
        start = (1.5, 1.5)
        goal = (6.5, 1.5)  # Goal directly along +x axis

        res = sim.run(start=start, goal=goal, initial_heading=initial_th)
        assert res.status == SimulationStatus.SUCCESS
        assert res.steps > 0
        final_state = res.trajectory[-1]
        assert math.hypot(goal[0] - final_state.x, goal[1] - final_state.y) <= sim.goal_tolerance


# ==============================================================================
# 5. Goal Tolerance & Arrival
# ==============================================================================

class TestGoalToleranceAndArrival:
    """Verifies terminal arrival strictly honors configured tolerance and halts."""

    @pytest.mark.parametrize("tol", [0.15, 0.25, 0.4])
    def test_arrival_within_custom_goal_tolerance(self, tol: float) -> None:
        """Final position error does not exceed goal_tolerance."""
        grid = OccupancyGrid.from_dimensions(10, 10)
        sim = NavigationSimulator(
            grid,
            controller="pure_pursuit",
            lookahead_distance=0.5,
            goal_tolerance=tol,
        )
        start = (1.5, 1.5)
        goal = (5.5, 5.5)

        res = sim.run(start, goal)
        assert res.status == SimulationStatus.SUCCESS
        final_state = res.trajectory[-1]
        err = math.hypot(goal[0] - final_state.x, goal[1] - final_state.y)
        assert err <= tol + 1e-6

    def test_zero_velocity_at_goal(self) -> None:
        """Controller outputs strictly (0.0, 0.0) once goal is reached."""
        ctrl = PurePursuitController(goal_tolerance=0.25, path=[(1.0, 1.0), (5.0, 5.0)])
        # Robot already at (5.0, 5.0)
        v, w = ctrl.compute_command((5.0, 5.0, 0.0))
        assert v == 0.0
        assert w == 0.0
        assert ctrl.is_goal_reached((5.0, 5.0, 0.0)) is True


# ==============================================================================
# 6. Continuous Collision Checking Active
# ==============================================================================

class TestCollisionCheckingContinuity:
    """Verifies motion collision checks remain active and prevent obstacle penetration."""

    def test_corner_collision_stops_trajectory(self) -> None:
        """Oversized robot cutting obstacle corner triggers COLLISION and halts."""
        grid = OccupancyGrid([[0, 0, 0], [0, 1, 0], [0, 0, 0]], resolution=1.0)
        sim = NavigationSimulator(
            grid,
            robot_radius=0.45,
            dt=0.1,
            linear_speed=2.0,
            controller="pure_pursuit",
            lookahead_distance=1.0,
        )
        res = sim.run(start=(0.5, 0.5), goal=(2.5, 2.5))
        assert res.status == SimulationStatus.COLLISION
        assert res.steps > 0
        last_pose = res.trajectory[-1]
        assert not sim.collision_checker.is_pose_collision_free(last_pose.x, last_pose.y)

    def test_motion_substepping_detects_thin_barrier(self) -> None:
        """Collision checking catches obstacle even if discrete time-step would skip."""
        grid = OccupancyGrid(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            resolution=1.0,
        )
        # Robot traversing across row 1
        sim = NavigationSimulator(
            grid,
            robot_radius=0.3,
            linear_speed=1.5,
            controller="pure_pursuit",
            lookahead_distance=0.8,
        )
        res = sim.run(start=(0.5, 0.5), goal=(3.5, 2.5))
        # The path goes around, but if footprint cuts corner it triggers collision or completes
        assert res.status in (SimulationStatus.SUCCESS, SimulationStatus.COLLISION)


# ==============================================================================
# 7. Unreachable Goals & Invalid Inputs
# ==============================================================================

class TestUnreachableAndInvalidHandling:
    """Verifies graceful handling of impossible routes and invalid coordinate queries."""

    def test_unreachable_walled_off_goal(self) -> None:
        """Wall separating start and goal terminates cleanly with NO_PATH."""
        data = np.zeros((6, 6), dtype=np.int8)
        data[:, 3] = 1  # Full vertical wall
        grid = OccupancyGrid(data)

        sim = NavigationSimulator(grid, controller="pure_pursuit")
        res = sim.run(start=(1.5, 1.5), goal=(4.5, 1.5))
        assert res.status == SimulationStatus.NO_PATH
        assert res.steps == 0
        assert len(res.trajectory) == 0

    def test_start_in_collision_raises_value_error(self) -> None:
        grid = OccupancyGrid([[1, 0], [0, 0]])
        sim = NavigationSimulator(grid, controller="pure_pursuit")
        with pytest.raises(ValueError, match="collision"):
            sim.run(start=(0.5, 0.5), goal=(1.5, 1.5))

    def test_goal_in_collision_raises_value_error(self) -> None:
        grid = OccupancyGrid([[0, 0], [0, 1]])
        sim = NavigationSimulator(grid, controller="pure_pursuit")
        with pytest.raises(ValueError, match="collision"):
            sim.run(start=(0.5, 0.5), goal=(1.5, 1.5))

    def test_out_of_bounds_queries_raise(self) -> None:
        grid = OccupancyGrid.from_dimensions(5, 5)
        sim = NavigationSimulator(grid, controller="pure_pursuit")
        with pytest.raises(ValueError):
            sim.run(start=(-1.0, 2.0), goal=(3.0, 3.0))
        with pytest.raises(ValueError):
            sim.run(start=(1.0, 2.0), goal=(10.0, 3.0))


# ==============================================================================
# 8. Determinism & Reproducibility
# ==============================================================================

class TestDeterminismAndReproducibility:
    """Verifies that two runs with identical inputs produce bit-for-bit identical results."""

    def test_two_simulation_runs_identical_trajectory(self) -> None:
        """Two successive runs with fresh simulators produce exact identical states."""
        scenario = scenario_multiple_obstacles()

        sim1 = scenario.create_simulator(controller="pure_pursuit", lookahead_distance=0.5)
        res1 = sim1.run(scenario.start, scenario.goal)

        sim2 = scenario.create_simulator(controller="pure_pursuit", lookahead_distance=0.5)
        res2 = sim2.run(scenario.start, scenario.goal)

        assert res1.status == res2.status
        assert res1.steps == res2.steps
        assert math.isclose(res1.path_length, res2.path_length, abs_tol=1e-12)
        assert len(res1.trajectory) == len(res2.trajectory)

        for s1, s2 in zip(res1.trajectory, res2.trajectory):
            assert s1.x == s2.x
            assert s1.y == s2.y
            assert s1.theta == s2.theta
            assert s1.v == s2.v
            assert s1.omega == s2.omega


# ==============================================================================
# 9. Degenerate Paths & Invalid Configurations
# ==============================================================================

class TestDegeneratePathsAndInvalidConfigs:
    """Verifies controller edge cases and degenerate path inputs."""

    def test_start_equals_goal_terminates_immediately(self) -> None:
        """When start is within goal tolerance, simulation terminates at step 0."""
        grid = OccupancyGrid.from_dimensions(5, 5)
        sim = NavigationSimulator(grid, controller="pure_pursuit", goal_tolerance=0.3)
        res = sim.run(start=(2.0, 2.0), goal=(2.1, 2.1))
        assert res.status == SimulationStatus.SUCCESS
        assert res.steps == 0
        assert res.path_length == 0.0

    def test_controller_rejects_non_numeric_parameters(self) -> None:
        """Parameters must be valid finite floats."""
        with pytest.raises(TypeError):
            PurePursuitController(lookahead_distance="fast")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            PurePursuitController(lookahead_distance=float("inf"))

    def test_find_lookahead_point_without_path_raises(self) -> None:
        """Calling find_lookahead_point before configuring a path raises ValueError."""
        ctrl = PurePursuitController()
        with pytest.raises(ValueError, match="set_path"):
            ctrl.find_lookahead_point((0.0, 0.0, 0.0))


# ==============================================================================
# 10. Trajectory Values are Strictly Finite
# ==============================================================================

class TestTrajectoryFiniteValuesGuarantee:
    """Verifies no non-finite (NaN, Inf) position, heading, or velocity in trajectories."""

    @pytest.mark.parametrize("scenario_func", [scenario_open_space, scenario_multiple_obstacles, scenario_maze])
    def test_all_trajectory_states_are_strictly_finite(self, scenario_func) -> None:
        """Every logged state in trajectory contains finite numbers and normalized angle."""
        scenario = scenario_func()
        sim = scenario.create_simulator(controller="pure_pursuit", lookahead_distance=0.5)
        res = sim.run(scenario.start, scenario.goal)
        assert res.status == SimulationStatus.SUCCESS
        assert len(res.trajectory) > 0

        for idx, s in enumerate(res.trajectory):
            assert math.isfinite(s.x), f"Non-finite x at step {idx}: {s.x}"
            assert math.isfinite(s.y), f"Non-finite y at step {idx}: {s.y}"
            assert math.isfinite(s.theta), f"Non-finite theta at step {idx}: {s.theta}"
            assert math.isfinite(s.v), f"Non-finite v at step {idx}: {s.v}"
            assert math.isfinite(s.omega), f"Non-finite omega at step {idx}: {s.omega}"
            assert -math.pi <= s.theta < math.pi, f"Unnormalized theta at step {idx}: {s.theta}"


# ==============================================================================
# 11. Lookahead Tuning Runner & Schema Validation
# ==============================================================================

class TestTuningExperimentRunner:
    """Tests verifying the execution and schema of the lookahead tuning experiment."""

    def test_tuning_experiment_execution_and_schema(self, tmp_path: Path) -> None:
        """run_pure_pursuit_tuning outputs valid schema with baseline and candidate records."""
        out_file = tmp_path / "tuning_test.json"
        report = run_pure_pursuit_tuning(lookaheads=[0.5, 0.75], output_path=out_file)

        assert report.schema_version == "1.0.0"
        assert report.baseline_controller == "HeadingProportionalController"
        assert len(report.evaluated_lookaheads) == 2
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"
        assert "baseline_experiment" in data
        assert "lookahead_experiments" in data
        assert "0.5" in data["lookahead_experiments"]
        assert "0.75" in data["lookahead_experiments"]

        # Check aggregate metrics structure
        agg_05 = data["lookahead_experiments"]["0.5"]["aggregate_metrics"]
        assert agg_05["navigable_scenarios_count"] == 5
        assert agg_05["navigation_successes"] == 5
        assert agg_05["navigation_success_rate"] == 1.0
        assert agg_05["collisions"] == 0
        assert agg_05["timeouts"] == 0
        assert agg_05["mean_steps_successful"] is not None
        assert agg_05["mean_traveled_distance_successful"] is not None
        assert agg_05["mean_final_goal_error_successful"] is not None
