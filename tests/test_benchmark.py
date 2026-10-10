"""Unit tests for the Benchmark Suite (Phase 3B).

Tests cover:
1. All seven scenarios are defined with unique names.
2. Each predefined scenario produces its specified expected outcome.
3. NO_PATH is counted as an expected-outcome match, not a navigation success.
4. Expected validation exceptions are classified correctly.
5. An unexpected outcome is reported as a benchmark failure.
6. JSON report serialization works and contains the required fields.
7. Two runs produce matching scenario outcomes and deterministic metrics.
8. Running a scenario twice does not mutate its original map or leak simulator state.
9. Empty or invalid scenario collections are handled clearly.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkSuiteResult,
    ScenarioResult,
)
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
from src.environment import OccupancyGrid
from src.simulator import SimulationStatus


class TestBenchmarkScenariosDefinition:
    """Tests verifying scenario definitions, uniqueness, and immutability."""

    def test_exactly_seven_predefined_scenarios(self) -> None:
        """Requirement 1: Verify exactly seven default scenarios are registered."""
        scenarios = get_default_scenarios()
        assert len(scenarios) == 7

    def test_all_scenarios_have_unique_names(self) -> None:
        """Requirement 1: All predefined scenarios must have distinct names."""
        scenarios = get_default_scenarios()
        names = [s.name for s in scenarios]
        assert len(names) == len(set(names)), f"Duplicate scenario names found: {names}"

    def test_lookup_scenario_by_name(self) -> None:
        """Scenarios can be retrieved by name, unknown names raise KeyError."""
        s = get_scenario_by_name("open_space")
        assert s.name == "open_space"

        with pytest.raises(KeyError, match="Unknown benchmark scenario 'nonexistent'"):
            get_scenario_by_name("nonexistent")

    def test_scenario_specification_validation(self) -> None:
        """BenchmarkScenario rejects invalid or conflicting configurations."""
        grid = OccupancyGrid.from_dimensions(5, 5)

        # Empty name rejected
        with pytest.raises(ValueError, match="name"):
            BenchmarkScenario(
                name="",
                description="desc",
                grid=grid,
                start=(1.0, 1.0),
                goal=(3.0, 3.0),
                expected_status=SimulationStatus.SUCCESS,
            )

        # Both status and exception specified
        with pytest.raises(ValueError, match="exactly one"):
            BenchmarkScenario(
                name="invalid_both",
                description="desc",
                grid=grid,
                start=(1.0, 1.0),
                goal=(3.0, 3.0),
                expected_status=SimulationStatus.SUCCESS,
                expected_exception=ValueError,
            )

        # Neither status nor exception specified
        with pytest.raises(ValueError, match="exactly one"):
            BenchmarkScenario(
                name="invalid_neither",
                description="desc",
                grid=grid,
                start=(1.0, 1.0),
                goal=(3.0, 3.0),
            )

        # Invalid start coordinate type
        with pytest.raises(TypeError):
            BenchmarkScenario(
                name="bad_coord",
                description="desc",
                grid=grid,
                start=(True, 1.0),  # type: ignore[arg-type]
                goal=(3.0, 3.0),
                expected_status=SimulationStatus.SUCCESS,
            )


class TestBenchmarkScenarioExecution:
    """Tests verifying individual scenario execution and outcome classification."""

    def test_open_space_scenario_execution(self) -> None:
        """Requirement 2: Open space scenario reaches goal successfully."""
        s = scenario_open_space()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is True
        assert res.actual_outcome == "SUCCESS"
        assert res.is_unexpected_failure is False
        assert res.metrics is not None
        assert res.metrics["success"] is True

    def test_one_obstacle_scenario_execution(self) -> None:
        """Requirement 2: One obstacle scenario bypasses barrier and succeeds."""
        s = scenario_one_obstacle()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is True
        assert res.actual_outcome == "SUCCESS"

    def test_multiple_obstacles_scenario_execution(self) -> None:
        """Requirement 2: Multiple obstacles demo scenario reaches goal."""
        s = scenario_multiple_obstacles()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is True
        assert res.actual_outcome == "SUCCESS"

    def test_narrow_corridor_scenario_execution(self) -> None:
        """Requirement 2: Narrow corridor scenario traverses passage."""
        s = scenario_narrow_corridor()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is True
        assert res.actual_outcome == "SUCCESS"

    def test_maze_scenario_execution(self) -> None:
        """Requirement 2: Maze scenario navigates serpentine turns."""
        s = scenario_maze()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is True
        assert res.actual_outcome == "SUCCESS"

    def test_no_path_counts_as_match_not_navigation_success(self) -> None:
        """Requirement 3: NO_PATH is an expected-outcome match, NOT navigation success."""
        s = scenario_unreachable_goal()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is False
        assert res.is_expected_failure is True
        assert res.is_unexpected_failure is False
        assert res.actual_outcome == "NO_PATH"
        assert res.simulation_status == "NO_PATH"
        assert res.metrics is not None
        assert res.metrics["is_no_path"] is True
        assert res.metrics["traveled_distance"] == 0.0

    def test_expected_validation_exception_classified_correctly(self) -> None:
        """Requirement 4: Expected validation exceptions are classified as expected matches."""
        s = scenario_invalid_start_goal()
        runner = BenchmarkRunner([s])
        res = runner.run_scenario(s)

        assert res.expectation_met is True
        assert res.navigation_success is False
        assert res.is_expected_failure is True
        assert res.is_unexpected_failure is False
        assert res.exception_type == "ValueError"
        assert "Start position" in str(res.exception_message)
        assert res.metrics is None


class TestBenchmarkRunnerAndFailures:
    """Tests verifying benchmark failure detection, reporting, and runner behaviors."""

    def test_unexpected_outcome_reported_as_benchmark_failure(self) -> None:
        """Requirement 5: A scenario expecting SUCCESS that terminates in NO_PATH fails the benchmark."""
        # Unreachable grid configured with expected_status=SUCCESS
        grid_data = np.zeros((6, 6), dtype=np.int8)
        grid_data[:, 3] = 1  # Dividing wall
        grid = OccupancyGrid(grid_data)

        flawed_scenario = BenchmarkScenario(
            name="deliberately_impossible_success",
            description="Expecting SUCCESS on an unreachable map",
            grid=grid,
            start=(1.5, 2.5),
            goal=(4.5, 2.5),
            expected_status=SimulationStatus.SUCCESS,
        )

        runner = BenchmarkRunner([flawed_scenario])
        res = runner.run_scenario(flawed_scenario)

        assert res.expectation_met is False
        assert res.is_unexpected_failure is True
        assert res.actual_outcome == "NO_PATH"
        assert res.expected_outcome == "SUCCESS"
        assert res.error_details is not None

        suite_res = runner.run_all()
        assert suite_res.total_scenarios == 1
        assert suite_res.passed_scenarios == 0
        assert suite_res.failed_scenarios == 1
        assert suite_res.benchmark_pass_rate == 0.0

    def test_unexpected_exception_reported_as_failure(self) -> None:
        """Requirement 5: An unexpected simulation exception is reported as a benchmark failure."""
        grid = OccupancyGrid.from_dimensions(5, 5)
        # Goal placed in collision but expected_status is SUCCESS
        grid_data = np.zeros((5, 5), dtype=np.int8)
        grid_data[4, 4] = 1
        colliding_grid = OccupancyGrid(grid_data)

        bad_scenario = BenchmarkScenario(
            name="unexpected_exception_case",
            description="Goal is occupied so sim raises ValueError",
            grid=colliding_grid,
            start=(1.5, 1.5),
            goal=(4.5, 4.5),
            expected_status=SimulationStatus.SUCCESS,
        )

        runner = BenchmarkRunner([bad_scenario])
        res = runner.run_scenario(bad_scenario)

        assert res.expectation_met is False
        assert res.is_unexpected_failure is True
        assert "EXCEPTION_ValueError" in res.actual_outcome

    def test_json_report_serialization(self, tmp_path: Path) -> None:
        """Requirement 6: Benchmark report serializes to valid JSON with all required fields."""
        runner = BenchmarkRunner()
        suite_res = runner.run_all()

        json_file = tmp_path / "benchmark_test_output.json"
        exported_path = runner.export_json(suite_res, output_path=json_file)

        assert exported_path.exists()
        with open(exported_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"
        assert data["total_scenarios"] == 7
        assert data["passed_scenarios"] == 7
        assert data["failed_scenarios"] == 0
        assert data["benchmark_pass_rate"] == 1.0
        assert data["navigation_success_count"] == 5
        assert len(data["results"]) == 7

        for r in data["results"]:
            assert "scenario_name" in r
            assert "expected_outcome" in r
            assert "actual_outcome" in r
            assert "expectation_met" in r
            assert "navigation_success" in r

    def test_two_runs_produce_deterministic_metrics(self) -> None:
        """Requirement 7: Two consecutive runs produce exactly identical outcomes and metrics."""
        runner = BenchmarkRunner()
        run1 = runner.run_all()
        run2 = runner.run_all()

        assert run1.total_scenarios == run2.total_scenarios
        assert run1.passed_scenarios == run2.passed_scenarios
        assert run1.navigation_success_count == run2.navigation_success_count

        for r1, r2 in zip(run1.results, run2.results):
            assert r1.scenario_name == r2.scenario_name
            assert r1.actual_outcome == r2.actual_outcome
            assert r1.expectation_met == r2.expectation_met
            if r1.metrics is not None and r2.metrics is not None:
                assert r1.metrics["steps"] == r2.metrics["steps"]
                assert r1.metrics["traveled_distance"] == pytest.approx(r2.metrics["traveled_distance"])
                assert r1.metrics["planned_path_length"] == r2.metrics["planned_path_length"]

    def test_scenario_execution_does_not_mutate_grid(self) -> None:
        """Requirement 8: Running a scenario does not mutate original grid or leak state."""
        scenario = scenario_one_obstacle()
        grid_before = scenario.grid.grid

        runner = BenchmarkRunner([scenario])
        _ = runner.run_all()
        grid_after_run1 = scenario.grid.grid
        assert np.array_equal(grid_before, grid_after_run1)

        _ = runner.run_all()
        grid_after_run2 = scenario.grid.grid
        assert np.array_equal(grid_before, grid_after_run2)

    def test_empty_and_invalid_scenario_collections(self) -> None:
        """Requirement 9: Empty sequence produces clean zero-summary; invalid types raise TypeError."""
        # Empty collection
        empty_runner = BenchmarkRunner([])
        report = empty_runner.run_all()
        assert report.total_scenarios == 0
        assert report.passed_scenarios == 0
        assert report.benchmark_pass_rate == 0.0
        assert len(report.results) == 0

        # Non-sequence input
        with pytest.raises(TypeError, match="sequence"):
            BenchmarkRunner("invalid_string")  # type: ignore[arg-type]

        # Sequence with invalid item
        with pytest.raises(TypeError, match="BenchmarkScenario"):
            BenchmarkRunner([123])  # type: ignore[list-item]
