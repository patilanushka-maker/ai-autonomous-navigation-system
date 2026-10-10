"""Unit tests for the end-to-end demonstration module (src/demo.py).

Tests verify:
- Deterministic grid structure and obstacle placement.
- End-to-end execution of `run_navigation_demo`.
- Valid artifact export (figure existence and JSON parsing).
- Matching between simulation result and exported metrics.
- Clean handling of stale output files.
- Command-line entry point execution and exit code.
"""

import json
from pathlib import Path
import pytest

from src.environment import OccupancyGrid
from src.simulator import SimulationStatus
from src.demo import create_demo_grid, main, run_navigation_demo


def test_create_demo_grid_structure():
    """Verify demo grid dimensions, resolution, start/goal clearance, and obstacles."""
    grid = create_demo_grid()

    assert grid.height == 15
    assert grid.width == 20
    assert grid.resolution == 1.0
    assert grid.origin == (0.0, 0.0)

    # Start cell (row 2, col 2) and Goal cell (row 12, col 17) must be free
    assert grid.is_free(2, 2)
    assert grid.is_free(12, 17)

    # Barrier 1 (cols 6-7, rows 0-9) must be occupied
    for r in range(10):
        for c in range(6, 8):
            assert grid.is_occupied(r, c)

    # Barrier 2 (cols 12-13, rows 5-14) must be occupied
    for r in range(5, 15):
        for c in range(12, 14):
            assert grid.is_occupied(r, c)

    # Obstacle 3 (cols 2-3, rows 8-10) must be occupied
    for r in range(8, 11):
        for c in range(2, 4):
            assert grid.is_occupied(r, c)

    # Obstacle 4 (cols 16-17, rows 2-4) must be occupied
    for r in range(2, 5):
        for c in range(16, 18):
            assert grid.is_occupied(r, c)

    # Northern passage (cols 6-7, rows 10-14) must be free
    for r in range(10, 15):
        for c in range(6, 8):
            assert grid.is_free(r, c)

    # Southern passage (cols 12-13, rows 0-4) must be free
    for r in range(5):
        for c in range(12, 14):
            assert grid.is_free(r, c)


def test_run_navigation_demo_artifacts_and_metrics(tmp_path: Path):
    """Verify complete demo execution, figure output, and JSON metrics generation."""
    test_fig = tmp_path / "test_demo.png"
    test_json = tmp_path / "test_demo_metrics.json"

    result, metrics, report = run_navigation_demo(
        figure_path=test_fig,
        metrics_path=test_json,
        show=False,
    )

    # Simulation outcome checks
    assert result.status == SimulationStatus.SUCCESS
    assert metrics.success is True
    assert metrics.steps > 0
    assert metrics.planned_path_length is not None
    assert metrics.planned_path_length == 37.0
    assert metrics.traveled_distance > 0.0
    assert metrics.actual_to_planned_ratio is not None
    assert metrics.final_goal_error is not None
    assert metrics.final_goal_error <= 0.25

    # Output artifact existence and non-zero size
    assert test_fig.exists()
    assert test_fig.stat().st_size > 0
    assert test_json.exists()
    assert test_json.stat().st_size > 0

    # JSON report parsing and data integrity
    with open(test_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["status"] == "SUCCESS"
    assert loaded["success"] is True
    assert loaded["steps"] == metrics.steps
    assert loaded["planned_path_length"] == metrics.planned_path_length
    assert loaded["traveled_distance"] == metrics.traveled_distance
    assert loaded["actual_to_planned_ratio"] == metrics.actual_to_planned_ratio
    assert loaded["final_goal_error"] == metrics.final_goal_error
    assert loaded["is_collision"] is False
    assert loaded["is_timeout"] is False
    assert loaded["is_no_path"] is False
    assert loaded["planning_duration"] is None


def test_stale_artifact_cleanup(tmp_path: Path):
    """Verify that existing output files are cleanly replaced."""
    test_fig = tmp_path / "stale.png"
    test_json = tmp_path / "stale.json"

    # Create stale placeholder files
    test_fig.write_text("stale image data", encoding="utf-8")
    test_json.write_text("{\"stale\": true}", encoding="utf-8")

    result, metrics, _ = run_navigation_demo(
        figure_path=test_fig,
        metrics_path=test_json,
        show=False,
    )

    assert result.status == SimulationStatus.SUCCESS
    # Fresh JSON should not contain stale data
    with open(test_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "stale" not in loaded
    assert loaded["status"] == "SUCCESS"


def test_demo_cli_entry_point(tmp_path: Path):
    """Verify demo CLI main function executes with exit code 0."""
    test_fig = tmp_path / "cli_demo.png"
    test_json = tmp_path / "cli_demo.json"

    exit_code = main([
        "--figure-path", str(test_fig),
        "--metrics-path", str(test_json),
    ])

    assert exit_code == 0
    assert test_fig.exists()
    assert test_json.exists()
