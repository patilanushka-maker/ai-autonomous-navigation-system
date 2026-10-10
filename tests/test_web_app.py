"""Unit and integration tests for the Streamlit web application.

Verifies:
1. Preset map catalog and collision validation functions.
2. Interactive simulation execution with Heading and Pure Pursuit controllers.
3. Streamlit AppTest headless execution across all pages without exceptions.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.app import (
    get_preset_map_catalog,
    run_interactive_simulation,
    validate_pose,
)
from src.environment import OccupancyGrid
from src.simulator import SimulationStatus


class TestWebAppHelpers:
    """Tests for web application helper logic and validation routines."""

    def test_preset_map_catalog_integrity(self) -> None:
        """Verify that all preset maps instantiate valid grids and valid default coordinates."""
        catalog = get_preset_map_catalog()
        expected_presets = [
            "Slalom Obstacles (15x20)",
            "Open Space (10x10)",
            "One Obstacle (10x10)",
            "Narrow Corridor (11x11)",
            "Maze Labyrinth (15x15)",
        ]
        for name in expected_presets:
            assert name in catalog, f"Missing expected preset map: {name}"
            entry = catalog[name]
            grid = entry["factory"]()
            assert isinstance(grid, OccupancyGrid)
            assert grid.width > 0 and grid.height > 0

            # Default start and goal must be valid and collision-free
            sx, sy = entry["default_start"]
            gx, gy = entry["default_goal"]
            valid_s, _ = validate_pose(grid, sx, sy, robot_radius=0.2)
            valid_g, _ = validate_pose(grid, gx, gy, robot_radius=0.2)
            assert valid_s, f"Default start ({sx}, {sy}) for {name} is invalid or in collision"
            assert valid_g, f"Default goal ({gx}, {gy}) for {name} is invalid or in collision"

    def test_validate_pose_out_of_bounds(self) -> None:
        """Verify out-of-bounds coordinate rejection."""
        grid = OccupancyGrid.from_dimensions(10, 10, resolution=1.0)
        # Outside grid extents [0, 10] x [0, 10]
        valid, msg = validate_pose(grid, -1.0, 5.0, robot_radius=0.2)
        assert not valid
        assert "Out of bounds" in msg

        valid_far, _ = validate_pose(grid, 15.0, 5.0, robot_radius=0.2)
        assert not valid_far

    def test_validate_pose_in_collision(self) -> None:
        """Verify obstacle collision detection."""
        catalog = get_preset_map_catalog()
        grid = catalog["One Obstacle (10x10)"]["factory"]()
        # Obstacle is at rows 4..5, cols 4..5 -> world [4, 6] x [4, 6]
        valid, msg = validate_pose(grid, 5.0, 5.0, robot_radius=0.2)
        assert not valid
        assert "Collision" in msg

    def test_run_interactive_simulation_heading(self) -> None:
        """Verify interactive simulation execution with Heading controller."""
        catalog = get_preset_map_catalog()
        grid = catalog["Open Space (10x10)"]["factory"]()
        result, metrics, err = run_interactive_simulation(
            grid=grid,
            start=(1.5, 1.5),
            goal=(8.5, 8.5),
            controller_type="heading",
            lookahead_distance=0.5,
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=0.1,
            max_steps=1000,
            initial_heading=0.0,
        )
        assert err is None
        assert result is not None
        assert metrics is not None
        assert result.status == SimulationStatus.SUCCESS
        assert metrics.success is True
        assert len(result.trajectory) > 0
        assert metrics.traveled_distance > 0.0

    def test_run_interactive_simulation_pure_pursuit(self) -> None:
        """Verify interactive simulation execution with Pure Pursuit controller."""
        catalog = get_preset_map_catalog()
        grid = catalog["One Obstacle (10x10)"]["factory"]()
        result, metrics, err = run_interactive_simulation(
            grid=grid,
            start=(2.0, 5.0),
            goal=(8.0, 5.0),
            controller_type="pure_pursuit",
            lookahead_distance=0.5,
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=0.1,
            max_steps=1000,
            initial_heading=0.0,
        )
        assert err is None
        assert result is not None
        assert metrics is not None
        assert result.status == SimulationStatus.SUCCESS
        assert metrics.success is True
        assert result.controller_name == "PurePursuitController"

    def test_run_interactive_simulation_error_handling(self) -> None:
        """Verify graceful error reporting when invalid parameters are supplied."""
        catalog = get_preset_map_catalog()
        grid = catalog["Open Space (10x10)"]["factory"]()
        # Supplying an invalid negative dt triggers ValueError in simulator
        result, metrics, err = run_interactive_simulation(
            grid=grid,
            start=(1.5, 1.5),
            goal=(8.5, 8.5),
            controller_type="heading",
            lookahead_distance=0.5,
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=-0.1,  # Invalid
            max_steps=1000,
            initial_heading=0.0,
        )
        assert result is None
        assert metrics is None
        assert err is not None
        assert "dt" in err

    def test_run_interactive_simulation_start_equals_goal(self) -> None:
        """Verify immediate success when start pose equals goal pose."""
        catalog = get_preset_map_catalog()
        grid = catalog["Open Space (10x10)"]["factory"]()
        result, metrics, err = run_interactive_simulation(
            grid=grid,
            start=(5.0, 5.0),
            goal=(5.0, 5.0),
            controller_type="pure_pursuit",
            lookahead_distance=0.5,
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=0.1,
            max_steps=1000,
            initial_heading=0.0,
        )
        assert err is None
        assert result is not None
        assert metrics is not None
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps == 0
        assert metrics.success is True
        assert metrics.traveled_distance == 0.0

    def test_run_interactive_simulation_unreachable_goal(self) -> None:
        """Verify NO_PATH status when goal is enclosed by impassable walls."""
        # Create a grid where row 5 is completely blocked by a solid wall
        grid_data = np.zeros((10, 10), dtype=np.int8)
        grid_data[5, :] = 1  # Solid wall dividing north and south
        grid = OccupancyGrid(grid_data, resolution=1.0)

        result, metrics, err = run_interactive_simulation(
            grid=grid,
            start=(2.0, 2.0),
            goal=(8.0, 8.0),
            controller_type="heading",
            lookahead_distance=0.5,
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=0.1,
            max_steps=1000,
            initial_heading=0.0,
        )
        assert err is None
        assert result is not None
        assert metrics is not None
        assert result.status == SimulationStatus.NO_PATH
        assert metrics.success is False
        assert metrics.is_no_path is True

    def test_run_interactive_simulation_invalid_lookahead(self) -> None:
        """Verify error message when lookahead distance is invalid or non-positive."""
        catalog = get_preset_map_catalog()
        grid = catalog["Open Space (10x10)"]["factory"]()
        result, metrics, err = run_interactive_simulation(
            grid=grid,
            start=(1.5, 1.5),
            goal=(8.5, 8.5),
            controller_type="pure_pursuit",
            lookahead_distance=-0.5,  # Invalid negative lookahead
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=0.1,
            max_steps=1000,
            initial_heading=0.0,
        )
        assert result is None
        assert metrics is None
        assert err is not None
        assert "lookahead" in err.lower() or "positive" in err.lower()

    def test_trajectory_csv_export_format(self) -> None:
        """Verify that trajectory records serialize into valid CSV content with expected headers."""
        import pandas as pd
        catalog = get_preset_map_catalog()
        grid = catalog["Open Space (10x10)"]["factory"]()
        result, _, _ = run_interactive_simulation(
            grid=grid,
            start=(1.5, 1.5),
            goal=(3.5, 1.5),
            controller_type="heading",
            lookahead_distance=0.5,
            angular_gain=2.0,
            linear_speed=0.5,
            angular_speed_limit=1.0,
            robot_radius=0.2,
            goal_tolerance=0.25,
            dt=0.1,
            max_steps=200,
            initial_heading=0.0,
        )
        assert result is not None
        assert len(result.trajectory) > 0

        records = [
            {
                "Step": k,
                "X (m)": round(s.x, 3),
                "Y (m)": round(s.y, 3),
                "Heading θ (rad)": round(s.theta, 3),
                "Heading θ (deg)": round(math.degrees(s.theta), 1),
                "Linear Velocity v (m/s)": round(s.v, 3),
                "Angular Velocity ω (rad/s)": round(s.omega, 3),
            }
            for k, s in enumerate(result.trajectory)
        ]
        df = pd.DataFrame(records)
        csv_str = df.to_csv(index=False)
        assert "Step,X (m),Y (m),Heading θ (rad),Heading θ (deg),Linear Velocity v (m/s),Angular Velocity ω (rad/s)" in csv_str
        assert len(csv_str.strip().split("\n")) == len(result.trajectory) + 1


class TestStreamlitAppTest:
    """Streamlit automated AppTest execution verifying full UI rendering."""

    def test_app_loads_without_exceptions(self) -> None:
        """Verify that the Streamlit app loads and renders initial page cleanly."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception, f"App raised unexpected exception: {at.exception}"
        # Title should contain Simulation Studio
        assert len(at.title) >= 1
        assert "Simulation Studio" in at.title[0].value

    def test_app_root_launcher_loads(self) -> None:
        """Verify that top-level app.py launcher loads identically."""
        at = AppTest.from_file(str(ROOT_DIR / "app.py"))
        at.run(timeout=10)
        assert not at.exception, f"app.py raised unexpected exception: {at.exception}"

    def test_app_navigate_to_benchmark_page(self) -> None:
        """Verify navigation to Benchmark Suite page."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Switch radio in sidebar
        at.sidebar.radio[0].set_value("📊 Benchmark Suite")
        at.run(timeout=10)
        assert not at.exception
        assert any("Benchmark" in t.value for t in at.title)

    def test_app_navigate_to_architecture_page(self) -> None:
        """Verify navigation to Architecture & Specs page."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        at.sidebar.radio[0].set_value("📘 Architecture & Specs")
        at.run(timeout=10)
        assert not at.exception
        assert any("Architecture" in t.value for t in at.title)

    def test_app_trigger_simulation_run(self) -> None:
        """Verify clicking Run Simulation button renders results."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Find Run Simulation button
        run_buttons = [b for b in at.button if "Run Simulation" in b.label]
        assert len(run_buttons) == 1
        run_buttons[0].click()
        at.run(timeout=15)
        assert not at.exception
        # Success banner should be visible
        assert any("SUCCESS" in s.value for s in at.success)

    def test_app_simulation_with_heading_controller(self) -> None:
        """Verify executing simulation with Heading Proportional controller in UI."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Switch controller selectbox to heading
        ctl_select = [s for s in at.selectbox if "Controller" in s.label]
        assert len(ctl_select) >= 1
        ctl_select[0].set_value("heading")
        at.run(timeout=10)
        assert not at.exception

        # Run simulation
        run_buttons = [b for b in at.button if "Run Simulation" in b.label]
        assert len(run_buttons) == 1
        run_buttons[0].click()
        at.run(timeout=15)
        assert not at.exception
        assert any("SUCCESS" in s.value for s in at.success)

    def test_app_benchmark_single_scenario_run(self) -> None:
        """Verify executing a single scenario on the Benchmark page."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Navigate to benchmark
        at.sidebar.radio[0].set_value("📊 Benchmark Suite")
        at.run(timeout=10)
        assert not at.exception

        # Click Run Scenario button
        run_scenario_btn = [b for b in at.button if "Run Scenario" in b.label]
        assert len(run_scenario_btn) == 1
        run_scenario_btn[0].click()
        at.run(timeout=15)
        assert not at.exception
        assert any("Expectation Met" in s.value for s in at.success)

    def test_app_side_by_side_comparison_run(self) -> None:
        """Verify executing Side-by-Side comparison on the Benchmark page."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Navigate to benchmark
        at.sidebar.radio[0].set_value("📊 Benchmark Suite")
        at.run(timeout=10)
        assert not at.exception

        # Switch to Side-by-Side comparison
        mode_radio = [r for r in at.radio if "Evaluation Mode" in r.label]
        assert len(mode_radio) == 1
        mode_radio[0].set_value("Side-by-Side Controller Comparison")
        at.run(timeout=10)
        assert not at.exception

        # Trigger Run Side-by-Side Comparison button
        comp_btn = [b for b in at.button if "Run Side-by-Side Comparison" in b.label]
        assert len(comp_btn) == 1
        comp_btn[0].click()
        at.run(timeout=30)
        assert not at.exception
        assert any("Comparative Overview" in m.value for m in at.markdown)

    def test_app_full_benchmark_suite_run(self) -> None:
        """Verify executing the full 7-scenario suite on the Benchmark page."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Navigate to benchmark
        at.sidebar.radio[0].set_value("📊 Benchmark Suite")
        at.run(timeout=10)
        assert not at.exception

        # Switch to Full Benchmark Suite
        mode_radio = [r for r in at.radio if "Evaluation Mode" in r.label]
        assert len(mode_radio) == 1
        mode_radio[0].set_value("Full Benchmark Suite (All 7 Scenarios)")
        at.run(timeout=10)
        assert not at.exception

        # Trigger Run button
        suite_btn = [b for b in at.button if "Execute Full Benchmark Suite" in b.label]
        assert len(suite_btn) == 1
        suite_btn[0].click()
        at.run(timeout=30)
        assert not at.exception
        assert any("Suite Summary" in m.value for m in at.markdown)


