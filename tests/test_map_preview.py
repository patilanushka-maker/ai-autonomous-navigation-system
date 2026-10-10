"""Tests for Phase 5A: Interactive Occupancy Grid Visualization and Map Preview.

Verifies:
1. Map preview rendering and data preparation (plot_occupancy_grid_preview & NavigationVisualizer).
2. World-to-grid coordinate alignment consistency with OccupancyGrid.
3. Start, goal, robot footprint, and obstacle representation in preview and post-simulation.
4. Post-simulation rendering with planned path, trajectory trail, and terminal status.
5. Edited map validation (rejecting obstacle collisions at start/goal).
6. Streamlit AppTest verification of preview rendering and editor actions.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.app import (
    get_preset_map_catalog,
    run_interactive_simulation,
    validate_pose,
)
from src.environment import OccupancyGrid
from src.simulator import NavigationSimulator, SimulationStatus
from src.visualization import (
    NavigationVisualizer,
    plot_occupancy_grid_preview,
    plot_simulation_result,
)


@pytest.fixture
def sample_grid() -> OccupancyGrid:
    """Fixture creating a 10x10 occupancy grid with a 2x2 central obstacle."""
    grid_matrix = np.zeros((10, 10), dtype=np.int8)
    grid_matrix[4:6, 4:6] = OccupancyGrid.OCCUPIED
    return OccupancyGrid(grid=grid_matrix, resolution=1.0, origin=(0.0, 0.0))


@pytest.fixture
def custom_frame_grid() -> OccupancyGrid:
    """Fixture creating a grid with non-zero origin and non-unit resolution."""
    grid_matrix = np.zeros((8, 12), dtype=np.int8)
    grid_matrix[3:5, 5:7] = OccupancyGrid.OCCUPIED
    return OccupancyGrid(grid=grid_matrix, resolution=0.5, origin=(2.0, -1.0))


# ===========================================================================
# 1. Map Preview Rendering and Data Preparation Tests
# ===========================================================================

class TestMapPreviewRendering:
    """Test map preview rendering, elements, and figure properties."""

    def test_map_preview_returns_figure_and_axes(self, sample_grid: OccupancyGrid) -> None:
        """Verify plot_occupancy_grid_preview returns valid Figure and Axes."""
        fig, ax = plot_occupancy_grid_preview(
            grid=sample_grid,
            start=(1.5, 1.5),
            goal=(8.5, 8.5),
            robot_radius=0.2,
        )
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
            assert isinstance(ax, matplotlib.axes.Axes)
            assert ax.get_xlim() == (0.0, 10.0)
            assert ax.get_ylim() == (0.0, 10.0)
            assert ax.get_aspect() in ("equal", 1.0)
            assert ax.get_xlabel() == "World X (m)"
            assert ax.get_ylabel() == "World Y (m)"
        finally:
            plt.close(fig)

    def test_map_preview_legend_entries(self, sample_grid: OccupancyGrid) -> None:
        """Verify legend includes obstacles, free space, start, goal, and footprint."""
        fig, ax = plot_occupancy_grid_preview(
            grid=sample_grid,
            start=(2.0, 2.0),
            goal=(8.0, 8.0),
            robot_radius=0.25,
        )
        try:
            legend = ax.get_legend()
            assert legend is not None
            labels = [text.get_text() for text in legend.get_texts()]

            assert "Obstacle" in labels
            assert "Free Space" in labels
            assert "Start Position" in labels
            assert "Goal Position" in labels
            assert any("Robot Footprint" in lbl for lbl in labels)
        finally:
            plt.close(fig)

    def test_map_preview_empty_grid_legend(self) -> None:
        """Verify obstacle-free grid correctly shows Free Space without crashing."""
        empty_grid = OccupancyGrid.from_dimensions(5, 5, resolution=1.0)
        fig, ax = plot_occupancy_grid_preview(
            grid=empty_grid,
            start=(1.0, 1.0),
            goal=(4.0, 4.0),
        )
        try:
            legend = ax.get_legend()
            assert legend is not None
            labels = [text.get_text() for text in legend.get_texts()]
            assert "Free Space" in labels
            assert "Obstacle" not in labels
        finally:
            plt.close(fig)

    def test_map_preview_without_start_or_goal(self, sample_grid: OccupancyGrid) -> None:
        """Verify rendering preview without start or goal still works."""
        fig, ax = plot_occupancy_grid_preview(grid=sample_grid)
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
            legend = ax.get_legend()
            labels = [text.get_text() for text in legend.get_texts()]
            assert "Obstacle" in labels
            assert "Free Space" in labels
            assert "Start Position" not in labels
            assert "Goal Position" not in labels
        finally:
            plt.close(fig)

    def test_map_preview_with_heading_arrow(self, sample_grid: OccupancyGrid) -> None:
        """Verify heading orientation arrow is drawn when initial_heading is passed."""
        fig, ax = plot_occupancy_grid_preview(
            grid=sample_grid,
            start=(2.0, 2.0),
            goal=(8.0, 8.0),
            initial_heading=math.pi / 2,  # 90 degrees pointing up
        )
        try:
            # Check for annotation child
            annotations = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
            assert len(annotations) >= 1
        finally:
            plt.close(fig)

    def test_map_preview_save_path(self, sample_grid: OccupancyGrid, tmp_path: Path) -> None:
        """Verify save_path writes an image file to disk."""
        target_file = tmp_path / "map_preview_test.png"
        assert not target_file.exists()
        fig, ax = plot_occupancy_grid_preview(
            grid=sample_grid,
            start=(1.0, 1.0),
            goal=(9.0, 9.0),
            save_path=target_file,
        )
        try:
            assert target_file.exists()
            assert target_file.stat().st_size > 0
        finally:
            plt.close(fig)

    def test_navigation_visualizer_plot_preview(self, sample_grid: OccupancyGrid) -> None:
        """Verify NavigationVisualizer.plot_preview method."""
        viz = NavigationVisualizer(robot_radius=0.3, figsize=(6.0, 6.0))
        fig, ax = viz.plot_preview(
            grid=sample_grid,
            start=(2.0, 2.0),
            goal=(7.0, 7.0),
            initial_heading=0.0,
        )
        try:
            assert isinstance(fig, matplotlib.figure.Figure)
            legend = ax.get_legend()
            labels = [t.get_text() for t in legend.get_texts()]
            assert any("r=0.30m" in lbl for lbl in labels)
        finally:
            plt.close(fig)


# ===========================================================================
# 2. World-to-Grid Coordinate Alignment Tests
# ===========================================================================

class TestCoordinateMappingConsistency:
    """Verify coordinate system rules required by OccupancyGrid."""

    def test_world_to_grid_and_grid_to_world_consistency(self, custom_frame_grid: OccupancyGrid) -> None:
        """Verify x maps to column, y maps to row, respecting origin and resolution."""
        ox, oy = custom_frame_grid.origin  # (2.0, -1.0)
        res = custom_frame_grid.resolution  # 0.5

        # Check cell (row=3, col=5)
        # Center should be at: x = ox + (col + 0.5) * res, y = oy + (row + 0.5) * res
        expected_cx = ox + (5 + 0.5) * res  # 2.0 + 2.75 = 4.75
        expected_cy = oy + (3 + 0.5) * res  # -1.0 + 1.75 = 0.75
        cx, cy = custom_frame_grid.grid_to_world(row=3, col=5, center=True)
        assert pytest.approx(cx) == expected_cx
        assert pytest.approx(cy) == expected_cy

        # Inverse world_to_grid
        row, col = custom_frame_grid.world_to_grid(cx, cy)
        assert row == 3
        assert col == 5

    def test_preview_extent_matches_world_boundaries(self, custom_frame_grid: OccupancyGrid) -> None:
        """Verify plot axes limits exactly match grid world boundaries."""
        fig, ax = plot_occupancy_grid_preview(custom_frame_grid)
        try:
            ox, oy = custom_frame_grid.origin
            res = custom_frame_grid.resolution
            expected_x_max = ox + custom_frame_grid.width * res
            expected_y_max = oy + custom_frame_grid.height * res

            assert ax.get_xlim() == (ox, expected_x_max)
            assert ax.get_ylim() == (oy, expected_y_max)
        finally:
            plt.close(fig)

    def test_obstacle_cell_placement_matches_world_collision(self, custom_frame_grid: OccupancyGrid) -> None:
        """Verify obstacle in grid matrix matches collision checker world coords."""
        # custom_frame_grid has obstacle at rows 3:5, cols 5:7
        # Cell (row=3, col=5) center is (4.75, 0.75)
        cx, cy = custom_frame_grid.grid_to_world(3, 5, center=True)
        assert custom_frame_grid.is_occupied(3, 5)

        # Pose validation at this obstacle cell center should fail
        valid, msg = validate_pose(custom_frame_grid, cx, cy, robot_radius=0.1)
        assert not valid
        assert "Collision" in msg


# ===========================================================================
# 3. Post-Simulation Rendering Tests
# ===========================================================================

class TestPostSimulationRendering:
    """Verify rendering after simulation includes all paths and terminal status."""

    def test_post_simulation_plot_contents(self, sample_grid: OccupancyGrid) -> None:
        """Verify plot_simulation_result contains trajectory, planned path, and status."""
        sim = NavigationSimulator(
            grid=sample_grid,
            robot_radius=0.2,
            controller="pure_pursuit",
            lookahead_distance=0.5,
            dt=0.1,
            max_steps=500,
        )
        result = sim.run(start=(1.5, 1.5), goal=(8.5, 8.5))
        assert result.status == SimulationStatus.SUCCESS

        fig, ax = plot_simulation_result(
            grid=sample_grid,
            result=result,
            robot_radius=0.2,
        )
        try:
            # Title should have status and step count
            title = ax.get_title()
            assert "SUCCESS" in title
            assert "Steps:" in title

            # Legend should have trajectory and planned path
            legend = ax.get_legend()
            assert legend is not None
            labels = [t.get_text() for t in legend.get_texts()]
            assert "Robot Trajectory" in labels
            assert "Planned Path (A*)" in labels
            assert "Start Position" in labels
            assert "Goal Position" in labels
        finally:
            plt.close(fig)

    def test_post_simulation_collision_status_display(self) -> None:
        """Verify collision simulation renders red footprint and collision title."""
        # Create a grid where obstacle directly blocks path
        grid = OccupancyGrid.from_dimensions(6, 6, resolution=1.0)
        # We manually construct a synthetic collision result
        from src.robot import RobotState
        from src.simulator import SimulationResult

        synthetic_res = SimulationResult(
            status=SimulationStatus.COLLISION,
            trajectory=[RobotState(x=1.0, y=1.0, theta=0.0, v=0.5, omega=0.0)],
            planned_path_grid=[(1, 1), (1, 2)],
            planned_path_world=[(1.0, 1.0), (2.0, 1.0)],
            steps=1,
            path_length=0.1,
            message="Collision detected",
            start_world=(1.0, 1.0),
            goal_world=(5.0, 1.0),
            controller_name="HeadingProportionalController",
        )

        fig, ax = plot_simulation_result(grid=grid, result=synthetic_res, robot_radius=0.2)
        try:
            assert "collision" in ax.get_title().lower()
        finally:
            plt.close(fig)


# ===========================================================================
# 4. Edited-Map Validation and Simulation Tests
# ===========================================================================

class TestEditedMapValidation:
    """Verify validation when an active map is edited."""

    def test_obstacle_added_at_start_fails_validation(self, sample_grid: OccupancyGrid) -> None:
        """Verify placing an obstacle directly at the start pose triggers collision error."""
        start_x, start_y = (1.5, 1.5)
        # Currently free
        val_before, _ = validate_pose(sample_grid, start_x, start_y, robot_radius=0.2)
        assert val_before is True

        # Toggle cell containing start to OCCUPIED
        row, col = sample_grid.world_to_grid(start_x, start_y)
        sample_grid.set_cell(row, col, OccupancyGrid.OCCUPIED)

        # Now validation MUST fail
        val_after, msg_after = validate_pose(sample_grid, start_x, start_y, robot_radius=0.2)
        assert val_after is False
        assert "Collision" in msg_after

    def test_obstacle_added_at_goal_fails_validation(self, sample_grid: OccupancyGrid) -> None:
        """Verify placing an obstacle at the goal pose triggers collision error."""
        goal_x, goal_y = (8.5, 8.5)
        val_before, _ = validate_pose(sample_grid, goal_x, goal_y, robot_radius=0.2)
        assert val_before is True

        row, col = sample_grid.world_to_grid(goal_x, goal_y)
        sample_grid.set_cell(row, col, OccupancyGrid.OCCUPIED)

        val_after, msg_after = validate_pose(sample_grid, goal_x, goal_y, robot_radius=0.2)
        assert val_after is False
        assert "Collision" in msg_after

    def test_edited_map_resimulation_adapts_path(self) -> None:
        """Verify that editing a map causes A* to find a bypass around the new obstacle."""
        grid = OccupancyGrid.from_dimensions(7, 7, resolution=1.0)
        start = (1.5, 1.5)
        goal = (5.5, 1.5)

        # First run: straight line path
        res1, m1, err1 = run_interactive_simulation(
            grid=grid,
            start=start,
            goal=goal,
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
        assert err1 is None
        assert res1.status == SimulationStatus.SUCCESS

        # Place a barrier directly in front of the straight line at col=3, rows 0..2
        for r in range(0, 3):
            grid.set_cell(r, 3, OccupancyGrid.OCCUPIED)

        # Re-run simulation: path planner must route around the new barrier
        res2, m2, err2 = run_interactive_simulation(
            grid=grid,
            start=start,
            goal=goal,
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
        assert err2 is None
        assert res2.status == SimulationStatus.SUCCESS
        assert res2.path_length > res1.path_length  # Detour required


# ===========================================================================
# 5. Input Validation Tests for plot_occupancy_grid_preview
# ===========================================================================

class TestPreviewInputValidation:
    """Verify robust exception handling for invalid inputs."""

    def test_invalid_grid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Expected OccupancyGrid"):
            plot_occupancy_grid_preview("not_a_grid")

    def test_invalid_robot_radius_raises(self, sample_grid: OccupancyGrid) -> None:
        with pytest.raises(TypeError):
            plot_occupancy_grid_preview(sample_grid, robot_radius=True)
        with pytest.raises(ValueError):
            plot_occupancy_grid_preview(sample_grid, robot_radius=-0.2)
        with pytest.raises(ValueError):
            plot_occupancy_grid_preview(sample_grid, robot_radius=float("nan"))

    def test_invalid_start_and_goal_coordinates_raises(self, sample_grid: OccupancyGrid) -> None:
        with pytest.raises(TypeError):
            plot_occupancy_grid_preview(sample_grid, start="bad_start")
        with pytest.raises(TypeError):
            plot_occupancy_grid_preview(sample_grid, goal=(1.0,))
        with pytest.raises(TypeError):
            plot_occupancy_grid_preview(sample_grid, start=(True, 1.0))
        with pytest.raises(ValueError):
            plot_occupancy_grid_preview(sample_grid, goal=(float("nan"), 1.0))

    def test_invalid_heading_raises(self, sample_grid: OccupancyGrid) -> None:
        with pytest.raises(TypeError):
            plot_occupancy_grid_preview(sample_grid, initial_heading="north")
        with pytest.raises(ValueError):
            plot_occupancy_grid_preview(sample_grid, initial_heading=float("inf"))

    def test_invalid_ax_raises(self, sample_grid: OccupancyGrid) -> None:
        with pytest.raises(TypeError):
            plot_occupancy_grid_preview(sample_grid, ax="not_axes")


# ===========================================================================
# 6. Streamlit AppTest Execution for Preview and Map Editor
# ===========================================================================

class TestStreamlitMapPreviewAndEditor:
    """Streamlit automated AppTest execution verifying Preview and Map Editor UI."""

    def test_app_displays_environment_map_preview(self) -> None:
        """Verify Simulation Studio displays the map preview header before simulation."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception
        # Check that Environment Map Preview markdown is present
        assert any("Environment Map Preview" in m.value for m in at.markdown)

    def test_app_editor_toggle_obstacle_button(self) -> None:
        """Verify clicking Toggle Obstacle button modifies the active grid."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Initial active_grid
        initial_grid: OccupancyGrid = at.session_state["active_grid"]
        initial_cell_state = initial_grid.is_occupied(0, 0)

        # Click Toggle Obstacle
        toggle_btns = [b for b in at.button if "Toggle Obstacle" in b.label]
        assert len(toggle_btns) == 1
        toggle_btns[0].click()
        at.run(timeout=10)
        assert not at.exception

        # Cell (0, 0) must now be flipped
        updated_grid: OccupancyGrid = at.session_state["active_grid"]
        assert updated_grid.is_occupied(0, 0) != initial_cell_state

    def test_app_editor_set_start_to_cell(self) -> None:
        """Verify Set Start to Cell button updates start coordinates."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Set edit row to 2 and col to 3
        row_inputs = [n for n in at.number_input if "Grid Row" in n.label]
        col_inputs = [n for n in at.number_input if "Grid Col" in n.label]
        assert len(row_inputs) >= 1 and len(col_inputs) >= 1
        row_inputs[0].set_value(2)
        col_inputs[0].set_value(3)
        at.run(timeout=10)
        assert not at.exception

        # Click Set Start to Cell
        set_start_btns = [b for b in at.button if "Set Start to Cell" in b.label]
        assert len(set_start_btns) == 1
        set_start_btns[0].click()
        at.run(timeout=10)
        assert not at.exception

        # Start coordinates in session state should match cell (2, 3) center
        grid: OccupancyGrid = at.session_state["active_grid"]
        expected_cx, expected_cy = grid.grid_to_world(2, 3, center=True)
        assert pytest.approx(at.session_state["start_x"]) == expected_cx
        assert pytest.approx(at.session_state["start_y"]) == expected_cy

    def test_app_editor_clear_and_reset_map(self) -> None:
        """Verify Clear Obstacles and Reset Preset buttons."""
        at = AppTest.from_file(str(ROOT_DIR / "src" / "app.py"))
        at.run(timeout=10)
        assert not at.exception

        # Click Clear Obstacles
        clear_btns = [b for b in at.button if "Clear Obstacles" in b.label]
        assert len(clear_btns) == 1
        clear_btns[0].click()
        at.run(timeout=10)
        assert not at.exception

        grid_cleared: OccupancyGrid = at.session_state["active_grid"]
        assert not np.any(grid_cleared.grid == OccupancyGrid.OCCUPIED)

        # Click Reset Preset
        reset_btns = [b for b in at.button if "Reset Preset" in b.label]
        assert len(reset_btns) == 1
        reset_btns[0].click()
        at.run(timeout=10)
        assert not at.exception

        grid_reset: OccupancyGrid = at.session_state["active_grid"]
        # Slalom course has obstacles
        assert np.any(grid_reset.grid == OccupancyGrid.OCCUPIED)
