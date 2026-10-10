"""Unit tests for the 2D Visualization module (Phase 2G).

Tests cover:
- Open grid rendering with a successful route.
- Environment with obstacles rendering.
- Unsuccessful results (NO_PATH, COLLISION, TIMEOUT) handled gracefully.
- Correct axis limits and coordinates for nonzero origin and non-unit resolution.
- Exporting/saving figures to a caller-specified path (and verifying no automatic save).
- Handling empty trajectory and path data safely without crashing.
- Rendering on pre-existing caller-supplied Matplotlib Axes.
- NavigationVisualizer object-oriented wrapper class.
- Input validation on types and parameters.
"""

import math
from pathlib import Path
import pytest
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test execution
import matplotlib.pyplot as plt

from src.environment import OccupancyGrid
from src.simulator import NavigationSimulator, SimulationResult, SimulationStatus
from src.visualization import NavigationVisualizer, plot_simulation_result


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def open_grid():
    """5x5 fully open grid with resolution=1.0, origin=(0.0, 0.0)."""
    return OccupancyGrid.from_dimensions(5, 5, initial_value=0, resolution=1.0)


@pytest.fixture
def corridor_grid():
    """7x7 grid with obstacles forming an L-shaped corridor."""
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
def successful_result(open_grid):
    """Successful simulation run on open grid."""
    sim = NavigationSimulator(open_grid, robot_radius=0.1, dt=0.1, goal_tolerance=0.3)
    return sim.run(start=(0.5, 0.5), goal=(4.5, 4.5))


# ─── Tests: Basic Rendering & Scenarios ──────────────────────────────────────


class TestVisualizationRendering:
    """Tests for rendering open grids, obstacles, and navigation routes."""

    def test_open_grid_successful_route(self, open_grid, successful_result):
        """Plotting a successful simulation on an open grid."""
        fig, ax = plot_simulation_result(open_grid, successful_result)
        try:
            assert fig is not None
            assert ax is not None
            assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"

            # World limits: [0.0, 5.0] along x and y
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            assert math.isclose(xlim[0], 0.0, abs_tol=1e-5)
            assert math.isclose(xlim[1], 5.0, abs_tol=1e-5)
            assert math.isclose(ylim[0], 0.0, abs_tol=1e-5)
            assert math.isclose(ylim[1], 5.0, abs_tol=1e-5)

            # Check title contains status and metrics
            title = ax.get_title()
            assert "SUCCESS" in title
            assert str(successful_result.steps) in title

            # Check labels
            assert "World X" in ax.get_xlabel()
            assert "World Y" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_obstacle_environment_visualization(self, corridor_grid):
        """Plotting a simulation navigating through an obstacle corridor."""
        sim = NavigationSimulator(corridor_grid, robot_radius=0.1, dt=0.1, goal_tolerance=0.3)
        result = sim.run(start=(0.5, 0.5), goal=(6.5, 6.5))
        assert result.status == SimulationStatus.SUCCESS

        fig, ax = plot_simulation_result(corridor_grid, result)
        try:
            # Legend should include obstacle and free space labels
            legend = ax.get_legend()
            assert legend is not None
            legend_texts = [t.get_text() for t in legend.get_texts()]
            assert "Obstacle" in legend_texts
            assert "Free Space" in legend_texts
            assert "Planned Path (A*)" in legend_texts
            assert "Robot Trajectory" in legend_texts
            assert "Start Position" in legend_texts
            assert "Goal Position" in legend_texts
        finally:
            plt.close(fig)

    def test_custom_title(self, open_grid, successful_result):
        """Custom title is respected when provided."""
        custom = "My Custom Benchmark Title"
        fig, ax = plot_simulation_result(open_grid, successful_result, title=custom)
        try:
            assert ax.get_title() == custom
        finally:
            plt.close(fig)


# ─── Tests: Unsuccessful Runs (NO_PATH, COLLISION, TIMEOUT) ──────────────────


class TestUnsuccessfulSimulations:
    """Tests verifying robust rendering of unsuccessful simulation outcomes."""

    def test_no_path_visualization(self, walled_grid):
        """Rendering a NO_PATH outcome with empty trajectory and no planned path."""
        sim = NavigationSimulator(walled_grid, robot_radius=0.1)
        result = sim.run(start=(0.5, 2.5), goal=(4.5, 2.5))
        assert result.status == SimulationStatus.NO_PATH
        assert result.planned_path_world is None
        assert len(result.trajectory) == 0

        fig, ax = plot_simulation_result(walled_grid, result)
        try:
            assert "NO_PATH" in ax.get_title()
            legend = ax.get_legend()
            assert legend is not None
            legend_texts = [t.get_text() for t in legend.get_texts()]
            # Start and Goal markers should be shown, but no trajectory
            assert "Start Position" in legend_texts
            assert "Goal Position" in legend_texts
            assert "Robot Trajectory" not in legend_texts
        finally:
            plt.close(fig)

    def test_collision_visualization(self):
        """Rendering a COLLISION outcome."""
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

        fig, ax = plot_simulation_result(grid, result)
        try:
            assert "COLLISION" in ax.get_title()
            # Trajectory exists
            legend = ax.get_legend()
            assert legend is not None
            legend_texts = [t.get_text() for t in legend.get_texts()]
            assert "Robot Trajectory" in legend_texts
        finally:
            plt.close(fig)

    def test_timeout_visualization(self, open_grid):
        """Rendering a TIMEOUT outcome."""
        sim = NavigationSimulator(
            open_grid,
            robot_radius=0.1,
            dt=0.1,
            max_steps=2,
            linear_speed=0.1,
        )
        result = sim.run(start=(0.5, 0.5), goal=(4.5, 4.5))
        assert result.status == SimulationStatus.TIMEOUT

        fig, ax = plot_simulation_result(open_grid, result)
        try:
            assert "TIMEOUT" in ax.get_title()
        finally:
            plt.close(fig)


# ─── Tests: Nonzero Origin & Non-Unit Resolution ─────────────────────────────


class TestCoordinateTransforms:
    """Tests verifying coordinate handling with non-standard origin and resolution."""

    def test_nonzero_origin_and_non_unit_resolution(self):
        """Plotting a grid with origin=(10.0, 20.0) and resolution=0.5."""
        grid = OccupancyGrid.from_dimensions(
            height=6,
            width=8,
            initial_value=0,
            resolution=0.5,
            origin=(10.0, 20.0),
        )
        sim = NavigationSimulator(grid, robot_radius=0.1, dt=0.1, goal_tolerance=0.25)
        # Start at cell (0, 0) center: (10.0 + 0.25, 20.0 + 0.25) = (10.25, 20.25)
        # Goal at cell (5, 7) center: (10.0 + 7.5*0.5, 20.0 + 5.5*0.5) = (13.75, 22.75)
        result = sim.run(start=(10.25, 20.25), goal=(13.75, 22.75))
        assert result.status == SimulationStatus.SUCCESS

        fig, ax = plot_simulation_result(grid, result)
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # x spans: [10.0, 10.0 + 8 * 0.5] = [10.0, 14.0]
            assert math.isclose(xlim[0], 10.0, abs_tol=1e-5)
            assert math.isclose(xlim[1], 14.0, abs_tol=1e-5)
            # y spans: [20.0, 20.0 + 6 * 0.5] = [20.0, 23.0]
            assert math.isclose(ylim[0], 20.0, abs_tol=1e-5)
            assert math.isclose(ylim[1], 23.0, abs_tol=1e-5)
        finally:
            plt.close(fig)


# ─── Tests: Saving and File Export ───────────────────────────────────────────


class TestFigureExport:
    """Tests verifying file export behavior."""

    def test_save_figure_to_caller_specified_path(self, open_grid, successful_result, tmp_path):
        """Figure is saved when save_path is provided."""
        out_file = tmp_path / "subdir" / "navigation_plot.png"
        assert not out_file.exists()

        fig, ax = plot_simulation_result(open_grid, successful_result, save_path=out_file)
        try:
            assert out_file.exists()
            assert out_file.stat().st_size > 0
        finally:
            plt.close(fig)

    def test_no_automatic_file_creation(self, open_grid, successful_result, tmp_path):
        """Figure is NOT saved to disk if save_path is None."""
        fig, ax = plot_simulation_result(open_grid, successful_result, save_path=None)
        try:
            # Check no new files were created in tmp_path
            assert list(tmp_path.iterdir()) == []
        finally:
            plt.close(fig)


# ─── Tests: Edge Cases & Safe Degradation ────────────────────────────────────


class TestEdgeCases:
    """Tests verifying robustness to sparse or edge-case data."""

    def test_empty_trajectory_and_path_data_safely(self, open_grid):
        """Synthetic result with empty trajectory and no path renders without crashing."""
        synthetic_result = SimulationResult(
            status=SimulationStatus.NO_PATH,
            steps=0,
            trajectory=[],
            planned_path_grid=None,
            planned_path_world=None,
            start_world=(1.0, 1.0),
            goal_world=(4.0, 4.0),
            path_length=0.0,
            message="No path found.",
        )
        fig, ax = plot_simulation_result(open_grid, synthetic_result)
        try:
            assert fig is not None
            assert ax is not None
            assert "NO_PATH" in ax.get_title()
        finally:
            plt.close(fig)

    def test_existing_axes_provided(self, open_grid, successful_result):
        """Caller passes an existing Matplotlib Axes instance."""
        fig_in, ax_in = plt.subplots(figsize=(6.0, 6.0))
        fig_out, ax_out = plot_simulation_result(open_grid, successful_result, ax=ax_in)
        try:
            assert ax_out is ax_in
            assert fig_out is fig_in
        finally:
            plt.close(fig_in)

    def test_start_equals_goal_rendering(self, open_grid):
        """Start equals goal with 1-element trajectory snapshot."""
        sim = NavigationSimulator(open_grid, robot_radius=0.1)
        result = sim.run(start=(2.5, 2.5), goal=(2.5, 2.5))
        assert result.status == SimulationStatus.SUCCESS
        assert result.steps == 0
        assert len(result.trajectory) == 1

        fig, ax = plot_simulation_result(open_grid, result)
        try:
            assert fig is not None
            assert "SUCCESS" in ax.get_title()
        finally:
            plt.close(fig)


# ─── Tests: NavigationVisualizer Class ───────────────────────────────────────


class TestNavigationVisualizerClass:
    """Tests for the NavigationVisualizer class."""

    def test_class_initialization_and_properties(self):
        """NavigationVisualizer exposes configured properties."""
        viz = NavigationVisualizer(robot_radius=0.25, figsize=(10.0, 10.0), dpi=200)
        assert viz.robot_radius == 0.25
        assert viz.figsize == (10.0, 10.0)
        assert viz.dpi == 200

    def test_class_plot_method(self, open_grid, successful_result):
        """NavigationVisualizer.plot produces valid figure and axes."""
        viz = NavigationVisualizer(robot_radius=0.15)
        fig, ax = viz.plot(open_grid, successful_result)
        try:
            assert fig is not None
            assert ax is not None
            assert "SUCCESS" in ax.get_title()
        finally:
            plt.close(fig)

    def test_class_plot_with_override_radius(self, open_grid, successful_result):
        """Override robot_radius in plot() call."""
        viz = NavigationVisualizer(robot_radius=0.1)
        fig, ax = viz.plot(open_grid, successful_result, robot_radius=0.3)
        try:
            assert fig is not None
        finally:
            plt.close(fig)


# ─── Tests: Input Validation ─────────────────────────────────────────────────


class TestInputValidation:
    """Tests verifying input parameter validation and error handling."""

    def test_invalid_grid_type(self, successful_result):
        """Rejects non-OccupancyGrid grid parameter."""
        with pytest.raises(TypeError, match="OccupancyGrid"):
            plot_simulation_result(grid="not_a_grid", result=successful_result)

    def test_invalid_result_type(self, open_grid):
        """Rejects non-SimulationResult result parameter."""
        with pytest.raises(TypeError, match="SimulationResult"):
            plot_simulation_result(grid=open_grid, result="not_a_result")

    def test_invalid_robot_radius_type(self, open_grid, successful_result):
        """Rejects boolean or non-numeric robot_radius."""
        with pytest.raises(TypeError, match="robot_radius"):
            plot_simulation_result(open_grid, successful_result, robot_radius=True)
        with pytest.raises(TypeError, match="robot_radius"):
            plot_simulation_result(open_grid, successful_result, robot_radius="large")

    def test_invalid_robot_radius_value(self, open_grid, successful_result):
        """Rejects zero, negative, or NaN robot_radius."""
        with pytest.raises(ValueError, match="strictly positive"):
            plot_simulation_result(open_grid, successful_result, robot_radius=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            plot_simulation_result(open_grid, successful_result, robot_radius=-0.5)
        with pytest.raises(ValueError, match="strictly positive"):
            plot_simulation_result(open_grid, successful_result, robot_radius=float("nan"))

    def test_invalid_ax_type(self, open_grid, successful_result):
        """Rejects invalid ax parameter."""
        with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
            plot_simulation_result(open_grid, successful_result, ax="not_axes")

    def test_class_invalid_radius(self):
        """NavigationVisualizer constructor validates robot_radius."""
        with pytest.raises(TypeError):
            NavigationVisualizer(robot_radius="bad")
        with pytest.raises(ValueError):
            NavigationVisualizer(robot_radius=-1.0)
