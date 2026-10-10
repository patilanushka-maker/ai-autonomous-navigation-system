r"""2D Visualization module for the Autonomous Navigation System.

This module provides 2D Matplotlib-based rendering for simulation results
and occupancy grid environments.

Coordinates and Geometric Alignment:
-------------------------------------
- Discrete grid: shape `(height, width)` indexed by `(row, col)`.
- Continuous world coordinates: `(x, y)`.
- Columns correspond to world $x$ ($x$ increases along columns).
- Rows correspond to world $y$ ($y$ increases along rows).
- Origin `(origin_x, origin_y)` marks the lower-left boundary of cell `(0, 0)`.
- Total extents:
    $x \in [\text{origin}_x, \text{origin}_x + \text{width} \cdot \text{resolution}]$
    $y \in [\text{origin}_y, \text{origin}_y + \text{height} \cdot \text{resolution}]$
- Equal aspect ratio is enforced to preserve Euclidean geometry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from src.environment import OccupancyGrid
from src.simulator import SimulationResult, SimulationStatus

# Visual styling tokens
STATUS_COLORS: Dict[SimulationStatus, str] = {
    SimulationStatus.SUCCESS: "#16a34a",     # Forest green
    SimulationStatus.COLLISION: "#dc2626",   # Bright red
    SimulationStatus.TIMEOUT: "#d97706",     # Amber / orange
    SimulationStatus.NO_PATH: "#7c3aed",     # Violet
    SimulationStatus.RUNNING: "#2563eb",     # Blue
}

# Free cells: light slate / off-white; Occupied cells: dark slate gray
GRID_CMAP = ListedColormap(["#f8fafc", "#334155"])


def plot_simulation_result(
    grid: OccupancyGrid,
    result: SimulationResult,
    robot_radius: float = 0.2,
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    dpi: int = 150,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot an occupancy grid and complete simulation result in 2D.

    Visualizes:
    1. Free space and occupied obstacle cells on the `OccupancyGrid`.
    2. Start and goal locations in continuous world coordinates.
    3. Planned A* path waypoints (if available).
    4. Continuous robot trajectory trail (if available).
    5. Final robot position and circular footprint (with heading orientation).
    6. Terminal simulation status, step count, and path length.

    Parameters
    ----------
    grid : OccupancyGrid
        The 2D occupancy grid representation of the environment.
    result : SimulationResult
        The result object returned by `NavigationSimulator.run()`.
    robot_radius : float, optional
        Radius of the circular robot footprint in meters. Defaults to 0.2.
    ax : Optional[matplotlib.axes.Axes], optional
        Existing Matplotlib Axes to draw on. If None, a new Figure and Axes
        are created. Defaults to None.
    save_path : Optional[Union[str, Path]], optional
        File path to save the generated figure. If None, the figure is not
        saved to disk. Defaults to None.
    show : bool, optional
        Whether to invoke `plt.show()` after rendering. Defaults to False.
    title : Optional[str], optional
        Custom title string. If None, a descriptive title containing the status,
        steps, and traversed length is generated. Defaults to None.
    figsize : Tuple[float, float], optional
        Dimensions `(width, height)` of the created figure in inches.
        Defaults to (8.0, 8.0).
    dpi : int, optional
        Resolution in dots per inch for the saved figure. Defaults to 150.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The Matplotlib Figure and Axes containing the rendered plot.

    Raises
    ------
    TypeError
        If `grid` is not an OccupancyGrid, `result` is not a SimulationResult,
        or numeric parameters have invalid types.
    ValueError
        If `robot_radius` is non-positive or non-finite.
    """
    if not isinstance(grid, OccupancyGrid):
        raise TypeError(f"Expected OccupancyGrid instance, got {type(grid).__name__}.")
    if not isinstance(result, SimulationResult):
        raise TypeError(f"Expected SimulationResult instance, got {type(result).__name__}.")

    if isinstance(robot_radius, bool) or not isinstance(robot_radius, (int, float)):
        raise TypeError(f"robot_radius must be a numeric float, got {type(robot_radius).__name__}.")
    if math.isnan(robot_radius) or math.isinf(robot_radius) or robot_radius <= 0.0:
        raise ValueError(f"robot_radius must be strictly positive and finite, got {robot_radius}.")

    if ax is not None and not isinstance(ax, matplotlib.axes.Axes):
        raise TypeError(f"Expected matplotlib.axes.Axes instance for ax, got {type(ax).__name__}.")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 1. Coordinate and boundary extents
    res = grid.resolution
    ox, oy = grid.origin
    x_min = ox
    x_max = ox + grid.width * res
    y_min = oy
    y_max = oy + grid.height * res

    # 2. Render occupancy grid background
    # origin="lower" ensures row 0 is at bottom (y_min) and row H-1 is at top (y_max)
    grid_matrix = grid.grid
    ax.imshow(
        grid_matrix,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=GRID_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )

    # Add subtle grid lines along cell boundaries
    if grid.width <= 100 and grid.height <= 100:
        ax.set_xticks(np.arange(x_min, x_max + res, res), minor=True)
        ax.set_yticks(np.arange(y_min, y_max + res, res), minor=True)
        ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=0.5, alpha=0.5)

    # 3. Render planned A* path if available
    if result.planned_path_world is not None and len(result.planned_path_world) > 0:
        px = [pt[0] for pt in result.planned_path_world]
        py = [pt[1] for pt in result.planned_path_world]
        ax.plot(
            px,
            py,
            color="#0284c7",
            linestyle="--",
            linewidth=2.0,
            marker=".",
            markersize=5,
            label="Planned Path (A*)",
            zorder=2,
        )

    # 4. Render actual robot trajectory if available
    if result.trajectory is not None and len(result.trajectory) > 0:
        tx = [s.x for s in result.trajectory]
        ty = [s.y for s in result.trajectory]
        ax.plot(
            tx,
            ty,
            color="#ea580c",
            linestyle="-",
            linewidth=2.2,
            label="Robot Trajectory",
            zorder=3,
        )

    # 5. Render start and goal markers
    sx, sy = result.start_world
    gx, gy = result.goal_world
    ax.scatter(
        [sx],
        [sy],
        color="#16a34a",
        s=120,
        marker="o",
        edgecolors="#14532d",
        linewidths=1.5,
        label="Start Position",
        zorder=5,
    )
    ax.scatter(
        [gx],
        [gy],
        color="#dc2626",
        s=160,
        marker="*",
        edgecolors="#7f1d1d",
        linewidths=1.5,
        label="Goal Position",
        zorder=5,
    )

    # 6. Render final robot pose and circular footprint if trajectory exists
    if result.trajectory is not None and len(result.trajectory) > 0:
        final_state = result.trajectory[-1]
        fx, fy, ftheta = final_state.x, final_state.y, final_state.theta

        is_collided = result.status == SimulationStatus.COLLISION
        fp_color = "#dc2626" if is_collided else "#f97316"
        edge_color = "#991b1b" if is_collided else "#c2410c"

        footprint = patches.Circle(
            (fx, fy),
            radius=robot_radius,
            facecolor=fp_color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=0.35,
            label=f"Robot Footprint (r={robot_radius:.2f}m)",
            zorder=4,
        )
        ax.add_patch(footprint)

        # Robot center point
        ax.scatter([fx], [fy], color=fp_color, s=35, marker="o", zorder=5)

        # Heading orientation indicator
        arrow_len = max(robot_radius * 1.2, 0.15)
        dx = arrow_len * math.cos(ftheta)
        dy = arrow_len * math.sin(ftheta)
        ax.annotate(
            "",
            xy=(fx + dx, fy + dy),
            xytext=(fx, fy),
            arrowprops=dict(
                arrowstyle="->",
                color="#7c2d12",
                lw=2.0,
            ),
            zorder=6,
        )

    # 7. Axes formatting, aspect ratio, labels, title
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)", fontsize=10)
    ax.set_ylabel("World Y (m)", fontsize=10)

    status_str = result.status.value
    status_color = STATUS_COLORS.get(result.status, "#334155")
    if title is None:
        ctrl_name = getattr(result, "controller_name", "HeadingProportionalController")
        title_str = (
            f"Simulation: {status_str} [{ctrl_name}] | Steps: {result.steps} | "
            f"Length: {result.path_length:.2f}m"
        )
    else:
        title_str = title
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=10, color=status_color)

    # 8. Legend construction (with proxies for obstacles and free space)
    handles, labels = ax.get_legend_handles_labels()
    # Add proxy artists for occupancy grid states
    obstacle_patch = patches.Patch(facecolor="#334155", edgecolor="none", label="Obstacle")
    free_patch = patches.Patch(facecolor="#f8fafc", edgecolor="#cbd5e1", label="Free Space")

    # Insert map layer proxies at top of legend
    if np.any(grid_matrix == OccupancyGrid.OCCUPIED):
        handles.insert(0, obstacle_patch)
        labels.insert(0, "Obstacle")
    handles.insert(1, free_patch)
    labels.insert(1, "Free Space")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        framealpha=0.9,
    )

    # 9. Save figure if requested
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)

    # 10. Display figure if requested
    if show:
        plt.show()

    return fig, ax


def plot_occupancy_grid_preview(
    grid: OccupancyGrid,
    start: Optional[Tuple[float, float]] = None,
    goal: Optional[Tuple[float, float]] = None,
    robot_radius: float = 0.2,
    initial_heading: Optional[float] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    dpi: int = 150,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot an occupancy grid environment preview with start and goal poses.

    Visualizes:
    1. Free space and occupied obstacle cells on the `OccupancyGrid`.
    2. Start position marker with circular robot footprint and optional heading orientation.
    3. Goal position marker.
    4. World coordinate axes and cell gridlines matching OccupancyGrid geometry.
    5. Legend identifying obstacles, free space, start, goal, and robot footprint.

    Parameters
    ----------
    grid : OccupancyGrid
        The 2D occupancy grid representation of the environment.
    start : Optional[Tuple[float, float]], optional
        Continuous world coordinates `(start_x, start_y)` of the start location.
    goal : Optional[Tuple[float, float]], optional
        Continuous world coordinates `(goal_x, goal_y)` of the goal location.
    robot_radius : float, optional
        Radius of the circular robot footprint in meters. Defaults to 0.2.
    initial_heading : Optional[float], optional
        Robot heading angle theta in radians at the start location. Defaults to None.
    ax : Optional[matplotlib.axes.Axes], optional
        Existing Matplotlib Axes to draw on. If None, a new Figure and Axes
        are created. Defaults to None.
    save_path : Optional[Union[str, Path]], optional
        File path to save the generated figure. Defaults to None.
    show : bool, optional
        Whether to invoke `plt.show()` after rendering. Defaults to False.
    title : Optional[str], optional
        Custom title string. If None, a descriptive title is generated. Defaults to None.
    figsize : Tuple[float, float], optional
        Dimensions `(width, height)` of the created figure in inches. Defaults to (8.0, 8.0).
    dpi : int, optional
        Resolution in dots per inch for the saved figure. Defaults to 150.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The Matplotlib Figure and Axes containing the rendered preview plot.

    Raises
    ------
    TypeError
        If `grid` is not an OccupancyGrid, `robot_radius` is invalid, or coordinate types are wrong.
    ValueError
        If `robot_radius` is non-positive or non-finite, or coordinates are NaN/inf.
    """
    if not isinstance(grid, OccupancyGrid):
        raise TypeError(f"Expected OccupancyGrid instance, got {type(grid).__name__}.")

    if isinstance(robot_radius, bool) or not isinstance(robot_radius, (int, float)):
        raise TypeError(f"robot_radius must be a numeric float, got {type(robot_radius).__name__}.")
    if math.isnan(robot_radius) or math.isinf(robot_radius) or robot_radius <= 0.0:
        raise ValueError(f"robot_radius must be strictly positive and finite, got {robot_radius}.")

    if start is not None:
        if not isinstance(start, (tuple, list)) or len(start) != 2:
            raise TypeError("start must be a sequence of two coordinates: (start_x, start_y).")
        sx, sy = start
        if isinstance(sx, bool) or isinstance(sy, bool) or not (isinstance(sx, (int, float)) and isinstance(sy, (int, float))):
            raise TypeError("start coordinates must be numeric floats.")
        if math.isnan(sx) or math.isinf(sx) or math.isnan(sy) or math.isinf(sy):
            raise ValueError(f"start coordinates must be finite, got ({sx}, {sy}).")

    if goal is not None:
        if not isinstance(goal, (tuple, list)) or len(goal) != 2:
            raise TypeError("goal must be a sequence of two coordinates: (goal_x, goal_y).")
        gx, gy = goal
        if isinstance(gx, bool) or isinstance(gy, bool) or not (isinstance(gx, (int, float)) and isinstance(gy, (int, float))):
            raise TypeError("goal coordinates must be numeric floats.")
        if math.isnan(gx) or math.isinf(gx) or math.isnan(gy) or math.isinf(gy):
            raise ValueError(f"goal coordinates must be finite, got ({gx}, {gy}).")

    if initial_heading is not None:
        if isinstance(initial_heading, bool) or not isinstance(initial_heading, (int, float)):
            raise TypeError(f"initial_heading must be a numeric float, got {type(initial_heading).__name__}.")
        if math.isnan(initial_heading) or math.isinf(initial_heading):
            raise ValueError(f"initial_heading must be finite, got {initial_heading}.")

    if ax is not None and not isinstance(ax, matplotlib.axes.Axes):
        raise TypeError(f"Expected matplotlib.axes.Axes instance for ax, got {type(ax).__name__}.")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 1. Coordinate and boundary extents
    res = grid.resolution
    ox, oy = grid.origin
    x_min = ox
    x_max = ox + grid.width * res
    y_min = oy
    y_max = oy + grid.height * res

    # 2. Render occupancy grid background
    grid_matrix = grid.grid
    ax.imshow(
        grid_matrix,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=GRID_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )

    # 3. Add subtle grid lines along cell boundaries
    if grid.width <= 100 and grid.height <= 100:
        ax.set_xticks(np.arange(x_min, x_max + res, res), minor=True)
        ax.set_yticks(np.arange(y_min, y_max + res, res), minor=True)
        ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=0.5, alpha=0.5)

    # 4. Render start position and robot footprint
    if start is not None:
        sx_val, sy_val = float(start[0]), float(start[1])
        footprint = patches.Circle(
            (sx_val, sy_val),
            radius=robot_radius,
            facecolor="#16a34a",
            edgecolor="#15803d",
            linewidth=1.5,
            linestyle="--",
            alpha=0.25,
            label=f"Robot Footprint (r={robot_radius:.2f}m)",
            zorder=4,
        )
        ax.add_patch(footprint)
        ax.scatter(
            [sx_val],
            [sy_val],
            color="#16a34a",
            s=120,
            marker="o",
            edgecolors="#14532d",
            linewidths=1.5,
            label="Start Position",
            zorder=5,
        )

        # Orientation arrow if provided
        if initial_heading is not None:
            arrow_len = max(robot_radius * 1.2, 0.15)
            dx = arrow_len * math.cos(initial_heading)
            dy = arrow_len * math.sin(initial_heading)
            ax.annotate(
                "",
                xy=(sx_val + dx, sy_val + dy),
                xytext=(sx_val, sy_val),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#14532d",
                    lw=2.0,
                ),
                zorder=6,
            )

    # 5. Render goal position
    if goal is not None:
        gx_val, gy_val = float(goal[0]), float(goal[1])
        ax.scatter(
            [gx_val],
            [gy_val],
            color="#dc2626",
            s=160,
            marker="*",
            edgecolors="#7f1d1d",
            linewidths=1.5,
            label="Goal Position",
            zorder=5,
        )

    # 6. Axes formatting, aspect ratio, labels, title
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)", fontsize=10)
    ax.set_ylabel("World Y (m)", fontsize=10)

    title_str = title if title is not None else f"Map Preview — Grid: {grid.width}x{grid.height} (res={res:.2f}m)"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=10, color="#1e293b")

    # 7. Legend construction
    handles, labels = ax.get_legend_handles_labels()
    obstacle_patch = patches.Patch(facecolor="#334155", edgecolor="none", label="Obstacle")
    free_patch = patches.Patch(facecolor="#f8fafc", edgecolor="#cbd5e1", label="Free Space")

    if np.any(grid_matrix == OccupancyGrid.OCCUPIED):
        handles.insert(0, obstacle_patch)
        labels.insert(0, "Obstacle")
        handles.insert(1, free_patch)
        labels.insert(1, "Free Space")
    else:
        handles.insert(0, free_patch)
        labels.insert(0, "Free Space")

    ax.legend(handles=handles, labels=labels, loc="upper right", framealpha=0.92, fontsize=8.5)

    # 8. Save figure if requested
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)

    # 9. Display figure if requested
    if show:
        plt.show()

    return fig, ax


class NavigationVisualizer:
    """Object-oriented 2D renderer for autonomous navigation simulation results.

    Parameters
    ----------
    robot_radius : float, optional
        Default circular robot footprint radius in meters. Defaults to 0.2.
    figsize : Tuple[float, float], optional
        Default figure dimensions `(width, height)` in inches. Defaults to (8.0, 8.0).
    dpi : int, optional
        Default dots-per-inch resolution for exported figures. Defaults to 150.

    Raises
    ------
    TypeError
        If `robot_radius` is non-numeric or boolean.
    ValueError
        If `robot_radius` is non-positive or non-finite.
    """

    def __init__(
        self,
        robot_radius: float = 0.2,
        figsize: Tuple[float, float] = (8.0, 8.0),
        dpi: int = 150,
    ) -> None:
        if isinstance(robot_radius, bool) or not isinstance(robot_radius, (int, float)):
            raise TypeError(f"robot_radius must be a numeric float, got {type(robot_radius).__name__}.")
        if math.isnan(robot_radius) or math.isinf(robot_radius) or robot_radius <= 0.0:
            raise ValueError(f"robot_radius must be strictly positive and finite, got {robot_radius}.")

        self._robot_radius = float(robot_radius)
        self._figsize = figsize
        self._dpi = dpi

    @property
    def robot_radius(self) -> float:
        """Return the default robot footprint radius in meters."""
        return self._robot_radius

    @property
    def figsize(self) -> Tuple[float, float]:
        """Return the default figure dimensions."""
        return self._figsize

    @property
    def dpi(self) -> int:
        """Return the default figure DPI."""
        return self._dpi

    def plot(
        self,
        grid: OccupancyGrid,
        result: SimulationResult,
        robot_radius: Optional[float] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        title: Optional[str] = None,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Render the simulation result on the given occupancy grid.

        Parameters
        ----------
        grid : OccupancyGrid
            The 2D occupancy grid environment.
        result : SimulationResult
            The simulation run outcome record.
        robot_radius : Optional[float], optional
            Override for robot radius. If None, uses `self.robot_radius`.
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to draw on. If None, a new Figure and Axes are created.
        save_path : Optional[Union[str, Path]], optional
            File path to save the generated figure.
        show : bool, optional
            Whether to call `plt.show()`.
        title : Optional[str], optional
            Custom title string.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The Matplotlib Figure and Axes.
        """
        effective_radius = self._robot_radius if robot_radius is None else robot_radius
        return plot_simulation_result(
            grid=grid,
            result=result,
            robot_radius=effective_radius,
            ax=ax,
            save_path=save_path,
            show=show,
            title=title,
            figsize=self._figsize,
            dpi=self._dpi,
        )

    def plot_preview(
        self,
        grid: OccupancyGrid,
        start: Optional[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        robot_radius: Optional[float] = None,
        initial_heading: Optional[float] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        title: Optional[str] = None,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Render an occupancy grid preview with start and goal poses.

        Parameters
        ----------
        grid : OccupancyGrid
            The 2D occupancy grid environment.
        start : Optional[Tuple[float, float]], optional
            Continuous world coordinates `(start_x, start_y)` of the start location.
        goal : Optional[Tuple[float, float]], optional
            Continuous world coordinates `(goal_x, goal_y)` of the goal location.
        robot_radius : Optional[float], optional
            Override for robot radius. If None, uses `self.robot_radius`.
        initial_heading : Optional[float], optional
            Robot heading angle theta in radians at the start location.
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to draw on. If None, a new Figure and Axes are created.
        save_path : Optional[Union[str, Path]], optional
            File path to save the generated figure.
        show : bool, optional
            Whether to call `plt.show()`.
        title : Optional[str], optional
            Custom title string.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The Matplotlib Figure and Axes containing the rendered preview plot.
        """
        effective_radius = self._robot_radius if robot_radius is None else robot_radius
        return plot_occupancy_grid_preview(
            grid=grid,
            start=start,
            goal=goal,
            robot_radius=effective_radius,
            initial_heading=initial_heading,
            ax=ax,
            save_path=save_path,
            show=show,
            title=title,
            figsize=self._figsize,
            dpi=self._dpi,
        )

