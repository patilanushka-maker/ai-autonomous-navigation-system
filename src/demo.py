"""End-to-end demonstration module for the AI-Based Autonomous Navigation System.

This module provides a reproducible demonstration that:
1. Constructs a deterministic 2D occupancy grid environment with obstacles.
2. Initializes the `NavigationSimulator` with closed-loop A* planning, kinematic updates,
   and heading proportional control.
3. Executes navigation from start to goal.
4. Computes comprehensive navigation performance metrics via `src.metrics`.
5. Visualizes the resulting trajectory, planned path, obstacles, and poses via `src.visualization`.
6. Exports the visualization figure to `outputs/figures/` and metrics report to `outputs/metrics/`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure workspace root is in sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.environment import OccupancyGrid
from src.metrics import NavigationMetrics, compute_metrics
from src.simulator import NavigationSimulator, SimulationResult, SimulationStatus
from src.visualization import plot_simulation_result

# Default output file locations relative to workspace root
DEFAULT_OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_FIGURE_PATH = DEFAULT_OUTPUTS_DIR / "figures" / "navigation_demo.png"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUTS_DIR / "metrics" / "navigation_demo_metrics.json"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default_config.json"


def create_demo_grid() -> OccupancyGrid:
    """Construct a deterministic 15x20 occupancy grid with multiple obstacles.

    The environment features:
    - Dimensions: 15 rows (height) x 20 columns (width), resolution = 1.0 m, origin = (0.0, 0.0).
    - Start cell: (row 2, col 2) -> world (2.5, 2.5).
    - Goal cell: (row 12, col 17) -> world (17.5, 12.5).
    - Obstacle 1 (South Wall): cols 6..7, rows 0..9 (forces northern transit around barrier).
    - Obstacle 2 (North Wall): cols 12..13, rows 5..14 (forces southern transit around barrier).
    - Obstacle 3 (West Block): cols 2..3, rows 8..10 (interior block above start).
    - Obstacle 4 (East Block): cols 16..17, rows 2..4 (interior block south of goal corridor).

    This layout establishes a structured slalom corridor where a straight-line route
    is blocked, requiring multi-phase obstacle avoidance while guaranteeing at least
    one feasible, collision-free route.

    Returns
    -------
    OccupancyGrid
        Configured deterministic occupancy grid.
    """
    height, width = 15, 20
    grid_matrix = np.zeros((height, width), dtype=np.int8)

    # Obstacle 1: Barrier at cols 6-7, rows 0-9 (height 10, width 2)
    grid_matrix[0:10, 6:8] = OccupancyGrid.OCCUPIED

    # Obstacle 2: Barrier at cols 12-13, rows 5-14 (height 10, width 2)
    grid_matrix[5:15, 12:14] = OccupancyGrid.OCCUPIED

    # Obstacle 3: Interior block at cols 2-3, rows 8-10 (height 3, width 2)
    grid_matrix[8:11, 2:4] = OccupancyGrid.OCCUPIED

    # Obstacle 4: Interior block at cols 16-17, rows 2-4 (height 3, width 2)
    grid_matrix[2:5, 16:18] = OccupancyGrid.OCCUPIED

    return OccupancyGrid(grid_matrix, resolution=1.0, origin=(0.0, 0.0))


def load_simulation_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from JSON or return robust fallback defaults.

    Parameters
    ----------
    config_path : Optional[Path], optional
        Path to JSON configuration file. Defaults to configs/default_config.json.

    Returns
    -------
    Dict[str, Any]
        Configuration parameters dictionary.
    """
    target = config_path or DEFAULT_CONFIG_PATH
    defaults: Dict[str, Any] = {
        "robot_radius": 0.2,
        "linear_speed": 0.5,
        "angular_speed_limit": 1.0,
        "angular_gain": 2.0,
        "simulation_dt": 0.1,
        "goal_tolerance": 0.25,
        "max_simulation_steps": 2000,
    }

    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                defaults.update({
                    "robot_radius": float(loaded.get("robot_radius", defaults["robot_radius"])),
                    "linear_speed": float(loaded.get("linear_speed", defaults["linear_speed"])),
                    "angular_speed_limit": float(loaded.get("angular_speed_limit", defaults["angular_speed_limit"])),
                    "simulation_dt": float(loaded.get("simulation_dt", defaults["simulation_dt"])),
                    "goal_tolerance": float(loaded.get("goal_tolerance", defaults["goal_tolerance"])),
                    "max_simulation_steps": int(loaded.get("max_simulation_steps", defaults["max_simulation_steps"])),
                })
        except Exception as exc:
            print(f"[Warning] Failed to parse config file '{target}': {exc}. Using defaults.")

    return defaults


def run_navigation_demo(
    figure_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    show: bool = False,
    controller: str = "heading",
    lookahead_distance: Optional[float] = None,
) -> Tuple[SimulationResult, NavigationMetrics, Dict[str, Any]]:
    """Execute the end-to-end navigation demonstration pipeline.

    Steps:
    1. Construct deterministic occupancy grid.
    2. Initialize `NavigationSimulator`.
    3. Run simulation from start (2.5, 2.5) to goal (17.5, 12.5).
    4. Compute quantitative performance metrics.
    5. Render and export visualization figure.
    6. Export structured metrics JSON report.

    Parameters
    ----------
    figure_path : Optional[Path], optional
        File path where the rendered figure PNG will be saved.
        Defaults to `outputs/figures/navigation_demo.png`.
    metrics_path : Optional[Path], optional
        File path where the metrics JSON report will be saved.
        Defaults to `outputs/metrics/navigation_demo_metrics.json`.
    config_path : Optional[Path], optional
        Path to custom JSON configuration file.
    show : bool, optional
        Whether to display the Matplotlib interactive plot window. Defaults to False.
    controller : str, optional
        Path-tracking controller type ('heading' or 'pure_pursuit'). Defaults to 'heading'.
    lookahead_distance : Optional[float], optional
        Lookahead distance for pure pursuit in meters. Defaults to None (uses 0.5m).

    Returns
    -------
    Tuple[SimulationResult, NavigationMetrics, Dict[str, Any]]
        The simulation result, metrics dataclass, and serializable metrics report dict.
    """
    out_fig = figure_path or DEFAULT_FIGURE_PATH
    out_json = metrics_path or DEFAULT_METRICS_PATH

    # Ensure output parent directories exist
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Clean up stale artifacts if present to ensure independent execution
    if out_fig.exists():
        try:
            out_fig.unlink()
        except OSError:
            pass
    if out_json.exists():
        try:
            out_json.unlink()
        except OSError:
            pass

    # 1. Construct map and load parameters
    cfg = load_simulation_config(config_path)
    grid = create_demo_grid()

    robot_radius = float(cfg["robot_radius"])
    dt = float(cfg["simulation_dt"])
    max_steps = int(cfg["max_simulation_steps"])
    goal_tolerance = float(cfg["goal_tolerance"])
    linear_speed = float(cfg["linear_speed"])
    angular_gain = float(cfg.get("angular_gain", 2.0))
    angular_speed_limit = float(cfg["angular_speed_limit"])

    start_pos: Tuple[float, float] = (2.5, 2.5)
    goal_pos: Tuple[float, float] = (17.5, 12.5)
    initial_heading: float = 0.0

    print("=" * 65)
    print(" AI-BASED AUTONOMOUS NAVIGATION SYSTEM -- DEMONSTRATION")
    print("=" * 65)
    print(f"Environment Shape     : {grid.height} rows x {grid.width} cols (Resolution: {grid.resolution} m)")
    print(f"Start Position        : {start_pos} (world coordinates)")
    print(f"Goal Position         : {goal_pos} (world coordinates)")
    print(f"Robot Radius          : {robot_radius} m")
    print(f"Linear Speed Limit    : {linear_speed} m/s")
    print(f"Angular Speed Limit   : {angular_speed_limit} rad/s")
    print(f"Goal Tolerance        : {goal_tolerance} m")
    print(f"Controller            : {controller}")
    if controller == "pure_pursuit" and lookahead_distance is not None:
        print(f"Lookahead Distance    : {lookahead_distance} m")
    print("-" * 65)
    print("Running closed-loop simulation...")

    # 2. Construct simulator and execute run
    sim = NavigationSimulator(
        grid=grid,
        robot_radius=robot_radius,
        dt=dt,
        max_steps=max_steps,
        goal_tolerance=goal_tolerance,
        linear_speed=linear_speed,
        angular_gain=angular_gain,
        angular_speed_limit=angular_speed_limit,
        controller=controller,
        lookahead_distance=lookahead_distance,
    )

    result = sim.run(start=start_pos, goal=goal_pos, initial_heading=initial_heading)

    # 3. Calculate quantitative performance metrics
    metrics = compute_metrics(result)

    # 4. Assemble comprehensive metrics report dictionary
    # Unavailable values serialize as JSON null (None)
    metrics_report: Dict[str, Any] = {
        "scenario": {
            "name": "Deterministic Multi-Obstacle Slalom Navigation",
            "grid_dimensions": {"width": grid.width, "height": grid.height},
            "resolution": grid.resolution,
            "origin": list(grid.origin),
            "start": list(start_pos),
            "goal": list(goal_pos),
            "initial_heading": initial_heading,
            "robot_radius": robot_radius,
        },
        "status": result.status.value,
        "success": bool(metrics.success),
        "planned_path_length": metrics.planned_path_length,
        "traveled_distance": metrics.traveled_distance,
        "actual_to_planned_ratio": metrics.actual_to_planned_ratio,
        "final_goal_error": metrics.final_goal_error,
        "steps": int(metrics.steps),
        "is_collision": bool(metrics.is_collision),
        "is_timeout": bool(metrics.is_timeout),
        "is_no_path": bool(metrics.is_no_path),
        "straight_line_distance": metrics.straight_line_distance,
        "planning_duration": metrics.planning_duration,
        "path_efficiency": metrics.path_efficiency,
        "message": result.message,
        "artifacts": {
            "figure_file": str(out_fig.as_posix()),
            "metrics_file": str(out_json.as_posix()),
        },
    }

    # 5. Render and export visualization figure
    print(f"Generating visualization plot...")
    plot_simulation_result(
        grid=grid,
        result=result,
        robot_radius=robot_radius,
        save_path=out_fig,
        show=show,
        title=f"Demo: {result.status.value} | Steps: {result.steps} | Traveled: {result.path_length:.2f}m",
    )
    print(f"  -> Figure saved to: {out_fig}")

    # 6. Save metrics report as formatted JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"  -> Metrics JSON saved to: {out_json}")

    # 7. Print summary block
    print("-" * 65)
    print(f"Outcome               : {result.status.value} (Success: {metrics.success})")
    print(f"Discrete Time Steps   : {metrics.steps} ({metrics.steps * dt:.1f} s simulated time)")
    plan_str = f"{metrics.planned_path_length:.3f} m" if metrics.planned_path_length is not None else "N/A"
    print(f"Planned Path Length   : {plan_str}")
    print(f"Actual Distance       : {metrics.traveled_distance:.3f} m")
    ratio_str = f"{metrics.actual_to_planned_ratio:.3f}" if metrics.actual_to_planned_ratio is not None else "N/A"
    print(f"Actual/Planned Ratio  : {ratio_str}")
    eff_str = f"{metrics.path_efficiency:.3f}" if metrics.path_efficiency is not None else "N/A"
    print(f"Path Efficiency       : {eff_str}")
    err_str = f"{metrics.final_goal_error:.3f} m" if metrics.final_goal_error is not None else "N/A"
    print(f"Final Goal Error      : {err_str}")
    print("=" * 65)

    return result, metrics, metrics_report


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run reproducible end-to-end demonstration for AI-Based Autonomous Navigation System."
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help=f"File path to save the output figure (default: {DEFAULT_FIGURE_PATH}).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"File path to save the metrics JSON report (default: {DEFAULT_METRICS_PATH}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to optional simulation configuration JSON.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the interactive Matplotlib figure window.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="heading",
        choices=["heading", "pure_pursuit"],
        help="Path-tracking controller ('heading' or 'pure_pursuit', default: 'heading').",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=None,
        help="Lookahead distance in meters for pure pursuit controller (default: 0.5).",
    )

    args = parser.parse_args(argv)

    try:
        result, metrics, _ = run_navigation_demo(
            figure_path=args.figure_path,
            metrics_path=args.metrics_path,
            config_path=args.config,
            show=args.show,
            controller=args.controller,
            lookahead_distance=args.lookahead,
        )
        return 0 if result.status == SimulationStatus.SUCCESS else 1
    except Exception as exc:
        print(f"[Error] Demonstration execution failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
