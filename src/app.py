"""Streamlit Web Application for the AI-Based Autonomous Navigation System.

This module provides an interactive, professional web interface integrating
directly with the existing simulation engine, motion controllers, collision
checker, visualization pipeline, and benchmark suite.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.benchmark.runner import BenchmarkRunner, ScenarioResult
from src.benchmark.suite import (
    get_default_scenarios,
    get_scenario_by_name,
    scenario_maze,
    scenario_multiple_obstacles,
    scenario_narrow_corridor,
    scenario_one_obstacle,
    scenario_open_space,
)
from src.benchmark.tuning import run_pure_pursuit_tuning
from src.collision_checker import CollisionChecker
from src.environment import OccupancyGrid
from src.metrics import NavigationMetrics, compute_metrics
from src.simulator import NavigationSimulator, SimulationResult, SimulationStatus
from src.visualization import plot_occupancy_grid_preview, plot_simulation_result


# ---------------------------------------------------------------------------
# Presets and Factory Helpers
# ---------------------------------------------------------------------------

def get_preset_map_catalog() -> Dict[str, Dict[str, Any]]:
    """Return catalog of preset simulation environments."""
    return {
        "Slalom Obstacles (15x20)": {
            "factory": lambda: scenario_multiple_obstacles().grid,
            "default_start": (2.5, 2.5),
            "default_goal": (17.5, 12.5),
            "description": "Slalom course with alternating obstacle barriers requiring multi-phase avoidance.",
        },
        "Open Space (10x10)": {
            "factory": lambda: scenario_open_space().grid,
            "default_start": (1.5, 1.5),
            "default_goal": (8.5, 8.5),
            "description": "Unobstructed open grid ideal for straight-line kinematic and speed profiling.",
        },
        "One Obstacle (10x10)": {
            "factory": lambda: scenario_one_obstacle().grid,
            "default_start": (2.0, 5.0),
            "default_goal": (8.0, 5.0),
            "description": "Central barrier requiring planar bypass around corner vertices.",
        },
        "Narrow Corridor (11x11)": {
            "factory": lambda: scenario_narrow_corridor().grid,
            "default_start": (1.5, 5.5),
            "default_goal": (9.5, 5.5),
            "description": "Single 1-meter wide channel testing lateral clearance and corridor tracking.",
        },
        "Maze Labyrinth (15x15)": {
            "factory": lambda: scenario_maze().grid,
            "default_start": (1.5, 1.5),
            "default_goal": (13.5, 13.5),
            "description": "Serpentine labyrinth with multiple sharp 90-degree orthogonal corners.",
        },
    }


def validate_pose(grid: OccupancyGrid, x: float, y: float, robot_radius: float) -> Tuple[bool, str]:
    """Validate whether a candidate coordinate is within bounds and collision-free.

    Parameters
    ----------
    grid : OccupancyGrid
        The environment grid.
    x : float
        World x-coordinate.
    y : float
        World y-coordinate.
    robot_radius : float
        Robot circular footprint radius.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, descriptive_message)
    """
    x_min, y_min = grid.origin
    x_max = x_min + grid.width * grid.resolution
    y_max = y_min + grid.height * grid.resolution

    if (x - robot_radius < x_min - 1e-9) or (x + robot_radius > x_max + 1e-9) or \
       (y - robot_radius < y_min - 1e-9) or (y + robot_radius > y_max + 1e-9):
        return False, f"Out of bounds: footprint extends beyond [{x_min:.1f}, {x_max:.1f}] x [{y_min:.1f}, {y_max:.1f}]."

    checker = CollisionChecker(grid, robot_radius=robot_radius)
    if not checker.is_pose_collision_free(x, y):
        return False, "Collision: candidate pose intersects an obstacle cell boundary."
    return True, "Valid: footprint is within bounds and collision-free."


def run_interactive_simulation(
    grid: OccupancyGrid,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    controller_type: str,
    lookahead_distance: float,
    angular_gain: float,
    linear_speed: float,
    angular_speed_limit: float,
    robot_radius: float,
    goal_tolerance: float,
    dt: float,
    max_steps: int,
    initial_heading: float,
) -> Tuple[Optional[SimulationResult], Optional[NavigationMetrics], Optional[str]]:
    """Execute the simulation engine and compute metrics.

    Returns
    -------
    Tuple[Optional[SimulationResult], Optional[NavigationMetrics], Optional[str]]
        (simulation_result, metrics, error_message)
    """
    try:
        sim = NavigationSimulator(
            grid=grid,
            robot_radius=robot_radius,
            dt=dt,
            max_steps=max_steps,
            goal_tolerance=goal_tolerance,
            linear_speed=linear_speed,
            angular_gain=angular_gain,
            angular_speed_limit=angular_speed_limit,
            controller=controller_type,
            lookahead_distance=lookahead_distance,
        )
        result = sim.run(start=start, goal=goal, initial_heading=initial_heading)
        metrics = compute_metrics(result)
        return result, metrics, None
    except Exception as exc:
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# Streamlit UI Components & Page Renderers
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render the sidebar branding and navigation selector."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="margin: 0; color: #0284c7;">🧭 Autonomous Nav</h2>
            <p style="margin: 4px 0; color: #64748b; font-size: 0.85rem;">
                Deterministic Grid Navigation Studio
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation Mode",
        [
            "🚀 Simulation Studio",
            "📊 Benchmark Suite",
            "📘 Architecture & Specs",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4;">
            <strong>System Engine:</strong><br>
            • A* Path Planner (4-connected)<br>
            • Unicycle Kinematics ($v, \\omega$)<br>
            • Heading P & Pure Pursuit<br>
            • Continuous Collision Checking<br>
            • Deterministic & GPU-free
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def render_simulation_studio() -> None:
    """Render the interactive Simulation Studio page."""
    st.title("🚀 Simulation Studio")
    st.markdown(
        "Configure environments, set initial/target coordinates, select motion controllers, "
        "and inspect real-time closed-loop navigation trajectories."
    )

    catalog = get_preset_map_catalog()

    # Layout: Setup controls in top columns
    col_map, col_ctrl = st.columns([1, 1], gap="medium")

    with col_map:
        st.subheader("1. Environment & Coordinates")
        map_name = st.selectbox("Preset Environment", list(catalog.keys()), index=0)
        selected_meta = catalog[map_name]

        # Manage active grid and preset state across runs and interactive edits
        if "active_map_preset" not in st.session_state or st.session_state["active_map_preset"] != map_name:
            st.session_state["active_map_preset"] = map_name
            st.session_state["active_grid"] = selected_meta["factory"]()
            st.session_state["start_x"] = float(selected_meta["default_start"][0])
            st.session_state["start_y"] = float(selected_meta["default_start"][1])
            st.session_state["goal_x"] = float(selected_meta["default_goal"][0])
            st.session_state["goal_y"] = float(selected_meta["default_goal"][1])
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_metrics", None)

        grid: OccupancyGrid = st.session_state["active_grid"]
        st.caption(selected_meta["description"])

        c_start_x, c_start_y = st.columns(2)
        with c_start_x:
            start_x = st.number_input(
                "Start X (m)",
                step=0.5,
                format="%.2f",
                key="start_x",
            )
        with c_start_y:
            start_y = st.number_input(
                "Start Y (m)",
                step=0.5,
                format="%.2f",
                key="start_y",
            )

        c_goal_x, c_goal_y = st.columns(2)
        with c_goal_x:
            goal_x = st.number_input(
                "Goal X (m)",
                step=0.5,
                format="%.2f",
                key="goal_x",
            )
        with c_goal_y:
            goal_y = st.number_input(
                "Goal Y (m)",
                step=0.5,
                format="%.2f",
                key="goal_y",
            )

        heading_deg = st.slider(
            "Initial Heading θ (degrees)",
            min_value=-180,
            max_value=180,
            value=0,
            step=15,
            help="Orientation of the robot at simulation start.",
        )
        initial_heading = math.radians(heading_deg)

    with col_ctrl:
        st.subheader("2. Controller Configuration")
        controller_choice = st.selectbox(
            "Path Tracking Controller",
            ["pure_pursuit", "heading"],
            format_func=lambda x: "Pure Pursuit Controller (Arc Tracking)" if x == "pure_pursuit" else "Heading Proportional Controller (Waypoint Pivot)",
            index=0,
        )

        lookahead_distance = 0.5
        angular_gain = 2.0

        if controller_choice == "pure_pursuit":
            lookahead_distance = st.slider(
                "Lookahead Distance L_d (meters)",
                min_value=0.10,
                max_value=1.50,
                value=0.50,
                step=0.05,
                help="Radius of target lookahead circle along planned path. (Default: 0.50 m).",
            )
            st.caption(
                "⚠️ *Note: L_d < 0.35m increases steering oscillations; L_d > 0.85m cuts corners in tight corridors.*"
            )
        else:
            angular_gain = st.slider(
                "Heading Proportional Gain K_p",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Proportional steering velocity scaling: omega = K_p * error_theta.",
            )

        c_spd, c_rot = st.columns(2)
        with c_spd:
            linear_speed = st.number_input(
                "Max Speed v_max (m/s)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
            )
        with c_rot:
            angular_speed_limit = st.number_input(
                "Max Yaw Rate ω_max (rad/s)",
                min_value=0.2,
                max_value=3.0,
                value=1.0,
                step=0.1,
            )

        with st.expander("⚙️ Advanced Simulation Parameters", expanded=False):
            robot_radius = st.number_input("Robot Radius R (m)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
            goal_tolerance = st.number_input("Goal Tolerance (m)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
            sim_dt = st.number_input("Time Step dt (s)", min_value=0.02, max_value=0.5, value=0.1, step=0.02)
            max_steps = st.number_input("Max Steps", min_value=100, max_value=5000, value=2000, step=100)

    # -----------------------------------------------------------------------
    # Interactive Map & Obstacle Cell Editor
    # -----------------------------------------------------------------------
    with st.expander("✏️ Interactive Map & Obstacle Cell Editor", expanded=False):
        st.markdown(
            "Modify the environment occupancy grid directly. Grid indices use discrete `(row, col)` with "
            "row 0 at bottom ($y_{\\min}$) and column 0 at left ($x_{\\min}$). "
            "Toggling obstacle cells immediately updates the active environment, map preview, and collision validation."
        )

        col_e1, col_e2, col_e3 = st.columns([1, 1, 1.2])
        with col_e1:
            edit_row = st.number_input(
                f"Grid Row (0 .. {grid.height - 1})",
                min_value=0,
                max_value=grid.height - 1,
                value=min(int(st.session_state.get("last_edit_row", 0)), grid.height - 1),
                step=1,
                key="edit_cell_row",
            )
        with col_e2:
            edit_col = st.number_input(
                f"Grid Col (0 .. {grid.width - 1})",
                min_value=0,
                max_value=grid.width - 1,
                value=min(int(st.session_state.get("last_edit_col", 0)), grid.width - 1),
                step=1,
                key="edit_cell_col",
            )
        with col_e3:
            r_idx, c_idx = int(edit_row), int(edit_col)
            cell_is_occ = grid.is_occupied(r_idx, c_idx)
            cell_cx, cell_cy = grid.grid_to_world(r_idx, c_idx, center=True)
            status_badge = "🛑 **Occupied (Obstacle)**" if cell_is_occ else "⬜ **Free Space**"
            st.markdown(
                f"**Cell Center:** `({cell_cx:.2f}, {cell_cy:.2f}) m`<br>"
                f"**Current State:** {status_badge}",
                unsafe_allow_html=True,
            )

        # Callbacks executing before widget instantiation to satisfy Streamlit session state lifecycle
        def toggle_cell_cb() -> None:
            curr = grid.is_occupied(r_idx, c_idx)
            grid.set_cell(r_idx, c_idx, OccupancyGrid.FREE if curr else OccupancyGrid.OCCUPIED)
            st.session_state["active_grid"] = grid
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_metrics", None)

        def set_start_cb() -> None:
            st.session_state["start_x"] = float(cell_cx)
            st.session_state["start_y"] = float(cell_cy)
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_metrics", None)

        def set_goal_cb() -> None:
            st.session_state["goal_x"] = float(cell_cx)
            st.session_state["goal_y"] = float(cell_cy)
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_metrics", None)

        def clear_obstacles_cb() -> None:
            empty = OccupancyGrid.from_dimensions(
                height=grid.height,
                width=grid.width,
                initial_value=OccupancyGrid.FREE,
                resolution=grid.resolution,
                origin=grid.origin,
            )
            st.session_state["active_grid"] = empty
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_metrics", None)

        def reset_map_cb() -> None:
            st.session_state["active_grid"] = selected_meta["factory"]()
            st.session_state["start_x"] = float(selected_meta["default_start"][0])
            st.session_state["start_y"] = float(selected_meta["default_start"][1])
            st.session_state["goal_x"] = float(selected_meta["default_goal"][0])
            st.session_state["goal_y"] = float(selected_meta["default_goal"][1])
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_metrics", None)

        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
        with bcol1:
            st.button("Toggle Obstacle", key="btn_toggle_cell", on_click=toggle_cell_cb)
        with bcol2:
            st.button("Set Start to Cell", key="btn_set_start_cell", on_click=set_start_cb)
        with bcol3:
            st.button("Set Goal to Cell", key="btn_set_goal_cell", on_click=set_goal_cb)
        with bcol4:
            st.button("Clear Obstacles", key="btn_clear_obstacles", on_click=clear_obstacles_cb)
        with bcol5:
            st.button("↺ Reset Preset", key="btn_reset_map", on_click=reset_map_cb)

    # Pre-flight Validation Indicators
    st.markdown("---")
    val_s, msg_s = validate_pose(grid, start_x, start_y, robot_radius)
    val_g, msg_g = validate_pose(grid, goal_x, goal_y, robot_radius)

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        if val_s:
            st.success(f"Start Pose: {msg_s}")
        else:
            st.error(f"Start Pose: {msg_s}")
    with vcol2:
        if val_g:
            st.success(f"Goal Pose: {msg_g}")
        else:
            st.error(f"Goal Pose: {msg_g}")

    can_run = val_s and val_g

    # Proximity notice
    if can_run:
        dist_sg = math.hypot(start_x - goal_x, start_y - goal_y)
        if dist_sg <= goal_tolerance:
            st.info(f"ℹ️ Start pose is within goal tolerance of goal ({dist_sg:.3f} m <= {goal_tolerance:.3f} m). Navigation will terminate immediately at step 0.")

    # -----------------------------------------------------------------------
    # Map Preview BEFORE Simulation Execution
    # -----------------------------------------------------------------------
    st.markdown("### 🗺️ Environment Map Preview")
    st.caption(
        f"Pre-simulation preview of **{map_name}** ({grid.width}×{grid.height} cells, resolution={grid.resolution:.2f} m/cell). "
        f"World extents: $[{grid.origin[0]:.1f}, {grid.origin[0] + grid.width * grid.resolution:.1f}] \\times "
        f"[{grid.origin[1]:.1f}, {grid.origin[1] + grid.height * grid.resolution:.1f}]$ m. "
        f"Start pose shown with robot footprint (r={robot_radius:.2f}m) and orientation arrow."
    )
    fig_prev, _ = plot_occupancy_grid_preview(
        grid=grid,
        start=(start_x, start_y),
        goal=(goal_x, goal_y),
        robot_radius=robot_radius,
        initial_heading=initial_heading,
        title=f"Environment Map Preview — {map_name}",
        figsize=(8.5, 6.0),
    )
    st.pyplot(fig_prev, use_container_width=True)
    plt.close(fig_prev)

    # Run Simulation Trigger
    run_btn = st.button("🚀 Run Simulation", type="primary", disabled=not can_run, use_container_width=True)

    if run_btn:
        with st.spinner("Executing deterministic simulation..."):
            result, metrics, err = run_interactive_simulation(
                grid=grid,
                start=(start_x, start_y),
                goal=(goal_x, goal_y),
                controller_type=controller_choice,
                lookahead_distance=lookahead_distance,
                angular_gain=angular_gain,
                linear_speed=linear_speed,
                angular_speed_limit=angular_speed_limit,
                robot_radius=robot_radius,
                goal_tolerance=goal_tolerance,
                dt=sim_dt,
                max_steps=int(max_steps),
                initial_heading=initial_heading,
            )

        if err:
            st.error(f"Simulation encountered an execution error: {err}")
            return

        if result is None or metrics is None:
            st.error("No simulation result returned.")
            return

        # Store in session state for inspection
        st.session_state["last_result"] = result
        st.session_state["last_metrics"] = metrics
        st.session_state["last_grid"] = grid
        st.session_state["last_radius"] = robot_radius

    # Display Results if available
    if "last_result" in st.session_state:
        result: SimulationResult = st.session_state["last_result"]
        metrics: NavigationMetrics = st.session_state["last_metrics"]
        grid: OccupancyGrid = st.session_state["last_grid"]
        robot_radius: float = st.session_state["last_radius"]

        st.markdown("### 📊 Simulation Execution Results")

        # Status Banner
        final_err = metrics.final_goal_error if metrics.final_goal_error is not None else result.final_goal_error
        err_str = f"{final_err:.3f}" if final_err is not None else "N/A"

        if result.status == SimulationStatus.SUCCESS:
            st.success(f"✅ Status: **SUCCESS** — Target goal reached within tolerance ({err_str} m).")
        elif result.status == SimulationStatus.COLLISION:
            st.error(f"💥 Status: **COLLISION** — Footprint collided with an obstacle boundary at step {result.steps}.")
        elif result.status == SimulationStatus.TIMEOUT:
            st.warning(f"⏱️ Status: **TIMEOUT** — Maximum simulation steps ({result.steps}) reached before reaching goal.")
        elif result.status == SimulationStatus.NO_PATH:
            st.error("🚫 Status: **NO_PATH** — A* search concluded no collision-free path exists to goal.")

        # Metric Badges
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Terminal Status", result.status.value)
        m2.metric("Simulation Steps", f"{result.steps} ({result.steps * sim_dt:.1f} s)")
        m3.metric("Traveled Distance", f"{metrics.traveled_distance:.2f} m")
        m4.metric("Planned Path", f"{metrics.planned_path_length:.2f} m" if metrics.planned_path_length else "N/A")
        m5.metric("Final Goal Error", f"{err_str} m")

        # Tabs for Visualization and Data
        tab_viz, tab_preview, tab_metrics, tab_data = st.tabs(
            ["🗺️ Post-Simulation Trajectory", "🔍 Base Map Preview", "📈 Detailed Metrics", "📋 Trajectory Pose Data"]
        )

        with tab_viz:
            fig, _ = plot_simulation_result(
                grid=grid,
                result=result,
                robot_radius=robot_radius,
                title=f"{result.controller_name} Navigation — Status: {result.status.value}",
                figsize=(9.0, 7.5),
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab_preview:
            fig_base, _ = plot_occupancy_grid_preview(
                grid=grid,
                start=result.start_world,
                goal=result.goal_world,
                robot_radius=robot_radius,
                initial_heading=initial_heading,
                title=f"Base Environment Map — {map_name}",
                figsize=(9.0, 7.5),
            )
            st.pyplot(fig_base, use_container_width=True)
            plt.close(fig_base)

        with tab_metrics:
            mean_v = float(np.mean([s.v for s in result.trajectory])) if result.trajectory else 0.0
            max_v = float(max([s.v for s in result.trajectory], default=0.0))
            mean_w = float(np.mean([abs(s.omega) for s in result.trajectory])) if result.trajectory else 0.0
            max_w = float(max([abs(s.omega) for s in result.trajectory], default=0.0))

            m_data = {
                "Metric Name": [
                    "Navigation Success",
                    "Collision Occurred",
                    "Timeout Occurred",
                    "Discrete Time Steps",
                    "Traveled Distance (m)",
                    "Planned Path Length (m)",
                    "Actual-to-Planned Ratio",
                    "Path Efficiency",
                    "Final Goal Error (m)",
                    "Mean Linear Velocity (m/s)",
                    "Max Linear Velocity (m/s)",
                    "Mean Angular Velocity (rad/s)",
                    "Max Angular Velocity (rad/s)",
                ],
                "Measured Value": [
                    str(metrics.success),
                    str(metrics.is_collision),
                    str(metrics.is_timeout),
                    str(metrics.steps),
                    f"{metrics.traveled_distance:.3f}",
                    f"{metrics.planned_path_length:.3f}" if metrics.planned_path_length is not None else "N/A",
                    f"{metrics.actual_to_planned_ratio:.3f}" if metrics.actual_to_planned_ratio is not None else "N/A",
                    f"{metrics.path_efficiency:.3f}" if metrics.path_efficiency is not None else "N/A",
                    f"{err_str}",
                    f"{mean_v:.3f}",
                    f"{max_v:.3f}",
                    f"{mean_w:.3f}",
                    f"{max_w:.3f}",
                ],
            }
            st.table(pd.DataFrame(m_data))

        with tab_data:
            if result.trajectory:
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
                df_traj = pd.DataFrame(records)
                st.dataframe(df_traj, use_container_width=True)
                csv = df_traj.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Trajectory CSV",
                    data=csv,
                    file_name=f"trajectory_{result.controller_name}_{result.status.value}.csv",
                    mime="text/csv",
                )
            else:
                st.info("ℹ️ No trajectory points recorded for this run (e.g. planner could not find an obstacle-free path).")


def render_benchmark_page() -> None:
    """Render the Benchmark Suite evaluation page."""
    st.title("📊 Standardized Benchmark Suite")
    st.markdown(
        "Evaluate motion controllers against the seven standardized benchmark scenarios "
        "established in Phase 3B and Phase 3D."
    )

    scenarios = get_default_scenarios()
    scenario_names = [s.name for s in scenarios]

    mode = st.radio(
        "Evaluation Mode",
        [
            "Single Scenario Run",
            "Full Benchmark Suite (All 7 Scenarios)",
            "Side-by-Side Controller Comparison",
            "Pure Pursuit Lookahead Tuning Experiment",
        ],
        horizontal=True,
    )

    st.markdown("---")

    if mode == "Single Scenario Run":
        c_scen, c_ctl, c_la = st.columns([1.5, 1, 1])
        with c_scen:
            selected_scen_name = st.selectbox("Select Scenario", scenario_names, index=0)
        with c_ctl:
            ctl_choice = st.selectbox("Controller", ["pure_pursuit", "heading"], index=0)
        with c_la:
            la_val = st.slider("Lookahead L_d (m)", min_value=0.25, max_value=1.0, value=0.5, step=0.25, disabled=(ctl_choice != "pure_pursuit"))

        scenario = get_scenario_by_name(selected_scen_name)
        exp_str = scenario.expected_status.value if scenario.expected_status else (scenario.expected_exception.__name__ if scenario.expected_exception else "N/A")
        st.info(f"**Scenario Description**: {scenario.description} (Expected: `{exp_str}`)")

        if st.button("Run Scenario", type="primary"):
            runner = BenchmarkRunner(controller=ctl_choice, lookahead_distance=la_val if ctl_choice == "pure_pursuit" else None)
            with st.spinner("Executing benchmark scenario..."):
                res: ScenarioResult = runner.run_scenario(scenario)

            if res.expectation_met:
                st.success(f"✅ Expectation Met: Observed `{res.actual_outcome}` (Expected `{res.expected_outcome}`)")
            else:
                st.error(f"❌ Expectation Diverged: Observed `{res.actual_outcome}` (Expected `{res.expected_outcome}`)")

            # Render scenario plot if simulation result exists
            if scenario.expected_status is not None:
                sim = scenario.create_simulator(
                    controller=ctl_choice,
                    lookahead_distance=la_val if ctl_choice == "pure_pursuit" else None,
                )
                try:
                    sim_result = sim.run(start=scenario.start, goal=scenario.goal)
                    fig, _ = plot_simulation_result(
                        grid=scenario.grid,
                        result=sim_result,
                        robot_radius=scenario.robot_radius,
                        title=f"Benchmark: {scenario.name} ({ctl_choice}) — Status: {sim_result.status.value}",
                        figsize=(8.0, 6.5),
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception:
                    pass

    elif mode == "Full Benchmark Suite (All 7 Scenarios)":
        c_ctl, c_la = st.columns([1, 1])
        with c_ctl:
            ctl_choice = st.selectbox("Controller", ["pure_pursuit", "heading"], index=0, key="suite_ctl")
        with c_la:
            la_val = st.slider("Lookahead L_d (m)", min_value=0.25, max_value=1.0, value=0.5, step=0.25, disabled=(ctl_choice != "pure_pursuit"), key="suite_la")

        if st.button("Execute Full Benchmark Suite", type="primary"):
            runner = BenchmarkRunner(controller=ctl_choice, lookahead_distance=la_val if ctl_choice == "pure_pursuit" else None)
            with st.spinner("Running all 7 benchmark scenarios..."):
                report = runner.run_all()

            st.markdown("### 🏆 Suite Summary")
            col_count = sum(1 for r in report.results if r.simulation_status == "COLLISION")
            to_count = sum(1 for r in report.results if r.simulation_status == "TIMEOUT")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Benchmark Pass Rate", f"{report.passed_scenarios}/{report.total_scenarios} ({report.benchmark_pass_rate * 100:.1f}%)")
            c2.metric("Navigable Success Rate", f"{report.navigation_success_rate * 100:.1f}%")
            c3.metric("Collisions", col_count)
            c4.metric("Timeouts", to_count)

            # Scenario Breakdown Table
            rows = []
            for r in report.results:
                m = r.metrics or {}
                rows.append({
                    "Scenario": r.scenario_name,
                    "Expected": r.expected_outcome,
                    "Actual": r.actual_outcome,
                    "Match": "✅ Met" if r.expectation_met else "❌ Diverged",
                    "Nav Success": "True" if r.navigation_success else "False",
                    "Steps": m.get("steps", "-"),
                    "Traveled (m)": round(m.get("traveled_distance", 0), 2) if m.get("traveled_distance") else "-",
                    "Goal Error (m)": round(m.get("final_goal_error", 0), 3) if m.get("final_goal_error") else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif mode == "Side-by-Side Controller Comparison":
        st.markdown("Compares the **Heading Proportional** controller against the **Pure Pursuit** controller ($L_d = 0.50$ m) on all scenarios.")
        if st.button("Run Side-by-Side Comparison", type="primary"):
            with st.spinner("Running comparative evaluations..."):
                runner_h = BenchmarkRunner(controller="heading")
                report_h = runner_h.run_all()

                runner_pp = BenchmarkRunner(controller="pure_pursuit", lookahead_distance=0.5)
                report_pp = runner_pp.run_all()

            st.markdown("### 📈 Comparative Overview")
            steps_h = [r.metrics.get("steps", 0) for r in report_h.results if r.navigation_success and r.metrics]
            steps_pp = [r.metrics.get("steps", 0) for r in report_pp.results if r.navigation_success and r.metrics]
            dist_h = [r.metrics.get("traveled_distance", 0.0) for r in report_h.results if r.navigation_success and r.metrics]
            dist_pp = [r.metrics.get("traveled_distance", 0.0) for r in report_pp.results if r.navigation_success and r.metrics]

            mean_steps_h = float(np.mean(steps_h)) if steps_h else 0.0
            mean_steps_pp = float(np.mean(steps_pp)) if steps_pp else 0.0
            mean_dist_h = float(np.mean(dist_h)) if dist_h else 0.0
            mean_dist_pp = float(np.mean(dist_pp)) if dist_pp else 0.0

            step_delta_pct = ((mean_steps_pp - mean_steps_h) / mean_steps_h * 100.0) if mean_steps_h > 0 else 0.0
            dist_delta_pct = ((mean_dist_pp - mean_dist_h) / mean_dist_h * 100.0) if mean_dist_h > 0 else 0.0

            col_count_h = sum(1 for r in report_h.results if r.simulation_status == "COLLISION")
            to_count_h = sum(1 for r in report_h.results if r.simulation_status == "TIMEOUT")
            col_count_pp = sum(1 for r in report_pp.results if r.simulation_status == "COLLISION")
            to_count_pp = sum(1 for r in report_pp.results if r.simulation_status == "TIMEOUT")

            comp_metrics = {
                "Metric": [
                    "Benchmark Pass Rate",
                    "Navigation Success Rate",
                    "Collisions",
                    "Timeouts",
                    "Total Steps (Navigable)",
                    "Mean Steps per Success",
                    "Mean Traveled Distance (m)",
                ],
                "Heading Proportional": [
                    f"{report_h.passed_scenarios}/{report_h.total_scenarios} ({report_h.benchmark_pass_rate*100:.1f}%)",
                    f"{report_h.navigation_success_rate*100:.1f}%",
                    str(col_count_h),
                    str(to_count_h),
                    str(sum(steps_h)),
                    f"{mean_steps_h:.1f}",
                    f"{mean_dist_h:.2f} m",
                ],
                "Pure Pursuit (L_d = 0.5 m)": [
                    f"{report_pp.passed_scenarios}/{report_pp.total_scenarios} ({report_pp.benchmark_pass_rate*100:.1f}%)",
                    f"{report_pp.navigation_success_rate*100:.1f}%",
                    str(col_count_pp),
                    str(to_count_pp),
                    str(sum(steps_pp)),
                    f"{mean_steps_pp:.1f} ({step_delta_pct:+.1f}%)",
                    f"{mean_dist_pp:.2f} m ({dist_delta_pct:+.1f}%)",
                ],
            }
            st.table(pd.DataFrame(comp_metrics))

            # Scenario Breakdown Table for side-by-side
            comp_rows = []
            for rh, rpp in zip(report_h.results, report_pp.results):
                mh = rh.metrics or {}
                mpp = rpp.metrics or {}
                comp_rows.append({
                    "Scenario": rh.scenario_name,
                    "Heading Outcome": rh.actual_outcome,
                    "Pure Pursuit Outcome": rpp.actual_outcome,
                    "Heading Steps": mh.get("steps", "-"),
                    "Pure Pursuit Steps": mpp.get("steps", "-"),
                    "Heading Dist (m)": round(mh.get("traveled_distance", 0), 2) if mh.get("traveled_distance") else "-",
                    "Pure Pursuit Dist (m)": round(mpp.get("traveled_distance", 0), 2) if mpp.get("traveled_distance") else "-",
                })
            st.markdown("#### Scenario Breakdown")
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    elif mode == "Pure Pursuit Lookahead Tuning Experiment":
        st.markdown("Evaluates lookahead distances $L_d \\in \\{0.25, 0.50, 0.75, 1.00\\}$ m on all seven scenarios against the Heading baseline.")
        if st.button("Run Lookahead Tuning Experiment", type="primary"):
            with st.spinner("Running full lookahead grid experiment (28 simulation runs)..."):
                report = run_pure_pursuit_tuning()

            st.markdown("### 🎯 Empirical Tuning Results")
            tuning_rows = []
            base_agg = report["baseline_experiment"]["aggregate_metrics"]
            tuning_rows.append({
                "Configuration": "Heading Controller (Base)",
                "Nav Success": f"{base_agg['navigation_successes']}/{base_agg['navigable_scenarios_count']} ({base_agg['navigation_success_rate']*100:.0f}%)",
                "Collisions": base_agg["collisions"],
                "Timeouts": base_agg["timeouts"],
                "Mean Steps": round(base_agg["mean_steps_successful"], 1),
                "Mean Dist (m)": round(base_agg["mean_traveled_distance_successful"], 2),
                "Mean Error (m)": round(base_agg["mean_final_goal_error_successful"], 3),
            })

            for exp in report["tuning_experiments"]:
                agg = exp["aggregate_metrics"]
                tuning_rows.append({
                    "Configuration": exp["configuration_label"],
                    "Nav Success": f"{agg['navigation_successes']}/{agg['navigable_scenarios_count']} ({agg['navigation_success_rate']*100:.0f}%)",
                    "Collisions": agg["collisions"],
                    "Timeouts": agg["timeouts"],
                    "Mean Steps": round(agg["mean_steps_successful"], 1) if agg["mean_steps_successful"] else "-",
                    "Mean Dist (m)": round(agg["mean_traveled_distance_successful"], 2) if agg["mean_traveled_distance_successful"] else "-",
                    "Mean Error (m)": round(agg["mean_final_goal_error_successful"], 3) if agg["mean_final_goal_error_successful"] else "-",
                })
            st.table(pd.DataFrame(tuning_rows))

            # Detailed scenario-by-scenario matrix
            st.markdown("#### Scenario-by-Scenario Matrix")
            scen_names = [s.name for s in get_default_scenarios()]
            base_runs = {r["scenario_name"]: r for r in report["baseline_experiment"]["scenario_runs"]}
            exp_runs_by_ld = {
                exp["lookahead_distance"]: {r["scenario_name"]: r for r in exp["scenario_runs"]}
                for exp in report["tuning_experiments"]
            }
            matrix_rows = []
            for sname in scen_names:
                r_base = base_runs.get(sname, {})
                base_str = f"{r_base.get('terminal_status', 'N/A')} ({r_base.get('steps', 0)}s)" if r_base.get('navigation_success') else r_base.get('terminal_status', 'N/A')
                row_dict = {
                    "Scenario": sname,
                    "Heading (Base)": base_str,
                }
                for ld in [0.25, 0.5, 0.75, 1.0]:
                    r_exp = exp_runs_by_ld.get(ld, {}).get(sname, {})
                    exp_str = f"{r_exp.get('terminal_status', 'N/A')} ({r_exp.get('steps', 0)}s)" if r_exp.get('navigation_success') else r_exp.get('terminal_status', 'N/A')
                    row_dict[f"PP (L_d={ld:.2f}m)"] = exp_str
                matrix_rows.append(row_dict)
            st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)
            st.caption("Machine-readable JSON saved to `outputs/metrics/pure_pursuit_tuning.json`.")


def render_architecture_page() -> None:
    """Render the System Architecture and Mathematical Specifications page."""
    st.title("📘 System Architecture & Specifications")
    st.markdown(
        "Mathematical formulations, algorithms, and scope disclosures implemented in the "
        "AI-Based Autonomous Navigation System."
    )

    t1, t2, t3, t4 = st.tabs(["🧩 Core Architecture", "📐 Motion Control", "🔍 Collision Checking", "⚠️ Scope & Limitations"])

    with t1:
        st.markdown(
            """
            ### 2D Occupancy Grid & A* Path Planning
            - **Map Representation**: 2D binary matrix $M \\in \\{0, 1\\}^{H \\times W}$ with spatial resolution $\\delta$ (m/cell).
            - **Search Space**: 4-connected discrete lattice avoiding obstacle cells.
            - **Admissible Heuristic**: Manhattan distance $h(n) = |r_n - r_g| + |c_n - c_g|$, guaranteeing shortest-path optimality under uniform movement cost.
            - **Kinematics**: Differential-drive / Unicycle model integrated via discrete forward Euler:
              $$x_{k+1} = x_k + v_k \\cos(\\theta_k) \\cdot \\Delta t$$
              $$y_{k+1} = y_k + v_k \\sin(\\theta_k) \\cdot \\Delta t$$
              $$\\theta_{k+1} = \\text{normalize}(\\theta_k + \\omega_k \\cdot \\Delta t)$$
            """
        )

    with t2:
        st.markdown(
            """
            ### Motion Controllers
            #### 1. Heading Proportional Controller
            - Targets waypoints sequentially.
            - Proportional yaw rate: $\\omega = \\text{clamp}(K_p \\cdot e_\\theta, -\\omega_{\\max}, \\omega_{\\max})$.
            - Translates forward when heading error is within threshold: $v = v_{\\max} \\cos(e_\\theta)$.

            #### 2. Pure Pursuit Path Tracking Controller
            - Targets a forward intersection point along path segments at lookahead distance $L_d$.
            - Relative bearing angle: $\\alpha = \\text{normalize}(\\text{atan2}(\\Delta y, \\Delta x) - \\theta)$.
            - Circular arc curvature: $\\kappa = \\frac{2 \\sin\\alpha}{L_d}$.
            - Yaw rate command: $\\omega = \\text{clamp}(v \\cdot \\kappa, -\\omega_{\\max}, \\omega_{\\max})$.
            - Smooth cornering enables **35.7% fewer simulation steps** compared to in-place pivot turns.
            """
        )

    with t3:
        st.markdown(
            """
            ### Continuous Substepping Collision Checker
            - **Footprint**: Circular disk of radius $R = 0.20$ m centered at $(x, y)$.
            - **Obstacle Cells**: Axis-aligned bounding boxes (AABBs).
            - **Distance Calculation**: Clamping circle center to rectangle bounds:
              $$x_{\\text{clamp}} = \\max(x_{\\min}, \\min(x_{\\max}, x)), \\quad y_{\\text{clamp}} = \\max(y_{\\min}, \\min(y_{\\max}, y))$$
              $$d = \\sqrt{(x - x_{\\text{clamp}})^2 + (y - y_{\\text{clamp}})^2}$$
            - **Substepping Guarantee**: Intermediate samples are evaluated along motion vectors $\\Delta \\mathbf{p}$ with step size $\\le \\delta_{\\text{sub}}$, ensuring the robot never tunnels through thin barriers.
            """
        )

    with t4:
        st.markdown(
            """
            ### Scope Boundaries & Disclaimers
            - **No Hardware / Physical Claim**: This simulation models idealized planar kinematics and geometry. It does not model motor slip, wheel inertia, or dynamics.
            - **No Machine Learning**: Path planning uses deterministic A* and motion tracking uses classical geometric control.
            - **Static Environment**: Obstacles are stationary; dynamic obstacle prediction is not included in this phase.
            - **Zero Heavy Dependencies**: Completely self-contained in standard Python with NumPy and Matplotlib. No ROS, Gazebo, or GPU required.
            """
        )


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit application root function."""
    st.set_page_config(
        page_title="AI Autonomous Navigation Studio",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page = render_sidebar()

    if page == "🚀 Simulation Studio":
        render_simulation_studio()
    elif page == "📊 Benchmark Suite":
        render_benchmark_page()
    elif page == "📘 Architecture & Specs":
        render_architecture_page()


if __name__ == "__main__":
    main()
