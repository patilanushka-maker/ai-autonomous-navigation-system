r"""Navigation performance metrics and evaluation module (Phase 2H).

This module provides quantitative performance metrics for evaluating
individual simulation runs and multi-run navigation benchmarks.

Available Metrics:
------------------
1. **status**: Terminal navigation status (`SUCCESS`, `COLLISION`, `TIMEOUT`, `NO_PATH`).
2. **success**: Boolean flag indicating if the goal was successfully reached.
3. **traveled_distance**: Actual Euclidean distance traversed by the robot, computed as
   the sum of Euclidean distances between consecutive recorded trajectory positions:
       $$D_{\text{act}} = \sum_{i=1}^{N-1} \sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2}$$
4. **planned_path_length**: Total length of the planned A* route, computed as the
   sum of Euclidean distances between consecutive world-coordinate waypoints:
       $$L_{\text{plan}} = \sum_{j=1}^{M-1} \sqrt{(w_{x,j} - w_{x,j-1})^2 + (w_{y,j} - w_{y,j-1})^2}$$
   Evaluates to `None` if no path was planned (e.g., in `NO_PATH` or start-at-goal).
5. **actual_to_planned_ratio**: Ratio of actual traveled distance to planned path length:
       $$R_{\text{act/plan}} = \frac{D_{\text{act}}}{L_{\text{plan}}}$$
   - Values $< 1.0$: Occurs when continuous trajectory smoothing rounds 90-degree discrete
     Manhattan grid turns within waypoint tolerance, and when the robot halts upon entering
     `goal_tolerance` before the exact cell center.
   - Values $> 1.0$: Indicates path expansion due to tracking errors, oscillation, or detours.
   - Evaluates to `None` for failed runs, absent paths, or zero-distance routes.
   - *Important*: A ratio $< 1.0$ does NOT indicate superior navigation or prove obstacle
     clearance/safety; cutting corners on discrete grids reduces clearance to obstacles.
6. **final_goal_error**: Euclidean distance between the robot's final state and the goal:
       $$e_{\text{goal}} = \sqrt{(x_{\text{final}} - g_x)^2 + (y_{\text{final}} - g_y)^2}$$
   Evaluates to `None` if the trajectory is empty or goal is unavailable.
   For successful runs, $e_{\text{goal}} \le \text{goal\_tolerance}$.
7. **steps**: Total number of executed discrete simulation time steps.
8. **planning_duration**: Duration of the path planning phase in seconds.
   *Note*: The current `NavigationSimulator` interface does not record planning execution
   timestamps; this field defaults to `None` to prevent fabricating untracked timing data.
9. **path_efficiency**: Ratio of planned route length to actual traveled distance:
       $$\eta = \frac{L_{\text{plan}}}{D_{\text{act}}} = \frac{1}{R_{\text{act/plan}}}$$
   Retained for backward compatibility. Evaluates to `None` for failed runs or zero distance.
10. **is_collision / is_timeout / is_no_path**: Boolean outcome indicator flags.
11. **straight_line_distance**: Direct Euclidean distance from start to goal coordinates:
       $$d_{\text{direct}} = \sqrt{(g_x - s_x)^2 + (g_y - s_y)^2}$$
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.simulator import SimulationResult, SimulationStatus

__all__ = [
    "SimulationStatus",
    "NavigationMetrics",
    "EvaluationSummary",
    "compute_traveled_distance",
    "compute_planned_path_length",
    "compute_actual_to_planned_ratio",
    "compute_final_goal_error",
    "compute_path_efficiency",
    "compute_metrics",
    "evaluate_multiple_runs",
]


@dataclass(frozen=True)
class NavigationMetrics:
    """Immutable evaluation metrics for a single navigation simulation run.

    Attributes
    ----------
    status : SimulationStatus
        Terminal simulation status code.
    success : bool
        True if the robot successfully reached the goal within tolerance.
    traveled_distance : float
        Actual Euclidean distance traveled by the robot in meters.
    planned_path_length : Optional[float]
        Total length of the planned A* path in meters, or None if no path was planned.
    steps : int
        Number of discrete simulation steps executed.
    planning_duration : Optional[float]
        Wall-clock or CPU planning duration in seconds. None if not recorded.
    path_efficiency : Optional[float]
        Ratio of planned path length to actual traveled distance (`planned / traveled`).
        Retained for backward compatibility. None if undefined or run was unsuccessful.
    actual_to_planned_ratio : Optional[float]
        Ratio of actual traveled distance to planned path length (`traveled / planned`).
        None for unsuccessful runs, absent paths, or zero planned length.
    final_goal_error : Optional[float]
        Euclidean distance from final robot position to goal position in meters.
        None if trajectory is empty.
    is_collision : bool
        True if the simulation terminated due to collision.
    is_timeout : bool
        True if the simulation terminated due to exceeding max_steps.
    is_no_path : bool
        True if the planner failed to find an obstacle-free path.
    straight_line_distance : float
        Euclidean distance from start to goal position in meters.
    """

    status: SimulationStatus
    success: bool
    traveled_distance: float
    planned_path_length: Optional[float]
    steps: int
    planning_duration: Optional[float] = None
    path_efficiency: Optional[float] = None
    actual_to_planned_ratio: Optional[float] = None
    final_goal_error: Optional[float] = None
    is_collision: bool = False
    is_timeout: bool = False
    is_no_path: bool = False
    straight_line_distance: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard dictionary with string enum values."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated evaluation summary for a collection of simulation runs.

    Attributes
    ----------
    total_runs : int
        Total number of simulation runs evaluated.
    success_count : int
        Number of runs that succeeded.
    success_rate : float
        Fraction of runs that succeeded: `success_count / total_runs` (0.0 if total_runs == 0).
    collision_count : int
        Number of runs terminating in collision.
    collision_rate : float
        Fraction of runs terminating in collision: `collision_count / total_runs`.
    timeout_count : int
        Number of runs terminating due to timeout.
    timeout_rate : float
        Fraction of runs terminating due to timeout: `timeout_count / total_runs`.
    no_path_count : int
        Number of runs where no path could be planned.
    no_path_rate : float
        Fraction of runs where no path was found: `no_path_count / total_runs`.
    mean_traveled_distance_successful : Optional[float]
        Average traveled distance for successful runs with valid trajectory data.
        None if no runs succeeded.
    mean_final_goal_error_successful : Optional[float]
        Average final goal distance error across successful runs.
        None if no runs succeeded.
    mean_actual_to_planned_ratio_successful : Optional[float]
        Average actual-to-planned distance ratio across successful runs.
        None if no runs succeeded or paths were absent.
    mean_steps_successful : Optional[float]
        Average simulation steps executed across successful runs.
        None if no runs succeeded.
    mean_path_efficiency_successful : Optional[float]
        Average path efficiency ratio across successful runs with defined efficiency.
        None if no valid efficiency values exist.
    """

    total_runs: int
    success_count: int
    success_rate: float
    collision_count: int
    collision_rate: float
    timeout_count: int
    timeout_rate: float
    no_path_count: int
    no_path_rate: float
    mean_traveled_distance_successful: Optional[float] = None
    mean_final_goal_error_successful: Optional[float] = None
    mean_actual_to_planned_ratio_successful: Optional[float] = None
    mean_steps_successful: Optional[float] = None
    mean_path_efficiency_successful: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert evaluation summary to a standard dictionary."""
        return asdict(self)

    def summary_table(self) -> str:
        """Format the evaluation summary as a human-readable text block."""
        dist_str = (
            f"{self.mean_traveled_distance_successful:.3f} m"
            if self.mean_traveled_distance_successful is not None
            else "N/A"
        )
        error_str = (
            f"{self.mean_final_goal_error_successful:.3f} m"
            if self.mean_final_goal_error_successful is not None
            else "N/A"
        )
        ratio_str = (
            f"{self.mean_actual_to_planned_ratio_successful:.3f}"
            if self.mean_actual_to_planned_ratio_successful is not None
            else "N/A"
        )
        steps_str = (
            f"{self.mean_steps_successful:.1f}"
            if self.mean_steps_successful is not None
            else "N/A"
        )
        eff_str = (
            f"{self.mean_path_efficiency_successful:.3f}"
            if self.mean_path_efficiency_successful is not None
            else "N/A"
        )

        lines = [
            "=" * 50,
            " NAVIGATION EVALUATION SUMMARY",
            "=" * 50,
            f"Total Runs Evaluated : {self.total_runs}",
            f"Success Rate         : {self.success_count}/{self.total_runs} ({self.success_rate * 100:.1f}%)",
            f"Collision Rate       : {self.collision_count}/{self.total_runs} ({self.collision_rate * 100:.1f}%)",
            f"Timeout Rate         : {self.timeout_count}/{self.total_runs} ({self.timeout_rate * 100:.1f}%)",
            f"No Path Rate         : {self.no_path_count}/{self.total_runs} ({self.no_path_rate * 100:.1f}%)",
            "-" * 50,
            f"Mean Traveled Dist   : {dist_str}",
            f"Mean Goal Error      : {error_str}",
            f"Actual/Planned Ratio : {ratio_str}",
            f"Mean Steps (Success) : {steps_str}",
            f"Mean Path Efficiency : {eff_str}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_traveled_distance(trajectory: Optional[Sequence[Any]]) -> float:
    """Calculate the total Euclidean distance traversed across consecutive trajectory states.

    Parameters
    ----------
    trajectory : Optional[Sequence[Any]]
        Sequence of robot states (exposing `.x` and `.y` or indexing `[0]`, `[1]`).

    Returns
    -------
    float
        Total distance in meters (0.0 if trajectory contains fewer than 2 states).
    """
    if not trajectory or len(trajectory) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        try:
            px = float(prev.x if hasattr(prev, "x") else prev[0])
            py = float(prev.y if hasattr(prev, "y") else prev[1])
            cx = float(curr.x if hasattr(curr, "x") else curr[0])
            cy = float(curr.y if hasattr(curr, "y") else curr[1])
            if math.isfinite(px) and math.isfinite(py) and math.isfinite(cx) and math.isfinite(cy):
                total += math.hypot(cx - px, cy - py)
        except (IndexError, TypeError, AttributeError, ValueError):
            continue
    return float(total)


def compute_planned_path_length(
    planned_path_world: Optional[Sequence[Tuple[float, float]]],
) -> Optional[float]:
    """Calculate the total Euclidean length across consecutive planned world waypoints.

    Parameters
    ----------
    planned_path_world : Optional[Sequence[Tuple[float, float]]]
        Sequence of (x, y) waypoints in world coordinates, or None.

    Returns
    -------
    Optional[float]
        Total planned path length in meters, or None if no path was planned.
    """
    if planned_path_world is None:
        return None
    if len(planned_path_world) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(planned_path_world)):
        try:
            p0 = planned_path_world[i - 1]
            p1 = planned_path_world[i]
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1):
                total += math.hypot(x1 - x0, y1 - y0)
        except (IndexError, TypeError, AttributeError, ValueError):
            continue
    return float(total)


def compute_actual_to_planned_ratio(
    traveled_distance: float,
    planned_length: Optional[float],
    success: bool,
) -> Optional[float]:
    """Calculate the actual-to-planned distance ratio: `traveled_distance / planned_length`.

    Parameters
    ----------
    traveled_distance : float
        Actual distance traversed by the robot in meters.
    planned_length : Optional[float]
        Total length of the planned path in meters.
    success : bool
        Whether the navigation goal was successfully achieved.

    Returns
    -------
    Optional[float]
        Ratio `traveled_distance / planned_length`, or None if undefined, unsuccessful,
        or planned_length is missing or non-positive.
    """
    if not success:
        return None
    if planned_length is None or not math.isfinite(planned_length) or planned_length <= 0.0:
        return None
    if not math.isfinite(traveled_distance) or traveled_distance < 0.0:
        return None
    return float(traveled_distance / planned_length)


def compute_final_goal_error(
    trajectory: Optional[Sequence[Any]],
    goal_world: Optional[Tuple[float, float]],
) -> Optional[float]:
    """Calculate the Euclidean distance between the robot's final pose and the goal position.

    Parameters
    ----------
    trajectory : Optional[Sequence[Any]]
        Sequence of recorded robot states.
    goal_world : Optional[Tuple[float, float]]
        Target world coordinates (goal_x, goal_y).

    Returns
    -------
    Optional[float]
        Euclidean distance in meters, or None if trajectory is empty or goal coordinates
        are missing or invalid.
    """
    if not trajectory or len(trajectory) == 0:
        return None
    if goal_world is None or len(goal_world) < 2:
        return None
    try:
        gx, gy = float(goal_world[0]), float(goal_world[1])
        if not (math.isfinite(gx) and math.isfinite(gy)):
            return None
        final_state = trajectory[-1]
        fx = float(final_state.x if hasattr(final_state, "x") else final_state[0])
        fy = float(final_state.y if hasattr(final_state, "y") else final_state[1])
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return None
        return float(math.hypot(fx - gx, fy - gy))
    except (IndexError, TypeError, AttributeError, ValueError):
        return None


def compute_path_efficiency(
    planned_length: Optional[float],
    traveled_distance: float,
    success: bool,
) -> Optional[float]:
    """Calculate the path efficiency ratio: `planned_path_length / traveled_distance`.

    Path efficiency evaluates how closely the executed robot trajectory followed
    the optimal planned route. It is only defensible when:
    1. The simulation succeeded (`success is True`).
    2. A planned path exists (`planned_length is not None` and `planned_length > 0`).
    3. The denominator is non-zero (`traveled_distance > 0`).

    Returns `None` for failed runs, start-at-goal (zero distance), or absent paths.

    Note: A ratio > 1.0 indicates that the executed path was shorter than the discrete
    4-connected grid path (due to corner smoothing and arrival within tolerance), NOT
    that navigation was "over 100% efficient". Retained for backward compatibility.
    See `actual_to_planned_ratio` for the standard distance ratio.

    Parameters
    ----------
    planned_length : Optional[float]
        Length of the planned path in meters.
    traveled_distance : float
        Actual distance traversed by the robot in meters.
    success : bool
        Whether the navigation goal was achieved.

    Returns
    -------
    Optional[float]
        Path efficiency ratio, or None if undefined / non-defensible.
    """
    if not success:
        return None
    if planned_length is None or not math.isfinite(planned_length) or planned_length <= 0.0:
        return None
    if not math.isfinite(traveled_distance) or traveled_distance <= 0.0:
        return None
    return float(planned_length / traveled_distance)


def compute_metrics(
    result: SimulationResult,
    planning_duration: Optional[float] = None,
) -> NavigationMetrics:
    """Extract and compute comprehensive performance metrics from a SimulationResult.

    Parameters
    ----------
    result : SimulationResult
        The result object produced by `NavigationSimulator.run()`.
    planning_duration : Optional[float], optional
        Externally recorded planning duration in seconds, if available.
        Defaults to None since `SimulationResult` does not track planning time.

    Returns
    -------
    NavigationMetrics
        Computed immutable performance metrics.

    Raises
    ------
    TypeError
        If `result` is not an instance of `SimulationResult`.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError(f"Expected SimulationResult instance, got {type(result).__name__}.")

    status = result.status
    success = status == SimulationStatus.SUCCESS
    is_collision = status == SimulationStatus.COLLISION
    is_timeout = status == SimulationStatus.TIMEOUT
    is_no_path = status == SimulationStatus.NO_PATH

    # 1. Traveled distance from trajectory
    traveled_dist = compute_traveled_distance(result.trajectory)

    # 2. Planned path length from planned_path_world
    planned_length = compute_planned_path_length(result.planned_path_world)

    # 3. Actual-to-planned distance ratio
    act_to_plan_ratio = compute_actual_to_planned_ratio(
        traveled_distance=traveled_dist,
        planned_length=planned_length,
        success=success,
    )

    # 4. Final goal distance error
    goal_error = compute_final_goal_error(
        trajectory=result.trajectory,
        goal_world=result.goal_world,
    )

    # 5. Path efficiency (planned / traveled, backward-compatible)
    efficiency = compute_path_efficiency(
        planned_length=planned_length,
        traveled_distance=traveled_dist,
        success=success,
    )

    # 6. Straight-line Euclidean distance
    try:
        sx, sy = float(result.start_world[0]), float(result.start_world[1])
        gx, gy = float(result.goal_world[0]), float(result.goal_world[1])
        straight_line_dist = float(math.hypot(gx - sx, gy - sy))
    except (IndexError, TypeError, AttributeError, ValueError):
        straight_line_dist = 0.0

    return NavigationMetrics(
        status=status,
        success=success,
        traveled_distance=traveled_dist,
        planned_path_length=planned_length,
        steps=int(result.steps),
        planning_duration=planning_duration,
        path_efficiency=efficiency,
        actual_to_planned_ratio=act_to_plan_ratio,
        final_goal_error=goal_error,
        is_collision=is_collision,
        is_timeout=is_timeout,
        is_no_path=is_no_path,
        straight_line_distance=straight_line_dist,
    )


def evaluate_multiple_runs(
    results: Sequence[Union[SimulationResult, NavigationMetrics]],
) -> EvaluationSummary:
    """Aggregate and summarize performance metrics across multiple simulation runs.

    Parameters
    ----------
    results : Sequence[Union[SimulationResult, NavigationMetrics]]
        A sequence of `SimulationResult` or `NavigationMetrics` objects.

    Returns
    -------
    EvaluationSummary
        Summary statistics including success rate, collision rate, timeout rate,
        mean traveled distance, mean goal error, and distance ratios.

    Raises
    ------
    TypeError
        If `results` is not an iterable sequence or contains items of unexpected types.
    """
    if not isinstance(results, (list, tuple)):
        raise TypeError(f"Expected a sequence (list or tuple), got {type(results).__name__}.")

    metrics_list: List[NavigationMetrics] = []
    for idx, item in enumerate(results):
        if isinstance(item, NavigationMetrics):
            metrics_list.append(item)
        elif isinstance(item, SimulationResult):
            metrics_list.append(compute_metrics(item))
        else:
            raise TypeError(
                f"Element at index {idx} must be a SimulationResult or NavigationMetrics, "
                f"got {type(item).__name__}."
            )

    total_runs = len(metrics_list)
    if total_runs == 0:
        return EvaluationSummary(
            total_runs=0,
            success_count=0,
            success_rate=0.0,
            collision_count=0,
            collision_rate=0.0,
            timeout_count=0,
            timeout_rate=0.0,
            no_path_count=0,
            no_path_rate=0.0,
            mean_traveled_distance_successful=None,
            mean_final_goal_error_successful=None,
            mean_actual_to_planned_ratio_successful=None,
            mean_steps_successful=None,
            mean_path_efficiency_successful=None,
        )

    success_count = sum(1 for m in metrics_list if m.success)
    collision_count = sum(1 for m in metrics_list if m.is_collision)
    timeout_count = sum(1 for m in metrics_list if m.is_timeout)
    no_path_count = sum(1 for m in metrics_list if m.is_no_path)

    success_rate = float(success_count / total_runs)
    collision_rate = float(collision_count / total_runs)
    timeout_rate = float(timeout_count / total_runs)
    no_path_rate = float(no_path_count / total_runs)

    # Statistics over successful runs only
    successful_runs = [m for m in metrics_list if m.success]
    if successful_runs:
        mean_traveled_dist = float(
            sum(m.traveled_distance for m in successful_runs) / len(successful_runs)
        )
        valid_errors = [m.final_goal_error for m in successful_runs if m.final_goal_error is not None]
        mean_goal_error = (
            float(sum(valid_errors) / len(valid_errors)) if valid_errors else None
        )
        valid_ratios = [
            m.actual_to_planned_ratio for m in successful_runs if m.actual_to_planned_ratio is not None
        ]
        mean_ratio = (
            float(sum(valid_ratios) / len(valid_ratios)) if valid_ratios else None
        )
        mean_steps = float(
            sum(m.steps for m in successful_runs) / len(successful_runs)
        )
        valid_efficiencies = [
            m.path_efficiency for m in successful_runs if m.path_efficiency is not None
        ]
        mean_eff = (
            float(sum(valid_efficiencies) / len(valid_efficiencies))
            if valid_efficiencies
            else None
        )
    else:
        mean_traveled_dist = None
        mean_goal_error = None
        mean_ratio = None
        mean_steps = None
        mean_eff = None

    return EvaluationSummary(
        total_runs=total_runs,
        success_count=success_count,
        success_rate=success_rate,
        collision_count=collision_count,
        collision_rate=collision_rate,
        timeout_count=timeout_count,
        timeout_rate=timeout_rate,
        no_path_count=no_path_count,
        no_path_rate=no_path_rate,
        mean_traveled_distance_successful=mean_traveled_dist,
        mean_final_goal_error_successful=mean_goal_error,
        mean_actual_to_planned_ratio_successful=mean_ratio,
        mean_steps_successful=mean_steps,
        mean_path_efficiency_successful=mean_eff,
    )
