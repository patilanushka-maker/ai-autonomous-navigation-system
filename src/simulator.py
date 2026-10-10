"""Navigation Simulator module for the 2D Autonomous Navigation System.

This module provides the `NavigationSimulator` class, which orchestrates the
discrete-time closed-loop simulation integrating:

- `OccupancyGrid`  — 2D environment with obstacles and world/grid transforms.
- `AStarPlanner`   — A* search producing grid-coordinate waypoints.
- `Robot`          — Unicycle kinematic model advancing pose each time step.
- `HeadingProportionalController` — Heading-based P-controller producing (v, ω).
- `CollisionChecker` — Circular footprint collision validation.

Simulation Loop (per time step):
---------------------------------
1. **Goal Check**: If the robot is within `goal_tolerance` of the final goal,
   terminate with `SUCCESS`.
2. **Waypoint Advancement**: If the robot is within `waypoint_tolerance` of
   the current waypoint, advance to the next waypoint in the planned path.
3. **Control Command**: The controller computes (v, ω) toward the current waypoint.
4. **Kinematic Update**: The robot integrates forward by `dt` using (v, ω).
5. **Collision Check**: Verify the new pose is collision-free.  If not,
   terminate with `COLLISION`.
6. **Step Limit Guard**: If `max_steps` is exceeded, terminate with `TIMEOUT`.
7. **Logging**: Record the robot state snapshot into the trajectory history.

Coordinate Convention Alignment:
---------------------------------
- A* plans in grid coordinates `(row, col)`.
- Robot operates in world coordinates `(x, y)`.
- `OccupancyGrid.grid_to_world(row, col)` converts grid waypoints to world targets.
- `OccupancyGrid.world_to_grid(x, y)` converts world poses to grid for planning.
- The controller and collision checker operate entirely in world coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.environment import OccupancyGrid
from src.path_planner import AStarPlanner
from src.robot import Robot, RobotState
from src.controller import HeadingProportionalController, PurePursuitController
from src.collision_checker import CollisionChecker


class SimulationStatus(Enum):
    """Terminal status codes for a completed simulation run."""

    SUCCESS = "SUCCESS"
    COLLISION = "COLLISION"
    TIMEOUT = "TIMEOUT"
    NO_PATH = "NO_PATH"
    RUNNING = "RUNNING"


@dataclass
class SimulationResult:
    """Immutable record of a completed simulation run.

    Attributes
    ----------
    status : SimulationStatus
        Terminal outcome of the simulation.
    steps : int
        Total number of simulation time steps executed.
    trajectory : List[RobotState]
        Ordered list of robot state snapshots at each time step.
    planned_path_grid : Optional[List[Tuple[int, int]]]
        A* planned path in grid coordinates, or None if no path was found.
    planned_path_world : Optional[List[Tuple[float, float]]]
        A* planned path converted to world coordinates (cell centers).
    start_world : Tuple[float, float]
        World coordinates of the start position.
    goal_world : Tuple[float, float]
        World coordinates of the goal position.
    path_length : float
        Total Euclidean path length traversed by the robot.
    message : str
        Human-readable description of the simulation outcome.
    controller_name : str
        Name of the controller class used in the simulation run.
    """

    status: SimulationStatus
    steps: int
    trajectory: List[RobotState]
    planned_path_grid: Optional[List[Tuple[int, int]]]
    planned_path_world: Optional[List[Tuple[float, float]]]
    start_world: Tuple[float, float]
    goal_world: Tuple[float, float]
    path_length: float
    message: str
    controller_name: str = "HeadingProportionalController"


class NavigationSimulator:
    """Closed-loop 2D navigation simulation engine.

    Parameters
    ----------
    grid : OccupancyGrid
        The 2D occupancy grid environment.
    robot_radius : float, optional
        Circular robot footprint radius in meters. Defaults to 0.2.
    dt : float, optional
        Simulation time step in seconds. Must be positive. Defaults to 0.1.
    max_steps : int, optional
        Maximum number of simulation steps before timeout. Defaults to 2000.
    goal_tolerance : float, optional
        Distance threshold to declare goal arrival (meters). Defaults to 0.25.
    waypoint_tolerance : float, optional
        Distance threshold to advance to next waypoint (meters).
        Defaults to `goal_tolerance` if not specified.
    linear_speed : float, optional
        Maximum forward linear speed in m/s. Defaults to 0.5.
    angular_gain : float, optional
        Proportional heading gain for the controller. Defaults to 2.0.
    angular_speed_limit : float, optional
        Maximum angular speed in rad/s. Defaults to 1.0.

    Raises
    ------
    TypeError
        If `grid` is not an `OccupancyGrid` instance or parameters have wrong types.
    ValueError
        If numeric parameters are non-finite, non-positive, or out of valid range.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        robot_radius: float = 0.2,
        dt: float = 0.1,
        max_steps: int = 2000,
        goal_tolerance: float = 0.25,
        waypoint_tolerance: float | None = None,
        linear_speed: float = 0.5,
        angular_gain: float = 2.0,
        angular_speed_limit: float = 1.0,
        max_substep_size: float | None = None,
        controller: Union[str, HeadingProportionalController, PurePursuitController, None] = None,
        lookahead_distance: float = 0.5,
    ) -> None:
        if not isinstance(grid, OccupancyGrid):
            raise TypeError(f"Expected OccupancyGrid instance, got {type(grid).__name__}.")

        self._validate_positive_float(dt, "dt")
        self._validate_positive_float(robot_radius, "robot_radius")
        self._validate_positive_float(goal_tolerance, "goal_tolerance")
        self._validate_positive_float(linear_speed, "linear_speed")
        self._validate_positive_float(angular_gain, "angular_gain")
        self._validate_positive_float(angular_speed_limit, "angular_speed_limit")

        if max_substep_size is not None:
            self._validate_positive_float(max_substep_size, "max_substep_size")
            self._max_substep_size: Optional[float] = float(max_substep_size)
        else:
            self._max_substep_size = None

        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer >= 1, got {max_steps}.")

        if waypoint_tolerance is not None:
            self._validate_positive_float(waypoint_tolerance, "waypoint_tolerance")
        else:
            waypoint_tolerance = goal_tolerance

        self._grid = grid
        self._robot_radius = float(robot_radius)
        self._dt = float(dt)
        self._max_steps = max_steps
        self._goal_tolerance = float(goal_tolerance)
        self._waypoint_tolerance = float(waypoint_tolerance)

        # Build sub-components
        self._planner = AStarPlanner(grid)
        self._collision_checker = CollisionChecker(grid, robot_radius=robot_radius)
        self._robot: Optional[Robot] = None

        # Build motion controller
        if controller is None or (
            isinstance(controller, str)
            and controller.lower() in ("heading", "heading_proportional", "headingproportional")
        ):
            self._controller: Union[HeadingProportionalController, PurePursuitController] = (
                HeadingProportionalController(
                    linear_speed=linear_speed,
                    angular_gain=angular_gain,
                    angular_speed_limit=angular_speed_limit,
                    goal_tolerance=goal_tolerance,
                )
            )
            self._controller_type = "heading"
            self._lookahead_distance: Optional[float] = None
        elif isinstance(controller, str) and controller.lower() in ("pure_pursuit", "purepursuit"):
            ld = 0.5 if lookahead_distance is None else lookahead_distance
            self._validate_positive_float(ld, "lookahead_distance")
            self._controller = PurePursuitController(
                lookahead_distance=ld,
                linear_speed=linear_speed,
                angular_speed_limit=angular_speed_limit,
                goal_tolerance=goal_tolerance,
                angular_gain=angular_gain,
            )
            self._controller_type = "pure_pursuit"
            self._lookahead_distance = float(ld)
        elif isinstance(controller, PurePursuitController):
            self._controller = controller
            self._controller_type = "pure_pursuit"
            self._lookahead_distance = controller.lookahead_distance
        elif isinstance(controller, HeadingProportionalController):
            self._controller = controller
            self._controller_type = "heading"
            self._lookahead_distance = None
        elif isinstance(controller, str):
            raise ValueError(
                f"Unknown controller type '{controller}'. Supported types: 'heading', 'pure_pursuit'."
            )
        else:
            raise TypeError(
                f"controller must be a string or controller instance, got {type(controller).__name__}."
            )

    @staticmethod
    def _validate_positive_float(val: Any, name: str) -> None:
        """Validate that a parameter is a finite, strictly positive number."""
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Parameter '{name}' must be numeric, got {type(val).__name__}.")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}.")
        if val <= 0.0:
            raise ValueError(f"Parameter '{name}' must be strictly positive (> 0.0), got {val}.")

    @property
    def grid(self) -> OccupancyGrid:
        """Return the underlying OccupancyGrid."""
        return self._grid

    @property
    def planner(self) -> AStarPlanner:
        """Return the A* planner instance."""
        return self._planner

    @property
    def controller(self) -> Union[HeadingProportionalController, PurePursuitController]:
        """Return the active motion controller instance."""
        return self._controller

    @property
    def controller_type(self) -> str:
        """Return the controller type identifier ('heading' or 'pure_pursuit')."""
        return self._controller_type

    @property
    def controller_name(self) -> str:
        """Return the class name of the active motion controller."""
        return type(self._controller).__name__

    @property
    def lookahead_distance(self) -> Optional[float]:
        """Return the configured lookahead distance in meters if using Pure Pursuit, else None."""
        return self._lookahead_distance

    @property
    def collision_checker(self) -> CollisionChecker:
        """Return the collision checker instance."""
        return self._collision_checker

    @property
    def robot_radius(self) -> float:
        """Return the robot footprint radius in meters."""
        return self._robot_radius

    @property
    def dt(self) -> float:
        """Return the simulation time step in seconds."""
        return self._dt

    @property
    def max_steps(self) -> int:
        """Return the maximum allowed simulation steps."""
        return self._max_steps

    @property
    def goal_tolerance(self) -> float:
        """Return the distance threshold to declare goal arrival in meters."""
        return self._goal_tolerance

    @property
    def waypoint_tolerance(self) -> float:
        """Return the distance threshold to advance to next waypoint in meters."""
        return self._waypoint_tolerance

    @property
    def max_substep_size(self) -> Optional[float]:
        """Return the maximum distance threshold for intermediate motion collision checks in meters."""
        return self._max_substep_size

    @property
    def robot(self) -> Optional[Robot]:
        """Return the current robot instance, or None if simulation has not been initialized."""
        return getattr(self, "_robot", None)

    def _world_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Compute Euclidean distance between two world points."""
        return math.hypot(x2 - x1, y2 - y1)

    def _convert_path_to_world(
        self, grid_path: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """Convert a list of grid-coordinate waypoints to world-coordinate cell centers."""
        world_path: List[Tuple[float, float]] = []
        for row, col in grid_path:
            wx, wy = self._grid.grid_to_world(row, col, center=True)
            world_path.append((wx, wy))
        return world_path

    def _compute_path_length(self, trajectory: List[RobotState]) -> float:
        """Compute the total Euclidean path length from a trajectory of robot states."""
        total = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            total += math.hypot(curr.x - prev.x, curr.y - prev.y)
        return total

    def run(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        initial_heading: float = 0.0,
    ) -> SimulationResult:
        """Execute a complete navigation simulation from start to goal.

        Parameters
        ----------
        start : Tuple[float, float]
            Starting world coordinates `(x, y)`.
        goal : Tuple[float, float]
            Goal world coordinates `(x, y)`.
        initial_heading : float, optional
            Initial robot heading in radians. Defaults to 0.0.

        Returns
        -------
        SimulationResult
            Complete record of the simulation outcome including trajectory,
            status, and computed metrics.

        Raises
        ------
        TypeError
            If start or goal coordinates are invalid.
        ValueError
            If start or goal positions are outside the grid or in collision.
        """
        # Validate start and goal coordinates
        sx, sy = self._validate_world_coords(start, "start")
        gx, gy = self._validate_world_coords(goal, "goal")

        # Check start position is collision-free
        if not self._collision_checker.is_pose_collision_free(sx, sy):
            raise ValueError(
                f"Start position ({sx}, {sy}) is in collision or outside grid bounds."
            )

        # Check goal position is collision-free
        if not self._collision_checker.is_pose_collision_free(gx, gy):
            raise ValueError(
                f"Goal position ({gx}, {gy}) is in collision or outside grid bounds."
            )

        # Handle start == goal edge case
        if self._world_distance(sx, sy, gx, gy) <= self._goal_tolerance:
            self._robot = Robot(x=sx, y=sy, theta=initial_heading)
            return SimulationResult(
                status=SimulationStatus.SUCCESS,
                steps=0,
                trajectory=[self._robot.state],
                planned_path_grid=None,
                planned_path_world=None,
                start_world=(sx, sy),
                goal_world=(gx, gy),
                path_length=0.0,
                message="Start position is already within goal tolerance.",
                controller_name=type(self._controller).__name__,
            )

        # Convert world start/goal to grid coordinates for A* planning
        start_grid = self._grid.world_to_grid(sx, sy)
        goal_grid = self._grid.world_to_grid(gx, gy)

        # Validate grid coordinates are in bounds and free
        if not self._grid.is_in_bounds(start_grid[0], start_grid[1]):
            raise ValueError(
                f"Start world position ({sx}, {sy}) maps to out-of-bounds grid cell {start_grid}."
            )
        if not self._grid.is_in_bounds(goal_grid[0], goal_grid[1]):
            raise ValueError(
                f"Goal world position ({gx}, {gy}) maps to out-of-bounds grid cell {goal_grid}."
            )

        # Run A* path planning
        grid_path = self._planner.plan(start_grid, goal_grid)

        if grid_path is None:
            return SimulationResult(
                status=SimulationStatus.NO_PATH,
                steps=0,
                trajectory=[],
                planned_path_grid=None,
                planned_path_world=None,
                start_world=(sx, sy),
                goal_world=(gx, gy),
                path_length=0.0,
                message="A* planner found no collision-free path from start to goal.",
                controller_name=type(self._controller).__name__,
            )

        # Convert grid waypoints to world coordinates
        world_path = self._convert_path_to_world(grid_path)

        # Initialize robot at start position
        self._robot = Robot(x=sx, y=sy, theta=initial_heading)
        robot = self._robot

        # Provide path to controller if supported (e.g. Pure Pursuit)
        if isinstance(self._controller, PurePursuitController) or hasattr(self._controller, "set_path"):
            path_for_controller = list(world_path)
            if path_for_controller:
                if path_for_controller[0] != (sx, sy):
                    path_for_controller.insert(0, (sx, sy))
                if path_for_controller[-1] != (gx, gy):
                    path_for_controller.append((gx, gy))
            self._controller.set_path(path_for_controller)

        # Trajectory log: record initial state
        trajectory: List[RobotState] = [robot.state]

        # Waypoint tracking: start from the second waypoint (first is start cell)
        waypoint_idx = 1 if len(world_path) > 1 else 0

        # Simulation loop
        for step in range(1, self._max_steps + 1):
            # 1. Check if robot has reached the final goal
            dist_to_goal = self._world_distance(robot.x, robot.y, gx, gy)
            if dist_to_goal <= self._goal_tolerance:
                return SimulationResult(
                    status=SimulationStatus.SUCCESS,
                    steps=step - 1,
                    trajectory=trajectory,
                    planned_path_grid=grid_path,
                    planned_path_world=world_path,
                    start_world=(sx, sy),
                    goal_world=(gx, gy),
                    path_length=self._compute_path_length(trajectory),
                    message=f"Goal reached in {step - 1} steps.",
                    controller_name=type(self._controller).__name__,
                )

            # 2 & 3. Compute control command toward current target
            if isinstance(self._controller, PurePursuitController):
                v_cmd, omega_cmd = self._controller.compute_command((robot.x, robot.y, robot.theta))
            else:
                current_wp = world_path[waypoint_idx]
                dist_to_wp = self._world_distance(robot.x, robot.y, current_wp[0], current_wp[1])

                if dist_to_wp <= self._waypoint_tolerance and waypoint_idx < len(world_path) - 1:
                    waypoint_idx += 1
                    current_wp = world_path[waypoint_idx]

                target_wp = (gx, gy) if waypoint_idx == len(world_path) - 1 else current_wp
                v_cmd, omega_cmd = self._controller.compute_command(
                    (robot.x, robot.y, robot.theta),
                    target_wp,
                )

            prev_pose = (robot.x, robot.y, robot.theta)

            # 4. Advance robot kinematics
            robot.step(self._dt, v=v_cmd, omega=omega_cmd)
            curr_pose = (robot.x, robot.y, robot.theta)

            # 5. Collision check along motion from prev_pose to curr_pose
            is_free, first_colliding_pose = self._collision_checker.check_motion(
                prev_pose,
                curr_pose,
                max_substep_size=self._max_substep_size,
            )

            if not is_free:
                if first_colliding_pose is not None:
                    robot.set_pose(
                        first_colliding_pose[0],
                        first_colliding_pose[1],
                        first_colliding_pose[2],
                    )
                trajectory.append(robot.state)
                return SimulationResult(
                    status=SimulationStatus.COLLISION,
                    steps=step,
                    trajectory=trajectory,
                    planned_path_grid=grid_path,
                    planned_path_world=world_path,
                    start_world=(sx, sy),
                    goal_world=(gx, gy),
                    path_length=self._compute_path_length(trajectory),
                    message=f"Collision detected at step {step} at pose ({robot.x:.3f}, {robot.y:.3f}).",
                    controller_name=type(self._controller).__name__,
                )

            # 6. Log state
            trajectory.append(robot.state)

        # Max steps exceeded
        return SimulationResult(
            status=SimulationStatus.TIMEOUT,
            steps=self._max_steps,
            trajectory=trajectory,
            planned_path_grid=grid_path,
            planned_path_world=world_path,
            start_world=(sx, sy),
            goal_world=(gx, gy),
            path_length=self._compute_path_length(trajectory),
            message=f"Simulation timed out after {self._max_steps} steps without reaching goal.",
            controller_name=type(self._controller).__name__,
        )

    @staticmethod
    def _validate_world_coords(
        coords: Any, name: str
    ) -> Tuple[float, float]:
        """Validate and extract (x, y) world coordinates."""
        if not isinstance(coords, (tuple, list)):
            raise TypeError(f"{name} must be a (x, y) tuple, got {type(coords).__name__}.")
        if len(coords) != 2:
            raise ValueError(f"{name} must have exactly 2 elements (x, y), got {len(coords)}.")
        x, y = coords
        for val, vname in [(x, f"{name}_x"), (y, f"{name}_y")]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{vname} must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"{vname} must be finite, got {val}.")
        return (float(x), float(y))

    def __repr__(self) -> str:
        return (
            f"NavigationSimulator(grid={self._grid!r}, "
            f"robot_radius={self._robot_radius}, dt={self._dt}, "
            f"max_steps={self._max_steps}, goal_tolerance={self._goal_tolerance})"
        )
