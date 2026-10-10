"""Benchmark scenario data model for the AI-Based Autonomous Navigation System.

This module defines the immutable `BenchmarkScenario` specification representing
a reproducible, self-contained test environment and navigation challenge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional, Sequence, Tuple, Type

from src.environment import OccupancyGrid
from src.simulator import NavigationSimulator, SimulationStatus


@dataclass(frozen=True)
class BenchmarkScenario:
    """Immutable specification for a single deterministic navigation benchmark scenario.

    Parameters
    ----------
    name : str
        Unique, human-readable identifier for the scenario.
    description : str
        Detailed description of the environment geometry and navigation objective.
    grid : OccupancyGrid
        Template occupancy grid defining the spatial boundaries and obstacles.
    start : Tuple[float, float]
        World-coordinate starting position `(x, y)` in meters.
    goal : Tuple[float, float]
        World-coordinate target position `(x, y)` in meters.
    expected_status : Optional[SimulationStatus], optional
        The expected terminal `SimulationStatus` (e.g. `SUCCESS`, `NO_PATH`).
        Must be provided if `expected_exception` is None.
    expected_exception : Optional[Type[Exception]], optional
        Expected exception class (e.g. `ValueError`) for validation scenarios.
        Must be provided if `expected_status` is None.
    additional_invalid_queries : Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
        Optional additional `(start, goal)` query pairs to test for validation scenarios.
    robot_radius : float, optional
        Circular footprint radius in meters. Defaults to 0.2.
    linear_speed : float, optional
        Linear cruising velocity limit in meters/second. Defaults to 0.5.
    angular_speed_limit : float, optional
        Maximum rotational speed limit in radians/second. Defaults to 1.0.
    angular_gain : float, optional
        Proportional gain for the heading controller. Defaults to 3.0.
    dt : float, optional
        Simulation time increment in seconds. Defaults to 0.1.
    goal_tolerance : float, optional
        Distance threshold in meters to declare arrival. Defaults to 0.25.
    waypoint_tolerance : Optional[float], optional
        Distance threshold in meters to advance waypoints. Defaults to None (uses goal_tolerance).
    max_steps : int, optional
        Maximum allowed simulation iterations. Defaults to 2000.
    """

    name: str
    description: str
    grid: OccupancyGrid
    start: Tuple[float, float]
    goal: Tuple[float, float]
    expected_status: Optional[SimulationStatus] = None
    expected_exception: Optional[Type[Exception]] = None
    additional_invalid_queries: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = field(
        default_factory=tuple
    )
    robot_radius: float = 0.2
    linear_speed: float = 0.5
    angular_speed_limit: float = 1.0
    angular_gain: float = 3.0
    dt: float = 0.1
    goal_tolerance: float = 0.25
    waypoint_tolerance: Optional[float] = None
    max_steps: int = 2000

    def __post_init__(self) -> None:
        """Validate scenario configuration consistency."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Scenario name must be a non-empty string, got {self.name!r}.")

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(f"Scenario description must be a non-empty string, got {self.description!r}.")

        if not isinstance(self.grid, OccupancyGrid):
            raise TypeError(f"Grid must be an OccupancyGrid instance, got {type(self.grid).__name__}.")

        # Ensure exactly one expectation type is defined
        has_status = self.expected_status is not None
        has_exception = self.expected_exception is not None
        if has_status == has_exception:
            raise ValueError(
                f"Scenario '{self.name}' must specify exactly one of 'expected_status' or 'expected_exception'."
            )

        if has_status and not isinstance(self.expected_status, SimulationStatus):
            raise TypeError(
                f"expected_status must be a SimulationStatus enum, got {type(self.expected_status).__name__}."
            )

        if has_exception and not (
            isinstance(self.expected_exception, type) and issubclass(self.expected_exception, Exception)
        ):
            raise TypeError(
                f"expected_exception must be an Exception subclass, got {self.expected_exception!r}."
            )

        # Validate start and goal tuples
        for pt, label in [(self.start, "start"), (self.goal, "goal")]:
            if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                raise TypeError(f"{label} coordinates must be a 2-element tuple/list of floats.")
            for coord, c_name in [(pt[0], f"{label}[0]"), (pt[1], f"{label}[1]")]:
                if isinstance(coord, bool) or not isinstance(coord, (int, float)):
                    raise TypeError(f"{c_name} must be numeric float, got {type(coord).__name__}.")
                if math.isnan(coord) or math.isinf(coord):
                    raise ValueError(f"{c_name} must be finite, got {coord}.")

    def create_grid(self) -> OccupancyGrid:
        """Return a fresh, defensive copy of the scenario's occupancy grid.

        Ensures zero mutable state sharing between simulation runs.
        """
        return OccupancyGrid(
            grid=self.grid.grid,
            resolution=self.grid.resolution,
            origin=self.grid.origin,
        )

    def create_simulator(
        self,
        controller: Union[str, Any] = "heading",
        lookahead_distance: Optional[float] = None,
    ) -> NavigationSimulator:
        """Instantiate a fresh NavigationSimulator configured for this scenario."""
        return NavigationSimulator(
            grid=self.create_grid(),
            robot_radius=self.robot_radius,
            dt=self.dt,
            max_steps=self.max_steps,
            goal_tolerance=self.goal_tolerance,
            waypoint_tolerance=self.waypoint_tolerance,
            linear_speed=self.linear_speed,
            angular_gain=self.angular_gain,
            angular_speed_limit=self.angular_speed_limit,
            controller=controller,
            lookahead_distance=lookahead_distance,
        )
