"""Predefined benchmark scenarios for the AI-Based Autonomous Navigation System.

This module provides the factory functions for the seven standardized benchmark
scenarios specified in Phase 3B.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from src.benchmark.scenario import BenchmarkScenario
from src.demo import create_demo_grid
from src.environment import OccupancyGrid
from src.simulator import SimulationStatus


def scenario_open_space() -> BenchmarkScenario:
    """Scenario 1: Open Space.

    An unobstructed 10x10 map with a reachable goal.
    Expected outcome: SimulationStatus.SUCCESS.
    """
    grid = OccupancyGrid.from_dimensions(height=10, width=10, initial_value=0, resolution=1.0)
    return BenchmarkScenario(
        name="open_space",
        description="Unobstructed 10x10 open grid with a clear, straight-line navigable route to a reachable goal.",
        grid=grid,
        start=(1.5, 1.5),
        goal=(8.5, 8.5),
        expected_status=SimulationStatus.SUCCESS,
        robot_radius=0.2,
        linear_speed=0.5,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
        max_steps=2000,
    )


def scenario_one_obstacle() -> BenchmarkScenario:
    """Scenario 2: One Obstacle.

    A 10x10 map with a central 2x2 obstacle block requiring navigation around the barrier.
    Obstacle spans rows 4..5, cols 4..5: world [4.0, 6.0] x [4.0, 6.0].
    Expected outcome: SimulationStatus.SUCCESS.
    """
    grid_data = np.zeros((10, 10), dtype=np.int8)
    grid_data[4:6, 4:6] = 1
    grid = OccupancyGrid(grid=grid_data, resolution=1.0)
    return BenchmarkScenario(
        name="one_obstacle",
        description="10x10 grid with a central 2x2 obstacle block requiring planar bypass around the barrier.",
        grid=grid,
        start=(2.0, 5.0),
        goal=(8.0, 5.0),
        expected_status=SimulationStatus.SUCCESS,
        robot_radius=0.2,
        linear_speed=0.5,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
        max_steps=2000,
    )


def scenario_multiple_obstacles() -> BenchmarkScenario:
    """Scenario 3: Multiple Obstacles.

    A 15x20 slalom environment based on the demonstration map with multiple barriers.
    Expected outcome: SimulationStatus.SUCCESS.
    """
    grid = create_demo_grid()
    return BenchmarkScenario(
        name="multiple_obstacles",
        description="15x20 slalom obstacle course based on the demonstration map requiring multi-turn obstacle avoidance.",
        grid=grid,
        start=(2.5, 2.5),
        goal=(17.5, 12.5),
        expected_status=SimulationStatus.SUCCESS,
        robot_radius=0.2,
        linear_speed=0.5,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
        max_steps=2000,
    )


def scenario_narrow_corridor() -> BenchmarkScenario:
    """Scenario 4: Narrow Corridor.

    An 11x11 map featuring a single 1-cell wide (1.0 m) horizontal passage between walls.
    With robot radius 0.2 m centered at y=5.5, lateral clearance is 0.3 m on each side.
    Expected outcome: SimulationStatus.SUCCESS.
    """
    grid_data = np.zeros((11, 11), dtype=np.int8)
    grid_data[4, 2:9] = 1  # Lower wall
    grid_data[6, 2:9] = 1  # Upper wall
    # Row 5 is the free corridor spanning y in [5.0, 6.0]
    grid = OccupancyGrid(grid=grid_data, resolution=1.0)
    return BenchmarkScenario(
        name="narrow_corridor",
        description="11x11 grid with a 1-meter wide straight passage physically navigable by the circular footprint.",
        grid=grid,
        start=(1.5, 5.5),
        goal=(9.5, 5.5),
        expected_status=SimulationStatus.SUCCESS,
        robot_radius=0.2,
        linear_speed=0.5,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
        max_steps=2000,
    )


def scenario_maze() -> BenchmarkScenario:
    """Scenario 5: Maze-like Map.

    A 15x15 labyrinth with interleaved serpentine barrier walls requiring multi-turn routing.
    Expected outcome: SimulationStatus.SUCCESS.
    """
    grid_data = np.zeros((15, 15), dtype=np.int8)
    grid_data[3, 1:13] = 1   # Wall 1: right opening at cols 13..14
    grid_data[6, 2:15] = 1   # Wall 2: left opening at cols 0..1
    grid_data[9, 1:13] = 1   # Wall 3: right opening at cols 13..14
    grid_data[12, 2:15] = 1  # Wall 4: left opening at cols 0..1
    grid = OccupancyGrid(grid=grid_data, resolution=1.0)
    return BenchmarkScenario(
        name="maze_like_map",
        description="15x15 serpentine labyrinth with interleaved interior barrier walls requiring multi-turn topological routing.",
        grid=grid,
        start=(1.5, 1.5),
        goal=(13.5, 13.5),
        expected_status=SimulationStatus.SUCCESS,
        robot_radius=0.2,
        linear_speed=0.5,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
        max_steps=4000,
    )


def scenario_unreachable_goal() -> BenchmarkScenario:
    """Scenario 6: Unreachable Goal.

    A 10x10 map with an unbroken vertical wall partitioning the left and right halves.
    Expected outcome: SimulationStatus.NO_PATH.
    """
    grid_data = np.zeros((10, 10), dtype=np.int8)
    grid_data[:, 5] = 1  # Solid barrier at column 5
    grid = OccupancyGrid(grid=grid_data, resolution=1.0)
    return BenchmarkScenario(
        name="unreachable_goal",
        description="10x10 grid partitioned by an unbroken wall completely isolating the goal from the start.",
        grid=grid,
        start=(2.5, 5.0),
        goal=(7.5, 5.0),
        expected_status=SimulationStatus.NO_PATH,
        robot_radius=0.2,
        linear_speed=0.5,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
        max_steps=2000,
    )


def scenario_invalid_start_goal() -> BenchmarkScenario:
    """Scenario 7: Invalid Start/Goal.

    A 10x10 map testing representative occupied and out-of-bounds start/goal inputs.
    Expected outcome: Raises ValueError.
    """
    grid_data = np.zeros((10, 10), dtype=np.int8)
    grid_data[2, 2] = 1  # Cell (2, 2) is occupied: [2.0, 3.0] x [2.0, 3.0]
    grid = OccupancyGrid(grid=grid_data, resolution=1.0)
    return BenchmarkScenario(
        name="invalid_start_goal",
        description="10x10 grid testing representative occupied and out-of-bounds start/goal inputs to verify rejection.",
        grid=grid,
        start=(2.5, 2.5),  # Occupied start point
        goal=(8.5, 8.5),
        expected_exception=ValueError,
        additional_invalid_queries=(
            ((-1.0, 5.0), (8.5, 8.5)),   # Out-of-bounds start
            ((1.5, 1.5), (15.0, 5.0)),   # Out-of-bounds goal
            ((1.5, 1.5), (2.5, 2.5)),    # Occupied goal
        ),
        robot_radius=0.2,
    )


def get_default_scenarios() -> List[BenchmarkScenario]:
    """Return the ordered list of all seven standardized benchmark scenarios."""
    return [
        scenario_open_space(),
        scenario_one_obstacle(),
        scenario_multiple_obstacles(),
        scenario_narrow_corridor(),
        scenario_maze(),
        scenario_unreachable_goal(),
        scenario_invalid_start_goal(),
    ]


def get_scenario_by_name(name: str) -> BenchmarkScenario:
    """Retrieve a benchmark scenario by its unique identifier.

    Parameters
    ----------
    name : str
        The unique name of the scenario to retrieve.

    Returns
    -------
    BenchmarkScenario
        The matching scenario specification.

    Raises
    ------
    KeyError
        If no scenario matches the provided name.
    """
    registry: Dict[str, BenchmarkScenario] = {s.name: s for s in get_default_scenarios()}
    if name not in registry:
        available = ", ".join(f"'{k}'" for k in registry.keys())
        raise KeyError(f"Unknown benchmark scenario '{name}'. Available scenarios: {available}")
    return registry[name]
