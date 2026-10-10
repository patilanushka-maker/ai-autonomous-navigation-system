"""Path Planner module for the 2D Autonomous Navigation Simulator.

This module provides the `AStarPlanner` class, implementing the A* graph search
algorithm for finding shortest collision-free paths on a 2D `OccupancyGrid`.

Movement Model:
---------------
- Four-connected grid motion: Up `(-1, 0)`, Down `(+1, 0)`, Left `(0, -1)`, Right `(0, +1)`.
- Uniform unit edge cost: each step has cost 1.0.
- Diagonal movements are strictly disallowed.
- Obstacle cells (`OCCUPIED == 1`) and out-of-bounds cells are non-traversable.

Heuristic Formulation:
----------------------
- Manhattan Distance: $h(n) = |r_n - r_{\\text{goal}}| + |c_n - c_{\\text{goal}}|$.
- Admissibility & Consistency: Under a four-connected grid with uniform step cost 1.0,
  the true shortest obstacle-free path length between $(r_1, c_1)$ and $(r_2, c_2)$
  is always $\\ge |r_1 - r_2| + |c_1 - c_2|$. Thus, $h(n)$ never overestimates the true
  remaining cost ($h(n) \\le h^*(n)$) and satisfies the triangle inequality
  ($h(n) \\le 1 + h(n')$), ensuring that A* is both admissible (finds the optimal shortest
  path) and monotonic (settles nodes with minimal redundant re-expansions).
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Set, Tuple

from src.environment import OccupancyGrid


class AStarPlanner:
    """A* Path Planner operating on a discrete 2D `OccupancyGrid`.

    Parameters
    ----------
    grid : OccupancyGrid
        The 2D occupancy grid representation of the environment.

    Raises
    ------
    TypeError
        If `grid` is not an instance of `OccupancyGrid`.
    """

    # 4-connected cardinal movements: Up, Down, Left, Right
    DIRECTIONS: Tuple[Tuple[int, int], ...] = (
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1),   # Right
    )

    def __init__(self, grid: OccupancyGrid) -> None:
        if not isinstance(grid, OccupancyGrid):
            raise TypeError(f"Expected OccupancyGrid instance, got {type(grid).__name__}.")
        self._grid = grid

    @property
    def grid(self) -> OccupancyGrid:
        """Return the underlying OccupancyGrid instance."""
        return self._grid

    @staticmethod
    def heuristic(r1: int, c1: int, r2: int, c2: int) -> int:
        """Compute the Manhattan distance heuristic between two cells.

        Parameters
        ----------
        r1, c1 : int
            Row and column of the source cell.
        r2, c2 : int
            Row and column of the target cell.

        Returns
        -------
        int
            Manhattan distance $|r_1 - r_2| + |c_1 - c_2|$.
        """
        return abs(r1 - r2) + abs(c1 - c2)

    def _validate_coordinate(self, coord: Any, name: str) -> Tuple[int, int]:
        """Validate format, types, bounds, and occupancy of a coordinate tuple.

        Raises
        ------
        TypeError
            If coordinate is not a sequence of 2 integers (or contains booleans).
        ValueError
            If coordinate is out of grid bounds or in an occupied obstacle cell.
        """
        if not isinstance(coord, (tuple, list)):
            raise TypeError(f"{name} must be a (row, column) tuple or list, got {type(coord).__name__}.")
        if len(coord) != 2:
            raise ValueError(f"{name} coordinate must contain exactly 2 elements, got {len(coord)} elements.")

        r, c = coord
        if isinstance(r, bool) or isinstance(c, bool) or not isinstance(r, int) or not isinstance(c, int):
            raise TypeError(f"{name} indices must be integers, got types ({type(r).__name__}, {type(c).__name__}).")

        if not self._grid.is_in_bounds(r, c):
            raise ValueError(
                f"{name} coordinate ({r}, {c}) is outside grid bounds "
                f"(height={self._grid.height}, width={self._grid.width})."
            )

        if self._grid.is_occupied(r, c):
            raise ValueError(f"{name} coordinate ({r}, {c}) is inside an occupied obstacle cell.")

        return (r, c)

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Compute the shortest collision-free 4-connected path from start to goal.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting coordinates `(start_row, start_col)`.
        goal : Tuple[int, int]
            Target goal coordinates `(goal_row, goal_col)`.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            A list of discrete `(row, column)` waypoints from `start` through `goal`
            inclusive if a valid path exists; `None` if the goal is disconnected/unreachable.

        Raises
        ------
        TypeError
            If `start` or `goal` is malformed, not a 2-tuple, or contains non-integer types.
        ValueError
            If `start` or `goal` is out of bounds or located on an occupied obstacle.
        """
        start_cell = self._validate_coordinate(start, "Start")
        goal_cell = self._validate_coordinate(goal, "Goal")

        # Edge case: start is already at goal
        if start_cell == goal_cell:
            return [start_cell]

        # Priority Queue entries: (f_score, h_score, (row, col))
        # Including h_score breaks ties in f_score towards nodes closer to the goal.
        start_h = self.heuristic(start_cell[0], start_cell[1], goal_cell[0], goal_cell[1])
        open_queue: List[Tuple[int, int, Tuple[int, int]]] = [(start_h, start_h, start_cell)]

        # Cost from start to each discovered node along the current best path
        g_score: Dict[Tuple[int, int], int] = {start_cell: 0}

        # Parent pointers for optimal path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Closed set to avoid re-expanding already settled optimal nodes
        closed_set: Set[Tuple[int, int]] = set()

        while open_queue:
            f_cost, h_cost, current = heapq.heappop(open_queue)

            if current in closed_set:
                continue
            closed_set.add(current)

            # Goal reached: reconstruct trajectory
            if current == goal_cell:
                path: List[Tuple[int, int]] = [goal_cell]
                curr = goal_cell
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                return path

            cur_r, cur_c = current
            cur_g = g_score[current]

            # Explore 4-connected cardinal neighbors
            for dr, dc in self.DIRECTIONS:
                nr, nc = cur_r + dr, cur_c + dc
                neighbor = (nr, nc)

                # Skip out-of-bounds or obstacle cells
                if not self._grid.is_in_bounds(nr, nc):
                    continue
                if self._grid.is_occupied(nr, nc):
                    continue
                if neighbor in closed_set:
                    continue

                tentative_g = cur_g + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = self.heuristic(nr, nc, goal_cell[0], goal_cell[1])
                    f = tentative_g + h
                    heapq.heappush(open_queue, (f, h, neighbor))

        # Goal is unreachable
        return None
