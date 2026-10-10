"""Collision Checker module for the 2D Autonomous Navigation Simulator.

This module provides the `CollisionChecker` class, which evaluates whether a
circular mobile robot footprint is collision-free at a stationary world pose
on an `OccupancyGrid`.

Coordinate System & Geometry Interpretation:
--------------------------------------------
- Grid Coordinates: Discrete indices `(row, column)`.
- World Coordinates: Continuous `(x, y)` in meters.
  - Horizontal $x$-axis aligns with grid columns ($x$ increases to the right).
  - Vertical $y$-axis aligns with grid rows ($y$ increases upwards).
- Map Boundaries:
  - $X_{\\min} = \\text{origin}_x, \\quad X_{\\max} = \\text{origin}_x + \\text{width} \\cdot \\text{resolution}$
  - $Y_{\\min} = \\text{origin}_y, \\quad Y_{\\max} = \\text{origin}_y + \\text{height} \\cdot \\text{resolution}$
- Occupied Cell Geometry:
  Each discrete occupied cell $(r, c)$ represents an axis-aligned rectangular region (AABB):
  $$[x_{\\min}(c), x_{\\max}(c)] \\times [y_{\\min}(r), y_{\\max}(r)]$$
  where:
    x_{\\min}(c) = \\text{origin}_x + c \\cdot \\text{resolution}
    x_{\\max}(c) = \\text{origin}_x + (c + 1) \\cdot \\text{resolution}
    y_{\\min}(r) = \\text{origin}_y + r \\cdot \\text{resolution}
    y_{\\max}(r) = \\text{origin}_y + (r + 1) \\cdot \\text{resolution}

Collision & Boundary Policy:
----------------------------
1. Robot Footprint:
   A circle centered at continuous world position $(x, y)$ with radius $R > 0$.
2. Distance Calculation:
   For any cell, the point on the rectangle closest to $(x, y)$ is found by clamping:
     x_{\\text{clamp}} = \\max(x_{\\min}, \\min(x_{\\max}, x))
     y_{\\text{clamp}} = \\max(y_{\\min}, \\min(y_{\\max}, y))
   The minimum Euclidean distance is $d = \\sqrt{(x - x_{\\text{clamp}})^2 + (y - y_{\\text{clamp}})^2}$.
3. Conservative Collision Rule:
   A collision occurs if $d \\le R + \\epsilon$ (where $\\epsilon = 10^{-9}$ handles floating-point
   precision). Touching an occupied cell boundary is treated as a collision.
4. Map Boundary Rule:
   The circular footprint must remain entirely within the represented map boundaries:
     x - R \\ge X_{\\min}, \\quad x + R \\le X_{\\max}
     y - R \\ge Y_{\\min}, \\quad y + R \\le Y_{\\max}
   Any pose whose circular footprint extends beyond the outer boundary is not collision-free.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from src.environment import OccupancyGrid


class CollisionChecker:
    """Evaluates whether a circular robot footprint is collision-free on an OccupancyGrid.

    Parameters
    ----------
    grid : OccupancyGrid
        The 2D occupancy grid representation of the environment.
    robot_radius : float, optional
        Effective radius of the robot's circular footprint in meters. Must be strictly
        positive and finite. Defaults to 0.2.

    Raises
    ------
    TypeError
        If `grid` is not an `OccupancyGrid` or `robot_radius` is non-numeric or boolean.
    ValueError
        If `robot_radius` is non-positive or non-finite.
    """

    TOLERANCE: float = 1e-9

    def __init__(self, grid: OccupancyGrid, robot_radius: float = 0.2) -> None:
        if not isinstance(grid, OccupancyGrid):
            raise TypeError(f"Expected OccupancyGrid instance, got {type(grid).__name__}.")

        if isinstance(robot_radius, bool) or not isinstance(robot_radius, (int, float)):
            raise TypeError(f"robot_radius must be a numeric float or int, got {type(robot_radius).__name__}.")
        if math.isnan(robot_radius) or math.isinf(robot_radius):
            raise ValueError(f"robot_radius must be finite, got {robot_radius}.")
        if robot_radius <= 0.0:
            raise ValueError(f"robot_radius must be strictly positive (> 0.0), got {robot_radius}.")

        self._grid: OccupancyGrid = grid
        self._radius: float = float(robot_radius)

        # Cache grid geometric extents
        self._res: float = grid.resolution
        self._ox: float = grid.origin[0]
        self._oy: float = grid.origin[1]
        self._x_min: float = self._ox
        self._x_max: float = self._ox + grid.width * self._res
        self._y_min: float = self._oy
        self._y_max: float = self._oy + grid.height * self._res

    @property
    def grid(self) -> OccupancyGrid:
        """Return the underlying OccupancyGrid."""
        return self._grid

    @property
    def robot_radius(self) -> float:
        """Return the robot circular footprint radius in meters."""
        return self._radius

    def _extract_coordinates(
        self,
        x: Union[float, Tuple[float, ...], Any],
        y: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Extract and validate numeric (x, y) coordinates from parameters."""
        if y is None:
            if hasattr(x, "x") and hasattr(x, "y"):
                raw_x, raw_y = x.x, x.y
            elif isinstance(x, (tuple, list)) and len(x) >= 2:
                raw_x, raw_y = x[0], x[1]
            else:
                raise TypeError(f"Expected (x, y) coordinates or object with x, y attributes, got {type(x).__name__}.")
        else:
            raw_x, raw_y = x, y

        for val, name in [(raw_x, "x"), (raw_y, "y")]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Coordinate '{name}' must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Coordinate '{name}' must be finite, got {val}.")

        return (float(raw_x), float(raw_y))

    def _extract_pose(
        self,
        pose: Union[float, Tuple[float, ...], Any],
        y: Optional[float] = None,
        theta: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Extract and validate numeric (x, y, theta) coordinates from parameters."""
        if y is None:
            if hasattr(pose, "x") and hasattr(pose, "y"):
                raw_x, raw_y = pose.x, pose.y
                raw_th = getattr(pose, "theta", 0.0)
            elif isinstance(pose, (tuple, list)):
                if len(pose) < 2 or len(pose) > 3:
                    raise ValueError(f"Pose sequence must have 2 or 3 elements, got {len(pose)}.")
                raw_x, raw_y = pose[0], pose[1]
                raw_th = pose[2] if len(pose) == 3 else 0.0
            else:
                raise TypeError(f"Expected (x, y) pose or object with x, y attributes, got {type(pose).__name__}.")
        else:
            raw_x, raw_y = pose, y
            raw_th = theta if theta is not None else 0.0

        for val, name in [(raw_x, "x"), (raw_y, "y"), (raw_th, "theta")]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Pose coordinate '{name}' must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Pose coordinate '{name}' must be finite, got {val}.")

        norm_th = (float(raw_th) + math.pi) % (2.0 * math.pi) - math.pi
        return (float(raw_x), float(raw_y), float(norm_th))

    def check_motion(
        self,
        start: Union[Tuple[float, ...], Any],
        end: Union[Tuple[float, ...], Any],
        max_substep_size: Optional[float] = None,
    ) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """Check whether continuous motion between two poses is collision-free.

        Substeps the linear translational segment between start and end positions,
        verifying that the circular footprint remains entirely inside map boundaries
        and free of obstacles at all intermediate sub-steps.

        Parameters
        ----------
        start : Union[Tuple[float, ...], Any]
            Starting pose `(x, y)` / `(x, y, theta)` or state object.
        end : Union[Tuple[float, ...], Any]
            Ending pose `(x, y)` / `(x, y, theta)` or state object.
        max_substep_size : Optional[float], optional
            Maximum distance (meters) between consecutive intermediate checks.
            Must be strictly positive and finite. Defaults to `robot_radius / 2.0`.

        Returns
        -------
        Tuple[bool, Optional[Tuple[float, float, float]]]
            `(True, None)` if entire motion is collision-free.
            `(False, first_colliding_pose)` if collision is detected.

        Raises
        ------
        TypeError
            If poses or `max_substep_size` contain non-numeric or boolean values.
        ValueError
            If coordinates or `max_substep_size` are non-finite or non-positive.
        """
        x0, y0, th0 = self._extract_pose(start)
        x1, y1, th1 = self._extract_pose(end)

        if max_substep_size is not None:
            if isinstance(max_substep_size, bool) or not isinstance(max_substep_size, (int, float)):
                raise TypeError(f"max_substep_size must be a numeric float, got {type(max_substep_size).__name__}.")
            if math.isnan(max_substep_size) or math.isinf(max_substep_size) or max_substep_size <= 0.0:
                raise ValueError(f"max_substep_size must be strictly positive and finite, got {max_substep_size}.")
            step_limit = float(max_substep_size)
        else:
            step_limit = self._radius / 2.0

        # Check starting pose
        if not self.is_pose_collision_free(x0, y0):
            return (False, (x0, y0, th0))

        dist = math.hypot(x1 - x0, y1 - y0)
        dtheta = (th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi

        if dist == 0.0:
            # Pure rotation in place: circular footprint is rotationally invariant
            return (True, None)

        num_steps = max(1, math.ceil(dist / step_limit))

        for i in range(1, num_steps + 1):
            alpha = float(i) / float(num_steps)
            xi = x0 + alpha * (x1 - x0)
            yi = y0 + alpha * (y1 - y0)
            thi = (th0 + alpha * dtheta + math.pi) % (2.0 * math.pi) - math.pi

            if not self.is_pose_collision_free(xi, yi):
                return (False, (float(xi), float(yi), float(thi)))

        return (True, None)

    def is_motion_collision_free(
        self,
        start: Union[Tuple[float, ...], Any],
        end: Union[Tuple[float, ...], Any],
        max_substep_size: Optional[float] = None,
    ) -> bool:
        """Check whether continuous motion between two poses is collision-free.

        Parameters
        ----------
        start : Union[Tuple[float, ...], Any]
            Starting pose `(x, y)` / `(x, y, theta)` or state object.
        end : Union[Tuple[float, ...], Any]
            Ending pose `(x, y)` / `(x, y, theta)` or state object.
        max_substep_size : Optional[float], optional
            Maximum distance (meters) between consecutive intermediate checks.

        Returns
        -------
        bool
            True if entire motion is collision-free; False otherwise.
        """
        is_free, _ = self.check_motion(start, end, max_substep_size=max_substep_size)
        return is_free

    def is_pose_collision_free(
        self,
        x: Union[float, Tuple[float, ...], Any],
        y: Optional[float] = None,
    ) -> bool:
        """Check whether a circular footprint centered at `(x, y)` is entirely collision-free.

        A pose is collision-free if:
        1. The footprint circle is completely within the outer map boundaries.
        2. The footprint circle does not intersect or touch any occupied obstacle cell.

        Parameters
        ----------
        x : Union[float, Tuple[float, ...], Any]
            World $x$ coordinate, a pose tuple `(x, y)` / `(x, y, theta)`, or a robot object.
        y : float, optional
            World $y$ coordinate if `x` is provided as a scalar float.

        Returns
        -------
        bool
            True if collision-free; False if in collision, touching an obstacle, or
            extending beyond the map boundary.

        Raises
        ------
        TypeError
            If coordinates are non-numeric or boolean.
        ValueError
            If coordinates are NaN or infinite.
        """
        cx, cy = self._extract_coordinates(x, y)
        r = self._radius

        # 1. Boundary Check: footprint must not extend outside map boundaries
        if (
            (cx - r) < (self._x_min - self.TOLERANCE)
            or (cx + r) > (self._x_max + self.TOLERANCE)
            or (cy - r) < (self._y_min - self.TOLERANCE)
            or (cy + r) > (self._y_max + self.TOLERANCE)
        ):
            return False

        # 2. Determine potential cell index bounds intersecting the circle's AABB
        col_min = max(0, math.floor((cx - r - self._ox) / self._res))
        col_max = min(self._grid.width - 1, math.floor((cx + r - self._ox) / self._res))
        row_min = max(0, math.floor((cy - r - self._oy) / self._res))
        row_max = min(self._grid.height - 1, math.floor((cy + r - self._oy) / self._res))

        r_sq = r * r

        # 3. Check distance to each occupied cell in the footprint bounding box
        for row in range(row_min, row_max + 1):
            cell_y_min = self._oy + row * self._res
            cell_y_max = cell_y_min + self._res

            # Clamp circle y to cell vertical interval
            closest_y = max(cell_y_min, min(cell_y_max, cy))
            dy = cy - closest_y
            dy_sq = dy * dy

            # If vertical distance alone exceeds footprint radius, skip entire row
            if dy_sq > (r_sq + self.TOLERANCE):
                continue

            for col in range(col_min, col_max + 1):
                if not self._grid.is_occupied(row, col):
                    continue

                cell_x_min = self._ox + col * self._res
                cell_x_max = cell_x_min + self._res

                # Clamp circle x to cell horizontal interval
                closest_x = max(cell_x_min, min(cell_x_max, cx))
                dx = cx - closest_x

                dist_sq = dx * dx + dy_sq

                # Conservative touching policy: collision if distance <= radius (with tolerance)
                # (dist_sq <= (r + tol)^2 is equivalent to dist <= r + tol)
                effective_r = r + self.TOLERANCE
                if dist_sq <= (effective_r * effective_r):
                    return False

        return True
