"""Environment module for the 2D Autonomous Navigation Simulator.

This module provides the `OccupancyGrid` abstraction representing a discrete
two-dimensional workspace for mobile robot navigation.

Coordinate and Occupancy Conventions:
-------------------------------------
1. Occupancy Semantics:
   - `0`: Free space (traversable by the robot).
   - `1`: Occupied space (obstacle / non-traversable).
   - Boolean arrays and boolean values are explicitly rejected to prevent semantic
     ambiguity between traversable and occupied states.

2. Discrete Grid Coordinates:
   - Indexing is specified as `(row, column)`.
   - `row` corresponds to the vertical grid index, where `0 <= row < height`.
   - `column` corresponds to the horizontal grid index, where `0 <= column < width`.
   - The internal matrix shape is `(height, width)` where `height` is the number of
     rows (axis 0) and `width` is the number of columns (axis 1).

3. Continuous World Coordinates:
   - Coordinates are defined as `(x, y)` in continuous space (e.g., meters).
   - The horizontal $x$-axis aligns with the grid columns (`x` increases with column index).
   - The vertical $y$-axis aligns with the grid rows (`y` increases with row index).
   - Origin `(origin_x, origin_y)` specifies the continuous world coordinates of the
     outer lower boundary of cell `(0, 0)`.
   - Each cell spans `[origin_x + col * res, origin_x + (col + 1) * res)` along $x$
     and `[origin_y + row * res, origin_y + (row + 1) * res)` along $y$.
   - Cell center coordinates are positioned at:
       x_center = origin_x + (col + 0.5) * resolution
       y_center = origin_y + (row + 0.5) * resolution
   - Continuous coordinates below the origin (e.g. $x < \text{origin}_x$ or $y < \text{origin}_y$)
     map to negative row or column indices via floor division.
   - Continuous coordinates at or beyond the outer boundary
     ($x \ge \text{origin}_x + \text{width} \cdot \text{res}$ or
      $y \ge \text{origin}_y + \text{height} \cdot \text{res}$)
     map to indices with $\text{col} \ge \text{width}$ or $\text{row} \ge \text{height}$.
   - All points falling outside the grid footprint can be safely verified via `is_in_bounds(row, col)`.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union
import numpy as np


class OccupancyGrid:
    """A two-dimensional discrete occupancy grid representation.

    Parameters
    ----------
    grid : Union[np.ndarray, Sequence[Sequence[int]]]
        Two-dimensional array or nested sequence containing cell occupancy states.
        Values must be exclusively 0 (free) or 1 (occupied). Boolean arrays are rejected.
    resolution : float, optional
        Physical size of each grid cell (e.g., meters per cell). Must be strictly positive
        and finite. Defaults to 1.0.
    origin : Tuple[float, float], optional
        World coordinates `(origin_x, origin_y)` of the grid origin. Must be finite.
        Defaults to (0.0, 0.0).

    Raises
    ------
    TypeError
        If input grid cannot be converted to a NumPy array, contains boolean or non-numeric
        types, or if origin/resolution types are invalid.
    ValueError
        If the grid is not two-dimensional, is empty, contains invalid cell values (not in {0, 1}),
        or if resolution is non-positive or non-finite.
    """

    FREE: int = 0
    OCCUPIED: int = 1

    def __init__(
        self,
        grid: Union[np.ndarray, Sequence[Sequence[int]]],
        resolution: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        # Validate and configure resolution
        if not isinstance(resolution, (int, float)) or isinstance(resolution, bool):
            raise TypeError(f"Resolution must be a numeric float, got {type(resolution).__name__}.")
        if math.isnan(resolution) or math.isinf(resolution) or resolution <= 0.0:
            raise ValueError(f"Resolution must be a strictly positive finite number, got {resolution}.")
        self._resolution = float(resolution)

        # Validate and configure origin
        if not isinstance(origin, (tuple, list)) or len(origin) != 2:
            raise TypeError("Origin must be a sequence of two numeric values: (origin_x, origin_y).")
        ox, oy = origin
        if not (isinstance(ox, (int, float)) and isinstance(oy, (int, float))) or isinstance(ox, bool) or isinstance(oy, bool):
            raise TypeError("Origin coordinates must be numeric floats.")
        if math.isnan(ox) or math.isinf(ox) or math.isnan(oy) or math.isinf(oy):
            raise ValueError(f"Origin coordinates must be finite, got ({ox}, {oy}).")
        self._origin: Tuple[float, float] = (float(ox), float(oy))

        # Check for boolean sequence inputs before numpy coercion
        if isinstance(grid, (list, tuple)):
            for row in grid:
                if isinstance(row, (list, tuple)):
                    for val in row:
                        if isinstance(val, bool):
                            raise TypeError("Boolean values are not accepted as occupancy states. Use integer 0 (free) and 1 (occupied).")

        # Validate input grid structure
        try:
            arr = np.array(grid)
        except Exception as err:
            raise TypeError(f"Failed to convert input grid to a NumPy array: {err}") from err

        if arr.dtype == bool:
            raise TypeError("Boolean arrays are not accepted as occupancy grids. Use integer 0 (free) and 1 (occupied).")

        if arr.ndim != 2:
            raise ValueError(f"Grid must be a 2D array, but got array with {arr.ndim} dimensions.")

        height, width = arr.shape
        if height == 0 or width == 0:
            raise ValueError(f"Grid dimensions must be non-empty, but got shape {arr.shape}.")

        # Check for non-numeric/NaN/inf values
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Grid elements must be numeric, got dtype {arr.dtype}.")

        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Grid contains NaN or infinite values.")

        # Strict validation of cell occupancy values: only 0 and 1 are permitted
        valid_mask = (arr == self.FREE) | (arr == self.OCCUPIED)
        if not np.all(valid_mask):
            invalid_examples = arr[~valid_mask].ravel()[:3]
            raise ValueError(
                f"Grid contains invalid occupancy values. All cells must be strictly 0 (free) "
                f"or 1 (occupied). Found invalid values such as: {invalid_examples.tolist()}."
            )

        # Store internal defensive copy
        self._grid: np.ndarray = arr.astype(np.int8, copy=True)
        self._height: int = height
        self._width: int = width

    @classmethod
    def from_dimensions(
        cls,
        height: int,
        width: int,
        initial_value: int = 0,
        resolution: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> OccupancyGrid:
        """Create a uniform occupancy grid with specified dimensions.

        Parameters
        ----------
        height : int
            Number of rows (vertical size). Must be >= 1.
        width : int
            Number of columns (horizontal size). Must be >= 1.
        initial_value : int, optional
            Initial state for all cells: 0 (free) or 1 (occupied). Defaults to 0.
        resolution : float, optional
            Grid resolution in meters per cell. Defaults to 1.0.
        origin : Tuple[float, float], optional
            World coordinates `(origin_x, origin_y)`. Defaults to (0.0, 0.0).
        """
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError(f"Height must be a positive integer >= 1, got {height}.")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError(f"Width must be a positive integer >= 1, got {width}.")
        if isinstance(initial_value, bool) or not isinstance(initial_value, (int, np.integer)):
            raise TypeError(f"initial_value must be an integer (0 or 1), got {type(initial_value).__name__}.")
        if initial_value not in (cls.FREE, cls.OCCUPIED):
            raise ValueError(f"initial_value must be 0 (free) or 1 (occupied), got {initial_value}.")

        arr = np.full((height, width), fill_value=initial_value, dtype=np.int8)
        return cls(grid=arr, resolution=resolution, origin=origin)

    @property
    def height(self) -> int:
        """Return the number of grid rows (vertical extent)."""
        return self._height

    @property
    def width(self) -> int:
        """Return the number of grid columns (horizontal extent)."""
        return self._width

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the grid dimensions as `(height, width)`."""
        return (self._height, self._width)

    @property
    def resolution(self) -> float:
        """Return the physical cell resolution (meters per cell)."""
        return self._resolution

    @property
    def origin(self) -> Tuple[float, float]:
        """Return the grid world origin `(origin_x, origin_y)`."""
        return self._origin

    @property
    def grid(self) -> np.ndarray:
        """Return a defensive copy of the internal 2D occupancy grid."""
        return self._grid.copy()

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check whether discrete grid coordinates `(row, col)` lie within the grid.

        Parameters
        ----------
        row : int
            Row index (0-indexed). Negative indices are considered out of bounds.
        col : int
            Column index (0-indexed). Negative indices are considered out of bounds.

        Returns
        -------
        bool
            True if coordinates are within [0, height) and [0, width), False otherwise.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            return False
        if not isinstance(col, int) or isinstance(col, bool):
            return False
        return 0 <= row < self._height and 0 <= col < self._width

    def get_cell(self, row: int, col: int) -> int:
        """Retrieve the occupancy value at discrete cell `(row, col)`.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        int
            0 if cell is free, 1 if cell is occupied.

        Raises
        ------
        TypeError
            If `row` or `col` is not an integer.
        IndexError
            If `(row, col)` is outside the grid boundaries.
        """
        if not (isinstance(row, int) and isinstance(col, int)) or isinstance(row, bool) or isinstance(col, bool):
            raise TypeError(f"Grid coordinates must be integers, got row={type(row).__name__}, col={type(col).__name__}.")
        if not self.is_in_bounds(row, col):
            raise IndexError(
                f"Coordinates (row={row}, col={col}) out of grid bounds for shape (height={self._height}, width={self._width})."
            )
        return int(self._grid[row, col])

    def set_cell(self, row: int, col: int, value: int) -> None:
        """Set the occupancy value at discrete cell `(row, col)`.

        Validates all coordinates and values before modifying the internal grid array.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        value : int
            Occupancy value: must be strictly integer 0 (free) or 1 (occupied).

        Raises
        ------
        TypeError
            If `row`, `col`, or `value` is of an invalid type.
        IndexError
            If `(row, col)` is outside the grid boundaries.
        ValueError
            If `value` is not 0 or 1.
        """
        if not (isinstance(row, int) and isinstance(col, int)) or isinstance(row, bool) or isinstance(col, bool):
            raise TypeError(f"Grid coordinates must be integers, got row={type(row).__name__}, col={type(col).__name__}.")
        if not self.is_in_bounds(row, col):
            raise IndexError(
                f"Coordinates (row={row}, col={col}) out of grid bounds for shape (height={self._height}, width={self._width})."
            )
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Cell value must be an integer (0 or 1), got {type(value).__name__}.")
        if value not in (self.FREE, self.OCCUPIED):
            raise ValueError(f"Cell value must be strictly 0 (free) or 1 (occupied), got {value}.")
        self._grid[row, col] = np.int8(value)

    def is_occupied(self, row: int, col: int) -> bool:
        """Check whether discrete cell `(row, col)` contains an obstacle.

        Raises
        ------
        TypeError
            If `row` or `col` is not an integer.
        IndexError
            If `(row, col)` is outside the grid boundaries.
        """
        return self.get_cell(row, col) == self.OCCUPIED

    def is_free(self, row: int, col: int) -> bool:
        """Check whether discrete cell `(row, col)` is free drivable space.

        Raises
        ------
        TypeError
            If `row` or `col` is not an integer.
        IndexError
            If `(row, col)` is outside the grid boundaries.
        """
        return self.get_cell(row, col) == self.FREE

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous world coordinates `(x, y)` to discrete grid indices `(row, col)`.

        The conversion uses floor discretization:
            col = floor((x - origin_x) / resolution)
            row = floor((y - origin_y) / resolution)

        Boundary & Negative Coordinate Behavior:
        - Coordinates below origin ($x < origin_x$ or $y < origin_y$) map to negative
          column or row indices, respectively.
        - Coordinates at or beyond outer boundaries map to indices with $col \ge width$
          or $row \ge height$.
        - Resulting indices are not clamped; callers should check `is_in_bounds(row, col)`
          to confirm if the continuous point lies inside the traversable workspace.

        Parameters
        ----------
        x : float
            Continuous horizontal coordinate.
        y : float
            Continuous vertical coordinate.

        Returns
        -------
        Tuple[int, int]
            Discrete grid indices as `(row, col)`.

        Raises
        ------
        TypeError
            If `x` or `y` is non-numeric or boolean.
        ValueError
            If `x` or `y` is NaN or infinite.
        """
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or isinstance(x, bool) or isinstance(y, bool):
            raise TypeError(f"World coordinates must be numeric, got x={type(x).__name__}, y={type(y).__name__}.")
        if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
            raise ValueError(f"World coordinates must be finite, got x={x}, y={y}.")

        col = math.floor((x - self._origin[0]) / self._resolution)
        row = math.floor((y - self._origin[1]) / self._resolution)
        return (int(row), int(col))

    def grid_to_world(self, row: int, col: int, center: bool = True) -> Tuple[float, float]:
        """Convert discrete grid indices `(row, col)` to continuous world coordinates `(x, y)`.

        Parameters
        ----------
        row : int
            Row index (vertical).
        col : int
            Column index (horizontal).
        center : bool, optional
            If True (default), returns the center coordinates of the cell:
                x = origin_x + (col + 0.5) * resolution
                y = origin_y + (row + 0.5) * resolution
            If False, returns the lower corner coordinate:
                x = origin_x + col * resolution
                y = origin_y + row * resolution

        Returns
        -------
        Tuple[float, float]
            Continuous world coordinates `(x, y)`.

        Raises
        ------
        TypeError
            If `row` or `col` is not an integer.
        IndexError
            If `(row, col)` is outside grid boundaries.
        """
        if not (isinstance(row, int) and isinstance(col, int)) or isinstance(row, bool) or isinstance(col, bool):
            raise TypeError(f"Grid coordinates must be integers, got row={type(row).__name__}, col={type(col).__name__}.")
        if not self.is_in_bounds(row, col):
            raise IndexError(
                f"Coordinates (row={row}, col={col}) out of grid bounds for shape (height={self._height}, width={self._width})."
            )

        offset = 0.5 if center else 0.0
        x = self._origin[0] + (col + offset) * self._resolution
        y = self._origin[1] + (row + offset) * self._resolution
        return (float(x), float(y))

    def __repr__(self) -> str:
        return (
            f"OccupancyGrid(shape=({self._height}, {self._width}), "
            f"resolution={self._resolution}, origin={self._origin})"
        )
