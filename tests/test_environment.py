"""Unit tests for the OccupancyGrid abstraction in src.environment.

Compatible with pytest when installed, and executable directly via Python.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Tuple, Type, Union
import numpy as np

# Ensure workspace root is in sys.path when invoked directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pytest
except ImportError:
    # Graceful shim allowing direct execution without pytest installed
    class _PytestFallback:
        class raises:
            def __init__(self, expected_exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> None:
                self.expected_exception = expected_exception

            def __enter__(self) -> _PytestFallback.raises:
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected_exception} was not raised.")
                return issubclass(exc_type, self.expected_exception)

        @staticmethod
        def approx(val: float, rel: float = 1e-6, abs: float = 1e-12) -> Any:
            class _Approx:
                def __init__(self, target: float) -> None:
                    self.target = target

                def __eq__(self, other: Any) -> bool:
                    return math.isclose(self.target, float(other), rel_tol=rel, abs_tol=abs)

                def __repr__(self) -> str:
                    return f"approx({self.target})"

            return _Approx(val)

    pytest = _PytestFallback()  # type: ignore[assignment]

from src.environment import OccupancyGrid


def test_construction_valid_free_grid() -> None:
    """Test 1: Construction from a valid completely free grid."""
    raw = np.zeros((10, 15), dtype=int)
    grid = OccupancyGrid(raw, resolution=0.5, origin=(1.0, 2.0))

    assert grid.height == 10
    assert grid.width == 15
    assert grid.shape == (10, 15)
    assert grid.resolution == 0.5
    assert grid.origin == (1.0, 2.0)
    for r in range(10):
        for c in range(15):
            assert grid.get_cell(r, c) == OccupancyGrid.FREE
            assert grid.is_free(r, c) is True
            assert grid.is_occupied(r, c) is False


def test_construction_with_obstacles() -> None:
    """Test 2: Construction from a grid containing obstacles."""
    raw = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
    ], dtype=int)
    grid = OccupancyGrid(raw)

    assert grid.get_cell(0, 1) == OccupancyGrid.OCCUPIED
    assert grid.is_occupied(0, 1) is True
    assert grid.is_free(0, 1) is False

    assert grid.get_cell(0, 0) == OccupancyGrid.FREE
    assert grid.is_free(0, 0) is True

    assert grid.get_cell(2, 2) == OccupancyGrid.OCCUPIED
    assert grid.is_occupied(2, 2) is True


def test_factory_from_dimensions() -> None:
    """Test 3: Correct dimensions and initialization via factory method."""
    grid_free = OccupancyGrid.from_dimensions(height=8, width=12, initial_value=0, resolution=0.25)
    assert grid_free.height == 8
    assert grid_free.width == 12
    assert grid_free.shape == (8, 12)
    assert grid_free.resolution == 0.25
    assert np.all(grid_free.grid == 0)

    grid_occ = OccupancyGrid.from_dimensions(height=4, width=5, initial_value=1)
    assert np.all(grid_occ.grid == 1)

    with pytest.raises(ValueError):
        OccupancyGrid.from_dimensions(height=0, width=5)
    with pytest.raises(ValueError):
        OccupancyGrid.from_dimensions(height=5, width=-1)
    with pytest.raises(TypeError):
        OccupancyGrid.from_dimensions(height=5, width=5, initial_value=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OccupancyGrid.from_dimensions(height=5, width=5, initial_value=2)


def test_rejection_empty_or_non_2d_inputs() -> None:
    """Test 4: Rejection of empty or non-2D inputs."""
    # 0D scalar
    with pytest.raises(ValueError):
        OccupancyGrid(0)  # type: ignore[arg-type]

    # 1D array
    with pytest.raises(ValueError):
        OccupancyGrid([0, 1, 0])

    # 3D array
    with pytest.raises(ValueError):
        OccupancyGrid(np.zeros((2, 3, 4)))

    # Empty array shape (0, 0)
    with pytest.raises(ValueError):
        OccupancyGrid(np.zeros((0, 0)))

    # Empty array with height=0 or width=0
    with pytest.raises(ValueError):
        OccupancyGrid(np.zeros((0, 5)))
    with pytest.raises(ValueError):
        OccupancyGrid(np.zeros((5, 0)))


def test_rejection_invalid_occupancy_values() -> None:
    """Test 5: Rejection of invalid cell values and deliberate boolean handling."""
    # Negative value
    with pytest.raises(ValueError):
        OccupancyGrid([[0, -1], [1, 0]])

    # Value greater than 1
    with pytest.raises(ValueError):
        OccupancyGrid([[0, 2], [1, 0]])

    # Fractional value
    with pytest.raises(ValueError):
        OccupancyGrid([[0.0, 0.5], [1.0, 0.0]])

    # NaN and Infinite values
    with pytest.raises(ValueError):
        OccupancyGrid([[0.0, np.nan], [1.0, 0.0]])
    with pytest.raises(ValueError):
        OccupancyGrid([[0.0, np.inf], [1.0, 0.0]])

    # Non-numeric string data
    with pytest.raises(TypeError):
        OccupancyGrid([["free", "occupied"], ["free", "free"]])  # type: ignore[list-item]

    # Boolean arrays (both NumPy bool_ array and nested Python bool lists)
    with pytest.raises(TypeError):
        OccupancyGrid(np.array([[True, False], [False, True]]))
    with pytest.raises(TypeError):
        OccupancyGrid([[True, False], [False, True]])
    with pytest.raises(TypeError):
        OccupancyGrid([[0, True], [0, 1]])


def test_protection_against_external_mutation() -> None:
    """Test 6: Internal grid cannot be mutated externally."""
    raw = np.array([[0, 1], [0, 0]], dtype=int)
    grid = OccupancyGrid(raw)

    # Mutating raw input should NOT affect the OccupancyGrid
    raw[0, 0] = 1
    assert grid.get_cell(0, 0) == 0

    # Mutating the array returned by .grid property should NOT affect internal state
    exported = grid.grid
    exported[0, 1] = 0
    assert grid.get_cell(0, 1) == 1

    # Safe internal modification via set_cell
    grid.set_cell(0, 0, 1)
    assert grid.get_cell(0, 0) == 1


def test_invalid_set_cell_does_not_modify_grid() -> None:
    """Test 7: Verify that invalid set_cell() calls fail and do not modify the grid."""
    grid = OccupancyGrid.from_dimensions(height=3, width=3, initial_value=0)

    # Value out of range
    with pytest.raises(ValueError):
        grid.set_cell(1, 1, 99)
    assert grid.get_cell(1, 1) == 0

    # Boolean value rejected
    with pytest.raises(TypeError):
        grid.set_cell(1, 1, True)  # type: ignore[arg-type]
    assert grid.get_cell(1, 1) == 0

    # Non-integer float rejected
    with pytest.raises(TypeError):
        grid.set_cell(1, 1, 0.5)  # type: ignore[arg-type]
    assert grid.get_cell(1, 1) == 0

    # String value rejected
    with pytest.raises(TypeError):
        grid.set_cell(1, 1, "1")  # type: ignore[arg-type]
    assert grid.get_cell(1, 1) == 0

    # Out of bounds indices
    with pytest.raises(IndexError):
        grid.set_cell(-1, 0, 1)
    with pytest.raises(IndexError):
        grid.set_cell(3, 0, 1)

    # Invalid coordinate types
    with pytest.raises(TypeError):
        grid.set_cell("a", 0, 1)  # type: ignore[arg-type]


def test_bounds_checking_and_cell_access() -> None:
    """Test 8: Handling of valid and out-of-bounds cell coordinates."""
    grid = OccupancyGrid.from_dimensions(height=5, width=7)

    # Valid bounds
    assert grid.is_in_bounds(0, 0) is True
    assert grid.is_in_bounds(4, 6) is True
    assert grid.is_in_bounds(2, 3) is True

    # Out of bounds
    assert grid.is_in_bounds(-1, 0) is False
    assert grid.is_in_bounds(0, -1) is False
    assert grid.is_in_bounds(5, 0) is False
    assert grid.is_in_bounds(0, 7) is False
    assert grid.is_in_bounds(10, 10) is False

    # Non-integer coordinates do not crash is_in_bounds
    assert grid.is_in_bounds(1.5, 2) is False  # type: ignore[arg-type]
    assert grid.is_in_bounds(True, 2) is False  # type: ignore[arg-type]

    # Non-integer coordinates raise TypeError on accessor methods
    with pytest.raises(TypeError):
        grid.get_cell("a", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        grid.is_occupied(0, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        grid.is_free(True, 0)  # type: ignore[arg-type]

    # IndexError raised on access out-of-bounds
    with pytest.raises(IndexError):
        grid.get_cell(-1, 0)
    with pytest.raises(IndexError):
        grid.get_cell(5, 0)
    with pytest.raises(IndexError):
        grid.get_cell(0, 7)
    with pytest.raises(IndexError):
        grid.is_occupied(10, 2)
    with pytest.raises(IndexError):
        grid.is_free(-2, 2)


def test_invalid_resolution_and_origin_handling() -> None:
    """Test 9: Rejection of invalid resolutions and origin specifications."""
    raw = np.zeros((3, 3), dtype=int)

    # Non-positive or infinite resolution
    with pytest.raises(ValueError):
        OccupancyGrid(raw, resolution=0.0)
    with pytest.raises(ValueError):
        OccupancyGrid(raw, resolution=-1.0)
    with pytest.raises(ValueError):
        OccupancyGrid(raw, resolution=float("nan"))
    with pytest.raises(ValueError):
        OccupancyGrid(raw, resolution=float("inf"))
    with pytest.raises(TypeError):
        OccupancyGrid(raw, resolution="1.0")  # type: ignore[arg-type]

    # Invalid origins
    with pytest.raises(TypeError):
        OccupancyGrid(raw, origin=(1.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OccupancyGrid(raw, origin=(float("nan"), 0.0))
    with pytest.raises(ValueError):
        OccupancyGrid(raw, origin=(0.0, float("inf")))
    with pytest.raises(ValueError):
        OccupancyGrid(raw, origin=(float("-inf"), 0.0))


def test_round_trip_world_and_grid_coordinates() -> None:
    """Test 10: World to grid and grid to world coordinate conversions."""
    resolution = 0.5
    origin = (10.0, 20.0)
    grid = OccupancyGrid.from_dimensions(height=6, width=8, resolution=resolution, origin=origin)

    # Test round trip from every cell center to world and back to grid
    for r in range(grid.height):
        for c in range(grid.width):
            wx, wy = grid.grid_to_world(r, c, center=True)
            expected_x = origin[0] + (c + 0.5) * resolution
            expected_y = origin[1] + (r + 0.5) * resolution
            assert pytest.approx(wx) == expected_x
            assert pytest.approx(wy) == expected_y

            # Invert back to grid
            round_r, round_c = grid.world_to_grid(wx, wy)
            assert (round_r, round_c) == (r, c)

    # Test corner conversion
    wx_corner, wy_corner = grid.grid_to_world(0, 0, center=False)
    assert pytest.approx(wx_corner) == origin[0]
    assert pytest.approx(wy_corner) == origin[1]


def test_world_coordinates_boundary_and_outer_limits() -> None:
    """Test 11: Boundaries just below origin, at outer boundary, and non-finite coordinates."""
    resolution = 0.5
    origin = (10.0, 20.0)
    h, w = 4, 6
    grid = OccupancyGrid.from_dimensions(height=h, width=w, resolution=resolution, origin=origin)

    # First cell center
    wx0, wy0 = grid.grid_to_world(0, 0, center=True)
    assert grid.world_to_grid(wx0, wy0) == (0, 0)
    assert grid.is_in_bounds(*grid.world_to_grid(wx0, wy0)) is True

    # Last cell center
    wxh, wyh = grid.grid_to_world(h - 1, w - 1, center=True)
    assert grid.world_to_grid(wxh, wyh) == (h - 1, w - 1)
    assert grid.is_in_bounds(*grid.world_to_grid(wxh, wyh)) is True

    # Coordinates just below origin (negative offset)
    r_below, c_below = grid.world_to_grid(origin[0] - 1e-6, origin[1] - 1e-6)
    assert r_below == -1 and c_below == -1
    assert grid.is_in_bounds(r_below, c_below) is False

    # Coordinates exactly at origin lower-left boundary
    r_orig, c_orig = grid.world_to_grid(origin[0], origin[1])
    assert r_orig == 0 and c_orig == 0
    assert grid.is_in_bounds(r_orig, c_orig) is True

    # Coordinates at the far outer boundary
    outer_x = origin[0] + w * resolution
    outer_y = origin[1] + h * resolution
    r_outer, c_outer = grid.world_to_grid(outer_x, outer_y)
    assert r_outer == h and c_outer == w
    assert grid.is_in_bounds(r_outer, c_outer) is False

    # Coordinates just inside the far outer boundary
    r_inside, c_inside = grid.world_to_grid(outer_x - 1e-6, outer_y - 1e-6)
    assert r_inside == h - 1 and c_inside == w - 1
    assert grid.is_in_bounds(r_inside, c_inside) is True

    # Non-finite coordinates
    with pytest.raises(ValueError):
        grid.world_to_grid(float("nan"), origin[1])
    with pytest.raises(ValueError):
        grid.world_to_grid(origin[0], float("inf"))
    with pytest.raises(ValueError):
        grid.world_to_grid(float("-inf"), origin[1])

    # Non-numeric coordinate types
    with pytest.raises(TypeError):
        grid.world_to_grid("10.0", origin[1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        grid.world_to_grid(True, origin[1])  # type: ignore[arg-type]


def test_boundary_behavior_first_and_last_cells() -> None:
    """Test 12: Explicit boundary conditions at first and last cells."""
    h, w = 4, 6
    grid = OccupancyGrid.from_dimensions(height=h, width=w, resolution=1.0, origin=(0.0, 0.0))

    # First cell (0, 0)
    assert grid.is_in_bounds(0, 0) is True
    grid.set_cell(0, 0, 1)
    assert grid.is_occupied(0, 0) is True

    # Last cell (h-1, w-1)
    assert grid.is_in_bounds(h - 1, w - 1) is True
    grid.set_cell(h - 1, w - 1, 1)
    assert grid.is_occupied(h - 1, w - 1) is True

    # Just outside lower bounds
    assert grid.is_in_bounds(-1, 0) is False
    assert grid.is_in_bounds(0, -1) is False

    # Just outside upper bounds
    assert grid.is_in_bounds(h, w - 1) is False
    assert grid.is_in_bounds(h - 1, w) is False
    assert grid.is_in_bounds(h, w) is False


def run_all_tests() -> None:
    """Direct execution runner for environments where pytest CLI is not installed."""
    tests = [
        test_construction_valid_free_grid,
        test_construction_with_obstacles,
        test_factory_from_dimensions,
        test_rejection_empty_or_non_2d_inputs,
        test_rejection_invalid_occupancy_values,
        test_protection_against_external_mutation,
        test_invalid_set_cell_does_not_modify_grid,
        test_bounds_checking_and_cell_access,
        test_invalid_resolution_and_origin_handling,
        test_round_trip_world_and_grid_coordinates,
        test_world_coordinates_boundary_and_outer_limits,
        test_boundary_behavior_first_and_last_cells,
    ]
    print(f"Executing {len(tests)} test functions directly...")
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  [PASS] {t.__name__}")
    print(f"All {passed}/{len(tests)} tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
