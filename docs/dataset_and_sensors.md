# Dataset & Environment Specification

This document describes how the environment data is represented and constructed in the **AI-Based Autonomous Navigation System**.

---

## Important Clarification: No External Dataset

The current system does **not** use any external dataset. The environment is a **programmatically constructed 2D occupancy grid** - a NumPy array of 0s and 1s. There are no images, point clouds, driving logs, or sensor recordings.

---

## 1. Environment Representation

### Occupancy Grid Format

The environment is a 2D integer matrix:

```python
import numpy as np

# Example: 5x7 grid with an L-shaped obstacle
grid_data = np.array([
    [0, 0, 0, 0, 0, 0, 0],   # Row 0 (y ≈ 0.5)
    [0, 1, 1, 1, 0, 0, 0],   # Row 1 - obstacle wall
    [0, 0, 0, 1, 0, 0, 0],   # Row 2
    [0, 0, 0, 1, 0, 0, 0],   # Row 3
    [0, 0, 0, 0, 0, 0, 0],   # Row 4 (y ≈ 4.5)
])
```

| Cell Value | Meaning | Color Convention |
| :--- | :--- | :--- |
| `0` | Free space (robot can traverse) | White |
| `1` | Obstacle (robot cannot enter) | Black |

### Grid Properties

| Property | Type | Example | Meaning |
| :--- | :--- | :--- | :--- |
| `height` | `int` | `5` | Number of rows (vertical extent) |
| `width` | `int` | `7` | Number of columns (horizontal extent) |
| `resolution` | `float` | `1.0` | Meters per grid cell |
| `origin` | `(float, float)` | `(0.0, 0.0)` | World coords of cell (0,0) lower corner |

### Physical Dimensions
- Grid physical width: `width × resolution` meters
- Grid physical height: `height × resolution` meters
- Total cells: `height × width`

---

## 2. Grid Construction Methods

### Method A: From a NumPy Array
```python
from src.environment import OccupancyGrid
import numpy as np

data = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
])
grid = OccupancyGrid(data, resolution=1.0, origin=(0.0, 0.0))
```

### Method B: From Python Lists
```python
grid = OccupancyGrid(
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    resolution=0.5
)
```

### Method C: Factory Method (Uniform Grid)
```python
# Create a 10x10 all-free grid
grid = OccupancyGrid.from_dimensions(height=10, width=10, initial_value=0)

# Create a 5x5 all-obstacles grid
grid = OccupancyGrid.from_dimensions(height=5, width=5, initial_value=1)
```

### Method D: Programmatic Obstacle Placement
```python
grid = OccupancyGrid.from_dimensions(20, 20)
# Add a horizontal wall
for col in range(5, 15):
    grid.set_cell(10, col, 1)
# Add a vertical wall
for row in range(3, 12):
    grid.set_cell(row, 7, 1)
```

---

## 3. Default Configuration

The file `configs/default_config.json` stores default simulation parameters:

```json
{
  "grid_width": 20,
  "grid_height": 15,
  "robot_radius": 0.2,
  "linear_speed": 0.5,
  "angular_speed_limit": 1.0,
  "simulation_dt": 0.1,
  "goal_tolerance": 0.25,
  "max_simulation_steps": 2000,
  "random_seed": 42
}
```

---

## 4. Input Validation Rules

The `OccupancyGrid` enforces strict input validation:

| Rule | Enforcement |
| :--- | :--- |
| Grid must be 2D | `ValueError` if `ndim ≠ 2` |
| Grid must be non-empty | `ValueError` if any dimension is 0 |
| Values must be only 0 or 1 | `ValueError` for any other value |
| No boolean arrays | `TypeError` for `dtype=bool` |
| No NaN or Inf values | `ValueError` |
| No string values | `TypeError` |
| Resolution must be > 0 | `ValueError` |
| Origin must be finite | `ValueError` |

---

## 5. Future: External Dataset Integration

A feasibility assessment for integrating external navigation datasets (BARN, HM3D, OpenLORIS-Scene) was conducted. Key findings:

- **BARN (Benchmark for Autonomous Robot Navigation)**: Most compatible - provides 2D occupancy grids in 300×300 format that could be loaded directly.
- **HM3D**: 3D mesh dataset - requires significant preprocessing (slicing to 2D, rasterization) and is overkill for a 2D simulator.
- **OpenLORIS-Scene**: RGB-D SLAM dataset - requires full perception pipeline, incompatible with current known-map approach.

> Integration of BARN grids is the recommended next step if external data is desired. See `docs/limitations_and_future_work.md` for details.
