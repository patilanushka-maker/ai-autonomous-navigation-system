# System Modules Specification

This document provides a detailed breakdown of every implemented module in `src/`, including class APIs, constructor parameters, public methods, return types, and exception behavior.

---

## Module File Map

```
src/
├── __init__.py             # Package declaration, version "0.1.0"
├── environment.py          # OccupancyGrid - 2D grid environment
├── path_planner.py         # AStarPlanner - A* search algorithm
├── robot.py                # Robot, RobotState - unicycle kinematics
├── controller.py           # HeadingProportionalController - P-controller
├── collision_checker.py    # CollisionChecker - circle-vs-grid collision
├── simulator.py            # NavigationSimulator, SimulationResult - main loop
├── config_loader.py        # Configuration loading utility (placeholder)
├── visualization.py        # Matplotlib visualizer (placeholder)
├── perception.py           # Known-map perception adapter (placeholder)
├── metrics.py              # Performance metrics collector (placeholder)
└── main.py                 # CLI entry point
```

---

## 1. `environment.py` - OccupancyGrid

**Class**: `OccupancyGrid`

Represents a 2D discrete occupancy grid where `0` = free and `1` = obstacle.

### Constructor

```python
OccupancyGrid(grid, resolution=1.0, origin=(0.0, 0.0))
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `grid` | `np.ndarray` or `List[List[int]]` | - | 2D array of 0s and 1s |
| `resolution` | `float` | `1.0` | Meters per cell (must be > 0) |
| `origin` | `Tuple[float, float]` | `(0.0, 0.0)` | World coords of cell (0,0) lower corner |

### Factory

```python
OccupancyGrid.from_dimensions(height, width, initial_value=0, resolution=1.0, origin=(0.0, 0.0))
```

### Properties
- `height: int` - Number of rows
- `width: int` - Number of columns
- `shape: (int, int)` - `(height, width)`
- `resolution: float` - Meters per cell
- `origin: (float, float)` - World origin
- `grid: np.ndarray` - Defensive copy of internal grid

### Key Methods
| Method | Returns | Raises |
| :--- | :--- | :--- |
| `is_in_bounds(row, col)` | `bool` | - |
| `get_cell(row, col)` | `int` (0 or 1) | `TypeError`, `IndexError` |
| `set_cell(row, col, value)` | `None` | `TypeError`, `IndexError`, `ValueError` |
| `is_occupied(row, col)` | `bool` | `TypeError`, `IndexError` |
| `is_free(row, col)` | `bool` | `TypeError`, `IndexError` |
| `world_to_grid(x, y)` | `(row, col)` | `TypeError`, `ValueError` |
| `grid_to_world(row, col, center=True)` | `(x, y)` | `TypeError`, `IndexError` |

---

## 2. `path_planner.py` - AStarPlanner

**Class**: `AStarPlanner`

A* search on a 4-connected grid with Manhattan distance heuristic.

### Constructor

```python
AStarPlanner(grid: OccupancyGrid)
```

### Key Methods
| Method | Parameters | Returns | Raises |
| :--- | :--- | :--- | :--- |
| `plan(start, goal)` | `(row, col)` tuples | `List[(row, col)]` or `None` | `TypeError`, `ValueError` |
| `heuristic(r1, c1, r2, c2)` | 4 ints | `int` (Manhattan distance) | - |

- Returns `None` if no path exists.
- Start and goal must be in-bounds and on free cells.
- 4-connected movement only: Up `(-1,0)`, Down `(+1,0)`, Left `(0,-1)`, Right `(0,+1)`.

---

## 3. `robot.py` - Robot & RobotState

### `normalize_angle(theta: float) → float`
Normalizes angle to `[-π, π)`.

### `RobotState` (frozen dataclass)
Immutable snapshot: `x, y, theta, v, omega`. Property: `pose → (x, y, theta)`.

### `Robot` class

```python
Robot(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)
```

| Method | Parameters | Description |
| :--- | :--- | :--- |
| `step(dt, v=None, omega=None)` | `dt ≥ 0` | Forward Euler integration |
| `set_pose(x, y, theta)` | floats | Update position and heading |
| `set_velocity(v, omega)` | floats | Update velocity commands |

Properties: `x`, `y`, `theta`, `v`, `omega`, `pose → (x, y, θ)`, `state → RobotState`.

---

## 4. `controller.py` - HeadingProportionalController

```python
HeadingProportionalController(
    linear_speed=0.5,
    angular_gain=2.0,
    angular_speed_limit=1.0,
    goal_tolerance=0.25,
    heading_error_threshold=π/2
)
```

### Key Methods
| Method | Parameters | Returns |
| :--- | :--- | :--- |
| `compute_command(current_pose, target_waypoint)` | pose `(x,y,θ)` + target `(x,y)` | `(v_cmd, ω_cmd)` |
| `is_waypoint_reached(current_pose, target_waypoint)` | pose + target | `bool` |

- Accepts pose as `(x, y, θ)` tuple or object with `x, y, theta` attributes.
- Forward speed is suppressed when heading error exceeds threshold.
- Velocity scales down near waypoint (slowdown distance = max(2·tolerance, 1.0)).

---

## 5. `collision_checker.py` - CollisionChecker

```python
CollisionChecker(grid: OccupancyGrid, robot_radius=0.2)
```

### Key Method

```python
is_pose_collision_free(x, y=None) → bool
```

- Accepts `(x, y)` as separate floats, a tuple, or an object with `x, y` attributes.
- Checks map boundary containment (footprint circle fully inside grid).
- Checks circle-vs-rectangle distance for each occupied cell in the footprint's AABB.
- Uses conservative touching policy: `distance ≤ radius + 1e-9` counts as collision.

---

## 6. `simulator.py` - NavigationSimulator

```python
NavigationSimulator(
    grid, robot_radius=0.2, dt=0.1, max_steps=2000,
    goal_tolerance=0.25, waypoint_tolerance=None,
    linear_speed=0.5, angular_gain=2.0, angular_speed_limit=1.0
)
```

### Main Method

```python
run(start: (float, float), goal: (float, float), initial_heading=0.0) → SimulationResult
```

### SimulationResult Fields
| Field | Type |
| :--- | :--- |
| `status` | `SimulationStatus` (SUCCESS, COLLISION, TIMEOUT, NO_PATH) |
| `steps` | `int` |
| `trajectory` | `List[RobotState]` |
| `planned_path_grid` | `List[(int,int)]` or `None` |
| `planned_path_world` | `List[(float,float)]` or `None` |
| `start_world` | `(float, float)` |
| `goal_world` | `(float, float)` |
| `path_length` | `float` |
| `message` | `str` |
