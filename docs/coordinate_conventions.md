# Coordinate Conventions

This document is a critical reference for understanding how coordinates are handled across all modules. Mismatched conventions are a common source of navigation bugs.

---

## 1. Two Coordinate Systems

The system uses two distinct coordinate systems:

| System | Notation | Units | Used By |
| :--- | :--- | :--- | :--- |
| **Grid Coordinates** | `(row, col)` | Integer cell indices | `OccupancyGrid`, `AStarPlanner` |
| **World Coordinates** | `(x, y)` | Meters (continuous) | `Robot`, `Controller`, `CollisionChecker`, `Simulator` |

---

## 2. Grid Coordinate System

```
        col 0   col 1   col 2   col 3
       ┌───────┬───────┬───────┬───────┐
row 0  │ (0,0) │ (0,1) │ (0,2) │ (0,3) │
       ├───────┼───────┼───────┼───────┤
row 1  │ (1,0) │ (1,1) │ (1,2) │ (1,3) │
       ├───────┼───────┼───────┼───────┤
row 2  │ (2,0) │ (2,1) │ (2,2) │ (2,3) │
       └───────┴───────┴───────┴───────┘
```

- **Row** is the vertical index (0 ≤ row < height)
- **Column** is the horizontal index (0 ≤ col < width)
- Internal array shape: `(height, width)` matching NumPy `(axis 0, axis 1)`
- `row` increases downward in memory, but **upward in world y-coordinates**

---

## 3. World Coordinate System

```
     y ↑
       │
  y=3  │  ·───────·───────·───────·
       │  │(r2,c0)│(r2,c1)│(r2,c2)│
  y=2  │  ·───────·───────·───────·
       │  │(r1,c0)│(r1,c1)│(r1,c2)│
  y=1  │  ·───────·───────·───────·
       │  │(r0,c0)│(r0,c1)│(r0,c2)│
  y=0  │  ·───────·───────·───────·
       └──────────────────────────→ x
       x=0       x=1      x=2     x=3
```

- **x-axis**: Horizontal, increasing to the right, aligned with grid columns
- **y-axis**: Vertical, increasing upward, aligned with grid rows
- **Origin** `(origin_x, origin_y)` is the world coordinate of the lower-left corner of cell `(0, 0)`
- With default origin `(0, 0)`: row 0 is at the bottom (y ≈ 0), row H-1 is at the top

---

## 4. Conversion Formulas (as implemented)

### World → Grid (floor discretization)
```python
col = floor((x - origin_x) / resolution)
row = floor((y - origin_y) / resolution)
```

### Grid → World (cell center)
```python
x = origin_x + (col + 0.5) * resolution
y = origin_y + (row + 0.5) * resolution
```

### Grid → World (cell lower corner)
```python
x = origin_x + col * resolution
y = origin_y + row * resolution
```

---

## 5. Heading Convention

The robot heading `theta` uses the standard mathematical convention:

| `theta` | Direction | Meaning |
| :--- | :--- | :--- |
| `0` | → (positive x) | Facing right |
| `+π/2` | ↑ (positive y) | Facing up |
| `±π` | ← (negative x) | Facing left |
| `−π/2` | ↓ (negative y) | Facing down |

- Angles are in **radians**
- All headings are normalized to **`[-π, π)`**
- Positive rotation is **counter-clockwise**

---

## 6. Per-Module Coordinate Usage

| Module | Coordinates Used | Notes |
| :--- | :--- | :--- |
| `OccupancyGrid` | Grid `(row, col)` and World `(x, y)` | Provides bidirectional conversion |
| `AStarPlanner` | Grid `(row, col)` only | Plans on discrete grid |
| `Robot` | World `(x, y, θ)` only | Continuous Euler integration |
| `Controller` | World `(x, y, θ)` and World `(x, y)` target | No grid awareness |
| `CollisionChecker` | World `(x, y)` input, internally converts to grid | Uses grid geometry for AABB |
| `Simulator` | World `(x, y)` externally, bridges to grid for planning | Converts A* grid path to world waypoints |

---

## 7. Critical Integration Point

The `NavigationSimulator.run()` method is the bridge between grid and world:

```
User provides: start=(x, y), goal=(x, y) in WORLD coordinates
  ↓
Simulator converts to grid: world_to_grid(x, y) → (row, col)
  ↓
A* plans in grid: plan(start_grid, goal_grid) → [(row, col), ...]
  ↓
Simulator converts back: grid_to_world(row, col) → (x, y) for each waypoint
  ↓
Controller steers robot in WORLD coordinates toward (x, y) waypoints
  ↓
Collision checker validates in WORLD coordinates using grid geometry
```
