# Maps Directory

This directory is intended to store 2D occupancy grid map definitions for the autonomous navigation simulator.

## Intended Format & Representation
- **Grid Representation**: 2D discrete arrays (e.g., CSV, JSON, or NumPy arrays) where each cell indicates free space (`0`) or obstacle (`1`).
- **Coordinate Conventions**:
  - **Grid Coordinates**: Discrete index pairs `(row, col)` or `(gx, gy)`.
  - **World Coordinates**: Continuous coordinates `(x, y)` in meters, mapped to grid cells via a designated resolution parameter (meters per cell).
- **Status**: No mock or static map files are pre-generated in this initialization phase. Specific maps will be added or generated programmatically during the environment implementation phase.
