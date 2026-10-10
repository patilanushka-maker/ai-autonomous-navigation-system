# Usage & Simulation Guide

This document describes how to configure, execute, and inspect simulations within the **AI-Based Autonomous Navigation System**.

> **Status Notice**: The simulator implementation is currently in the scaffolding and foundation phase. The workflows described below outline the operational workflow planned for the implementation phase.

---

## 1. Configuration (`configs/default_config.json`)

Simulation parameters are centralized in `configs/default_config.json`:

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

### Parameter Reference
- `grid_width`, `grid_height`: Dimensions of the 2D occupancy grid in discrete cells.
- `robot_radius`: Effective circular collision radius in meters.
- `linear_speed`: Nominal forward linear velocity ($m/s$).
- `angular_speed_limit`: Maximum allowable turning velocity ($\text{rad}/s$).
- `simulation_dt`: Discrete time increment per simulation step (seconds).
- `goal_tolerance`: Radial distance threshold to register successful goal arrival (meters).
- `max_simulation_steps`: Maximum allowed steps before terminating runaway simulations.
- `random_seed`: Deterministic seed for reproducible obstacle generation and noise.

---

## 2. Planned Execution Workflow

### Running the Simulator (Phase 2)
```bash
python main.py
```
Or with custom configuration overrides:
```bash
python main.py --config configs/custom_config.json
```

---

## 3. Output Inspection

Simulation runs will output results into the `outputs/` directory:
- `outputs/figures/`: Saved trajectory plots showing the grid map, planned A* path, and robot trail.
- `outputs/logs/`: Simulation execution logs with timestamps and step telemetry.
- `outputs/metrics/`: JSON performance summaries detailing path efficiency, clearance, and success flags.
