# Simulation Supporting Package

This directory provides a supporting package and staging area for simulation interfaces, batch runners, and scenario integration.

## Distinction from `src/simulator.py`
- **`src/simulator.py`**: Houses the core discrete-time step loop, robot state updates, collision checks, and termination condition logic.
- **`simulation/`**: Reserved for higher-level simulation utilities, external harness bindings, or batch scenario execution scripts, without duplicating the core engine logic.

## Current Status
Currently in initial foundation phase. No simulation execution code is added here yet.
