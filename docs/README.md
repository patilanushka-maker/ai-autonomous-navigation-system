# Technical Documentation - AI-Based Autonomous Navigation System

Welcome to the technical documentation repository for the **AI-Based Autonomous Navigation System**. This folder is the single source of truth for all architecture, data flow, module details, algorithmic formulations, and development history.

---

## Documentation Index

| Document | What It Covers |
| :--- | :--- |
| [architecture.md](architecture.md) | High-level system architecture, component hierarchy, perception-planning-control pipeline, and explicit scope boundaries. |
| [data_flow.md](data_flow.md) | End-to-end data flow from grid input through A* planning, control, kinematics, collision checking, and simulation result output. |
| [modules.md](modules.md) | Detailed specification of every `src/` module: classes, public API signatures, parameters, return types, and exception behavior. |
| [algorithms.md](algorithms.md) | Mathematical formulations: occupancy grid, A* search, unicycle kinematics, heading proportional control, circle-rectangle collision, and simulation loop. |
| [model_and_learning.md](model_and_learning.md) | Clarification that the current system is algorithmic (no ML). Documents the deterministic decision-making pipeline and evaluation metrics. |
| [dataset_and_sensors.md](dataset_and_sensors.md) | Grid-based environment representation, programmatic grid construction, and no external dataset dependency. |
| [coordinate_conventions.md](coordinate_conventions.md) | Critical reference for grid ↔ world coordinate mapping, heading orientation, and how each module uses coordinates. |
| [execution_log.md](execution_log.md) | Chronological development log tracking every implementation phase, milestone, and technical decision. |
| [installation.md](installation.md) | Setup instructions: Python version, virtual environment, dependency installation, and running the project. |
| [usage.md](usage.md) | Configuration parameters, running simulations, interpreting results, and running the test suite. |
| [limitations_and_future_work.md](limitations_and_future_work.md) | Current scope boundaries and planned future enhancements. |
| [project_report.md](project_report.md) | Student portfolio summary for academic submission. |

---

## 🔧 Technology Stack (Actual Implementation)

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.10+ |
| Numerical Computing | NumPy |
| Visualization | Matplotlib (planned) |
| Testing | pytest |
| Configuration | JSON (`configs/default_config.json`) |
| Build/Packaging | `pyproject.toml` |

> **Note:** This project does **not** use ROS, Gazebo, PyTorch, TensorFlow, OpenCV, SLAM, or any deep learning framework. All navigation logic is purely algorithmic.
