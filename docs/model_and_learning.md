# Model & Learning Methodology

This document clarifies the decision-making approach used in the **AI-Based Autonomous Navigation System**.

---

## Important Clarification: No Machine Learning

The current system is **entirely algorithmic** - it does not use any machine learning, neural networks, or training loops. All navigation decisions are made by classical algorithms:

| Decision | Algorithm | ML Equivalent |
| :--- | :--- | :--- |
| Path finding | A* search with Manhattan heuristic | Replaces learned navigation policies |
| Steering control | Proportional heading controller | Replaces neural network controllers |
| Environment sensing | Known occupancy grid (ground truth) | Replaces perception models (YOLO, etc.) |
| Collision avoidance | Geometric circle-rectangle distance | Replaces learned collision predictors |

---

## How the System "Decides" - Deterministic Pipeline

### Step 1: Environment Knowledge (No Perception Model)
The system has **complete ground-truth knowledge** of the environment via the occupancy grid. There is no camera, LiDAR, or sensor simulation. The grid is directly provided as a NumPy array.

### Step 2: Path Planning (A* Search, Not Learned)
A* search finds the optimal shortest path by:
1. Maintaining a priority queue ordered by $f(n) = g(n) + h(n)$
2. Expanding nodes greedily toward the goal
3. Guaranteeing optimality via admissible Manhattan heuristic

**This is not reinforcement learning** - A* is a classical graph search algorithm that exhaustively explores the space and mathematically guarantees the shortest path.

### Step 3: Control (Proportional Controller, Not a Neural Network)
The heading controller uses a simple proportional control law:
$$\omega = K_p \cdot e_\theta$$

**This is not a learned policy** - it's a hand-tuned feedback controller with three parameters (`angular_gain`, `angular_speed_limit`, `heading_error_threshold`).

### Step 4: Safety (Geometric Collision Check, Not Learned)
Collision checking uses exact circle-rectangle distance computation:
$$d = \sqrt{(x - x_{\text{clamp}})^2 + (y - y_{\text{clamp}})^2}$$

**This is not a learned safety model** - it's analytical geometry.

---

## Evaluation Metrics

The system evaluates navigation performance using deterministic metrics:

| Metric | Formula | What It Measures |
| :--- | :--- | :--- |
| **Success Rate** | `status == SUCCESS` | Whether the robot reaches the goal |
| **Path Length** | $L = \sum_{k=1}^N \|p_k - p_{k-1}\|_2$ | Total distance traversed |
| **Step Count** | Number of simulation steps | Time efficiency |
| **Termination Type** | SUCCESS / COLLISION / TIMEOUT / NO_PATH | Failure mode |

---

## Future Work: Potential ML Integration

If machine learning were added in future versions, the following approaches could augment the current system:

| Paradigm | How It Would Integrate |
| :--- | :--- |
| **Imitation Learning** | Train a neural network to mimic the A* + P-controller pipeline from simulation data |
| **Reinforcement Learning (DRL)** | Replace the controller with a learned policy trained via reward shaping |
| **Learned Perception** | Replace the known grid with a learned occupancy predictor from sensor data |
| **Neural Motion Planning** | Replace A* with a learned planner (e.g., MPNet, VIN) |

> **Note:** These are future possibilities, not current features. The current system is fully deterministic and does not require training data, GPU computation, or any ML framework.
