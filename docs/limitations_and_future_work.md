# Limitations & Future Work

This document outlines the design boundaries, deliberate assumptions, and planned evolutionary roadmap for the **AI-Based Autonomous Navigation System**.

---

## 1. Deliberate Scope Boundaries & Limitations

To ensure an interpretable, testable, and beginner-friendly portfolio project, the initial version adopts several deliberate simplifications:

1. **Known Occupancy Grid Perception**:
   - Perception operates strictly as a known-map adapter reading from the ground-truth grid.
   - Camera vision, LiDAR point cloud processing, neural network detectors, and SLAM are excluded from this initial version.

2. **Kinematic Unicycle Model**:
   - The robot is modeled using idealized unicycle kinematics ($v, \omega$).
   - Dynamic effects such as wheel slip, inertia, tire friction, motor torque saturation, and center-of-mass shifts are omitted.

3. **Heading Proportional Control**:
   - Trajectory following utilizes a heading-based proportional controller.
   - Advanced multi-variable controllers (e.g., Model Predictive Control, LQR, or full PID) are deferred.

4. **Static Obstacles**:
   - The 2D grid contains only static obstacles; dynamic pedestrians or moving vehicles are not simulated in the initial phase.

5. **No Middleware / Heavy Simulator Dependencies**:
   - ROS 2, Gazebo, Webots, and CARLA are intentionally excluded to keep the repository lightweight, portable, and easily verifiable across any standard Python installation.

---

## 2. Planned Future Extensions

| Phase / Milestone | Proposed Feature | Technical Benefit |
| :--- | :--- | :--- |
| **Phase 2.x** | **Dynamic Window Approach (DWA)** | Local reactive collision avoidance for unexpected obstacles. |
| **Phase 3.0** | **Continuous Path Smoothing** | Replaces jagged 8-connected grid lines with Dubins paths or B-splines. |
| **Phase 3.5** | **Stanley / Pure Pursuit Controller** | Cross-track error minimization for higher tracking accuracy at speed. |
| **Phase 4.0** | **Noisy Sensor & EKF Localization** | Adds Gaussian sensor noise and Extended Kalman Filter state estimation. |
| **Phase 5.0** | **ROS 2 Nav2 Node Adapter** | Wraps the Python modules into standard ROS 2 publishers, subscribers, and action servers. |
