"""Heading-based proportional controller module for the 2D Autonomous Navigation Simulator.

This module provides the `HeadingProportionalController` class that translates
a robot's continuous world pose $(x, y, \\theta)$ and a target waypoint $(x_{\\text{target}}, y_{\\text{target}})$
into linear velocity $v$ and angular velocity $\\omega$ control commands.

Control Law & Equations:
------------------------
1. Position Error:
   $$dx = x_{\\text{target}} - x, \\quad dy = y_{\\text{target}} - y$$
2. Euclidean Distance:
   $$d = \\sqrt{dx^2 + dy^2}$$
3. Waypoint Arrival Check:
   If $d \\le \\text{goal\\_tolerance}$:
   $$v = 0.0, \\quad \\omega = 0.0$$
4. Desired Heading:
   $$\\theta_{\\text{desired}} = \\text{atan2}(dy, dx)$$
5. Wrapped Heading Error (in $[-\\pi, \\pi)$):
   $$e_\\theta = \\text{normalize\\_angle}(\\theta_{\\text{desired}} - \\theta)$$
6. Proportional Angular Velocity:
   $$\\omega_{\\text{cmd}} = \\text{clamp}(K_p \\cdot e_\\theta, -\\omega_{\\max}, \\omega_{\\max})$$
7. Linear Velocity Scaling:
   - If $|e_\\theta| \\ge \\text{heading\\_error\\_threshold}$ (default $\\pi/2$), the robot faces
     away from the waypoint; forward motion is suppressed ($v = 0.0$) while the robot rotates in place.
   - Otherwise, forward velocity scales down as heading error increases and as distance approaches
     the target:
     $$v_{\\text{cmd}} = v_{\\max} \\cdot \\cos(e_\\theta) \\cdot \\min\\left(1.0, \\frac{d}{d_{\\text{slowdown}}}\\right)$$
     where $d_{\\text{slowdown}} = \\max(2.0 \\cdot \\text{goal\\_tolerance}, 1.0)$.
"""

from __future__ import annotations

import math
from typing import Any, Tuple, Union

from src.robot import normalize_angle

__all__ = ["HeadingProportionalController", "PurePursuitController"]


class HeadingProportionalController:
    """Proportional heading controller for waypoint tracking.

    Parameters
    ----------
    linear_speed : float, optional
        Maximum forward linear speed in m/s. Must be strictly positive. Defaults to 0.5.
    angular_gain : float, optional
        Proportional gain $K_p$ for heading steering. Must be strictly positive. Defaults to 2.0.
    angular_speed_limit : float, optional
        Maximum allowable absolute turning rate $\\omega_{\\max}$ in rad/s. Must be strictly positive.
        Defaults to 1.0.
    goal_tolerance : float, optional
        Radial distance threshold in meters to declare waypoint arrival. Must be non-negative.
        Defaults to 0.25.
    heading_error_threshold : float, optional
        Maximum absolute heading error (radians) permitting forward motion. If $|e_\\theta|$ exceeds
        this threshold, linear velocity is set to 0.0. Must be in $(0.0, \\pi]$. Defaults to $\\pi/2$.

    Raises
    ------
    TypeError
        If any constructor parameter is non-numeric or boolean.
    ValueError
        If any parameter is non-finite, or fails positivity / range requirements.
    """

    def __init__(
        self,
        linear_speed: float = 0.5,
        angular_gain: float = 2.0,
        angular_speed_limit: float = 1.0,
        goal_tolerance: float = 0.25,
        heading_error_threshold: float = math.pi / 2,
    ) -> None:
        self._validate_numeric(linear_speed, "linear_speed", positive=True)
        self._validate_numeric(angular_gain, "angular_gain", positive=True)
        self._validate_numeric(angular_speed_limit, "angular_speed_limit", positive=True)
        self._validate_numeric(goal_tolerance, "goal_tolerance", non_negative=True)
        self._validate_numeric(heading_error_threshold, "heading_error_threshold", positive=True)

        if heading_error_threshold > math.pi:
            raise ValueError(
                f"heading_error_threshold cannot exceed pi radians, got {heading_error_threshold}."
            )

        self._linear_speed = float(linear_speed)
        self._angular_gain = float(angular_gain)
        self._angular_speed_limit = float(angular_speed_limit)
        self._goal_tolerance = float(goal_tolerance)
        self._heading_error_threshold = float(heading_error_threshold)
        self._slowdown_distance = max(2.0 * self._goal_tolerance, 1.0)

    @staticmethod
    def _validate_numeric(
        val: Any,
        name: str,
        positive: bool = False,
        non_negative: bool = False,
    ) -> None:
        """Validate that a configuration parameter is finite, numeric, and meets sign constraints."""
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Parameter '{name}' must be a numeric float or int, got {type(val).__name__}.")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}.")
        if positive and val <= 0.0:
            raise ValueError(f"Parameter '{name}' must be strictly positive (> 0.0), got {val}.")
        if non_negative and val < 0.0:
            raise ValueError(f"Parameter '{name}' must be non-negative (>= 0.0), got {val}.")

    @property
    def linear_speed(self) -> float:
        """Maximum forward linear speed in m/s."""
        return self._linear_speed

    @property
    def angular_gain(self) -> float:
        """Proportional angular gain $K_p$."""
        return self._angular_gain

    @property
    def angular_speed_limit(self) -> float:
        """Maximum angular speed limit in rad/s."""
        return self._angular_speed_limit

    @property
    def goal_tolerance(self) -> float:
        """Radial distance tolerance to reach target waypoint in meters."""
        return self._goal_tolerance

    @property
    def heading_error_threshold(self) -> float:
        """Heading error threshold in radians for forward translation suppression."""
        return self._heading_error_threshold

    @staticmethod
    def _extract_pose(pose: Union[Tuple[float, float, float], Any]) -> Tuple[float, float, float]:
        """Extract and validate `(x, y, theta)` from a tuple or an object with pose attributes."""
        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            x, y, theta = pose.x, pose.y, pose.theta
        elif isinstance(pose, (tuple, list)):
            if len(pose) != 3:
                raise ValueError(f"Pose tuple must have exactly 3 elements (x, y, theta), got {len(pose)}.")
            x, y, theta = pose
        else:
            raise TypeError(f"Pose must be a 3-element tuple or an object with x, y, theta, got {type(pose).__name__}.")

        for val, name in [(x, "x"), (y, "y"), (theta, "theta")]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Pose element '{name}' must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Pose element '{name}' must be finite, got {val}.")

        return (float(x), float(y), float(theta))

    @staticmethod
    def _extract_waypoint(target: Any) -> Tuple[float, float]:
        """Extract and validate `(x, y)` target coordinates."""
        if not isinstance(target, (tuple, list)):
            raise TypeError(f"Target waypoint must be a (x, y) tuple or list, got {type(target).__name__}.")
        if len(target) != 2:
            raise ValueError(f"Target waypoint must have exactly 2 elements (target_x, target_y), got {len(target)}.")

        tx, ty = target
        for val, name in [(tx, "target_x"), (ty, "target_y")]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Waypoint coordinate '{name}' must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Waypoint coordinate '{name}' must be finite, got {val}.")

        return (float(tx), float(ty))

    def is_waypoint_reached(
        self,
        current_pose: Union[Tuple[float, float, float], Any],
        target_waypoint: Tuple[float, float],
    ) -> bool:
        """Determine if current pose is within goal tolerance of target waypoint.

        Parameters
        ----------
        current_pose : Union[Tuple[float, float, float], Any]
            Robot pose `(x, y, theta)` or an object exposing `x, y, theta`.
        target_waypoint : Tuple[float, float]
            Target coordinates `(target_x, target_y)`.

        Returns
        -------
        bool
            True if Euclidean distance $\\le \\text{goal\\_tolerance}$, False otherwise.
        """
        x, y, _ = self._extract_pose(current_pose)
        tx, ty = self._extract_waypoint(target_waypoint)
        dist = math.hypot(tx - x, ty - y)
        return dist <= self._goal_tolerance

    def compute_command(
        self,
        current_pose: Union[Tuple[float, float, float], Any],
        target_waypoint: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Compute `(linear_velocity, angular_velocity)` commands toward target waypoint.

        Does not modify the robot's internal pose or velocity.

        Parameters
        ----------
        current_pose : Union[Tuple[float, float, float], Any]
            Robot pose `(x, y, theta)` or an object exposing `x, y, theta`.
        target_waypoint : Tuple[float, float]
            Target coordinates `(target_x, target_y)`.

        Returns
        -------
        Tuple[float, float]
            Command velocities `(linear_velocity, angular_velocity)`.
        """
        x, y, theta = self._extract_pose(current_pose)
        tx, ty = self._extract_waypoint(target_waypoint)

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        # 1. Waypoint reached: halt completely
        if dist <= self._goal_tolerance:
            return (0.0, 0.0)

        # 2. Desired heading and wrapped error
        desired_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(desired_heading - theta)

        # 3. Proportional angular velocity clamped to limits
        raw_omega = self._angular_gain * heading_error
        omega = max(-self._angular_speed_limit, min(self._angular_speed_limit, raw_omega))

        # 4. Linear velocity scaling
        # Suppress forward movement when facing substantially away from target
        if abs(heading_error) >= self._heading_error_threshold:
            v = 0.0
        else:
            heading_factor = math.cos(heading_error)
            dist_factor = min(1.0, dist / self._slowdown_distance)
            v = self._linear_speed * heading_factor * dist_factor
            v = max(0.0, min(self._linear_speed, v))

        return (float(v), float(omega))


class PurePursuitController:
    """Pure Pursuit path-tracking controller for differential-drive / unicycle mobile robots.

    Control Law & Geometric Formulation:
    -----------------------------------
    Pure Pursuit operates by calculating the curvature $\\kappa$ of a circular arc that
    connects the current robot pose $(x, y, \\theta)$ to a lookahead target point $(x_{la}, y_{la})$
    located at lookahead distance $L_d$ along the planned route.

    1. Lookahead Vector & Distance:
       $$\\Delta x = x_{la} - x, \\quad \\Delta y = y_{la} - y$$
       $$D = \\sqrt{\\Delta x^2 + \\Delta y^2}$$

    2. Angle to Target & Heading Error:
       $$\\theta_{la} = \\text{atan2}(\\Delta y, \\Delta x)$$
       $$\\alpha = \\text{normalize\\_angle}(\\theta_{la} - \\theta) \\in [-\\pi, \\pi)$$

    3. Lateral Offset in Robot Frame:
       $$y_{\\text{local}} = -\\Delta x \\sin\\theta + \\Delta y \\cos\\theta = D \\sin\\alpha$$

    4. Circular Arc Curvature:
       From the geometry of the chord $D$ and circular radius $R_c$:
       $$2 R_c y_{\\text{local}} = D^2 \\implies \\kappa = \\frac{1}{R_c} = \\frac{2 \\sin\\alpha}{D}$$

    5. Unicycle Control Mapping:
       Unlike Ackermann steering (which controls a front-wheel steering angle),
       the unicycle robot turns directly via angular velocity:
       $$\\omega = v \\cdot \\kappa = \\frac{2 v \\sin\\alpha}{D}$$
       $$\\omega_{\\text{cmd}} = \\text{clamp}(\\omega, -\\omega_{\\max}, \\omega_{\\max})$$

    6. Velocity Scaling & Goal Arrival:
       - Goal Arrival: If Euclidean distance to the final path waypoint $d_{\\text{goal}} \\le \\text{goal\\_tolerance}$:
         $$v_{\\text{cmd}} = 0.0, \\quad \\omega_{\\text{cmd}} = 0.0$$
       - Facing Away ($|\\alpha| \\ge \\text{heading\\_error\\_threshold}$):
         Forward translation is suppressed ($v = 0.0$), and the robot rotates in place:
         $$\\omega_{\\text{cmd}} = \\text{clamp}(K_{\\text{turn}} \\cdot \\alpha, -\\omega_{\\max}, \\omega_{\\max})$$
       - Deceleration Profile: As $d_{\\text{goal}} \\to 0$, speed tapers down:
         $$v_{\\text{cmd}} = v_{\\max} \\cdot \\cos\\alpha \\cdot \\min\\left(1.0, \\frac{d_{\\text{goal}}}{d_{\\text{slowdown}}}\\right)$$

    Parameters
    ----------
    lookahead_distance : float, optional
        Lookahead distance $L_d$ in meters along the path. Must be strictly positive.
        Defaults to 0.5.
    linear_speed : float, optional
        Maximum forward linear speed in m/s. Must be strictly positive. Defaults to 0.5.
    angular_speed_limit : float, optional
        Maximum angular turning speed limit in rad/s. Must be strictly positive. Defaults to 1.0.
    goal_tolerance : float, optional
        Distance threshold in meters to declare arrival at the goal. Must be non-negative.
        Defaults to 0.25.
    heading_error_threshold : float, optional
        Maximum absolute heading error (radians) permitting forward motion. Defaults to $\\pi/2$.
    angular_gain : float, optional
        Proportional gain for in-place orientation alignment when facing away. Defaults to 2.0.
    path : Optional[Sequence[Tuple[float, float]]], optional
        Initial path waypoints. Can also be set later via `set_path()`.
    """

    def __init__(
        self,
        lookahead_distance: float = 0.5,
        linear_speed: float = 0.5,
        angular_speed_limit: float = 1.0,
        goal_tolerance: float = 0.25,
        heading_error_threshold: float = math.pi / 2,
        angular_gain: float = 2.0,
        path: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        self._validate_numeric(lookahead_distance, "lookahead_distance", positive=True)
        self._validate_numeric(linear_speed, "linear_speed", positive=True)
        self._validate_numeric(angular_speed_limit, "angular_speed_limit", positive=True)
        self._validate_numeric(goal_tolerance, "goal_tolerance", non_negative=True)
        self._validate_numeric(heading_error_threshold, "heading_error_threshold", positive=True)
        self._validate_numeric(angular_gain, "angular_gain", positive=True)

        if heading_error_threshold > math.pi:
            raise ValueError(
                f"heading_error_threshold cannot exceed pi radians, got {heading_error_threshold}."
            )

        self._lookahead_distance = float(lookahead_distance)
        self._linear_speed = float(linear_speed)
        self._angular_speed_limit = float(angular_speed_limit)
        self._goal_tolerance = float(goal_tolerance)
        self._heading_error_threshold = float(heading_error_threshold)
        self._angular_gain = float(angular_gain)
        self._slowdown_distance = max(2.0 * self._goal_tolerance, 1.0)

        self._path: List[Tuple[float, float]] = []
        self._last_index: int = 0
        self._current_target_point: Optional[Tuple[float, float]] = None

        if path is not None:
            self.set_path(path)

    @staticmethod
    def _validate_numeric(
        val: Any,
        name: str,
        positive: bool = False,
        non_negative: bool = False,
    ) -> None:
        """Validate that a configuration parameter is finite, numeric, and meets sign constraints."""
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Parameter '{name}' must be a numeric float or int, got {type(val).__name__}.")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}.")
        if positive and val <= 0.0:
            raise ValueError(f"Parameter '{name}' must be strictly positive (> 0.0), got {val}.")
        if non_negative and val < 0.0:
            raise ValueError(f"Parameter '{name}' must be non-negative (>= 0.0), got {val}.")

    @property
    def lookahead_distance(self) -> float:
        """Configured lookahead distance $L_d$ in meters."""
        return self._lookahead_distance

    @property
    def linear_speed(self) -> float:
        """Maximum forward linear speed in m/s."""
        return self._linear_speed

    @property
    def angular_speed_limit(self) -> float:
        """Maximum angular speed limit in rad/s."""
        return self._angular_speed_limit

    @property
    def goal_tolerance(self) -> float:
        """Goal arrival tolerance in meters."""
        return self._goal_tolerance

    @property
    def heading_error_threshold(self) -> float:
        """Heading error threshold in radians for forward translation suppression."""
        return self._heading_error_threshold

    @property
    def angular_gain(self) -> float:
        """Proportional gain for orientation alignment."""
        return self._angular_gain

    @property
    def path(self) -> List[Tuple[float, float]]:
        """Return a copy of the current active path waypoints."""
        return list(self._path)

    @property
    def current_target_point(self) -> Optional[Tuple[float, float]]:
        """Return the most recently computed lookahead target point."""
        return self._current_target_point

    def set_path(self, path: Sequence[Tuple[float, float]]) -> None:
        """Assign a new planned path to track and reset trajectory progress.

        Parameters
        ----------
        path : Sequence[Tuple[float, float]]
            Ordered sequence of world-coordinate waypoints `[(x0, y0), (x1, y1), ...]`.
            Must contain at least 1 point.

        Raises
        ------
        TypeError
            If path is not a sequence of 2-element numeric coordinates.
        ValueError
            If path is empty or contains non-finite coordinates.
        """
        if not isinstance(path, (list, tuple)):
            raise TypeError(f"Path must be a sequence of (x, y) tuples, got {type(path).__name__}.")
        if len(path) == 0:
            raise ValueError("Path must contain at least one point.")

        validated: List[Tuple[float, float]] = []
        for i, pt in enumerate(path):
            if not isinstance(pt, (tuple, list)):
                raise TypeError(f"Path waypoint at index {i} must be a (x, y) tuple, got {type(pt).__name__}.")
            if len(pt) != 2:
                raise ValueError(f"Path waypoint at index {i} must have exactly 2 elements, got {len(pt)}.")
            x, y = pt
            for val, label in [(x, f"path[{i}].x"), (y, f"path[{i}].y")]:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"Coordinate '{label}' must be numeric, got {type(val).__name__}.")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Coordinate '{label}' must be finite, got {val}.")
            validated.append((float(x), float(y)))

        self._path = validated
        self._last_index = 0
        self._current_target_point = self._path[0] if len(self._path) == 1 else None

    @staticmethod
    def _extract_pose(pose: Union[Tuple[float, float, float], Any]) -> Tuple[float, float, float]:
        """Extract and validate `(x, y, theta)` from a tuple or an object with pose attributes."""
        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            x, y, theta = pose.x, pose.y, pose.theta
        elif isinstance(pose, (tuple, list)):
            if len(pose) != 3:
                raise ValueError(f"Pose tuple must have exactly 3 elements (x, y, theta), got {len(pose)}.")
            x, y, theta = pose
        else:
            raise TypeError(f"Pose must be a 3-element tuple or an object with x, y, theta, got {type(pose).__name__}.")

        for val, name in [(x, "x"), (y, "y"), (theta, "theta")]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Pose element '{name}' must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Pose element '{name}' must be finite, got {val}.")

        return (float(x), float(y), float(theta))

    def is_goal_reached(self, current_pose: Union[Tuple[float, float, float], Any]) -> bool:
        """Check whether the current pose is within goal tolerance of the path terminus.

        Parameters
        ----------
        current_pose : Union[Tuple[float, float, float], Any]
            Robot pose `(x, y, theta)` or state object.

        Returns
        -------
        bool
            True if distance to final waypoint is $\\le \\text{goal\\_tolerance}$, False otherwise.
        """
        if not self._path:
            return False
        x, y, _ = self._extract_pose(current_pose)
        gx, gy = self._path[-1]
        return math.hypot(gx - x, gy - y) <= self._goal_tolerance

    def find_lookahead_point(
        self,
        current_pose: Union[Tuple[float, float, float], Any],
    ) -> Tuple[float, float]:
        """Identify the lookahead target point along the path.

        Finds the forward-most intersection between the circle of radius $L_d$
        centered at the robot and the piecewise-linear path segments.

        Parameters
        ----------
        current_pose : Union[Tuple[float, float, float], Any]
            Robot pose `(x, y, theta)` or state object.

        Returns
        -------
        Tuple[float, float]
            World coordinates `(x_{la}, y_{la})` of the target lookahead point.
        """
        if not self._path:
            raise ValueError("No path configured. Call set_path() before finding lookahead point.")

        x, y, _ = self._extract_pose(current_pose)
        num_points = len(self._path)

        # Single-point path edge case
        if num_points == 1:
            return self._path[0]

        ld = self._lookahead_distance
        ld_sq = ld * ld

        # 1. Monotonically advance closest segment index to prevent path backtracking
        min_dist_sq = float("inf")
        best_seg_idx = self._last_index
        t_proj_best = 0.0

        for i in range(self._last_index, num_points - 1):
            x1, y1 = self._path[i]
            x2, y2 = self._path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            l_sq = dx * dx + dy * dy
            if l_sq < 1e-12:
                proj_x, proj_y = x1, y1
                t = 0.0
            else:
                t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / l_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy

            d_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_seg_idx = i
                t_proj_best = t

        self._last_index = max(self._last_index, best_seg_idx)

        # Fast path: if on the final segment and goal is within lookahead distance, target goal directly
        gx, gy = self._path[-1]
        if self._last_index >= num_points - 2:
            dist_to_goal_sq = (gx - x) ** 2 + (gy - y) ** 2
            if dist_to_goal_sq <= ld_sq:
                return (gx, gy)

        # 2. Search forward along path segments for circle-segment intersection
        intersect_pt: Optional[Tuple[float, float]] = None

        for i in range(self._last_index, num_points - 1):
            x1, y1 = self._path[i]
            x2, y2 = self._path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            a = dx * dx + dy * dy
            if a < 1e-12:
                continue

            fx = x1 - x
            fy = y1 - y
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - ld_sq

            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)

                # For the current segment, only accept intersections ahead of the robot's projection
                min_t = t_proj_best if i == self._last_index else 0.0

                # Prefer the furthest forward intersection along the segment
                if min_t <= t2 <= 1.0:
                    intersect_pt = (x1 + t2 * dx, y1 + t2 * dy)
                elif min_t <= t1 <= 1.0:
                    intersect_pt = (x1 + t1 * dx, y1 + t1 * dy)

        # 3. Fallback handling: if no forward intersection found, target final waypoint
        if intersect_pt is not None:
            return intersect_pt

        return (gx, gy)

    def compute_command(
        self,
        current_pose: Union[Tuple[float, float, float], Any],
        target_waypoint: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """Compute `(linear_velocity, angular_velocity)` commands via Pure Pursuit.

        Parameters
        ----------
        current_pose : Union[Tuple[float, float, float], Any]
            Robot pose `(x, y, theta)` or state object.
        target_waypoint : Optional[Tuple[float, float]], optional
            Optional fallback target waypoint if no path was pre-configured.

        Returns
        -------
        Tuple[float, float]
            Command velocities `(linear_velocity, angular_velocity)`.
        """
        # If path is not yet configured, allow dynamic single-waypoint fallback
        if not self._path:
            if target_waypoint is not None:
                self.set_path([target_waypoint])
            else:
                raise ValueError("PurePursuitController requires a path. Call set_path() first.")

        x, y, theta = self._extract_pose(current_pose)
        gx, gy = self._path[-1]
        dist_to_goal = math.hypot(gx - x, gy - y)

        # 1. Goal arrival check: halt completely when within tolerance
        if dist_to_goal <= self._goal_tolerance:
            self._current_target_point = (gx, gy)
            return (0.0, 0.0)

        # 2. Identify lookahead point
        tx, ty = self.find_lookahead_point(current_pose)
        self._current_target_point = (tx, ty)

        # 3. Compute relative angle and heading error
        dx = tx - x
        dy = ty - y
        dist_to_target = math.hypot(dx, dy)
        d_effective = max(1e-6, dist_to_target)

        target_angle = math.atan2(dy, dx)
        heading_error = normalize_angle(target_angle - theta)

        # 4. Pure Pursuit curvature: kappa = 2 * sin(alpha) / D
        curvature = 2.0 * math.sin(heading_error) / d_effective

        # 5. Velocity command generation
        if abs(heading_error) >= self._heading_error_threshold:
            # Facing substantially away from target: rotate in place toward lookahead point
            v = 0.0
            raw_omega = self._angular_gain * heading_error
            omega = max(-self._angular_speed_limit, min(self._angular_speed_limit, raw_omega))
        else:
            # Scale forward speed by cosine of heading error and goal deceleration profile
            heading_factor = math.cos(heading_error)
            dist_factor = min(1.0, dist_to_goal / self._slowdown_distance)
            v = self._linear_speed * heading_factor * dist_factor
            v = max(0.0, min(self._linear_speed, v))

            # Unicycle angular velocity: omega = v * kappa
            raw_omega = v * curvature
            omega = max(-self._angular_speed_limit, min(self._angular_speed_limit, raw_omega))

        return (float(v), float(omega))

