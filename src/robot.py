"""Robot Kinematics module for the 2D Autonomous Navigation Simulator.

This module provides the differential-drive / unicycle-style kinematic model
for a planar mobile robot.

Coordinate & Heading Conventions:
---------------------------------
- World Coordinates: Position is represented as continuous `(x, y)` coordinates in meters.
  In alignment with `OccupancyGrid`:
  - The $x$-axis runs horizontally along columns ($x$ increases to the right).
  - The $y$-axis runs vertically along rows ($y$ increases upwards).
- Heading Orientation (`theta`):
  - Expressed in radians.
  - `theta = 0.0`: facing in the positive $x$ direction (along horizontal columns).
  - `theta = +pi/2`: facing in the positive $y$ direction (along vertical rows).
  - `theta = +pi` or `-pi`: facing in the negative $x$ direction (left).
  - `theta = -pi/2`: facing in the negative $y$ direction (down).
  - Angle normalization wraps all heading values into the half-open interval `[-pi, pi)`.

Kinematics Update Equations:
----------------------------
For time step $\\Delta t \\ge 0$, linear forward speed $v$, and angular turning rate $\\omega$:
  x_{k+1} = x_k + v_k * cos(theta_k) * dt
  y_{k+1} = y_k + v_k * sin(theta_k) * dt
  theta_{k+1} = normalize_angle(theta_k + omega_k * dt)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Tuple


def normalize_angle(theta: float) -> float:
    """Normalize an angular heading in radians to the interval `[-pi, pi)`.

    Parameters
    ----------
    theta : float
        Raw angle in radians.

    Returns
    -------
    float
        Normalized angle in `[-pi, pi)`.

    Raises
    ------
    TypeError
        If `theta` is non-numeric or boolean.
    ValueError
        If `theta` is NaN or infinite.
    """
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise TypeError(f"Angle must be a numeric float or int, got {type(theta).__name__}.")
    if math.isnan(theta) or math.isinf(theta):
        raise ValueError(f"Angle must be finite, got {theta}.")

    # Maps angle to [-pi, pi)
    normalized = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return float(normalized)


@dataclass(frozen=True)
class RobotState:
    """Immutable snapshot of the mobile robot kinematic state.

    Attributes
    ----------
    x : float
        World position along the horizontal $x$-axis (meters).
    y : float
        World position along the vertical $y$-axis (meters).
    theta : float
        Heading angle in radians, normalized in `[-pi, pi)`.
    v : float
        Linear forward velocity in meters/second.
    omega : float
        Angular velocity in radians/second.
    """

    x: float
    y: float
    theta: float
    v: float
    omega: float

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Return the pose as `(x, y, theta)`."""
        return (self.x, self.y, self.theta)


class Robot:
    """Unicycle / differential-drive mobile robot kinematic model.

    Parameters
    ----------
    x : float, optional
        Initial $x$ position in world coordinates (meters). Defaults to 0.0.
    y : float, optional
        Initial $y$ position in world coordinates (meters). Defaults to 0.0.
    theta : float, optional
        Initial heading orientation in radians. Normalized to `[-pi, pi)`. Defaults to 0.0.
    v : float, optional
        Initial forward linear velocity in meters/second. Defaults to 0.0.
    omega : float, optional
        Initial angular velocity in radians/second. Defaults to 0.0.

    Raises
    ------
    TypeError
        If any input parameter is non-numeric or boolean.
    ValueError
        If any input parameter is NaN or infinite.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v: float = 0.0,
        omega: float = 0.0,
    ) -> None:
        self._validate_numeric(x, "x")
        self._validate_numeric(y, "y")
        self._validate_numeric(theta, "theta")
        self._validate_numeric(v, "v")
        self._validate_numeric(omega, "omega")

        self._x: float = float(x)
        self._y: float = float(y)
        self._theta: float = normalize_angle(theta)
        self._v: float = float(v)
        self._omega: float = float(omega)

    @staticmethod
    def _validate_numeric(val: Any, name: str) -> None:
        """Validate that an input value is finite, numeric, and not a boolean."""
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Parameter '{name}' must be a numeric float or int, got {type(val).__name__}.")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}.")

    @property
    def x(self) -> float:
        """World $x$ coordinate in meters."""
        return self._x

    @property
    def y(self) -> float:
        """World $y$ coordinate in meters."""
        return self._y

    @property
    def theta(self) -> float:
        """Heading orientation in radians, normalized in `[-pi, pi)`."""
        return self._theta

    @property
    def v(self) -> float:
        """Linear forward velocity in m/s."""
        return self._v

    @property
    def omega(self) -> float:
        """Angular velocity in rad/s."""
        return self._omega

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Return the current pose as `(x, y, theta)`."""
        return (self._x, self._y, self._theta)

    @property
    def state(self) -> RobotState:
        """Return an immutable snapshot of the current robot state."""
        return RobotState(
            x=self._x,
            y=self._y,
            theta=self._theta,
            v=self._v,
            omega=self._omega,
        )

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Update the robot's pose coordinates.

        Parameters
        ----------
        x : float
            New world $x$ position in meters.
        y : float
            New world $y$ position in meters.
        theta : float
            New heading in radians. Automatically normalized to `[-pi, pi)`.

        Raises
        ------
        TypeError
            If any parameter is non-numeric or boolean.
        ValueError
            If any parameter is NaN or infinite.
        """
        self._validate_numeric(x, "x")
        self._validate_numeric(y, "y")
        self._validate_numeric(theta, "theta")

        self._x = float(x)
        self._y = float(y)
        self._theta = normalize_angle(theta)

    def set_velocity(self, v: float, omega: float) -> None:
        """Set the robot control velocity commands.

        Parameters
        ----------
        v : float
            Linear forward velocity in meters/second.
        omega : float
            Angular velocity in radians/second.

        Raises
        ------
        TypeError
            If any parameter is non-numeric or boolean.
        ValueError
            If any parameter is NaN or infinite.
        """
        self._validate_numeric(v, "v")
        self._validate_numeric(omega, "omega")

        self._v = float(v)
        self._omega = float(omega)

    def step(
        self,
        dt: float,
        v: float | None = None,
        omega: float | None = None,
    ) -> None:
        """Advance robot kinematic state forward by time increment `dt`.

        Kinematic Update:
          x_{k+1} = x_k + v * cos(theta_k) * dt
          y_{k+1} = y_k + v * sin(theta_k) * dt
          theta_{k+1} = normalize_angle(theta_k + omega * dt)

        Parameters
        ----------
        dt : float
            Time step duration in seconds. Must be non-negative and finite.
        v : float, optional
            Optional new linear velocity command to apply during this step.
        omega : float, optional
            Optional new angular velocity command to apply during this step.

        Raises
        ------
        TypeError
            If `dt`, `v`, or `omega` is non-numeric or boolean.
        ValueError
            If `dt` is negative, NaN, or infinite.
        """
        self._validate_numeric(dt, "dt")
        if dt < 0.0:
            raise ValueError(f"Time step dt must be non-negative (>= 0.0), got {dt}.")

        if v is not None:
            self._validate_numeric(v, "v")
            self._v = float(v)

        if omega is not None:
            self._validate_numeric(omega, "omega")
            self._omega = float(omega)

        if dt == 0.0:
            return

        # Discrete-time forward Euler integration
        self._x += self._v * math.cos(self._theta) * dt
        self._y += self._v * math.sin(self._theta) * dt
        self._theta = normalize_angle(self._theta + self._omega * dt)

    def __repr__(self) -> str:
        return (
            f"Robot(x={self._x:.4f}, y={self._y:.4f}, "
            f"theta={self._theta:.4f}, v={self._v:.4f}, omega={self._omega:.4f})"
        )
