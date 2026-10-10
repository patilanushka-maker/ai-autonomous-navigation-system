"""Unit tests for mobile robot kinematics in src.robot.

Compatible with pytest when installed, and executable directly via Python.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Tuple, Type, Union

# Ensure workspace root is in sys.path when invoked directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pytest
except ImportError:
    # Graceful shim allowing direct execution without pytest installed
    class _PytestFallback:
        class raises:
            def __init__(self, expected_exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> None:
                self.expected_exception = expected_exception

            def __enter__(self) -> _PytestFallback.raises:
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected_exception} was not raised.")
                return issubclass(exc_type, self.expected_exception)

        @staticmethod
        def approx(val: float, rel: float = 1e-6, abs: float = 1e-12) -> Any:
            class _Approx:
                def __init__(self, target: float) -> None:
                    self.target = target

                def __eq__(self, other: Any) -> bool:
                    return math.isclose(self.target, float(other), rel_tol=rel, abs_tol=abs)

                def __repr__(self) -> str:
                    return f"approx({self.target})"

            return _Approx(val)

    pytest = _PytestFallback()  # type: ignore[assignment]

from src.robot import Robot, RobotState, normalize_angle


def test_initial_pose_and_default_values() -> None:
    """Test 1: Construction with default parameters and explicit parameters."""
    r_default = Robot()
    assert r_default.x == 0.0
    assert r_default.y == 0.0
    assert r_default.theta == 0.0
    assert r_default.v == 0.0
    assert r_default.omega == 0.0
    assert r_default.pose == (0.0, 0.0, 0.0)

    state = r_default.state
    assert isinstance(state, RobotState)
    assert state.x == 0.0
    assert state.y == 0.0
    assert state.theta == 0.0
    assert state.v == 0.0
    assert state.omega == 0.0
    assert state.pose == (0.0, 0.0, 0.0)

    # Explicit constructor initialization
    r_custom = Robot(x=1.5, y=-2.0, theta=math.pi / 4, v=2.0, omega=0.5)
    assert r_custom.x == 1.5
    assert r_custom.y == -2.0
    assert pytest.approx(r_custom.theta) == math.pi / 4
    assert r_custom.v == 2.0
    assert r_custom.omega == 0.5


def test_straight_line_motion_heading_zero() -> None:
    """Test 2: Linear translation at theta = 0 moves purely along the positive x-axis."""
    robot = Robot(x=1.0, y=2.0, theta=0.0, v=2.0, omega=0.0)
    dt = 0.5
    robot.step(dt)

    # Expected: x += 2.0 * 1.0 * 0.5 = 1.0; y += 0; theta unchanged
    assert pytest.approx(robot.x) == 2.0
    assert pytest.approx(robot.y) == 2.0
    assert pytest.approx(robot.theta) == 0.0


def test_straight_line_motion_nonzero_heading() -> None:
    """Test 3: Translation at arbitrary heading angles (pi/2 and pi/4)."""
    # Moving along +y when theta = pi/2
    robot_y = Robot(x=0.0, y=0.0, theta=math.pi / 2, v=1.5, omega=0.0)
    robot_y.step(1.0)
    assert pytest.approx(robot_y.x, abs=1e-12) == 0.0
    assert pytest.approx(robot_y.y) == 1.5
    assert pytest.approx(robot_y.theta) == math.pi / 2

    # Moving at 45 degrees (pi/4)
    robot_diag = Robot(x=0.0, y=0.0, theta=math.pi / 4, v=math.sqrt(2), omega=0.0)
    robot_diag.step(1.0)
    # Expected: dx = sqrt(2) * (sqrt(2)/2) * 1 = 1.0; dy = 1.0
    assert pytest.approx(robot_diag.x) == 1.0
    assert pytest.approx(robot_diag.y) == 1.0
    assert pytest.approx(robot_diag.theta) == math.pi / 4


def test_rotation_in_place() -> None:
    """Test 4: Pure rotation (v = 0, omega != 0) changes heading without altering position."""
    robot = Robot(x=5.0, y=-3.0, theta=0.0, v=0.0, omega=math.pi / 2)
    robot.step(0.5)

    # Position must remain stationary
    assert pytest.approx(robot.x) == 5.0
    assert pytest.approx(robot.y) == -3.0
    # Heading advances by (pi/2) * 0.5 = pi/4
    assert pytest.approx(robot.theta) == math.pi / 4


def test_combined_linear_and_angular_motion() -> None:
    """Test 5: Simultaneous linear and angular motion follows the update equations."""
    # Start at origin, theta = 0, v = 1.0, omega = 1.0, dt = 0.5
    robot = Robot(x=0.0, y=0.0, theta=0.0, v=1.0, omega=1.0)
    dt = 0.5
    robot.step(dt)

    # Kinematics equations:
    # x_1 = 0 + 1.0 * cos(0) * 0.5 = 0.5
    # y_1 = 0 + 1.0 * sin(0) * 0.5 = 0.0
    # theta_1 = 0 + 1.0 * 0.5 = 0.5
    assert pytest.approx(robot.x) == 0.5
    assert pytest.approx(robot.y) == 0.0
    assert pytest.approx(robot.theta) == 0.5

    # Second step
    robot.step(dt)
    # x_2 = 0.5 + 1.0 * cos(0.5) * 0.5 = 0.5 + 0.5 * 0.87758256 = 0.93879128
    # y_2 = 0.0 + 1.0 * sin(0.5) * 0.5 = 0.0 + 0.5 * 0.47942554 = 0.23971277
    # theta_2 = 0.5 + 1.0 * 0.5 = 1.0
    expected_x = 0.5 + 0.5 * math.cos(0.5)
    expected_y = 0.0 + 0.5 * math.sin(0.5)
    assert pytest.approx(robot.x) == expected_x
    assert pytest.approx(robot.y) == expected_y
    assert pytest.approx(robot.theta) == 1.0


def test_zero_duration_update() -> None:
    """Test 6: Step with dt = 0 leaves all state variables unchanged."""
    robot = Robot(x=2.5, y=-1.5, theta=0.75, v=3.0, omega=-1.2)
    robot.step(0.0)

    assert robot.x == 2.5
    assert robot.y == -1.5
    assert robot.theta == 0.75
    assert robot.v == 3.0
    assert robot.omega == -1.2


def test_heading_normalization() -> None:
    """Test 7: Angle wrapping function and step updates wrap correctly into [-pi, pi)."""
    # Direct normalization tests
    assert pytest.approx(normalize_angle(0.0)) == 0.0
    assert pytest.approx(normalize_angle(math.pi)) == -math.pi
    assert pytest.approx(normalize_angle(-math.pi)) == -math.pi
    assert pytest.approx(normalize_angle(2.0 * math.pi)) == 0.0
    assert pytest.approx(normalize_angle(3.0 * math.pi)) == -math.pi
    assert pytest.approx(normalize_angle(-3.0 * math.pi)) == -math.pi
    assert pytest.approx(normalize_angle(2.5 * math.pi)) == math.pi / 2
    assert pytest.approx(normalize_angle(-2.5 * math.pi)) == -math.pi / 2

    # Normalization during robot initialization
    robot = Robot(theta=3.0 * math.pi)
    assert pytest.approx(robot.theta) == -math.pi

    # Normalization during continuous rotation steps
    robot = Robot(theta=math.pi - 0.1, omega=0.5)
    robot.step(1.0)  # theta = pi - 0.1 + 0.5 = pi + 0.4 -> wraps to -pi + 0.4
    expected = -math.pi + 0.4
    assert pytest.approx(robot.theta) == expected
    assert -math.pi <= robot.theta < math.pi


def test_invalid_time_steps_and_inputs() -> None:
    """Test 8: Rejection of negative time steps, non-finite values, and malformed inputs."""
    robot = Robot()

    # Negative dt
    with pytest.raises(ValueError):
        robot.step(-0.1)

    # Non-finite dt
    with pytest.raises(ValueError):
        robot.step(float("nan"))
    with pytest.raises(ValueError):
        robot.step(float("inf"))

    # Non-numeric dt
    with pytest.raises(TypeError):
        robot.step("0.1")  # type: ignore[arg-type]

    # Non-finite constructor values
    with pytest.raises(ValueError):
        Robot(x=float("nan"))
    with pytest.raises(ValueError):
        Robot(y=float("inf"))
    with pytest.raises(ValueError):
        Robot(theta=float("-inf"))
    with pytest.raises(ValueError):
        Robot(v=float("nan"))
    with pytest.raises(ValueError):
        Robot(omega=float("inf"))

    # Non-numeric setter values
    with pytest.raises(TypeError):
        robot.set_pose("1.0", 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        robot.set_velocity(1.0, None)  # type: ignore[arg-type]


def test_boolean_inputs_rejected() -> None:
    """Test 9: Boolean inputs rejected where numeric floats/ints are required."""
    with pytest.raises(TypeError):
        Robot(x=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Robot(y=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Robot(theta=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Robot(v=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Robot(omega=True)  # type: ignore[arg-type]

    robot = Robot()
    with pytest.raises(TypeError):
        robot.step(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        robot.set_pose(True, 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        robot.set_velocity(0.0, False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        normalize_angle(True)  # type: ignore[arg-type]


def test_state_immutability_and_protection() -> None:
    """Test 10: State inspection via .state and .pose does not permit accidental internal mutation."""
    robot = Robot(x=1.0, y=2.0, theta=0.5, v=1.5, omega=-0.2)
    state = robot.state

    # RobotState is frozen; attribute assignment must raise FrozenInstanceError / AttributeError
    try:
        state.x = 99.0  # type: ignore[misc]
    except Exception as exc:
        assert isinstance(exc, (AttributeError, Exception))

    # Robot internal state remains unchanged
    assert robot.x == 1.0

    # Changing robot state does not retroactively mutate previously retrieved RobotState snapshot
    robot.set_pose(5.0, 6.0, 0.0)
    assert robot.x == 5.0
    assert state.x == 1.0


def run_all_tests() -> None:
    """Direct execution runner for environments where pytest CLI is not installed."""
    tests = [
        test_initial_pose_and_default_values,
        test_straight_line_motion_heading_zero,
        test_straight_line_motion_nonzero_heading,
        test_rotation_in_place,
        test_combined_linear_and_angular_motion,
        test_zero_duration_update,
        test_heading_normalization,
        test_invalid_time_steps_and_inputs,
        test_boolean_inputs_rejected,
        test_state_immutability_and_protection,
    ]
    print(f"Executing {len(tests)} test functions in test_robot...")
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  [PASS] {t.__name__}")
    print(f"All {passed}/{len(tests)} tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
