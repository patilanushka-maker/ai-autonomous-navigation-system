"""Unit tests for the heading-based proportional controller in src.controller.

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

from src.controller import HeadingProportionalController
from src.robot import Robot


def test_target_directly_ahead() -> None:
    """Test 1: Target directly in front of the robot results in zero angular velocity and full forward speed."""
    controller = HeadingProportionalController(
        linear_speed=0.5,
        angular_gain=2.0,
        angular_speed_limit=1.0,
        goal_tolerance=0.25,
    )
    # Robot at (0, 0) facing theta = 0, target at (5.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    target = (5.0, 0.0)

    v, omega = controller.compute_command(pose, target)
    assert pytest.approx(omega, abs=1e-9) == 0.0
    assert pytest.approx(v) == 0.5
    assert controller.is_waypoint_reached(pose, target) is False


def test_target_directly_behind() -> None:
    """Test 2: Target directly behind causes heading error of pi, suppressing linear speed (v = 0.0)."""
    controller = HeadingProportionalController(
        linear_speed=0.5,
        angular_gain=2.0,
        angular_speed_limit=1.0,
        heading_error_threshold=math.pi / 2,
    )
    # Robot at (0, 0) facing +x (theta=0), target at (-5.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    target = (-5.0, 0.0)

    v, omega = controller.compute_command(pose, target)
    # Heading error is -pi; |heading_error| >= pi/2 -> forward motion suppressed
    assert v == 0.0
    # Angular velocity should saturate at the maximum limit
    assert abs(omega) == controller.angular_speed_limit


def test_target_to_left_and_right() -> None:
    """Test 3: Target to the left commands positive omega; target to the right commands negative omega."""
    controller = HeadingProportionalController(
        linear_speed=0.5,
        angular_gain=1.0,
        angular_speed_limit=1.0,
    )
    pose = (0.0, 0.0, 0.0)

    # Target 90 degrees to the left at (0.0, 5.0) -> desired_heading = +pi/2
    v_left, omega_left = controller.compute_command(pose, (0.0, 5.0))
    assert omega_left > 0.0
    assert v_left == 0.0  # At threshold pi/2, linear speed is suppressed

    # Target 90 degrees to the right at (0.0, -5.0) -> desired_heading = -pi/2
    v_right, omega_right = controller.compute_command(pose, (0.0, -5.0))
    assert omega_right < 0.0
    assert v_right == 0.0


def test_wrapped_heading_error_near_pi_boundary() -> None:
    """Test 4: Heading error wrapping prevents discontinuous spin near the -pi / +pi boundary."""
    controller = HeadingProportionalController(
        angular_gain=2.0,
        angular_speed_limit=1.5,
    )
    # Robot heading is slightly below +pi (e.g., +3.14 rad)
    # Target is in the quadrant slightly below -pi (desired heading ~ -3.14 rad)
    # Without angle wrapping, difference would be (-3.14 - 3.14) = -6.28 rad
    # With wrapping, the angular difference is only ~0.003 rad
    pose = (0.0, 0.0, 3.14)
    target = (-10.0, -0.01)  # atan2(-0.01, -10.0) is approx -3.14059

    v, omega = controller.compute_command(pose, target)
    # The magnitude of omega should be very small rather than saturating at 1.5
    assert abs(omega) < 0.1


def test_angular_command_saturation() -> None:
    """Test 5: Large angular errors clamp cleanly at angular_speed_limit."""
    controller = HeadingProportionalController(
        angular_gain=10.0,
        angular_speed_limit=0.8,
    )
    pose = (0.0, 0.0, 0.0)
    target = (1.0, 2.0)  # heading error approx 1.1 rad -> raw omega = 11.0

    v, omega = controller.compute_command(pose, target)
    assert pytest.approx(omega) == 0.8


def test_forward_speed_decreases_as_heading_error_increases() -> None:
    """Test 6: Forward speed scales down smoothly with increasing heading deviation."""
    controller = HeadingProportionalController(linear_speed=1.0)
    pose_ahead = (0.0, 0.0, 0.0)
    pose_30deg = (0.0, 0.0, -math.pi / 6)
    pose_60deg = (0.0, 0.0, -math.pi / 3)
    pose_90deg = (0.0, 0.0, -math.pi / 2)
    target = (10.0, 0.0)

    v_ahead, _ = controller.compute_command(pose_ahead, target)
    v_30deg, _ = controller.compute_command(pose_30deg, target)
    v_60deg, _ = controller.compute_command(pose_60deg, target)
    v_90deg, _ = controller.compute_command(pose_90deg, target)

    assert v_ahead > v_30deg > v_60deg > v_90deg == 0.0


def test_forward_speed_decreases_as_target_gets_closer() -> None:
    """Test 7: Forward speed ramps down smoothly as the robot approaches the target."""
    controller = HeadingProportionalController(linear_speed=1.0, goal_tolerance=0.2)
    pose = (0.0, 0.0, 0.0)

    v_far, _ = controller.compute_command(pose, (5.0, 0.0))
    v_mid, _ = controller.compute_command(pose, (0.8, 0.0))
    v_near, _ = controller.compute_command(pose, (0.3, 0.0))

    assert v_far > v_mid > v_near > 0.0


def test_zero_commands_when_target_within_tolerance() -> None:
    """Test 8: Controller halts (v = 0, omega = 0) when within goal tolerance."""
    controller = HeadingProportionalController(goal_tolerance=0.25)
    pose = (1.0, 1.0, 0.0)

    # Waypoint at distance 0.15 <= 0.25
    target_inside = (1.1, 1.1)
    assert controller.is_waypoint_reached(pose, target_inside) is True
    v, omega = controller.compute_command(pose, target_inside)
    assert v == 0.0
    assert omega == 0.0

    # Exactly at target
    assert controller.is_waypoint_reached(pose, (1.0, 1.0)) is True
    v_exact, omega_exact = controller.compute_command(pose, (1.0, 1.0))
    assert v_exact == 0.0
    assert omega_exact == 0.0


def test_invalid_constructor_parameters() -> None:
    """Test 9: Rejection of non-positive speeds/gains, negative tolerance, and invalid types."""
    with pytest.raises(ValueError):
        HeadingProportionalController(linear_speed=-0.5)
    with pytest.raises(ValueError):
        HeadingProportionalController(linear_speed=0.0)
    with pytest.raises(ValueError):
        HeadingProportionalController(angular_gain=0.0)
    with pytest.raises(ValueError):
        HeadingProportionalController(angular_speed_limit=-1.0)
    with pytest.raises(ValueError):
        HeadingProportionalController(goal_tolerance=-0.1)
    with pytest.raises(ValueError):
        HeadingProportionalController(heading_error_threshold=4.0)  # > pi

    # Non-numeric or boolean constructor arguments
    with pytest.raises(TypeError):
        HeadingProportionalController(linear_speed=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HeadingProportionalController(angular_gain="2.0")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HeadingProportionalController(linear_speed=float("nan"))


def test_invalid_poses_and_target_coordinates() -> None:
    """Test 10: Malformed poses and target coordinates raise TypeError or ValueError."""
    controller = HeadingProportionalController()

    # Wrong length pose
    with pytest.raises(ValueError):
        controller.compute_command((0.0, 0.0), (1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        controller.compute_command((0.0, 0.0, 0.0, 0.0), (1.0, 1.0))  # type: ignore[arg-type]

    # Wrong length target
    with pytest.raises(ValueError):
        controller.compute_command((0.0, 0.0, 0.0), (1.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        controller.compute_command((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # type: ignore[arg-type]

    # Non-sequence inputs
    with pytest.raises(TypeError):
        controller.compute_command(123, (1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        controller.compute_command((0.0, 0.0, 0.0), "target")  # type: ignore[arg-type]


def test_boolean_and_non_finite_inputs() -> None:
    """Test 11: Boolean and non-finite inputs in pose and target coordinates are rejected."""
    controller = HeadingProportionalController()

    with pytest.raises(TypeError):
        controller.compute_command((True, 0.0, 0.0), (1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        controller.compute_command((0.0, 0.0, 0.0), (1.0, False))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        controller.compute_command((float("nan"), 0.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError):
        controller.compute_command((0.0, 0.0, 0.0), (float("inf"), 1.0))


def test_controller_does_not_mutate_robot_or_pose() -> None:
    """Test 12: Passing a Robot instance does not alter its pose or velocity."""
    controller = HeadingProportionalController()
    robot = Robot(x=1.0, y=2.0, theta=0.5, v=0.2, omega=-0.1)

    v, omega = controller.compute_command(robot, (5.0, 5.0))
    # Command values are computed
    assert v > 0.0
    assert abs(omega) > 0.0

    # Robot state must remain untouched
    assert robot.x == 1.0
    assert robot.y == 2.0
    assert robot.theta == 0.5
    assert robot.v == 0.2
    assert robot.omega == -0.1


def run_all_tests() -> None:
    """Direct execution runner for environments where pytest CLI is not installed."""
    tests = [
        test_target_directly_ahead,
        test_target_directly_behind,
        test_target_to_left_and_right,
        test_wrapped_heading_error_near_pi_boundary,
        test_angular_command_saturation,
        test_forward_speed_decreases_as_heading_error_increases,
        test_forward_speed_decreases_as_target_gets_closer,
        test_zero_commands_when_target_within_tolerance,
        test_invalid_constructor_parameters,
        test_invalid_poses_and_target_coordinates,
        test_boolean_and_non_finite_inputs,
        test_controller_does_not_mutate_robot_or_pose,
    ]
    print(f"Executing {len(tests)} test functions in test_controller...")
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  [PASS] {t.__name__}")
    print(f"All {passed}/{len(tests)} tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
