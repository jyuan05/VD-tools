"""Geometry-derived spring, rocker, damper and anti-roll-bar actuation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import CornerState


_EPS = 1e-12
_ANGLE_TOL = 1e-10
_AXIS_TOL = 1e-6


def _rotate_point(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return origin + Rotation.from_rotvec(axis * angle).apply(point - origin)


def _wishbone_angle(corner: dict, state: CornerState, name: str) -> float:
    arm = corner[name]
    pivot_rearward = np.asarray(arm["inboard_rearward"], dtype=float)
    pivot_forward = np.asarray(arm["inboard_forward"], dtype=float)
    axis = pivot_forward - pivot_rearward
    axis /= np.linalg.norm(axis)
    reference = np.asarray(arm["outboard"], dtype=float) - pivot_rearward
    actual = np.asarray(getattr(state, f"{name}_outboard"), dtype=float) - pivot_rearward
    reference -= axis * np.dot(reference, axis)
    actual -= axis * np.dot(actual, axis)
    return float(np.arctan2(np.dot(axis, np.cross(reference, actual)), np.dot(reference, actual)))


def _rotating_link_solution(
    *,
    pivot: np.ndarray,
    axis: np.ndarray,
    rotating_point: np.ndarray,
    target: np.ndarray,
    reference_length: float,
    angle_limits: list[float] | tuple[float, float],
    preferred_angle: float = 0.0,
) -> tuple[float, float, str]:
    """Solve the exact circle/sphere closure for one rotating link endpoint."""
    axis = np.asarray(axis, dtype=float)
    axis_length = float(np.linalg.norm(axis))
    if not np.isfinite(axis_length) or axis_length <= _EPS:
        return float("nan"), float("nan"), "link_closure"
    axis = axis / axis_length
    pivot = np.asarray(pivot, dtype=float)
    rotating_point = np.asarray(rotating_point, dtype=float)
    target = np.asarray(target, dtype=float)
    radius = rotating_point - pivot
    parallel = axis * np.dot(axis, radius)
    radial = radius - parallel
    tangent = np.cross(axis, radial)
    other = pivot - target + parallel

    constant = float(np.dot(other, other) + np.dot(radial, radial))
    cosine = float(2.0 * np.dot(other, radial))
    sine = float(2.0 * np.dot(other, tangent))
    amplitude = float(np.hypot(cosine, sine))
    target_value = float(reference_length * reference_length - constant)
    if amplitude <= _EPS:
        if abs(target_value) > 1e-10 * max(reference_length * reference_length, 1.0):
            return float("nan"), float("nan"), "link_closure"
        roots = [float(preferred_angle)]
    else:
        ratio = target_value / amplitude
        if ratio < -1.0 - 1e-10 or ratio > 1.0 + 1e-10:
            return float("nan"), float("nan"), "link_closure"
        ratio = float(np.clip(ratio, -1.0, 1.0))
        phase = float(np.arctan2(sine, cosine))
        offset = float(np.arccos(ratio))
        roots: list[float] = []
        low, high = map(float, angle_limits)
        for base in (phase - offset, phase + offset):
            for turns in range(-2, 3):
                candidate = base + turns * 2.0 * np.pi
                if low - _ANGLE_TOL <= candidate <= high + _ANGLE_TOL:
                    candidate = float(np.clip(candidate, low, high))
                    if not any(abs(candidate - prior) <= 1e-9 for prior in roots):
                        roots.append(candidate)
        if not roots:
            return float("nan"), float("nan"), "angle_limit"

    angle = min(roots, key=lambda value: (abs(value - preferred_angle), abs(value)))
    moved = _rotate_point(rotating_point, pivot, axis, angle)
    delta = moved - target
    residual = float(np.linalg.norm(delta) - reference_length)
    if not np.isfinite(residual) or abs(residual) > 2e-9:
        return float("nan"), residual, "link_closure"

    radial_length = float(np.linalg.norm(radial))
    derivative = abs(float(np.dot(delta, np.cross(axis, moved - pivot))))
    if radial_length > _EPS and reference_length > _EPS:
        normalized_slope = derivative / (reference_length * radial_length)
        if normalized_slope <= 1e-7:
            return float("nan"), residual, "link_toggle"
    return angle, residual, "ok"


def _rocker_solution(corner: dict, state: CornerState) -> tuple[float, float, str]:
    if "rocker" not in corner:
        return float("nan"), float("nan"), "no_rocker"
    rocker = corner["rocker"]
    target = _attachment_position(corner, state, rocker["rod_mount"], rocker_solution=None)
    if not np.all(np.isfinite(target)):
        return float("nan"), float("nan"), "attachment_invalid"
    rod_point = np.asarray(rocker["rod_point"], dtype=float)
    mount_reference = np.asarray(rocker["rod_mount"]["point"], dtype=float)
    rod_length = float(np.linalg.norm(rod_point - mount_reference))
    return _rotating_link_solution(
        pivot=np.asarray(rocker["pivot"], dtype=float),
        axis=np.asarray(rocker["axis"], dtype=float),
        rotating_point=rod_point,
        target=target,
        reference_length=rod_length,
        angle_limits=rocker["angle_limits"],
    )


def _attachment_position(
    corner: dict,
    state: CornerState,
    attachment: dict,
    *,
    rocker_solution: tuple[float, float, str] | None,
) -> np.ndarray:
    point = np.asarray(attachment["point"], dtype=float)
    body = attachment["body"]
    if body == "chassis":
        return point.copy()
    if body == "upright":
        center = np.asarray(corner["wheel_center"], dtype=float)
        return np.asarray(state.wheel_center, dtype=float) + np.asarray(state.rotation, dtype=float) @ (point - center)
    if body in {"lower", "upper"}:
        arm = corner[body]
        pivot = np.asarray(arm["inboard_rearward"], dtype=float)
        axis = np.asarray(arm["inboard_forward"], dtype=float) - pivot
        axis /= np.linalg.norm(axis)
        angle = _wishbone_angle(corner, state, body)
        return _rotate_point(point, pivot, axis, angle)
    if body == "rocker":
        if rocker_solution is None:
            rocker_solution = _rocker_solution(corner, state)
        angle, _, reason = rocker_solution
        if reason != "ok" or not np.isfinite(angle):
            return np.full(3, np.nan, dtype=float)
        rocker = corner["rocker"]
        return _rotate_point(
            point,
            np.asarray(rocker["pivot"], dtype=float),
            np.asarray(rocker["axis"], dtype=float),
            angle,
        )
    return np.full(3, np.nan, dtype=float)


def attachment_position(corner: dict, state: CornerState, attachment: dict) -> np.ndarray:
    """Return a reference attachment after its configured body's rigid motion."""
    if not state.valid:
        return np.full(3, np.nan, dtype=float)
    return _attachment_position(corner, state, attachment, rocker_solution=None)


def _component_result(
    corner: dict,
    state: CornerState,
    component: dict,
    rocker_solution: tuple[float, float, str] | None,
) -> tuple[float, float, np.ndarray, str]:
    moving = _attachment_position(corner, state, component["moving"], rocker_solution=rocker_solution)
    if not np.all(np.isfinite(moving)):
        return float("nan"), float("nan"), np.full(2, np.nan), "attachment_invalid"
    fixed = np.asarray(component["fixed"], dtype=float)
    length = float(np.linalg.norm(moving - fixed))
    reference_length = float(np.linalg.norm(np.asarray(component["moving"]["point"]) - fixed))
    low, high = map(float, component["length_limits"])
    margin = np.asarray([length - low, high - length], dtype=float)
    if length < low - 1e-10 or length > high + 1e-10:
        return float("nan"), float("nan"), np.full(2, np.nan), "stroke_limit"
    return length, reference_length - length, margin, "ok"


def actuation_state(corner: dict, state: CornerState) -> dict[str, Any]:
    """Return geometry-derived spring/damper compression and travel margins.

    Stroke margin is ``[remaining_compression, remaining_extension]`` in
    metres, calculated as ``[length - min_length, max_length - length]``.
    """
    output: dict[str, Any] = {
        "valid": False,
        "reason": "invalid_corner",
        "spring_length": float("nan"),
        "spring_compression": float("nan"),
        "spring_stroke_margin": np.full(2, np.nan, dtype=float),
        "damper_length": float("nan"),
        "damper_compression": float("nan"),
        "damper_stroke_margin": np.full(2, np.nan, dtype=float),
        "rocker_angle": float("nan"),
        "rod_residual": float("nan"),
    }
    if not state.valid:
        return output

    rocker_solution: tuple[float, float, str] | None = None
    if "rocker" in corner:
        rocker_solution = _rocker_solution(corner, state)
        rocker_angle, rod_residual, rocker_reason = rocker_solution
        if rocker_reason != "ok":
            if rocker_reason == "angle_limit":
                output["reason"] = "rocker_limit"
            elif rocker_reason == "link_closure":
                output["reason"] = "rocker_closure"
            elif rocker_reason == "link_toggle":
                output["reason"] = "rocker_toggle"
            else:
                output["reason"] = rocker_reason
            output["rod_residual"] = rod_residual
            return output
        low, high = map(float, corner["rocker"]["angle_limits"])
        if rocker_angle < low - 1e-10 or rocker_angle > high + 1e-10:
            output["reason"] = "rocker_limit"
            output["rod_residual"] = rod_residual
            return output
        output["rocker_angle"] = rocker_angle
        output["rod_residual"] = rod_residual

    spring = _component_result(corner, state, corner["spring"], rocker_solution)
    damper = _component_result(corner, state, corner["damper"], rocker_solution)
    if spring[3] != "ok" or damper[3] != "ok":
        output["reason"] = "stroke_limit" if "stroke_limit" in {spring[3], damper[3]} else "attachment_invalid"
        return output

    output.update(
        {
            "valid": True,
            "reason": "ok",
            "spring_length": spring[0],
            "spring_compression": spring[1],
            "spring_stroke_margin": spring[2],
            "damper_length": damper[0],
            "damper_compression": damper[1],
            "damper_stroke_margin": damper[2],
        }
    )
    return output


def _arb_invalid(reason: str) -> dict[str, Any]:
    return {
        "left_angle": float("nan"),
        "right_angle": float("nan"),
        "twist": float("nan"),
        "valid": False,
        "reason": reason,
    }


def _arb_side(
    lever: dict,
    corner: dict,
    state: CornerState,
    shaft_axis: np.ndarray,
) -> tuple[float, str]:
    if not state.valid:
        return float("nan"), "invalid_corner"
    axis = np.asarray(lever["axis"], dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _EPS:
        return float("nan"), "bar_axis_invalid"
    axis /= axis_norm
    alignment = float(np.dot(axis, shaft_axis))
    if abs(abs(alignment) - 1.0) > _AXIS_TOL:
        return float("nan"), "bar_axis_invalid"

    pickup = attachment_position(corner, state, lever["pickup"])
    if not np.all(np.isfinite(pickup)):
        return float("nan"), "attachment_invalid"
    tip = np.asarray(lever["tip"], dtype=float)
    pickup_reference = np.asarray(lever["pickup"]["point"], dtype=float)
    reference_length = float(np.linalg.norm(tip - pickup_reference))
    raw_angle, _, reason = _rotating_link_solution(
        pivot=np.asarray(lever["pivot"], dtype=float),
        axis=axis,
        rotating_point=tip,
        target=pickup,
        reference_length=reference_length,
        angle_limits=lever["angle_limits"],
    )
    if reason != "ok":
        return float("nan"), reason
    # Local axis sign may be reversed in an otherwise equivalent input.  The
    # exposed value is always projected onto the shaft direction right-to-left.
    return raw_angle * (1.0 if alignment > 0.0 else -1.0), "ok"


def arb_angles(geometry: dict, states: dict[str, CornerState]) -> dict[str, dict[str, Any]]:
    """Solve both ARB drop links and return signed shaft-relative angles.

    The common torsional shaft direction is from the right pivot to the left
    pivot.  Lever axes must be parallel to that shaft.  Each local solved
    angle is projected onto this common direction, and torsional twist is the
    declared relative rotation ``left_angle - right_angle``.
    """
    result: dict[str, dict[str, Any]] = {}
    for axle, levers in geometry.get("arbs", {}).items():
        left = levers["left"]
        right = levers["right"]
        shaft = np.asarray(left["pivot"], dtype=float) - np.asarray(right["pivot"], dtype=float)
        shaft_length = float(np.linalg.norm(shaft))
        if not np.isfinite(shaft_length) or shaft_length <= _EPS:
            result[axle] = _arb_invalid("bar_axis_invalid")
            continue
        shaft_axis = shaft / shaft_length
        left_corner = "FL" if axle == "front" else "RL"
        right_corner = "FR" if axle == "front" else "RR"
        left_angle, left_reason = _arb_side(left, geometry["corners"][left_corner], states[left_corner], shaft_axis)
        right_angle, right_reason = _arb_side(
            right, geometry["corners"][right_corner], states[right_corner], shaft_axis
        )
        if left_reason != "ok" or right_reason != "ok":
            reason = left_reason if left_reason != "ok" else right_reason
            result[axle] = _arb_invalid(reason)
            continue
        result[axle] = {
            "left_angle": left_angle,
            "right_angle": right_angle,
            "twist": left_angle - right_angle,
            "valid": True,
            "reason": "ok",
        }
    return result
