"""Finite chassis pose composition, generalized rates, and tyre condensation.

The corner solver works in the configured chassis reference frame.  This
module composes those exact corner solutions with a finite body rotation and
solves the scalar jounce required to keep each hub on its fixed world-height
road support.  The same map is then used for force and Hessian composition.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.optimize import brentq, least_squares

from .actuation import actuation_state
from .elasticity import CORNER_ORDER, component_law, wheel_energy, wheel_response
from .kinematics import CornerState, solve_corner


_CORNER_INDEX = {name: index for index, name in enumerate(CORNER_ORDER)}
_NAN4 = np.full(4, np.nan, dtype=float)
_NAN43 = np.full((4, 3), np.nan, dtype=float)
_NAN33 = np.full((3, 3), np.nan, dtype=float)
_NAN44 = np.full((4, 4), np.nan, dtype=float)
_Q_NAMES = ("heave", "roll", "pitch")
Q_LABELS = ("heave_m", "roll_rad", "pitch_rad")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


def body_rotation(q: Any) -> np.ndarray:
    """Return ``Ry(pitch) @ Rx(roll)`` for ``q=[heave, roll, pitch]``."""
    values = _finite_array(q, (3,), "q")
    roll = float(values[1])
    pitch = float(values[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return ry @ rx


def _solver_value(solver: Any, name: str, default: Any) -> Any:
    if solver is None:
        return default
    if isinstance(solver, Mapping):
        return solver.get(name, default)
    return getattr(solver, name, default)


def _solver_options(solver: Any) -> tuple[int, float, np.ndarray, float]:
    max_nfev = _solver_value(solver, "max_nfev", 100)
    tolerance = _solver_value(solver, "residual_tolerance", 1e-9)
    derivative_steps = _solver_value(solver, "derivative_steps", [1e-4, 1e-4, 1e-4])
    jounce_step = _solver_value(solver, "jounce_step", 1e-4)
    if isinstance(max_nfev, bool) or int(max_nfev) < 1:
        raise ValueError("solver.max_nfev must be a positive integer")
    max_nfev = int(max_nfev)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("solver.residual_tolerance must be finite and positive")
    derivative_steps = np.asarray(derivative_steps, dtype=float)
    if derivative_steps.ndim == 0:
        derivative_steps = np.full(3, float(derivative_steps))
    if derivative_steps.shape != (3,) or not np.all(np.isfinite(derivative_steps)) or np.any(derivative_steps <= 0.0):
        raise ValueError("solver.derivative_steps must contain three positive finite values")
    jounce_step = float(jounce_step)
    if not np.isfinite(jounce_step) or jounce_step <= 0.0:
        raise ValueError("solver.jounce_step must be positive and finite")
    return max_nfev, tolerance, derivative_steps, jounce_step


def _origin(geometry: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(geometry.get("reference_origin_world", [0.0, 0.0, 0.0]), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("geometry.reference_origin_world must be finite with shape (3,)")
    return value


def _world_center(origin: np.ndarray, rotation: np.ndarray, local: np.ndarray, heave: float) -> np.ndarray:
    # Hard points are stored in the chassis reference frame.  The configured
    # origin is their nominal world translation, not a second point about
    # which the local coordinates should be rotated.
    return origin + np.array([0.0, 0.0, -heave]) + rotation @ local


def _transform_state(origin: np.ndarray, rotation: np.ndarray, heave: float, state: CornerState) -> dict[str, Any]:
    if not state.valid:
        return {
            "valid": False,
            "reason": state.reason,
            "wheel_center": np.full(3, np.nan),
            "upper_ball_joint": np.full(3, np.nan),
            "lower_ball_joint": np.full(3, np.nan),
            "tie_outboard": np.full(3, np.nan),
            "rotation": np.full((3, 3), np.nan),
        }
    transform = lambda point: _world_center(origin, rotation, np.asarray(point, dtype=float), heave)
    return {
        "valid": True,
        "reason": "ok",
        "wheel_center": transform(state.wheel_center),
        "upper_ball_joint": transform(state.upper_ball_joint),
        "lower_ball_joint": transform(state.lower_ball_joint),
        "tie_outboard": transform(state.tie_outboard),
        "rotation": rotation @ np.asarray(state.rotation, dtype=float),
    }


def _corner_height_solution(
    geometry: Mapping[str, Any],
    corner_name: str,
    q: np.ndarray,
    wheel_rise: float,
    *,
    max_nfev: int,
    tolerance: float,
) -> tuple[float, CornerState, float, str]:
    """Solve one exact local jounce for the finite world-height constraint."""
    corner = geometry["corners"][corner_name]
    origin = _origin(geometry)
    rotation = body_rotation(q)
    reference_center = np.asarray(corner["wheel_center"], dtype=float)
    target_height = float(origin[2] + reference_center[2] + wheel_rise)
    low, high = map(float, corner["jounce_limits"])
    root_tolerance = max(1e-12, min(tolerance * 1e-3, 1e-10))
    cache: dict[float, tuple[CornerState, float]] = {}

    def sample(jounce: float) -> tuple[CornerState, float]:
        key = float(jounce)
        if key not in cache:
            state = solve_corner(corner, key, max_nfev=max_nfev, tolerance=tolerance)
            if state.valid:
                world = _world_center(origin, rotation, state.wheel_center, float(q[0]))
                residual = float(world[2] - target_height)
            else:
                residual = float("nan")
            cache[key] = (state, residual)
        return cache[key]

    estimate = float(q[0] - reference_center[1] * q[1] + reference_center[0] * q[2] + wheel_rise)
    estimate = float(np.clip(estimate, low, high))
    state0, residual0 = sample(estimate)
    if state0.valid and abs(residual0) <= root_tolerance:
        return estimate, state0, residual0, "ok"

    # Most finite body perturbations remain on the same local branch as the
    # small-pose estimate.  Refine a nearby exact bracket first so a tiny
    # O(q^2) height residual is not hidden by the linear estimate and to keep
    # ordinary derivative calls away from a full travel-grid sweep.
    if state0.valid and np.isfinite(residual0):
        local_width = max(2.0e-7, min(0.01, 0.25 * (high - low)))
        for fraction in (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0):
            delta = local_width * fraction
            left_value = max(low, estimate - delta)
            right_value = min(high, estimate + delta)
            left_state, left_residual = sample(left_value)
            right_state, right_residual = sample(right_value)
            if left_state.valid and np.isfinite(left_residual) and left_residual * residual0 <= 0.0:
                bracket = (left_value, estimate)
            elif right_state.valid and np.isfinite(right_residual) and residual0 * right_residual <= 0.0:
                bracket = (estimate, right_value)
            else:
                continue
            try:
                root = brentq(
                    lambda value: sample(float(value))[1],
                    bracket[0], bracket[1], xtol=root_tolerance, rtol=1e-12,
                    maxiter=max(25, max_nfev * 3),
                )
                state, residual = sample(float(root))
                if state.valid and np.isfinite(residual) and abs(residual) <= root_tolerance * 10.0:
                    return float(root), state, float(residual), "ok"
            except (ValueError, RuntimeError, FloatingPointError):
                pass

    # A finite set of exact solves provides a bracket while retaining any
    # valid neighboring branch when an endpoint is outside the mechanism's
    # actual travel.  No interpolation is used for the returned state.
    grid = np.linspace(low, high, max(9, min(33, max_nfev // 3 + 1)))
    samples = [sample(float(value)) for value in grid]
    brackets: list[tuple[float, float]] = []
    best: tuple[float, float] | None = None
    for value, (state, residual) in zip(grid, samples, strict=True):
        if state.valid and np.isfinite(residual):
            if best is None or abs(residual) < best[1]:
                best = (float(value), abs(float(residual)))
    for left, right, left_result, right_result in zip(
        grid[:-1], grid[1:], samples[:-1], samples[1:], strict=True
    ):
        left_state, left_residual = left_result
        right_state, right_residual = right_result
        if not (left_state.valid and right_state.valid):
            continue
        if abs(left_residual) <= root_tolerance:
            return float(left), left_state, float(left_residual), "ok"
        if abs(right_residual) <= root_tolerance:
            return float(right), right_state, float(right_residual), "ok"
        if left_residual * right_residual < 0.0:
            brackets.append((float(left), float(right)))
    if brackets:
        bracket = min(brackets, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - estimate))
        try:
            root = brentq(
                lambda value: sample(float(value))[1],
                bracket[0],
                bracket[1],
                xtol=root_tolerance,
                rtol=1e-12,
                maxiter=max(25, max_nfev * 3),
            )
        except (ValueError, RuntimeError, FloatingPointError):
            root = float("nan")
        if np.isfinite(root):
            state, residual = sample(float(root))
            if state.valid and np.isfinite(residual) and abs(residual) <= root_tolerance * 10.0:
                return float(root), state, float(residual), "ok"

    if best is not None and best[1] <= root_tolerance * 10.0:
        state, residual = sample(best[0])
        if state.valid:
            return best[0], state, residual, "ok"
    reason = "pose_out_of_range" if any(state.valid for state, _ in samples) else "corner_invalid"
    invalid_state = state0 if state0.valid is False else solve_corner(
        corner, float(estimate), max_nfev=max_nfev, tolerance=tolerance
    )
    return float("nan"), invalid_state, float("nan"), reason


def pose_jounces(
    geometry: Mapping[str, Any],
    q: Any,
    *,
    u: Any | None = None,
    wheel_height_offsets: Any | None = None,
    solver: Any | None = None,
    max_nfev: int | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Solve exact corner jounces for fixed world hub heights.

    ``u`` is an optional world wheel-centre rise relative to the nominal hub
    height.  It is positive upward; the tyre compression change is therefore
    ``-u``.  With ``u=0`` the returned world hub heights equal the configured
    reference heights while x/y migration remains free.
    """
    values = _finite_array(q, (3,), "q")
    if u is not None and wheel_height_offsets is not None:
        raise ValueError("pass only one of u or wheel_height_offsets")
    if wheel_height_offsets is not None:
        u = wheel_height_offsets
    rises = np.zeros(4, dtype=float) if u is None else _finite_array(u, (4,), "u")
    configured_nfev, configured_tolerance, _, _ = _solver_options(solver)
    if max_nfev is not None:
        configured_nfev = int(max_nfev)
    if tolerance is not None:
        configured_tolerance = float(tolerance)
    if configured_nfev < 1 or configured_tolerance <= 0.0:
        raise ValueError("max_nfev and tolerance must be positive")
    rotation = body_rotation(values)
    origin = _origin(geometry)
    jounce = np.full(4, np.nan, dtype=float)
    residual = np.full(4, np.nan, dtype=float)
    corner_valid = np.zeros(4, dtype=bool)
    corner_reason: dict[str, str] = {}
    states: dict[str, CornerState] = {}
    world_states: dict[str, dict[str, Any]] = {}
    world_centers = np.full((4, 3), np.nan, dtype=float)

    for index, name in enumerate(CORNER_ORDER):
        solved_jounce, state, state_residual, reason = _corner_height_solution(
            geometry,
            name,
            values,
            float(rises[index]),
            max_nfev=configured_nfev,
            tolerance=configured_tolerance,
        )
        states[name] = state
        corner_reason[name] = reason
        if reason == "ok" and state.valid:
            jounce[index] = solved_jounce
            residual[index] = state_residual
            corner_valid[index] = True
            transformed = _transform_state(origin, rotation, float(values[0]), state)
            world_states[name] = transformed
            world_centers[index] = transformed["wheel_center"]
        else:
            transformed = _transform_state(origin, rotation, float(values[0]), state)
            transformed["valid"] = False
            transformed["reason"] = reason
            world_states[name] = transformed

    valid = bool(corner_valid.all())
    reason = "ok" if valid else next((value for value in corner_reason.values() if value != "ok"), "invalid_pose")
    nominal_centers = origin + np.asarray(
        [geometry["corners"][name]["wheel_center"] for name in CORNER_ORDER], dtype=float
    )
    migration = world_centers[:, :2] - nominal_centers[:, :2]
    return {
        "valid": valid,
        "reason": reason,
        "q": values,
        "u": rises.copy(),
        "rotation": rotation,
        "jounce": jounce,
        "residual": residual,
        "corner_valid": corner_valid,
        "corner_reason": corner_reason,
        "corner_states": states,
        "world_corner_states": world_states,
        "world_wheel_centers": world_centers,
        "wheel_center_migration": migration,
        # Short aliases retained for callers that consume the pose object
        # directly rather than through evaluate_pose.
        "jounces": jounce.copy(),
        "world_hub_centers": world_centers.copy(),
        "wheel_centers": world_centers.copy(),
        "states": states,
        "validity": {"corner": corner_valid.copy()},
        "reasons": {"corner": dict(corner_reason)},
    }


def _pose_scalar_map(
    geometry: Mapping[str, Any],
    q: np.ndarray,
    u: np.ndarray,
    *,
    max_nfev: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    result = pose_jounces(geometry, q, u=u, max_nfev=max_nfev, tolerance=tolerance)
    return result["jounce"], result["corner_valid"], result["corner_reason"]


def _pose_map_derivatives(
    geometry: Mapping[str, Any],
    q: np.ndarray,
    u: np.ndarray,
    steps: np.ndarray,
    *,
    max_nfev: int,
    tolerance: float,
) -> dict[str, Any]:
    center = pose_jounces(geometry, q, u=u, max_nfev=max_nfev, tolerance=tolerance)
    center_values = np.asarray(center["jounce"], dtype=float)
    center_valid = np.asarray(center["corner_valid"], dtype=bool)
    jacobian = np.full((4, 3), np.nan, dtype=float)
    hessian = np.full((4, 3, 3), np.nan, dtype=float)
    jacobian_valid = np.zeros((4, 3), dtype=bool)
    hessian_valid = np.zeros((4, 3, 3), dtype=bool)
    jacobian_method = np.full((4, 3), "invalid", dtype=object)
    hessian_method = np.full((4, 3, 3), "invalid", dtype=object)
    cache: dict[tuple[int, ...], dict[str, Any]] = {(): center}

    def shifted(offset: tuple[int, ...]) -> dict[str, Any]:
        if offset not in cache:
            point = q + np.asarray(offset, dtype=float) * steps
            cache[offset] = pose_jounces(geometry, point, u=u, max_nfev=max_nfev, tolerance=tolerance)
        return cache[offset]

    for axis, step in enumerate(steps):
        plus = shifted(tuple(1 if index == axis else 0 for index in range(3)))
        minus = shifted(tuple(-1 if index == axis else 0 for index in range(3)))
        plus2 = shifted(tuple(2 if index == axis else 0 for index in range(3)))
        minus2 = shifted(tuple(-2 if index == axis else 0 for index in range(3)))
        for index in range(4):
            f0 = center_values[index]
            fp, fm = plus["jounce"][index], minus["jounce"][index]
            vp = bool(plus["corner_valid"][index] and np.isfinite(fp))
            vm = bool(minus["corner_valid"][index] and np.isfinite(fm))
            if center_valid[index] and vp and vm:
                jacobian[index, axis] = (fp - fm) / (2.0 * step)
                jacobian_valid[index, axis] = True
                jacobian_method[index, axis] = "central2"
            elif center_valid[index] and vp and plus2["corner_valid"][index] and np.isfinite(plus2["jounce"][index]):
                jacobian[index, axis] = (-3.0 * f0 + 4.0 * fp - plus2["jounce"][index]) / (2.0 * step)
                jacobian_valid[index, axis] = True
                jacobian_method[index, axis] = "one_sided2"
            elif center_valid[index] and vm and minus2["corner_valid"][index] and np.isfinite(minus2["jounce"][index]):
                jacobian[index, axis] = (3.0 * f0 - 4.0 * fm + minus2["jounce"][index]) / (2.0 * step)
                jacobian_valid[index, axis] = True
                jacobian_method[index, axis] = "one_sided2"
            elif center_valid[index] and vp:
                jacobian[index, axis] = (fp - f0) / step
                jacobian_valid[index, axis] = True
                jacobian_method[index, axis] = "one_sided1"
            elif center_valid[index] and vm:
                jacobian[index, axis] = (f0 - fm) / step
                jacobian_valid[index, axis] = True
                jacobian_method[index, axis] = "one_sided1"

            if center_valid[index] and vp and vm:
                hessian[index, axis, axis] = (fp - 2.0 * f0 + fm) / step**2
                hessian_valid[index, axis, axis] = True
                hessian_method[index, axis, axis] = "central2"
            elif center_valid[index] and vp and plus2["corner_valid"][index] and np.isfinite(plus2["jounce"][index]):
                hessian[index, axis, axis] = (f0 - 2.0 * fp + plus2["jounce"][index]) / step**2
                hessian_valid[index, axis, axis] = True
                hessian_method[index, axis, axis] = "one_sided1"
            elif center_valid[index] and vm and minus2["corner_valid"][index] and np.isfinite(minus2["jounce"][index]):
                hessian[index, axis, axis] = (f0 - 2.0 * fm + minus2["jounce"][index]) / step**2
                hessian_valid[index, axis, axis] = True
                hessian_method[index, axis, axis] = "one_sided1"

    for axis in range(3):
        for other in range(axis):
            step = steps[axis] * steps[other]
            offsets = {}
            for sign_axis in (-1, 1):
                for sign_other in (-1, 1):
                    offset = [0, 0, 0]
                    offset[axis] = sign_axis
                    offset[other] = sign_other
                    offsets[(sign_axis, sign_other)] = shifted(tuple(offset))
            for index in range(4):
                values = []
                valid = True
                for key in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    item = offsets[key]
                    value = item["jounce"][index]
                    if not item["corner_valid"][index] or not np.isfinite(value):
                        valid = False
                    values.append(value)
                if valid:
                    hessian[index, axis, other] = (values[0] - values[1] - values[2] + values[3]) / (4.0 * step)
                    hessian[index, other, axis] = hessian[index, axis, other]
                    hessian_valid[index, axis, other] = True
                    hessian_valid[index, other, axis] = True
                    hessian_method[index, axis, other] = "central2"
                    hessian_method[index, other, axis] = "central2"
    return {
        "center": center,
        "jacobian": jacobian,
        "hessian": hessian,
        "jacobian_valid": jacobian_valid,
        "hessian_valid": hessian_valid,
        "jacobian_method": jacobian_method,
        "hessian_method": hessian_method,
        "sample_count": len(cache),
    }


def _local_compression_derivative(
    geometry: Mapping[str, Any],
    name: str,
    jounce: float,
    component: str,
    step: float,
    *,
    max_nfev: int,
    tolerance: float,
) -> tuple[float, bool, str]:
    corner = geometry["corners"][name]

    def value(j: float) -> tuple[float, bool, str]:
        state = solve_corner(corner, float(j), max_nfev=max_nfev, tolerance=tolerance)
        if not state.valid:
            return float("nan"), False, state.reason
        actuation = actuation_state(corner, state)
        if not actuation["valid"]:
            return float("nan"), False, actuation["reason"]
        return float(actuation[f"{component}_compression"]), True, "ok"

    center, center_valid, reason = value(jounce)
    plus, plus_valid, _ = value(jounce + step)
    minus, minus_valid, _ = value(jounce - step)
    if center_valid and plus_valid and minus_valid:
        return float((plus - minus) / (2.0 * step)), True, "central2"
    if center_valid and plus_valid:
        return float((plus - center) / step), True, "one_sided1"
    if center_valid and minus_valid:
        return float((center - minus) / step), True, "one_sided1"
    return float("nan"), False, reason


def _canonical_normal(normal: np.ndarray) -> np.ndarray:
    value = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return np.full(3, np.nan)
    value = value / norm
    if value[1] < 0.0:
        value = -value
    return value


def _corner_attitude(name: str, corner: Mapping[str, Any], state: CornerState, rotation: np.ndarray) -> dict[str, float]:
    if not state.valid:
        return {key: float("nan") for key in ("camber_chassis", "camber_road", "toe", "caster")}
    side = 1.0 if name.endswith("L") else -1.0
    local_normal = _canonical_normal(np.asarray(state.rotation, dtype=float) @ np.asarray(corner["spindle_axis"], dtype=float))
    world_normal = _canonical_normal(rotation @ (np.asarray(state.rotation, dtype=float) @ np.asarray(corner["spindle_axis"], dtype=float)))
    if not np.all(np.isfinite(local_normal)) or not np.all(np.isfinite(world_normal)):
        return {key: float("nan") for key in ("camber_chassis", "camber_road", "toe", "caster")}
    lower = np.asarray(state.lower_ball_joint, dtype=float)
    upper = np.asarray(state.upper_ball_joint, dtype=float)
    return {
        "camber_chassis": float(-side * np.arctan2(local_normal[2], local_normal[1])),
        "camber_road": float(-side * np.arctan2(world_normal[2], world_normal[1])),
        "toe": float(side * np.arctan2(local_normal[0], local_normal[1])),
        "caster": float(np.arctan2(lower[0] - upper[0], upper[2] - lower[2])),
    }


def _invalid_wheel_response(jounce: np.ndarray, reason: str = "pose_invalid") -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "energy": float("nan"),
        "gradient": np.full(4, np.nan),
        "stiffness": np.full((4, 4), np.nan),
        "gradient_valid": np.zeros(4, dtype=bool),
        "stiffness_valid": np.zeros((4, 4), dtype=bool),
        "gradient_reason": np.full(4, reason, dtype=object),
        "stiffness_reason": np.full((4, 4), reason, dtype=object),
        "material_rates": np.full(4, np.nan),
        "material_rate_valid": np.zeros(4, dtype=bool),
        "material_rate_reason": np.full(4, reason, dtype=object),
        "motion_ratios": np.full(4, np.nan),
        "motion_ratio_valid": np.zeros(4, dtype=bool),
        "motion_ratio_reason": np.full(4, reason, dtype=object),
        "jounce": jounce.copy(),
    }


def _axle_matrix(
    indices: tuple[int, int],
    jacobian: np.ndarray,
    hessian: np.ndarray,
    response: Mapping[str, Any],
    gradient_valid: np.ndarray,
    hessian_valid: np.ndarray,
) -> tuple[np.ndarray, bool]:
    values = np.full((3, 3), np.nan, dtype=float)
    local_stiffness = np.asarray(response["stiffness"])[np.ix_(indices, indices)]
    local_valid = np.asarray(response["stiffness_valid"])[np.ix_(indices, indices)]
    forces = np.asarray(response["gradient"], dtype=float)
    for first in range(3):
        for second in range(3):
            valid = bool(local_valid.all() and gradient_valid[list(indices)].all() and hessian_valid[list(indices), first, second].all())
            if not valid:
                continue
            value = float(jacobian[list(indices), first] @ local_stiffness @ jacobian[list(indices), second])
            value += float(np.sum(forces[list(indices)] * hessian[list(indices), first, second]))
            values[first, second] = value
    valid = bool(np.isfinite(values).all())
    return values, valid


def evaluate_pose(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    q: Any,
    solver: Any | None = None,
    *,
    u: Any | None = None,
    include_tyres: bool = False,
) -> dict[str, Any]:
    """Evaluate geometry, suspension forces, and finite body derivatives."""
    values = _finite_array(q, (3,), "q")
    max_nfev, tolerance, derivative_steps, jounce_step = _solver_options(solver)
    origin = _origin(geometry)
    rises = np.zeros(4, dtype=float) if u is None else _finite_array(u, (4,), "u")
    start = perf_counter()
    pose = pose_jounces(
        geometry,
        values,
        u=rises,
        max_nfev=max_nfev,
        tolerance=tolerance,
    )
    jounce = np.asarray(pose["jounce"], dtype=float)
    response = (
        wheel_response(geometry, setup, jounce, derivative_steps=jounce_step, max_nfev=max_nfev, tolerance=tolerance)
        if pose["valid"]
        else _invalid_wheel_response(jounce, pose["reason"])
    )
    derivative = _pose_map_derivatives(
        geometry,
        values,
        rises,
        derivative_steps,
        max_nfev=max_nfev,
        tolerance=tolerance,
    )
    jacobian = derivative["jacobian"]
    curvature = derivative["hessian"]
    jacobian_valid = derivative["jacobian_valid"]
    curvature_valid = derivative["hessian_valid"]
    body_gradient = np.full(3, np.nan, dtype=float)
    body_gradient_valid = np.zeros(3, dtype=bool)
    body_stiffness = np.full((3, 3), np.nan, dtype=float)
    body_stiffness_valid = np.zeros((3, 3), dtype=bool)
    forces = np.asarray(response.get("gradient", np.full(4, np.nan)), dtype=float)
    force_valid = np.asarray(response.get("gradient_valid", np.zeros(4, dtype=bool)), dtype=bool)
    wheel_stiffness = np.asarray(response.get("stiffness", _NAN44), dtype=float)
    wheel_stiffness_valid = np.asarray(response.get("stiffness_valid", np.zeros((4, 4), dtype=bool)), dtype=bool)
    for axis in range(3):
        if bool(np.all(jacobian_valid[:, axis]) and np.all(force_valid)):
            body_gradient[axis] = float(np.dot(jacobian[:, axis], forces))
            body_gradient_valid[axis] = np.isfinite(body_gradient[axis])
        for other in range(3):
            if not (
                np.all(jacobian_valid[:, axis])
                and np.all(jacobian_valid[:, other])
                and np.all(curvature_valid[:, axis, other])
                and np.all(force_valid)
                and np.all(wheel_stiffness_valid)
            ):
                continue
            value = float(jacobian[:, axis] @ wheel_stiffness @ jacobian[:, other])
            value += float(np.sum(forces * curvature[:, axis, other]))
            if np.isfinite(value):
                body_stiffness[axis, other] = value
                body_stiffness_valid[axis, other] = True
    if body_stiffness_valid.any():
        # A valid Hessian is analytically symmetric.  Symmetrize only entries
        # whose paired chain-rule entries are both valid.
        for axis in range(3):
            for other in range(axis):
                if body_stiffness_valid[axis, other] and body_stiffness_valid[other, axis]:
                    average = 0.5 * (body_stiffness[axis, other] + body_stiffness[other, axis])
                    body_stiffness[axis, other] = average
                    body_stiffness[other, axis] = average
    body_stiffness_reason = np.full((3, 3), "invalid_pose_derivative", dtype=object)
    body_stiffness_reason[body_stiffness_valid] = "ok"
    body_gradient_reason = np.full(3, "invalid_pose_derivative", dtype=object)
    body_gradient_reason[body_gradient_valid] = "ok"

    spring_compression = np.full(4, np.nan, dtype=float)
    damper_compression = np.full(4, np.nan, dtype=float)
    rocker_angle = np.full(4, np.nan, dtype=float)
    spring_margin = np.full((4, 2), np.nan, dtype=float)
    damper_margin = np.full((4, 2), np.nan, dtype=float)
    spring_ratio = np.full(4, np.nan, dtype=float)
    damper_ratio = np.full(4, np.nan, dtype=float)
    gains_spring = np.full((4, 3), np.nan, dtype=float)
    gains_damper = np.full((4, 3), np.nan, dtype=float)
    geometry_validity: dict[str, np.ndarray] = {
        "jounce": pose["corner_valid"].copy(),
        "wheel_migration": np.repeat(pose["corner_valid"][:, None], 2, axis=1),
        "camber_chassis": pose["corner_valid"].copy(),
        "camber_road": pose["corner_valid"].copy(),
        "toe": pose["corner_valid"].copy(),
        "caster": pose["corner_valid"].copy(),
        "spring_compression": np.zeros(4, dtype=bool),
        "damper_compression": np.zeros(4, dtype=bool),
        "spring_motion_ratio": np.zeros(4, dtype=bool),
        "damper_motion_ratio": np.zeros(4, dtype=bool),
        "spring_actuation_gains": np.zeros((4, 3), dtype=bool),
        "damper_actuation_gains": np.zeros((4, 3), dtype=bool),
        "rocker_angle": pose["corner_valid"].copy(),
        "spring_stroke_margin": np.zeros((4, 2), dtype=bool),
        "damper_stroke_margin": np.zeros((4, 2), dtype=bool),
    }
    reasons: dict[str, Any] = {
        name: np.full(np.asarray(mask).shape, "invalid_pose", dtype=object)
        for name, mask in geometry_validity.items()
    }
    attitude = {key: np.full(4, np.nan, dtype=float) for key in ("camber_chassis", "camber_road", "toe", "caster")}
    for index, name in enumerate(CORNER_ORDER):
        state = pose["corner_states"].get(name)
        if state is None or not state.valid or not pose["corner_valid"][index]:
            continue
        values_attitude = _corner_attitude(name, geometry["corners"][name], state, pose["rotation"])
        for key, value in values_attitude.items():
            attitude[key][index] = value
            geometry_validity[key][index] = np.isfinite(value)
            reasons[key][index] = "ok" if np.isfinite(value) else "invalid_orientation"
        motion = actuation_state(geometry["corners"][name], state)
        if not motion["valid"]:
            for key in ("spring_compression", "damper_compression", "spring_motion_ratio", "damper_motion_ratio"):
                reasons[key][index] = motion["reason"]
            continue
        spring_compression[index] = motion["spring_compression"]
        damper_compression[index] = motion["damper_compression"]
        spring_margin[index] = motion["spring_stroke_margin"]
        damper_margin[index] = motion["damper_stroke_margin"]
        rocker_angle[index] = motion["rocker_angle"]
        geometry_validity["spring_compression"][index] = True
        geometry_validity["damper_compression"][index] = True
        geometry_validity["rocker_angle"][index] = np.isfinite(motion["rocker_angle"])
        geometry_validity["spring_stroke_margin"][index, :] = np.isfinite(motion["spring_stroke_margin"])
        geometry_validity["damper_stroke_margin"][index, :] = np.isfinite(motion["damper_stroke_margin"])
        reasons["spring_compression"][index] = "ok"
        reasons["damper_compression"][index] = "ok"
        reasons["rocker_angle"][index] = "ok" if geometry_validity["rocker_angle"][index] else "no_rocker"
        reasons["spring_stroke_margin"][index, :] = np.where(
            geometry_validity["spring_stroke_margin"][index, :], "ok", "invalid_actuation"
        )
        reasons["damper_stroke_margin"][index, :] = np.where(
            geometry_validity["damper_stroke_margin"][index, :], "ok", "invalid_actuation"
        )
        for component, target, target_valid, key in (
            ("spring", spring_ratio, geometry_validity["spring_motion_ratio"], "spring_motion_ratio"),
            ("damper", damper_ratio, geometry_validity["damper_motion_ratio"], "damper_motion_ratio"),
        ):
            ratio, valid, ratio_reason = _local_compression_derivative(
                geometry, name, float(jounce[index]), component, jounce_step,
                max_nfev=max_nfev, tolerance=tolerance,
            )
            target[index] = ratio
            target_valid[index] = valid
            reasons[key][index] = "ok" if valid else ratio_reason
            if component == "spring":
                if valid and jacobian_valid[index].all():
                    gains_spring[index] = ratio * jacobian[index]
                    geometry_validity["spring_actuation_gains"][index] = True
                    reasons["spring_actuation_gains"][index, :] = "ok"
            elif valid and jacobian_valid[index].all():
                gains_damper[index] = ratio * jacobian[index]
                geometry_validity["damper_actuation_gains"][index] = True
                reasons["damper_actuation_gains"][index, :] = "ok"

    front_matrix, front_valid = _axle_matrix((0, 1), jacobian, curvature, response, force_valid, curvature_valid)
    rear_matrix, rear_valid = _axle_matrix((2, 3), jacobian, curvature, response, force_valid, curvature_valid)
    axle_heave = np.array([front_matrix[0, 0], rear_matrix[0, 0]], dtype=float)
    axle_roll = np.array([front_matrix[1, 1], rear_matrix[1, 1]], dtype=float)
    axle_pitch = np.array([front_matrix[2, 2], rear_matrix[2, 2]], dtype=float)
    axle_valid = np.isfinite
    front_fraction = float(axle_roll[0] / np.sum(axle_roll)) if np.all(np.isfinite(axle_roll)) and abs(np.sum(axle_roll)) > 1e-15 else float("nan")

    tyre_result = None
    if include_tyres:
        tyre_result = tyre_effective_response(
            geometry, setup, values, solver=solver, base_result={"pose": pose, "response": response}
        )

    world_centers = np.asarray(pose["world_wheel_centers"], dtype=float)
    nominal_centers = origin + np.asarray(
        [geometry["corners"][name]["wheel_center"] for name in CORNER_ORDER], dtype=float
    )
    metrics: dict[str, Any] = {
        "jounce": jounce.copy(),
        "wheel_center_x": world_centers[:, 0].copy(),
        "wheel_center_y": world_centers[:, 1].copy(),
        "wheel_center_z": world_centers[:, 2].copy(),
        "wheel_migration_x": world_centers[:, 0] - nominal_centers[:, 0],
        "wheel_migration_y": world_centers[:, 1] - nominal_centers[:, 1],
        "wheel_migration": world_centers[:, :2] - nominal_centers[:, :2],
        **attitude,
        "spring_compression": spring_compression,
        "damper_compression": damper_compression,
        "rocker_angle": rocker_angle,
        "spring_stroke_margin": spring_margin,
        "damper_stroke_margin": damper_margin,
        "spring_motion_ratio": spring_ratio,
        "damper_motion_ratio": damper_ratio,
        "wheel_force": forces.copy(),
        "wheel_material_rate": np.asarray(response.get("material_rates", _NAN4), dtype=float).copy(),
        "wheel_tangent_rate": np.diag(wheel_stiffness).copy() if wheel_stiffness.shape == (4, 4) else _NAN4.copy(),
        "wheel_energy": float(response.get("energy", np.nan)),
        "wheel_stiffness": wheel_stiffness.copy(),
        "body_gradient": body_gradient,
        "body_restoring_reaction": -body_gradient,
        "body_stiffness": body_stiffness,
        "spring_actuation_gains": gains_spring,
        "damper_actuation_gains": gains_damper,
        "axle_heave_stiffness": axle_heave,
        "axle_roll_stiffness": axle_roll,
        "axle_pitch_stiffness": axle_pitch,
        "heave_stiffness": body_stiffness[0, 0],
        "pitch_stiffness": body_stiffness[2, 2],
        "front_roll_stiffness": axle_roll[0],
        "rear_roll_stiffness": axle_roll[1],
        "total_roll_stiffness": body_stiffness[1, 1],
        "front_elastic_roll_fraction": front_fraction,
    }
    validity: dict[str, Any] = dict(geometry_validity)
    reasons.update({
        "wheel_force": np.asarray(response.get("gradient_reason", np.full(4, "invalid_response", dtype=object)), dtype=object).copy(),
        "wheel_material_rate": np.asarray(response.get("material_rate_reason", np.full(4, "invalid_response", dtype=object)), dtype=object).copy(),
        "wheel_tangent_rate": np.asarray(
            np.diag(response.get("stiffness_reason", np.full((4, 4), "invalid_response", dtype=object))),
            dtype=object,
        ),
    })
    validity["wheel_force"] = np.asarray(response.get("gradient_valid", np.zeros(4, dtype=bool)), dtype=bool)
    validity["wheel_material_rate"] = np.asarray(response.get("material_rate_valid", np.zeros(4, dtype=bool)), dtype=bool)
    validity["wheel_tangent_rate"] = np.diag(wheel_stiffness_valid)
    validity["wheel_energy"] = bool(response.get("valid", False) and np.isfinite(response.get("energy", np.nan)))
    reasons["wheel_energy"] = "ok" if validity["wheel_energy"] else str(response.get("reason", "invalid_response"))
    validity["wheel_stiffness"] = wheel_stiffness_valid.copy()
    reasons["wheel_stiffness"] = np.asarray(response.get("stiffness_reason", np.full((4, 4), "invalid_response", dtype=object)), dtype=object).copy()
    validity["body_gradient"] = body_gradient_valid.copy()
    validity["body_restoring_reaction"] = body_gradient_valid.copy()
    validity["body_stiffness"] = body_stiffness_valid.copy()
    validity["heave_stiffness"] = bool(body_stiffness_valid[0, 0])
    validity["pitch_stiffness"] = bool(body_stiffness_valid[2, 2])
    validity["front_roll_stiffness"] = bool(np.isfinite(axle_roll[0]) and front_valid)
    validity["rear_roll_stiffness"] = bool(np.isfinite(axle_roll[1]) and rear_valid)
    validity["total_roll_stiffness"] = bool(body_stiffness_valid[1, 1])
    validity["front_elastic_roll_fraction"] = bool(np.isfinite(front_fraction))
    reasons["body_gradient"] = body_gradient_reason.copy()
    reasons["body_restoring_reaction"] = body_gradient_reason.copy()
    reasons["body_stiffness"] = body_stiffness_reason.copy()
    for name in ("heave_stiffness", "pitch_stiffness", "total_roll_stiffness"):
        index = {"heave_stiffness": (0, 0), "pitch_stiffness": (2, 2), "total_roll_stiffness": (1, 1)}[name]
        reasons[name] = "ok" if validity[name] else str(body_stiffness_reason[index])
    reasons["front_roll_stiffness"] = "ok" if validity["front_roll_stiffness"] else "invalid_axle"
    reasons["rear_roll_stiffness"] = "ok" if validity["rear_roll_stiffness"] else "invalid_axle"
    reasons["front_elastic_roll_fraction"] = "ok" if validity["front_elastic_roll_fraction"] else "zero_roll_stiffness"

    if tyre_result is not None:
        metrics.update(tyre_result["metrics"])
        validity.update(tyre_result["validity"])
        reasons.update(tyre_result["reasons"])

    full_valid = bool(pose["valid"] and response.get("valid", False) and body_gradient_valid.all())
    reason = "ok" if full_valid else (pose["reason"] if not pose["valid"] else str(response.get("reason", "invalid_derivative")))
    return {
        "valid": full_valid,
        "reason": reason,
        "q": values,
        "u": rises.copy(),
        "pose": pose,
        "jounce": jounce.copy(),
        "corner_valid": pose["corner_valid"].copy(),
        "corner_reason": dict(pose["corner_reason"]),
        "corner_states": pose["corner_states"],
        "world_corner_states": pose["world_corner_states"],
        "response": response,
        "wheel_response": response,
        "pose_derivatives": derivative,
        "metrics": metrics,
        "validity": validity,
        "reasons": reasons,
        "metadata": {
            "q_labels": Q_LABELS,
            "q_definition": "q=[heave_m, roll_rad, pitch_rad], positive heave lowers chassis",
            "rotation_order": "Ry(pitch) @ Rx(roll)",
            "road_support": "fixed nominal world hub heights with free x/y migration",
            "runtime_seconds": perf_counter() - start,
        },
        "tyre": tyre_result,
        # Common fields are mirrored at the top level for lightweight
        # consumers; ``metrics`` remains the authoritative grouped mapping.
        "body_gradient": body_gradient.copy(),
        "body_restoring_reaction": (-body_gradient).copy(),
        "body_stiffness": body_stiffness.copy(),
        "wheel_stiffness": wheel_stiffness.copy(),
        "wheel_force": forces.copy(),
        "validity_masks": validity,
        "reason_codes": reasons,
    }


def _tyre_spec(setup: Mapping[str, Any], index: int) -> Mapping[str, Any] | None:
    try:
        value = setup["corners"][CORNER_ORDER[index]].get("tyre")
    except (KeyError, AttributeError, TypeError):
        return None
    return value if isinstance(value, Mapping) else None


def _tyre_law(spec: Mapping[str, Any], compression: float) -> tuple[float, float, float]:
    # Tyre setup uses ``reference_force`` rather than the spring law's
    # ``preload_force`` name.  Preserve that operating-point force when
    # evaluating the stored-energy derivative.
    if "reference_force" in spec and "rate" in spec and "curve" not in spec:
        law = {"rate": spec["rate"], "preload_force": spec["reference_force"]}
        return component_law(law, compression)
    return component_law(spec, compression)


def _energy_at_internal(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    q: np.ndarray,
    u: np.ndarray,
    *,
    max_nfev: int,
    tolerance: float,
) -> tuple[float, dict[str, Any], bool, str, np.ndarray]:
    pose = pose_jounces(geometry, q, u=u, max_nfev=max_nfev, tolerance=tolerance)
    if not pose["valid"]:
        return float("nan"), pose, False, pose["reason"], np.full(4, np.nan)
    energy = wheel_energy(geometry, setup, pose["jounce"], max_nfev=max_nfev, tolerance=tolerance)
    if not energy["valid"]:
        return float("nan"), pose, False, energy["reason"], pose["jounce"]
    tyre_energy = 0.0
    for index in range(4):
        try:
            tyre_energy += _tyre_law(_tyre_spec(setup, index), float(-u[index]))[0]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan"), pose, False, "tyre_law_invalid", pose["jounce"]
    return float(energy["energy"] + tyre_energy), pose, True, "ok", pose["jounce"]


def _implicit_jounce_per_hub_rise(
    geometry: Mapping[str, Any],
    corner_name: str,
    q: np.ndarray,
    jounce: float,
    *,
    step: float,
    max_nfev: int,
    tolerance: float,
) -> tuple[float, bool, str]:
    """Return ``dj/du`` from one exact corner height map.

    At fixed body pose the corner constraint is
    ``e_z.T R p(j) = z_reference + u``.  Only this corner contributes to its
    own internal coordinate, so differentiating that scalar constraint avoids
    perturbing all four hubs for every residual evaluation.
    """
    corner = geometry["corners"][corner_name]
    low, high = map(float, corner["jounce_limits"])
    j = float(jounce)
    plus_step = min(float(step), max(0.0, high - j))
    minus_step = min(float(step), max(0.0, j - low))
    rotation = body_rotation(q)

    def height(offset: float) -> tuple[float, bool, str]:
        state = solve_corner(
            corner, j + offset, max_nfev=max_nfev, tolerance=tolerance
        )
        if not state.valid:
            return float("nan"), False, state.reason
        return float((rotation @ np.asarray(state.wheel_center, dtype=float))[2]), True, "ok"

    if plus_step > 1e-14 and minus_step > 1e-14:
        plus, plus_valid, plus_reason = height(plus_step)
        minus, minus_valid, minus_reason = height(-minus_step)
        if plus_valid and minus_valid:
            dz_dj = (plus - minus) / (plus_step + minus_step)
            if np.isfinite(dz_dj) and abs(dz_dj) > 1e-12:
                return float(1.0 / dz_dj), True, "central2"
        reason = plus_reason if not plus_valid else minus_reason
    elif plus_step > 1e-14:
        center, center_valid, center_reason = height(0.0)
        plus, plus_valid, plus_reason = height(plus_step)
        if center_valid and plus_valid:
            dz_dj = (plus - center) / plus_step
            if np.isfinite(dz_dj) and abs(dz_dj) > 1e-12:
                return float(1.0 / dz_dj), True, "one_sided1"
        reason = center_reason if not center_valid else plus_reason
    elif minus_step > 1e-14:
        center, center_valid, center_reason = height(0.0)
        minus, minus_valid, minus_reason = height(-minus_step)
        if center_valid and minus_valid:
            dz_dj = (center - minus) / minus_step
            if np.isfinite(dz_dj) and abs(dz_dj) > 1e-12:
                return float(1.0 / dz_dj), True, "one_sided1"
        reason = center_reason if not center_valid else minus_reason
    else:
        reason = "jounce_limit"
    return float("nan"), False, str(reason)


def _body_rotation_derivatives(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``R``, first, and second derivatives for ``Ry(theta)Rx(phi)``."""
    roll = float(q[1])
    pitch = float(q[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rx_roll = np.array([[0.0, 0.0, 0.0], [0.0, -sr, -cr], [0.0, cr, -sr]])
    rx_roll_roll = np.array([[0.0, 0.0, 0.0], [0.0, -cr, sr], [0.0, -sr, -cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    ry_pitch = np.array([[-sp, 0.0, cp], [0.0, 0.0, 0.0], [-cp, 0.0, -sp]])
    ry_pitch_pitch = np.array([[-cp, 0.0, -sp], [0.0, 0.0, 0.0], [sp, 0.0, -cp]])
    rotation = ry @ rx
    first = np.zeros((3, 3, 3), dtype=float)
    first[1] = ry @ rx_roll
    first[2] = ry_pitch @ rx
    second = np.zeros((3, 3, 3, 3), dtype=float)
    second[1, 1] = ry @ rx_roll_roll
    second[2, 2] = ry_pitch_pitch @ rx
    second[1, 2] = second[2, 1] = ry_pitch @ rx_roll
    return rotation, first, second


def _corner_path_derivatives(
    corner: Mapping[str, Any],
    jounce: float,
    *,
    step: float,
    max_nfev: int,
    tolerance: float,
    center_state: CornerState | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, str, str, int]:
    """Differentiate one exact local wheel-centre path with strict solves."""
    low, high = map(float, corner["jounce_limits"])
    center = center_state or solve_corner(corner, jounce, max_nfev=max_nfev, tolerance=tolerance)
    if not center.valid:
        return np.full(3, np.nan), np.full(3, np.nan), False, center.reason, "invalid", 1

    def solve(offset: float) -> CornerState:
        return solve_corner(corner, jounce + offset, max_nfev=max_nfev, tolerance=tolerance)

    distance_plus = max(0.0, high - jounce)
    distance_minus = max(0.0, jounce - low)
    requested = float(step)
    if distance_plus >= requested and distance_minus >= requested:
        plus = solve(requested)
        minus = solve(-requested)
        if plus.valid and minus.valid:
            p0 = np.asarray(center.wheel_center, dtype=float)
            pp = np.asarray(plus.wheel_center, dtype=float)
            pm = np.asarray(minus.wheel_center, dtype=float)
            return (
                (pp - pm) / (2.0 * requested),
                (pp - 2.0 * p0 + pm) / requested**2,
                True,
                "ok",
                "central2",
                3,
            )

    # A second-order one-sided stencil is used only when two points fit on
    # one side.  This keeps the method label truthful and preserves p''.
    if distance_plus > 1e-12:
        one_step = min(requested, 0.5 * distance_plus)
        first = solve(one_step)
        second = solve(2.0 * one_step)
        if first.valid and second.valid and one_step > 1e-12:
            p0 = np.asarray(center.wheel_center, dtype=float)
            p1 = np.asarray(first.wheel_center, dtype=float)
            p2 = np.asarray(second.wheel_center, dtype=float)
            return (
                (-3.0 * p0 + 4.0 * p1 - p2) / (2.0 * one_step),
                (p0 - 2.0 * p1 + p2) / one_step**2,
                True,
                "ok",
                "one_sided2",
                3,
            )
    if distance_minus > 1e-12:
        one_step = min(requested, 0.5 * distance_minus)
        first = solve(-one_step)
        second = solve(-2.0 * one_step)
        if first.valid and second.valid and one_step > 1e-12:
            p0 = np.asarray(center.wheel_center, dtype=float)
            p1 = np.asarray(first.wheel_center, dtype=float)
            p2 = np.asarray(second.wheel_center, dtype=float)
            return (
                (3.0 * p0 - 4.0 * p1 + p2) / (2.0 * one_step),
                (p0 - 2.0 * p1 + p2) / one_step**2,
                True,
                "ok",
                "one_sided2",
                3,
            )
    return np.full(3, np.nan), np.full(3, np.nan), False, "path_derivative_invalid", "invalid", 1


def _implicit_pose_map_derivatives(
    geometry: Mapping[str, Any],
    q: np.ndarray,
    u: np.ndarray,
    jounce: np.ndarray,
    pose: Mapping[str, Any],
    *,
    step: float,
    max_nfev: int,
    tolerance: float,
) -> dict[str, Any]:
    """Compose exact q/u derivatives of the constrained jounce map."""
    rotation, rotation_first, rotation_second = _body_rotation_derivatives(q)
    jacobian_q = np.full((4, 3), np.nan)
    hessian_qq = np.full((4, 3, 3), np.nan)
    jacobian_u = np.full((4, 4), np.nan)
    hessian_qu = np.full((4, 3, 4), np.nan)
    hessian_uu = np.full((4, 4, 4), np.nan)
    valid_q = np.zeros((4, 3), dtype=bool)
    valid_qq = np.zeros((4, 3, 3), dtype=bool)
    valid_u = np.zeros((4, 4), dtype=bool)
    valid_qu = np.zeros((4, 3, 4), dtype=bool)
    valid_uu = np.zeros((4, 4, 4), dtype=bool)
    methods = np.full(4, "invalid", dtype=object)
    reasons = np.full(4, "invalid_pose_derivative", dtype=object)
    sample_count = 0
    ez = np.array([0.0, 0.0, 1.0])

    for index, name in enumerate(CORNER_ORDER):
        if not bool(np.asarray(pose.get("corner_valid", np.zeros(4, dtype=bool)))[index]):
            reasons[index] = str(pose.get("corner_reason", {}).get(name, "invalid_pose"))
            continue
        state = pose.get("corner_states", {}).get(name)
        if state is None or not state.valid or not np.isfinite(jounce[index]):
            reasons[index] = "invalid_pose"
            continue
        path_first, path_second, path_valid, path_reason, method, count = _corner_path_derivatives(
            geometry["corners"][name], float(jounce[index]), step=step,
            max_nfev=max_nfev, tolerance=tolerance, center_state=state,
        )
        sample_count += count
        if not path_valid:
            reasons[index] = path_reason
            continue
        point = np.asarray(state.wheel_center, dtype=float)
        fj = float(ez @ rotation @ path_first)
        fjj = float(ez @ rotation @ path_second)
        if not np.isfinite(fj) or abs(fj) <= 1e-12:
            reasons[index] = "height_tangent_singular"
            continue
        fa = np.array([-1.0, ez @ rotation_first[1] @ point, ez @ rotation_first[2] @ point])
        jacobian_q[index] = -fa / fj
        # Each corner's jounce depends on its own hub rise only.  The zero
        # cross-coordinate derivatives are exact and therefore valid masks,
        # rather than unavailable values.
        jacobian_u[index, :] = 0.0
        jacobian_u[index, index] = 1.0 / fj
        hessian_qu[index, :, :] = 0.0
        hessian_uu[index, :, :] = 0.0
        valid_u[index, :] = np.isfinite(jacobian_u[index])
        valid_qu[index, :, :] = True
        valid_uu[index, :, :] = True
        valid_q[index] = np.isfinite(jacobian_q[index])
        for first in range(3):
            for second in range(3):
                fab = float(ez @ rotation_second[first, second] @ point)
                faj_first = 0.0 if first == 0 else float(ez @ rotation_first[first] @ path_first)
                faj_second = 0.0 if second == 0 else float(ez @ rotation_first[second] @ path_first)
                value = -(fab + faj_first * jacobian_q[index, second] + faj_second * jacobian_q[index, first] + fjj * jacobian_q[index, first] * jacobian_q[index, second]) / fj
                hessian_qq[index, first, second] = value
                valid_qq[index, first, second] = np.isfinite(value)
        for axis in range(3):
            faj = 0.0 if axis == 0 else float(ez @ rotation_first[axis] @ path_first)
            value = -(faj + fjj * jacobian_q[index, axis]) * jacobian_u[index, index] / fj
            hessian_qu[index, axis, index] = value
            valid_qu[index, axis, index] = np.isfinite(value)
        value = -fjj * jacobian_u[index, index] ** 2 / fj
        hessian_uu[index, index, index] = value
        valid_uu[index, index, index] = np.isfinite(value)
        methods[index] = method
        reasons[index] = "ok" if np.all(valid_q[index]) and np.all(valid_qq[index]) else "invalid_pose_derivative"

    return {
        "rotation": rotation,
        "jacobian_q": jacobian_q,
        "hessian_qq": hessian_qq,
        "jacobian_u": jacobian_u,
        "hessian_qu": hessian_qu,
        "hessian_uu": hessian_uu,
        "jacobian_valid": valid_q,
        "hessian_qq_valid": valid_qq,
        "jacobian_u_valid": valid_u,
        "hessian_qu_valid": valid_qu,
        "hessian_uu_valid": valid_uu,
        "method": methods,
        "reason": reasons,
        "sample_count": sample_count,
    }


def _compose_pose_hessian(
    response: Mapping[str, Any],
    derivatives: Mapping[str, Any],
    tyre_tangent: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply the full jounce-map chain rule to q/u Hessian blocks."""
    force = np.asarray(response.get("gradient", _NAN4), dtype=float)
    stiffness = np.asarray(response.get("stiffness", _NAN44), dtype=float)
    force_valid = np.asarray(response.get("gradient_valid", np.zeros(4, dtype=bool)), dtype=bool)
    stiffness_valid = np.asarray(response.get("stiffness_valid", np.zeros((4, 4), dtype=bool)), dtype=bool)
    jq = np.asarray(derivatives["jacobian_q"], dtype=float)
    ju = np.asarray(derivatives["jacobian_u"], dtype=float)
    hqq_map = np.asarray(derivatives["hessian_qq"], dtype=float)
    hqu_map = np.asarray(derivatives["hessian_qu"], dtype=float)
    huu_map = np.asarray(derivatives["hessian_uu"], dtype=float)
    valid_q = np.asarray(derivatives["jacobian_valid"], dtype=bool)
    valid_qq = np.asarray(derivatives["hessian_qq_valid"], dtype=bool)
    valid_u = np.asarray(derivatives["jacobian_u_valid"], dtype=bool)
    valid_qu = np.asarray(derivatives["hessian_qu_valid"], dtype=bool)
    valid_uu = np.asarray(derivatives["hessian_uu_valid"], dtype=bool)
    body_gradient = jq.T @ force if np.all(valid_q) and np.all(force_valid) else np.full(3, np.nan)
    hqq = np.full((3, 3), np.nan)
    hqu = np.full((3, 4), np.nan)
    huu = np.full((4, 4), np.nan)
    if np.all(force_valid) and np.all(stiffness_valid):
        for first in range(3):
            for second in range(3):
                if np.all(valid_q[:, first]) and np.all(valid_q[:, second]) and np.all(valid_qq[:, first, second]):
                    hqq[first, second] = jq[:, first] @ stiffness @ jq[:, second] + np.sum(force * hqq_map[:, first, second])
        for first in range(3):
            for second in range(4):
                if np.all(valid_q[:, first]) and valid_u[second, second] and np.all(valid_qu[:, first, second]):
                    hqu[first, second] = jq[:, first] @ stiffness @ ju[:, second] + np.sum(force * hqu_map[:, first, second])
        for first in range(4):
            for second in range(4):
                if valid_u[first, first] and valid_u[second, second] and np.all(valid_uu[:, first, second]):
                    huu[first, second] = ju[:, first] @ stiffness @ ju[:, second] + np.sum(force * huu_map[:, first, second])
    if tyre_tangent is not None and np.all(np.isfinite(tyre_tangent)):
        huu = huu + np.diag(np.asarray(tyre_tangent, dtype=float))
    hqq = 0.5 * (hqq + hqq.T) if np.all(np.isfinite(hqq)) else hqq
    huu = 0.5 * (huu + huu.T) if np.all(np.isfinite(huu)) else huu
    full = np.block([[hqq, hqu], [hqu.T, huu]])
    return {"body_gradient": body_gradient, "hqq": hqq, "hqu": hqu, "huu": huu, "full": full}


def _internal_force(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    q: np.ndarray,
    u: np.ndarray,
    *,
    max_nfev: int,
    tolerance: float,
    step: float,
) -> tuple[np.ndarray, dict[str, Any], bool, str]:
    total_energy, pose, valid, reason, jounce = _energy_at_internal(
        geometry, setup, q, u, max_nfev=max_nfev, tolerance=tolerance
    )
    if not valid:
        return np.full(4, np.nan), pose, False, reason
    response = wheel_response(geometry, setup, jounce, derivative_steps=step, max_nfev=max_nfev, tolerance=tolerance)
    if not np.asarray(response.get("gradient_valid", np.zeros(4, dtype=bool))).all():
        return np.full(4, np.nan), pose, False, "suspension_force_invalid"
    # For fixed q each u_i only changes its own wheel jounce.  Differentiate
    # that corner's exact scalar height constraint instead of solving a full
    # four-corner pose for every finite-difference perturbation.
    jacobian = np.full(4, np.nan)
    jacobian_reason: list[str] = []
    for index, name in enumerate(CORNER_ORDER):
        jacobian[index], valid, reason = _implicit_jounce_per_hub_rise(
            geometry, name, q, float(jounce[index]), step=step,
            max_nfev=max_nfev, tolerance=tolerance,
        )
        if not valid:
            jacobian_reason.append(reason)
    if not np.all(np.isfinite(jacobian)):
        return np.full(4, np.nan), pose, False, jacobian_reason[0] if jacobian_reason else "pose_derivative_invalid"
    values = np.full(4, np.nan)
    for index in range(4):
        try:
            tyre_force = _tyre_law(_tyre_spec(setup, index), float(-u[index]))[1]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return np.full(4, np.nan), pose, False, "tyre_law_invalid"
        values[index] = response["gradient"][index] * jacobian[index] - tyre_force
    return values, pose, bool(np.all(np.isfinite(values))), "ok" if np.all(np.isfinite(values)) else "invalid_force"


def _energy_hessian(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    q: np.ndarray,
    u: np.ndarray,
    q_steps: np.ndarray,
    u_step: float,
    *,
    max_nfev: int,
    tolerance: float,
    include_tyres: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """Finite-difference total/suspension Hessians at one equilibrium."""
    n = 7
    x0 = np.concatenate([q, u])
    steps = np.concatenate([q_steps, np.full(4, u_step)])
    values: dict[tuple[int, ...], float] = {}
    valid = True
    reason = "ok"

    def energy(offset: tuple[int, ...]) -> float:
        nonlocal valid, reason
        if offset in values:
            return values[offset]
        x = x0 + np.asarray(offset, dtype=float) * steps
        total, _, ok, why, _ = _energy_at_internal(
            geometry, setup, x[:3], x[3:], max_nfev=max_nfev, tolerance=tolerance
        )
        if not ok:
            valid = False
            reason = why
            values[offset] = float("nan")
            return values[offset]
        values[offset] = total
        return total

    center = energy((0,) * n)
    hessian = np.full((n, n), np.nan)
    if not np.isfinite(center):
        return hessian[:3, :3], hessian[:3, 3:], hessian[3:, 3:], False, reason
    for i in range(n):
        plus = [0] * n; plus[i] = 1
        minus = [0] * n; minus[i] = -1
        fplus, fminus = energy(tuple(plus)), energy(tuple(minus))
        if np.isfinite(fplus) and np.isfinite(fminus):
            hessian[i, i] = (fplus - 2.0 * center + fminus) / steps[i] ** 2
        else:
            valid = False
    for i in range(n):
        for j in range(i):
            pp = [0] * n; pp[i] = 1; pp[j] = 1
            pm = [0] * n; pm[i] = 1; pm[j] = -1
            mp = [0] * n; mp[i] = -1; mp[j] = 1
            mm = [0] * n; mm[i] = -1; mm[j] = -1
            samples = [energy(tuple(item)) for item in (pp, pm, mp, mm)]
            if np.all(np.isfinite(samples)):
                value = (samples[0] - samples[1] - samples[2] + samples[3]) / (4.0 * steps[i] * steps[j])
                hessian[i, j] = value
                hessian[j, i] = value
            else:
                valid = False
    # Recompute the suspension-only u Hessian separately so ride_stiffness
    # can expose the coupled series result with a diagonal tyre tangent.
    suspension_huu = np.full((4, 4), np.nan)
    base_values: dict[tuple[int, ...], float] = {}

    def suspension_energy(offset: tuple[int, ...]) -> float:
        if offset in base_values:
            return base_values[offset]
        uu = u + np.asarray(offset, dtype=float) * u_step
        pose = pose_jounces(geometry, q, u=uu, max_nfev=max_nfev, tolerance=tolerance)
        if not pose["valid"]:
            base_values[offset] = float("nan")
        else:
            result = wheel_energy(geometry, setup, pose["jounce"], max_nfev=max_nfev, tolerance=tolerance)
            base_values[offset] = float(result["energy"]) if result["valid"] else float("nan")
        return base_values[offset]

    center_suspension = suspension_energy((0, 0, 0, 0))
    for i in range(4):
        plus = [0] * 4; plus[i] = 1
        minus = [0] * 4; minus[i] = -1
        fp, fm = suspension_energy(tuple(plus)), suspension_energy(tuple(minus))
        if np.all(np.isfinite([fp, fm, center_suspension])):
            suspension_huu[i, i] = (fp - 2.0 * center_suspension + fm) / u_step**2
        for j in range(i):
            pp = [0] * 4; pp[i] = 1; pp[j] = 1
            pm = [0] * 4; pm[i] = 1; pm[j] = -1
            mp = [0] * 4; mp[i] = -1; mp[j] = 1
            mm = [0] * 4; mm[i] = -1; mm[j] = -1
            samples = [suspension_energy(tuple(item)) for item in (pp, pm, mp, mm)]
            if np.all(np.isfinite(samples)):
                suspension_huu[i, j] = suspension_huu[j, i] = (samples[0] - samples[1] - samples[2] + samples[3]) / (4.0 * u_step**2)
    return hessian[:3, :3], hessian[:3, 3:], suspension_huu, bool(valid and np.all(np.isfinite(hessian))), reason


def _condense_internal_hessian(
    suspension_huu: Any,
    tyre_tangent: Any,
    body_hqq: Any | None = None,
    body_hqu: Any | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Condense internal hub coordinates from a quadratic energy model.

    ``suspension_huu`` is the suspension-only Hessian with respect to world
    hub rises, while ``tyre_tangent`` is the diagonal tyre Hessian.  The
    returned wheel matrix is ``A - A(A+T)^-1A``; optional body blocks use the
    matching Schur term ``Hqq-Hqu(A+T)^-1Huq``.  Keeping this algebra in one
    small helper makes the unit-MR quadratic fixture and the nonlinear solver
    share the same condensation path.
    """
    suspension = np.asarray(suspension_huu, dtype=float)
    tangent = np.asarray(tyre_tangent, dtype=float)
    if suspension.shape != (4, 4) or tangent.shape != (4,):
        raise ValueError("internal Hessian must be [4,4] and tyre tangent [4]")
    if not np.all(np.isfinite(suspension)) or not np.all(np.isfinite(tangent)):
        raise ValueError("internal Hessian and tyre tangent must be finite")
    total = suspension + np.diag(tangent)
    solved = np.linalg.solve(total, suspension)
    ride = suspension - suspension @ solved
    ride = 0.5 * (ride + ride.T)
    body: np.ndarray | None = None
    if body_hqq is not None or body_hqu is not None:
        if body_hqq is None or body_hqu is None:
            raise ValueError("body_hqq and body_hqu must be supplied together")
        hqq = np.asarray(body_hqq, dtype=float)
        hqu = np.asarray(body_hqu, dtype=float)
        if hqq.shape != (3, 3) or hqu.shape != (3, 4):
            raise ValueError("body Hessian blocks must be [3,3] and [3,4]")
        body = hqq - hqu @ np.linalg.solve(total, hqu.T)
        body = 0.5 * (body + body.T)
    return ride, body, total


def tyre_effective_response(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    q: Any,
    solver: Any | None = None,
    *,
    base_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve internal vertical hub/tyre equilibrium and condense its Hessian."""
    values = _finite_array(q, (3,), "q")
    max_nfev, tolerance, derivative_steps, jounce_step = _solver_options(solver)
    tyre_specs = [_tyre_spec(setup, index) for index in range(4)]
    if any(spec is None for spec in tyre_specs):
        nan_metrics = {
            "ride_rate": _NAN4.copy(), "ride_stiffness": _NAN44.copy(), "body_ride_stiffness": _NAN33.copy(),
            "tyre_equilibrium_hub_displacement": _NAN4.copy(), "tyre_equilibrium_jounce": _NAN4.copy(),
            "tyre_contact_force": _NAN4.copy(), "tyre_effective_reaction": np.full(3, np.nan),
        }
        validity = {key: (np.zeros_like(value, dtype=bool) if isinstance(value, np.ndarray) else False) for key, value in nan_metrics.items()}
        reasons = {key: (np.full(value.shape, "tyres_unavailable", dtype=object) if isinstance(value, np.ndarray) else "tyres_unavailable") for key, value in nan_metrics.items()}
        return {"valid": False, "reason": "tyres_unavailable", "metrics": nan_metrics, "validity": validity, "reasons": reasons,
                "ride_rate": _NAN4.copy(), "ride_stiffness": _NAN44.copy(), "body_ride_stiffness": _NAN33.copy(),
                "tyre_equilibrium_hub_displacement": _NAN4.copy(), "tyre_equilibrium_jounce": _NAN4.copy(), "tyre_contact_force": _NAN4.copy()}

    # Internal u bounds are world-height offsets of exact local jounce limits,
    # not jounce limits copied into a different coordinate.  The initial guess
    # is the u that places local jounce at zero (or the closest feasible
    # assembly sample), which remains useful when q moves the prescribed u=0
    # support outside a tight stroke range.
    rotation = body_rotation(values)
    origin = _origin(geometry)
    low = np.full(4, np.nan, dtype=float)
    high = np.full(4, np.nan, dtype=float)
    initial = np.full(4, np.nan, dtype=float)
    for index, name in enumerate(CORNER_ORDER):
        corner = geometry["corners"][name]
        reference = np.asarray(corner["wheel_center"], dtype=float)
        target = float(origin[2] + reference[2])
        j_low, j_high = map(float, corner["jounce_limits"])
        candidates = np.linspace(j_low, j_high, 25)
        offset_samples: list[tuple[float, float]] = []
        closest: tuple[float, float] | None = None
        for j_candidate in candidates:
            state = solve_corner(corner, float(j_candidate), max_nfev=max_nfev, tolerance=tolerance)
            if not state.valid:
                continue
            world = _world_center(origin, rotation, state.wheel_center, float(values[0]))
            offset = float(world[2] - target)
            if np.isfinite(offset):
                offset_samples.append((float(j_candidate), offset))
                if closest is None or abs(j_candidate) < abs(closest[0]):
                    closest = (float(j_candidate), offset)
        if not offset_samples:
            return _invalid_tyre_result("tyre_equilibrium_bounds")
        offsets = np.asarray([item[1] for item in offset_samples], dtype=float)
        low[index] = float(np.min(offsets))
        high[index] = float(np.max(offsets))
        if low[index] >= high[index] - 1e-12:
            return _invalid_tyre_result("tyre_equilibrium_bounds")
        zero_state = solve_corner(corner, 0.0, max_nfev=max_nfev, tolerance=tolerance)
        if zero_state.valid:
            world_zero = _world_center(origin, rotation, zero_state.wheel_center, float(values[0]))
            initial[index] = float(world_zero[2] - target)
        else:
            initial[index] = float(closest[1]) if closest is not None else float(0.5 * (low[index] + high[index]))
    initial = np.clip(initial, low + 1e-10, high - 1e-10)

    def residual(internal: np.ndarray) -> np.ndarray:
        forces, _, valid, _, = _internal_force(
            geometry, setup, values, np.asarray(internal, dtype=float), max_nfev=max_nfev, tolerance=tolerance, step=jounce_step
        )
        if not valid:
            return np.full(4, 1e9, dtype=float)
        return forces

    fit = least_squares(
        residual,
        initial,
        bounds=(low, high),
        x_scale=np.maximum(0.01, high - low),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max(25, max_nfev),
    )
    u_eq = np.asarray(fit.x, dtype=float)
    equilibrium_force = residual(u_eq)
    if not fit.success or not np.all(np.isfinite(equilibrium_force)) or np.max(np.abs(equilibrium_force)) > max(2e-5, tolerance * 100.0):
        return _invalid_tyre_result("tyre_equilibrium_failed", u_eq=u_eq)
    total_energy, pose_eq, energy_valid, energy_reason, jounce_eq = _energy_at_internal(
        geometry, setup, values, u_eq, max_nfev=max_nfev, tolerance=tolerance
    )
    if not energy_valid:
        return _invalid_tyre_result(energy_reason, u_eq=u_eq)
    contact = np.full(4, np.nan)
    tangents = np.full(4, np.nan)
    for index, spec in enumerate(tyre_specs):
        try:
            _, contact[index], tangents[index] = _tyre_law(spec, float(-u_eq[index]))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _invalid_tyre_result("tyre_law_invalid", u_eq=u_eq)
    # Zero force is the unilateral contact-release boundary.  It cannot
    # support a bilateral local ride tangent, so only strictly positive
    # contact is exported as a valid tyre-inclusive state.
    if not np.all(np.isfinite(contact)) or np.any(contact <= max(1e-9, tolerance * 10.0)):
        return _invalid_tyre_result("nonpositive_contact", u_eq=u_eq)
    if not np.all(np.isfinite(tangents)) or np.any(tangents < -1e-9):
        return _invalid_tyre_result("invalid_tyre_tangent", u_eq=u_eq)

    # A kink at the initial guess is allowed; the final law tangents are the
    # acceptance gate.  Never report a central-difference average at a final
    # bump-stop/force-curve kink as a valid ride tangent.
    final_response = wheel_response(
        geometry,
        setup,
        jounce_eq,
        derivative_steps=jounce_step,
        max_nfev=max_nfev,
        tolerance=tolerance,
    )
    if not np.asarray(final_response.get("stiffness_valid", np.zeros((4, 4), dtype=bool))).all():
        return _invalid_tyre_result("suspension_tangent_invalid", u_eq=u_eq)
    hqq, hqu, suspension_huu, hessian_valid, hessian_reason = _energy_hessian(
        geometry, setup, values, u_eq, derivative_steps, jounce_step,
        max_nfev=max_nfev, tolerance=tolerance, include_tyres=True,
    )
    total_huu = suspension_huu + np.diag(tangents)
    stable = bool(np.all(np.isfinite(total_huu)))
    condition = float(np.linalg.cond(total_huu)) if stable else float("inf")
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (total_huu + total_huu.T))
    except np.linalg.LinAlgError:
        eigenvalues = np.full(4, np.nan)
    stable = bool(stable and np.isfinite(condition) and condition < 1e10 and np.all(eigenvalues > 0.0))
    if not stable or not hessian_valid:
        return _invalid_tyre_result("unstable_internal_hessian" if not stable else hessian_reason, u_eq=u_eq)
    try:
        ride_stiffness, body_ride, _ = _condense_internal_hessian(
            suspension_huu, tangents, hqq, hqu
        )
    except (np.linalg.LinAlgError, ValueError):
        return _invalid_tyre_result("singular_internal_hessian", u_eq=u_eq)
    base = base_result or evaluate_pose(geometry, setup, values, solver=solver, include_tyres=False, u=np.zeros(4))
    base_response = base.get("response", base.get("wheel_response", {}))
    base_reaction = np.asarray(base.get("metrics", {}).get("body_restoring_reaction", np.full(3, np.nan)), dtype=float)
    effective_response = evaluate_pose(geometry, setup, values, solver=solver, include_tyres=False, u=u_eq)
    effective_reaction = np.asarray(effective_response["metrics"].get("body_restoring_reaction", np.full(3, np.nan)), dtype=float)
    metrics = {
        "ride_rate": np.diag(ride_stiffness).copy(),
        "ride_stiffness": ride_stiffness,
        "body_ride_stiffness": body_ride,
        "tyre_equilibrium_hub_displacement": u_eq.copy(),
        "tyre_equilibrium_jounce": np.asarray(jounce_eq, dtype=float).copy(),
        "tyre_contact_force": contact.copy(),
        "tyre_effective_reaction": effective_reaction.copy(),
    }
    validity = {
        "ride_rate": np.ones(4, dtype=bool),
        "ride_stiffness": np.ones((4, 4), dtype=bool),
        "body_ride_stiffness": np.ones((3, 3), dtype=bool),
        "tyre_equilibrium_hub_displacement": np.ones(4, dtype=bool),
        "tyre_equilibrium_jounce": np.ones(4, dtype=bool),
        "tyre_contact_force": np.ones(4, dtype=bool),
        "tyre_effective_reaction": np.ones(3, dtype=bool),
    }
    reasons = {key: np.full(np.asarray(value).shape, "ok", dtype=object) for key, value in metrics.items()}
    return {
        "valid": True,
        "reason": "ok",
        "metrics": metrics,
        "validity": validity,
        "reasons": reasons,
        "tyre_equilibrium_hub_displacement": u_eq.copy(),
        "tyre_equilibrium_jounce": np.asarray(jounce_eq, dtype=float).copy(),
        "tyre_contact_force": contact.copy(),
        "ride_rate": np.diag(ride_stiffness).copy(),
        "ride_stiffness": ride_stiffness,
        "body_ride_stiffness": body_ride,
        "internal_hessian": total_huu,
        "suspension_hub_hessian": suspension_huu,
        "tyre_tangent": tangents,
        "equilibrium_force": equilibrium_force,
        "base_pose_reaction": base_reaction,
        "tyre_effective_reaction": effective_reaction,
        "base_response": base_response,
        "pose": pose_eq,
        "energy": total_energy,
        "condition": condition,
    }


def _invalid_tyre_result(reason: str, *, u_eq: np.ndarray | None = None) -> dict[str, Any]:
    if u_eq is None:
        u_eq = _NAN4.copy()
    metrics = {
        "ride_rate": _NAN4.copy(), "ride_stiffness": _NAN44.copy(), "body_ride_stiffness": _NAN33.copy(),
        "tyre_equilibrium_hub_displacement": np.asarray(u_eq, dtype=float).copy(),
        "tyre_equilibrium_jounce": _NAN4.copy(), "tyre_contact_force": _NAN4.copy(),
        "tyre_effective_reaction": np.full(3, np.nan),
    }
    validity = {key: np.zeros_like(np.asarray(value), dtype=bool) for key, value in metrics.items()}
    reasons = {key: np.full(np.asarray(value).shape, reason, dtype=object) for key, value in metrics.items()}
    return {
        "valid": False,
        "reason": reason,
        "metrics": metrics,
        "validity": validity,
        "reasons": reasons,
        "ride_rate": metrics["ride_rate"].copy(),
        "ride_stiffness": metrics["ride_stiffness"].copy(),
        "body_ride_stiffness": metrics["body_ride_stiffness"].copy(),
        "tyre_equilibrium_hub_displacement": np.asarray(u_eq, dtype=float).copy(),
        "tyre_equilibrium_jounce": _NAN4.copy(),
        "tyre_contact_force": _NAN4.copy(),
    }


# Descriptive aliases used by downstream callers and early Task 4 probes.
solve_pose = pose_jounces
solve_tyre_equilibrium = tyre_effective_response


__all__ = [
    "body_rotation", "pose_jounces", "solve_pose", "evaluate_pose",
    "tyre_effective_response", "solve_tyre_equilibrium",
]
