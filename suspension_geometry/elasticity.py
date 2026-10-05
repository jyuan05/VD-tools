"""Elastic component energies and geometry-derived wheel responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .actuation import actuation_state, arb_angles
from .kinematics import solve_corner
from .numerics import finite_difference_derivatives


CORNER_ORDER = ("FL", "FR", "RL", "RR")
_CORNER_INDEX = {name: index for index, name in enumerate(CORNER_ORDER)}
_CURVE_ZERO_TOL = 1e-12
_KINK_TOL = 1e-12
_SLOPE_TOL = 1e-9

# Curves interpolate force linearly and integrate that interpolant exactly.
# At a knot with unequal adjacent slopes, force is defined but tangent is not.
CURVE_INTERPOLATION = "piecewise_linear_force"


class LawDomainError(ValueError):
    """A force law has no configured value at the requested displacement."""


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _validated_curve(spec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = spec.get("curve")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("force curve must contain at least two [displacement, force] points")
    points = np.asarray(raw, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("force curve points must be finite [displacement, force] pairs")
    x, force = points.T
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("force curve displacement values must be strictly increasing")
    slopes = np.diff(force) / np.diff(x)
    if np.any(slopes < -_SLOPE_TOL):
        raise ValueError("force curve must have a passive non-negative tangent")
    if x[0] > _CURVE_ZERO_TOL or x[-1] < -_CURVE_ZERO_TOL:
        raise LawDomainError("force curve domain must include zero displacement")
    return x, force


def _curve_primitive(x: np.ndarray, force: np.ndarray, displacement: float) -> float:
    if displacement < x[0] - _CURVE_ZERO_TOL or displacement > x[-1] + _CURVE_ZERO_TOL:
        raise LawDomainError("displacement is outside the configured force curve domain")
    value = float(np.clip(displacement, x[0], x[-1]))
    segment = int(np.searchsorted(x, value, side="right") - 1)
    segment = min(max(segment, 0), x.size - 2)
    widths = np.diff(x)
    slopes = np.diff(force) / widths
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (force[:-1] + force[1:]) * widths)))
    offset = value - x[segment]
    return float(cumulative[segment] + force[segment] * offset + 0.5 * slopes[segment] * offset**2)


def _curve_law(spec: Mapping[str, Any], displacement: float) -> tuple[float, float, float]:
    x, force_values = _validated_curve(spec)
    if displacement < x[0] - _CURVE_ZERO_TOL or displacement > x[-1] + _CURVE_ZERO_TOL:
        raise LawDomainError("displacement is outside the configured force curve domain")
    value = float(np.clip(displacement, x[0], x[-1]))
    segment = int(np.searchsorted(x, value, side="right") - 1)
    segment = min(max(segment, 0), x.size - 2)
    slopes = np.diff(force_values) / np.diff(x)
    force = float(np.interp(value, x, force_values))
    tangent = float(slopes[segment])
    for knot in range(1, x.size - 1):
        if abs(value - x[knot]) <= _KINK_TOL * max(1.0, abs(value), abs(x[knot])):
            left_slope = float(slopes[knot - 1])
            right_slope = float(slopes[knot])
            scale = max(1.0, abs(left_slope), abs(right_slope))
            tangent = 0.5 * (left_slope + right_slope) if abs(left_slope - right_slope) <= 1e-10 * scale else float("nan")
            break
    energy = _curve_primitive(x, force_values, value) - _curve_primitive(x, force_values, 0.0)
    return float(energy), force, tangent


def component_law(spec: Mapping[str, Any], displacement: float) -> tuple[float, float, float]:
    """Return stored energy, force, and tangent for one component.

    Linear spring ``preload_force`` is the force at zero displacement. Force
    curves use piecewise-linear force interpolation, integrate that curve from
    zero to obtain energy, and reject values outside the declared domain. A
    bump-stop curve's abscissa is compression beyond its engagement point.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("component law must be a mapping")
    coordinate = _finite_scalar(displacement, "displacement")

    if "engagement" in spec:
        engagement = _finite_scalar(spec["engagement"], "engagement")
        penetration = coordinate - engagement
        base = {key: value for key, value in spec.items() if key != "engagement"}
        reference_penetration = -engagement
        reference_energy = 0.0
        if "curve" in base:
            _, force_at_engagement, _ = _curve_law(base, 0.0)
            if abs(force_at_engagement) > 1e-10:
                raise ValueError("bump-stop force curve must have zero force at engagement")
            if reference_penetration > 0.0:
                reference_energy = _curve_law(base, reference_penetration)[0]
        elif reference_penetration > 0.0:
            rate = _finite_scalar(base.get("rate"), "rate")
            if rate < 0.0:
                raise ValueError("rate must be non-negative")
            reference_energy = 0.5 * rate * reference_penetration**2
        if penetration < -_KINK_TOL * max(1.0, abs(coordinate), abs(engagement)):
            return -reference_energy, 0.0, 0.0
        if abs(penetration) <= _KINK_TOL * max(1.0, abs(coordinate), abs(engagement)):
            return -reference_energy, 0.0, float("nan")
        energy, force, tangent = component_law(base, penetration)
        return energy - reference_energy, force, tangent

    if "curve" in spec:
        if set(spec) != {"curve"}:
            raise ValueError("force-curve component cannot also define a rate or preload")
        return _curve_law(spec, coordinate)

    rate = _finite_scalar(spec.get("rate"), "rate")
    preload = _finite_scalar(spec.get("preload_force", 0.0), "preload_force")
    if rate < 0.0:
        raise ValueError("rate must be non-negative")
    energy = preload * coordinate + 0.5 * rate * coordinate**2
    force = preload + rate * coordinate
    return float(energy), float(force), float(rate)


def _compose_scalar_law(
    law_result: tuple[float, float, float],
    displacement_gradient: Any,
    displacement_hessian: Any,
) -> dict[str, Any]:
    """Apply the full scalar chain rule, including force times map curvature."""
    energy, force, tangent = map(float, law_result)
    geometric_gradient = np.asarray(displacement_gradient, dtype=float)
    geometric_hessian = np.asarray(displacement_hessian, dtype=float)
    if geometric_gradient.ndim != 1 or geometric_hessian.shape != (geometric_gradient.size, geometric_gradient.size):
        raise ValueError("displacement derivative shapes must be [n] and [n,n]")
    gradient = force * geometric_gradient
    outer = np.outer(geometric_gradient, geometric_gradient)
    material = tangent * outer if np.isfinite(tangent) else np.zeros_like(geometric_hessian)
    reported_material = material.copy()
    stiffness = material + force * geometric_hessian
    stiffness_valid = np.isfinite(stiffness)
    stiffness_reason = np.full(stiffness.shape, "ok", dtype=object)
    if not np.isfinite(tangent):
        affected = np.abs(outer) > 1e-14
        stiffness_valid[affected] = False
        stiffness_reason[affected] = "law_kink"
        stiffness[affected] = np.nan
        reported_material[affected] = np.nan
    stiffness_valid &= np.isfinite(geometric_hessian)
    stiffness_reason[~stiffness_valid & (stiffness_reason == "ok")] = "invalid_geometry_derivative"
    return {
        "energy": energy,
        "component_force": force,
        "gradient": gradient,
        "stiffness": stiffness,
        "stiffness_valid": stiffness_valid,
        "stiffness_reason": stiffness_reason,
        "material_stiffness": reported_material,
        "geometric_stiffness": force * geometric_hessian,
    }


def _active_bar(spec: Mapping[str, Any] | None) -> bool:
    if spec is None:
        return False
    if "rate" in spec:
        return float(spec["rate"]) != 0.0
    return "curve" in spec


def _bar_component_law(
    spec: Mapping[str, Any],
    geometric_twist_change: float,
) -> tuple[float, float, float]:
    """Evaluate torque at reference plus twist change with zero reference energy."""
    reference = _finite_scalar(spec.get("reference_twist", 0.0), "reference_twist")
    physical_twist = reference + _finite_scalar(geometric_twist_change, "twist_change")
    energy, torque, tangent = component_law(spec, physical_twist)
    reference_energy = component_law(spec, reference)[0]
    return float(energy - reference_energy), torque, tangent


def _component_energy_defaults(setup: Mapping[str, Any]) -> dict[str, Any]:
    corners = setup.get("corners", {})
    stop_active = np.array(
        ["bump_stop" in corners.get(name, {}) for name in CORNER_ORDER],
        dtype=bool,
    )
    active_bars = {
        axle: _active_bar(setup.get("arbs", {}).get(axle))
        for axle in ("front", "rear")
    }
    component_energies = {
        "spring": np.full(4, np.nan, dtype=float),
        "bump_stop": np.where(stop_active, np.nan, 0.0),
        "arb_front": float("nan") if active_bars["front"] else 0.0,
        "arb_rear": float("nan") if active_bars["rear"] else 0.0,
    }
    component_energy_validity = {
        "spring": np.zeros(4, dtype=bool),
        "bump_stop": ~stop_active,
        "arb_front": not active_bars["front"],
        "arb_rear": not active_bars["rear"],
    }
    component_energy_reason = {
        "spring": np.full(4, "not_evaluated", dtype=object),
        "bump_stop": np.where(stop_active, "not_evaluated", "inactive").astype(object),
        "arb_front": "not_evaluated" if active_bars["front"] else "inactive",
        "arb_rear": "not_evaluated" if active_bars["rear"] else "inactive",
    }
    return {
        "stop_active": stop_active,
        "active_bars": active_bars,
        "component_energies": component_energies,
        "component_energy_validity": component_energy_validity,
        "component_energy_reason": component_energy_reason,
    }


def _invalid_energy(reason: str, jounce: np.ndarray, setup: Mapping[str, Any]) -> dict[str, Any]:
    defaults = _component_energy_defaults(setup)
    component_energy_reason = defaults["component_energy_reason"]
    component_energy_reason["spring"][:] = reason
    component_energy_reason["bump_stop"][defaults["stop_active"]] = reason
    for axle, active in defaults["active_bars"].items():
        if active:
            component_energy_reason[f"arb_{axle}"] = reason
    return {
        "valid": False,
        "reason": reason,
        "energy": float("nan"),
        "component_energies": defaults["component_energies"],
        "component_energy_validity": defaults["component_energy_validity"],
        "component_energy_reason": component_energy_reason,
        "spring_compression": np.full(4, np.nan, dtype=float),
        "corner_states": {},
        "actuation": {},
        "arb_state": {},
        "jounce": jounce.copy(),
        "corner_valid": np.zeros(4, dtype=bool),
        "corner_reason": {name: reason for name in CORNER_ORDER},
        "component_reason": {},
    }


def wheel_energy(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    jounce_vector: Any,
    *,
    max_nfev: int = 100,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate spring, bump-stop, and configured ARB energy at four jounces."""
    jounce = np.asarray(jounce_vector, dtype=float)
    if jounce.shape != (4,):
        raise ValueError("jounce_vector must have shape [4] in FL, FR, RL, RR order")
    if not np.all(np.isfinite(jounce)):
        return _invalid_energy("nonfinite_jounce", jounce, setup)

    defaults = _component_energy_defaults(setup)
    states: dict[str, Any] = {}
    actuation: dict[str, Any] = {}
    corner_valid = np.zeros(4, dtype=bool)
    corner_reason: dict[str, str] = {}
    component_reason: dict[str, str] = {}
    spring_compression = np.full(4, np.nan, dtype=float)
    spring_energy = defaults["component_energies"]["spring"].copy()
    stop_energy = defaults["component_energies"]["bump_stop"].copy()
    component_energy_validity = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in defaults["component_energy_validity"].items()
    }
    component_energy_reason = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in defaults["component_energy_reason"].items()
    }

    for index, name in enumerate(CORNER_ORDER):
        corner = geometry["corners"][name]
        state = solve_corner(corner, float(jounce[index]), max_nfev=max_nfev, tolerance=tolerance)
        states[name] = state
        if not state.valid:
            corner_reason[name] = state.reason
            component_energy_reason["spring"][index] = state.reason
            if defaults["stop_active"][index]:
                component_energy_reason["bump_stop"][index] = state.reason
            continue
        motion = actuation_state(corner, state)
        actuation[name] = motion
        if not motion["valid"]:
            corner_reason[name] = motion["reason"]
            component_energy_reason["spring"][index] = motion["reason"]
            if defaults["stop_active"][index]:
                component_energy_reason["bump_stop"][index] = motion["reason"]
            continue
        spring_compression[index] = float(motion["spring_compression"])
        try:
            spring_energy[index], _, _ = component_law(
                setup["corners"][name]["spring"], spring_compression[index]
            )
        except ValueError as error:
            component_reason[name] = "curve_domain" if isinstance(error, LawDomainError) else "invalid_law"
            corner_reason[name] = str(error)
            component_energy_reason["spring"][index] = component_reason[name]
            if defaults["stop_active"][index]:
                component_energy_reason["bump_stop"][index] = component_reason[name]
            continue
        component_energy_validity["spring"][index] = True
        component_energy_reason["spring"][index] = "ok"
        if defaults["stop_active"][index]:
            try:
                stop_energy[index], _, _ = component_law(
                    setup["corners"][name]["bump_stop"], spring_compression[index]
                )
            except ValueError as error:
                component_reason[name] = "curve_domain" if isinstance(error, LawDomainError) else "invalid_law"
                corner_reason[name] = str(error)
                component_energy_reason["bump_stop"][index] = component_reason[name]
                continue
            component_energy_validity["bump_stop"][index] = True
            component_energy_reason["bump_stop"][index] = "ok"
        corner_valid[index] = True
        corner_reason[name] = "ok"

    bars: dict[str, Any] = {}
    total_bar_energy: dict[str, float] = {
        axle: defaults["component_energies"][f"arb_{axle}"] for axle in ("front", "rear")
    }
    active_bar_validity = {
        axle: bool(defaults["component_energy_validity"][f"arb_{axle}"])
        for axle in ("front", "rear")
    }
    active_bar_reason = {
        axle: str(defaults["component_energy_reason"][f"arb_{axle}"])
        for axle in ("front", "rear")
    }
    active_axles = [
        axle for axle in ("front", "rear") if _active_bar(setup.get("arbs", {}).get(axle))
    ]
    if active_axles:
        if all(corner_valid):
            bars = arb_angles(geometry, states)
        for axle in active_axles:
            if not all(corner_valid):
                active_bar_reason[axle] = "corner_invalid"
                component_reason[f"arb_{axle}"] = "corner_invalid"
                continue
            spec = setup["arbs"][axle]
            if axle not in geometry.get("arbs", {}):
                active_bar_reason[axle] = "arb_geometry_missing"
                component_reason[f"arb_{axle}"] = "arb_geometry_missing"
                continue
            bar = bars.get(axle, {})
            if not bar.get("valid", False):
                active_bar_reason[axle] = str(bar.get("reason", "invalid_arb"))
                component_reason[f"arb_{axle}"] = active_bar_reason[axle]
                continue
            try:
                total_bar_energy[axle] = _bar_component_law(spec, float(bar["twist"]))[0]
            except ValueError as error:
                active_bar_reason[axle] = "curve_domain" if isinstance(error, LawDomainError) else "invalid_law"
                component_reason[f"arb_{axle}"] = active_bar_reason[axle]
                continue
            active_bar_validity[axle] = True
            active_bar_reason[axle] = "ok"

    for axle in ("front", "rear"):
        component_energy_validity[f"arb_{axle}"] = active_bar_validity[axle]
        component_energy_reason[f"arb_{axle}"] = active_bar_reason[axle]

    valid = bool(corner_valid.all() and all(active_bar_validity.values()))
    component_energies: dict[str, Any] = {
        "spring": spring_energy,
        "bump_stop": stop_energy,
        "arb_front": total_bar_energy["front"],
        "arb_rear": total_bar_energy["rear"],
    }
    total_energy = (
        float(np.sum(spring_energy) + np.sum(stop_energy) + sum(total_bar_energy.values()))
        if valid
        else float("nan")
    )
    return {
        "valid": valid,
        "reason": "ok" if valid else "invalid_component",
        "energy": total_energy,
        "component_energies": component_energies,
        "component_energy_validity": component_energy_validity,
        "component_energy_reason": component_energy_reason,
        "spring_compression": spring_compression,
        "corner_states": states,
        "actuation": actuation,
        "arb_state": bars,
        "jounce": jounce.copy(),
        "corner_valid": corner_valid,
        "corner_reason": corner_reason,
        "component_reason": component_reason,
    }


def _step_vector(value: Any) -> np.ndarray:
    steps = np.asarray(value, dtype=float)
    if steps.ndim == 0:
        steps = np.full(4, float(steps), dtype=float)
    if steps.shape != (4,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("derivative_steps must be positive finite scalar or length-4 vector")
    return steps


def _compression_derivatives(
    geometry: Mapping[str, Any],
    corner_name: str,
    jounce: float,
    step: float,
    law_spec: Mapping[str, Any],
    *,
    max_nfev: int,
    tolerance: float,
) -> dict[str, Any]:
    corner = geometry["corners"][corner_name]

    def compression_map(point: np.ndarray) -> tuple[float, bool, str]:
        state = solve_corner(corner, float(point[0]), max_nfev=max_nfev, tolerance=tolerance)
        if not state.valid:
            return float("nan"), False, state.reason
        motion = actuation_state(corner, state)
        if not motion["valid"]:
            return float("nan"), False, motion["reason"]
        compression = float(motion["spring_compression"])
        try:
            component_law(law_spec, compression)
        except LawDomainError:
            return float("nan"), False, "curve_domain"
        return compression, True, "ok"

    return finite_difference_derivatives(
        compression_map,
        [jounce],
        steps=[step],
    )


def _bar_twist_derivatives(
    geometry: Mapping[str, Any],
    axle: str,
    jounce: np.ndarray,
    steps: np.ndarray,
    law_spec: Mapping[str, Any],
    *,
    max_nfev: int,
    tolerance: float,
) -> dict[str, Any]:
    names = ("FL", "FR") if axle == "front" else ("RL", "RR")
    indices = [_CORNER_INDEX[name] for name in names]
    axle_geometry = {"corners": geometry["corners"], "arbs": {axle: geometry["arbs"][axle]}}

    def twist_map(point: np.ndarray) -> tuple[float, bool, str]:
        states: dict[str, Any] = {}
        for local_index, name in enumerate(names):
            state = solve_corner(
                geometry["corners"][name],
                float(point[local_index]),
                max_nfev=max_nfev,
                tolerance=tolerance,
            )
            if not state.valid:
                return float("nan"), False, state.reason
            states[name] = state
        bar = arb_angles(axle_geometry, states).get(axle, {})
        if not bar.get("valid", False):
            return float("nan"), False, str(bar.get("reason", "invalid_arb"))
        twist = float(bar["twist"])
        try:
            _bar_component_law(law_spec, twist)
        except LawDomainError:
            return float("nan"), False, "curve_domain"
        return twist, True, "ok"

    return finite_difference_derivatives(
        twist_map,
        jounce[indices],
        steps=steps[indices],
    )


def _empty_contributions() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for component in ("spring", "bump_stop"):
        values[f"{component}_energy"] = np.zeros(4, dtype=float)
        values[f"{component}_component_force"] = np.zeros(4, dtype=float)
        values[f"{component}_force"] = np.zeros(4, dtype=float)
        values[f"{component}_material_rate"] = np.zeros(4, dtype=float)
        values[f"{component}_geometric_rate"] = np.zeros(4, dtype=float)
        values[f"{component}_stiffness"] = np.zeros((4, 4), dtype=float)
    for axle in ("front", "rear"):
        values[f"arb_{axle}_energy"] = 0.0
        values[f"arb_{axle}_torque"] = 0.0
        values[f"arb_{axle}_force"] = np.zeros(4, dtype=float)
        values[f"arb_{axle}_stiffness"] = np.zeros((4, 4), dtype=float)
    return values


def _invalid_contributions(energy: Mapping[str, Any]) -> dict[str, Any]:
    values = _empty_contributions()
    values["spring_energy"] = np.asarray(energy["component_energies"]["spring"], dtype=float).copy()
    values["bump_stop_energy"] = np.asarray(energy["component_energies"]["bump_stop"], dtype=float).copy()
    values["arb_front_energy"] = float(energy["component_energies"]["arb_front"])
    values["arb_rear_energy"] = float(energy["component_energies"]["arb_rear"])

    for key, value in values.items():
        if key.endswith("_energy"):
            continue
        if isinstance(value, np.ndarray):
            value.fill(np.nan)
        else:
            values[key] = float("nan")

    stop_inactive = np.asarray(energy["component_energy_reason"]["bump_stop"], dtype=object) == "inactive"
    for field in ("component_force", "force", "material_rate", "geometric_rate"):
        values[f"bump_stop_{field}"][stop_inactive] = 0.0
    values["bump_stop_stiffness"][stop_inactive, :] = 0.0
    values["bump_stop_stiffness"][:, stop_inactive] = 0.0

    for axle in ("front", "rear"):
        component = f"arb_{axle}"
        if energy["component_energy_reason"][component] == "inactive":
            values[f"{component}_torque"] = 0.0
            values[f"{component}_force"].fill(0.0)
            values[f"{component}_stiffness"].fill(0.0)
            continue
        indices = (0, 1) if axle == "front" else (2, 3)
        outside = [index for index in range(4) if index not in indices]
        values[f"{component}_force"][outside] = 0.0
        values[f"{component}_stiffness"][outside, :] = 0.0
        values[f"{component}_stiffness"][:, outside] = 0.0
    return values


def _component_contribution_status(
    contributions: Mapping[str, Any],
    energy_validity: Mapping[str, Any],
    energy_reason: Mapping[str, Any],
    *,
    invalid_reason: str = "invalid_contribution",
) -> tuple[dict[str, Any], dict[str, Any]]:
    validity: dict[str, Any] = {}
    reasons: dict[str, Any] = {}

    def finite_status(value: Any) -> tuple[Any, Any]:
        array = np.asarray(value)
        finite = np.isfinite(array)
        reason = np.full(array.shape, invalid_reason, dtype=object)
        reason[finite] = "ok"
        if array.ndim == 0:
            return bool(finite), str(reason.item())
        return finite.astype(bool), reason

    for component in ("spring", "bump_stop"):
        component_validity: dict[str, Any] = {}
        component_reasons: dict[str, Any] = {}
        energy_mask = energy_validity[component]
        energy_reason_values = energy_reason[component]
        component_validity["energy"] = (
            energy_mask.copy() if isinstance(energy_mask, np.ndarray) else bool(energy_mask)
        )
        component_reasons["energy"] = (
            energy_reason_values.copy()
            if isinstance(energy_reason_values, np.ndarray)
            else str(energy_reason_values)
        )
        inactive = (
            np.asarray(energy_reason_values, dtype=object) == "inactive"
            if component == "bump_stop"
            else np.zeros(4, dtype=bool)
        )
        for field in ("component_force", "force", "material_rate", "geometric_rate", "stiffness"):
            field_valid, field_reason = finite_status(contributions[f"{component}_{field}"])
            if field == "stiffness":
                field_valid[inactive, :] = True
                field_valid[:, inactive] = True
                field_reason[inactive, :] = "inactive"
                field_reason[:, inactive] = "inactive"
            else:
                field_valid[inactive] = True
                field_reason[inactive] = "inactive"
            component_validity[field] = field_valid
            component_reasons[field] = field_reason
        validity[component] = component_validity
        reasons[component] = component_reasons

    for axle in ("front", "rear"):
        component = f"arb_{axle}"
        component_validity = {}
        component_reasons = {}
        energy_mask = energy_validity[component]
        energy_reason_value = str(energy_reason[component])
        component_validity["energy"] = bool(energy_mask)
        component_reasons["energy"] = energy_reason_value
        torque_valid, torque_reason = finite_status(contributions[f"{component}_torque"])
        component_validity["torque"] = torque_valid
        component_reasons["torque"] = torque_reason
        indices = (0, 1) if axle == "front" else (2, 3)
        outside = [index for index in range(4) if index not in indices]
        force_valid, force_reason = finite_status(contributions[f"{component}_force"])
        force_valid[outside] = True
        force_reason[outside] = "inactive"
        stiffness_valid, stiffness_reason = finite_status(contributions[f"{component}_stiffness"])
        stiffness_valid[outside, :] = True
        stiffness_valid[:, outside] = True
        stiffness_reason[outside, :] = "inactive"
        stiffness_reason[:, outside] = "inactive"
        component_validity["force"] = force_valid
        component_validity["stiffness"] = stiffness_valid
        component_reasons["force"] = force_reason
        component_reasons["stiffness"] = stiffness_reason
        validity[component] = component_validity
        reasons[component] = component_reasons
    return validity, reasons


def _invalid_component_validity(energy: Mapping[str, Any]) -> dict[str, Any]:
    validity: dict[str, Any] = {
        "spring": np.zeros(4, dtype=bool),
        "bump_stop": np.asarray(energy["component_energy_reason"]["bump_stop"], dtype=object) == "inactive",
    }
    for axle in ("front", "rear"):
        component = f"arb_{axle}"
        matrix_valid = np.ones((4, 4), dtype=bool)
        matrix_reason = np.full((4, 4), "response_invalid", dtype=object)
        if energy["component_energy_reason"][component] == "inactive":
            matrix_reason[:, :] = "inactive"
        else:
            indices = (0, 1) if axle == "front" else (2, 3)
            outside = [index for index in range(4) if index not in indices]
            matrix_valid[np.ix_(indices, indices)] = False
            matrix_reason[np.ix_(indices, indices)] = "response_invalid"
            matrix_reason[outside, :] = "inactive"
            matrix_reason[:, outside] = "inactive"
        validity[component] = {
            "stiffness_valid": matrix_valid,
            "stiffness_reason": matrix_reason,
        }
    return validity


def _invalid_response(energy: dict[str, Any], steps: np.ndarray) -> dict[str, Any]:
    reason = str(energy["reason"])
    contributions = _invalid_contributions(energy)
    contribution_validity, contribution_reason = _component_contribution_status(
        contributions,
        energy["component_energy_validity"],
        energy["component_energy_reason"],
        invalid_reason="response_invalid",
    )
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
        "component_contributions": contributions,
        "component_energies": energy["component_energies"],
        "component_energy_validity": energy["component_energy_validity"],
        "component_energy_reason": energy["component_energy_reason"],
        "component_contribution_validity": contribution_validity,
        "component_contribution_reason": contribution_reason,
        "component_validity": _invalid_component_validity(energy),
        "derivative_steps": steps.copy(),
        "derivative_quality": {},
        "corner_states": energy["corner_states"],
        "actuation": energy["actuation"],
        "arb_state": energy["arb_state"],
        "corner_valid": energy["corner_valid"],
        "corner_reason": energy["corner_reason"],
        "component_reason": energy["component_reason"],
        "curve_interpolation": CURVE_INTERPOLATION,
    }


def wheel_response(
    geometry: Mapping[str, Any],
    setup: Mapping[str, Any],
    jounce_vector: Any,
    *,
    derivative_steps: Any = 1e-4,
    max_nfev: int = 100,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Return energy gradient, full wheel stiffness, and component rates.

    Spring compression and ARB twist are differentiated from exact geometry
    solves. Their scalar energy laws are then composed with those derivatives,
    retaining the geometric ``force * curvature`` term. Result masks and
    reason arrays use [4] and [4,4] FL,FR,RL,RR ordering.
    """
    jounce = np.asarray(jounce_vector, dtype=float)
    if jounce.shape != (4,):
        raise ValueError("jounce_vector must have shape [4] in FL, FR, RL, RR order")
    steps = _step_vector(derivative_steps)
    energy_result = wheel_energy(
        geometry,
        setup,
        jounce,
        max_nfev=max_nfev,
        tolerance=tolerance,
    )
    if not energy_result["valid"]:
        return _invalid_response(energy_result, steps)

    gradient = np.zeros(4, dtype=float)
    stiffness = np.zeros((4, 4), dtype=float)
    gradient_valid = np.ones(4, dtype=bool)
    stiffness_valid = np.ones((4, 4), dtype=bool)
    gradient_reason = np.full(4, "ok", dtype=object)
    stiffness_reason = np.full((4, 4), "ok", dtype=object)
    motion_ratios = np.full(4, np.nan, dtype=float)
    component_validity: dict[str, Any] = {
        "spring": np.ones(4, dtype=bool),
        "bump_stop": np.ones(4, dtype=bool),
    }
    quality: dict[str, Any] = {}
    contributions = _empty_contributions()
    contributions["spring_energy"] = np.asarray(energy_result["component_energies"]["spring"], dtype=float).copy()
    contributions["bump_stop_energy"] = np.asarray(
        energy_result["component_energies"]["bump_stop"], dtype=float
    ).copy()

    def add_local(component: str, index: int, composed: dict[str, Any], derivative: dict[str, Any]) -> None:
        key = f"{component}_{index}"
        law_gradient_valid = bool(derivative["gradient_valid"][0])
        if law_gradient_valid and np.isfinite(composed["gradient"][0]):
            value = float(composed["gradient"][0])
            contributions[f"{component}_force"][index] += value
            gradient[index] += value
            component_validity.setdefault(component, np.ones(4, dtype=bool))[index] = True
            if component == "spring":
                motion_ratios[index] = float(derivative["gradient"][0])
        else:
            gradient[index] = np.nan
            gradient_valid[index] = False
            gradient_reason[index] = str(derivative["gradient_reason"][0])
            contributions[f"{component}_force"][index] = np.nan
            component_validity.setdefault(component, np.zeros(4, dtype=bool))[index] = False

        entry = (index, index)
        if derivative["hessian_valid"][0, 0] and composed["stiffness_valid"][0, 0]:
            value = float(composed["stiffness"][0, 0])
            stiffness[entry] += value
            contributions[f"{component}_stiffness"][entry] += value
            material = float(composed["material_stiffness"][0, 0])
            geometric = float(composed["geometric_stiffness"][0, 0])
            contributions[f"{component}_material_rate"][index] += material
            contributions[f"{component}_geometric_rate"][index] += geometric
        else:
            stiffness[entry] = np.nan
            stiffness_valid[entry] = False
            reason = (
                str(composed["stiffness_reason"][0, 0])
                if not composed["stiffness_valid"][0, 0]
                else str(derivative["hessian_reason"][0, 0])
            )
            stiffness_reason[entry] = reason
            contributions[f"{component}_stiffness"][entry] = np.nan
        if np.isfinite(composed["material_stiffness"][0, 0]):
            contributions[f"{component}_material_rate"][index] = float(composed["material_stiffness"][0, 0])
        else:
            contributions[f"{component}_material_rate"][index] = np.nan
        if derivative["hessian_valid"][0, 0] and np.isfinite(composed["geometric_stiffness"][0, 0]):
            contributions[f"{component}_geometric_rate"][index] = float(composed["geometric_stiffness"][0, 0])
        else:
            contributions[f"{component}_geometric_rate"][index] = np.nan
        quality[key] = {
            "steps": derivative["steps"].copy(),
            "gradient_method": derivative["gradient_method"].copy(),
            "hessian_method": derivative["hessian_method"].copy(),
            "sample_count": derivative["sample_count"],
        }

    for index, name in enumerate(CORNER_ORDER):
        compression = float(energy_result["spring_compression"][index])
        derivative_specs = [("spring", setup["corners"][name]["spring"])]
        if "bump_stop" in setup["corners"][name]:
            derivative_specs.append(("bump_stop", setup["corners"][name]["bump_stop"]))
        for component, law_spec in derivative_specs:
            derivative = _compression_derivatives(
                geometry,
                name,
                float(jounce[index]),
                float(steps[index]),
                law_spec,
                max_nfev=max_nfev,
                tolerance=tolerance,
            )
            try:
                law_result = component_law(law_spec, compression)
            except LawDomainError:
                # The center was already validated by wheel_energy; this is a
                # defensive guard for mutable input mappings.
                law_result = (float("nan"), float("nan"), float("nan"))
            contributions[f"{component}_component_force"][index] = float(law_result[1])
            composed = _compose_scalar_law(
                law_result,
                derivative["gradient"],
                derivative["hessian"],
            )
            add_local(component, index, composed, derivative)

    active_axles = [
        axle for axle in ("front", "rear") if _active_bar(setup.get("arbs", {}).get(axle))
    ]
    for axle in active_axles:
        spec = setup["arbs"][axle]
        bar_state = energy_result["arb_state"][axle]
        law_result = _bar_component_law(spec, float(bar_state["twist"]))
        names = ("FL", "FR") if axle == "front" else ("RL", "RR")
        indices = [_CORNER_INDEX[name] for name in names]
        derivative = _bar_twist_derivatives(
            geometry,
            axle,
            jounce,
            steps,
            spec,
            max_nfev=max_nfev,
            tolerance=tolerance,
        )
        composed = _compose_scalar_law(law_result, derivative["gradient"], derivative["hessian"])
        component_force = np.zeros(4, dtype=float)
        component_stiffness = np.zeros((4, 4), dtype=float)
        component_stiffness_valid = np.ones((4, 4), dtype=bool)
        component_stiffness_reason = np.full((4, 4), "ok", dtype=object)
        for local_i, global_i in enumerate(indices):
            if derivative["gradient_valid"][local_i] and np.isfinite(composed["gradient"][local_i]):
                component_force[global_i] = composed["gradient"][local_i]
                gradient[global_i] += component_force[global_i]
            else:
                component_force[global_i] = np.nan
                gradient[global_i] = np.nan
                gradient_valid[global_i] = False
                gradient_reason[global_i] = str(derivative["gradient_reason"][local_i])
            for local_j, global_j in enumerate(indices):
                if composed["stiffness_valid"][local_i, local_j] and derivative["hessian_valid"][local_i, local_j]:
                    value = float(composed["stiffness"][local_i, local_j])
                    component_stiffness[global_i, global_j] = value
                    stiffness[global_i, global_j] += value
                else:
                    component_stiffness[global_i, global_j] = np.nan
                    component_stiffness_valid[global_i, global_j] = False
                    reason = (
                        str(composed["stiffness_reason"][local_i, local_j])
                        if not composed["stiffness_valid"][local_i, local_j]
                        else str(derivative["hessian_reason"][local_i, local_j])
                    )
                    component_stiffness_reason[global_i, global_j] = reason
                    stiffness[global_i, global_j] = np.nan
                    stiffness_valid[global_i, global_j] = False
                    stiffness_reason[global_i, global_j] = reason
        contributions[f"arb_{axle}_energy"] = float(law_result[0])
        contributions[f"arb_{axle}_torque"] = float(law_result[1])
        contributions[f"arb_{axle}_force"] = component_force
        contributions[f"arb_{axle}_stiffness"] = component_stiffness
        component_validity[f"arb_{axle}"] = {
            "stiffness_valid": component_stiffness_valid,
            "stiffness_reason": component_stiffness_reason,
        }
        quality[f"arb_{axle}"] = {
            "steps": derivative["steps"].copy(),
            "gradient_method": derivative["gradient_method"].copy(),
            "hessian_method": derivative["hessian_method"].copy(),
            "sample_count": derivative["sample_count"],
        }

    material_rates = contributions["spring_material_rate"] + contributions["bump_stop_material_rate"]
    material_rate_valid = np.isfinite(material_rates)
    material_rate_reason = np.where(material_rate_valid, "ok", stiffness_reason.diagonal()).astype(object)
    motion_ratio_valid = np.isfinite(motion_ratios)
    motion_ratio_reason = np.where(motion_ratio_valid, "ok", gradient_reason).astype(object)
    valid = bool(energy_result["valid"] and gradient_valid.all())
    contribution_validity, contribution_reason = _component_contribution_status(
        contributions,
        energy_result["component_energy_validity"],
        energy_result["component_energy_reason"],
    )
    return {
        "valid": valid,
        "reason": "ok" if valid else "invalid_gradient",
        "energy": float(energy_result["energy"]),
        "gradient": gradient,
        "stiffness": stiffness,
        "gradient_valid": gradient_valid,
        "stiffness_valid": stiffness_valid,
        "gradient_reason": gradient_reason,
        "stiffness_reason": stiffness_reason,
        "material_rates": material_rates,
        "material_rate_valid": material_rate_valid,
        "material_rate_reason": material_rate_reason,
        "motion_ratios": motion_ratios,
        "motion_ratio_valid": motion_ratio_valid,
        "motion_ratio_reason": motion_ratio_reason,
        "component_contributions": contributions,
        "component_energies": energy_result["component_energies"],
        "component_energy_validity": energy_result["component_energy_validity"],
        "component_energy_reason": energy_result["component_energy_reason"],
        "component_contribution_validity": contribution_validity,
        "component_contribution_reason": contribution_reason,
        "component_validity": component_validity,
        "derivative_steps": steps.copy(),
        "derivative_quality": quality,
        "corner_states": energy_result["corner_states"],
        "actuation": energy_result["actuation"],
        "arb_state": energy_result["arb_state"],
        "corner_valid": energy_result["corner_valid"],
        "corner_reason": energy_result["corner_reason"],
        "component_reason": energy_result["component_reason"],
        "curve_interpolation": CURVE_INTERPOLATION,
    }
