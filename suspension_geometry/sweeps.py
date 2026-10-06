"""Bounded finite-pose studies and their in-memory result contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .metrics import CORNER_ORDER, known_metrics, metric_definitions
from .actuation import actuation_state
from .kinematics import solve_corner_sweep
from .vehicle import _corner_attitude, _local_compression_derivative, evaluate_pose


@dataclass
class MapResult:
    """Study arrays with explicit validity and reason tails.

    Every metric has leading dimensions ``(Nh, Nr, Np)``.  Its declared tail
    dimensions follow the metric registry: corner fields append ``(4,)``,
    wheel/body matrices append ``(4,4)``/``(3,3)``, and scalar fields append
    nothing.  ``reasons`` is intentionally a first-class field so an invalid
    stencil or unavailable tyre cannot be inferred as a finite zero later.
    """

    axes: dict[str, np.ndarray]
    metrics: dict[str, np.ndarray]
    validity: dict[str, np.ndarray]
    metadata: dict[str, Any]
    corner_maps: dict[str, Any]
    reasons: dict[str, np.ndarray] = field(default_factory=dict)
    definitions: dict[str, dict[str, Any]] = field(default_factory=metric_definitions)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(len(self.axes[key]) for key in ("heave_m", "roll_rad", "pitch_rad"))  # type: ignore[return-value]

    @property
    def raw_corner_maps(self) -> dict[str, Any]:
        """Alias used by consumers that distinguish raw corner data from metrics."""
        return self.corner_maps


def _axis_value(study: Mapping[str, Any], canonical: str, legacy: str) -> np.ndarray:
    axes = study.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError("study.axes must be a mapping")
    value = axes.get(canonical, axes.get(legacy))
    if value is None:
        raise ValueError(f"study.axes.{legacy} is required")
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"study.axes.{legacy} must be a finite non-empty axis")
    if result.size > 1 and np.any(np.diff(result) <= 0.0):
        raise ValueError(f"study.axes.{legacy} must be strictly increasing")
    return result.copy()


def _selected_metrics(study: Mapping[str, Any], definitions: Mapping[str, Any]) -> list[str]:
    selected = study.get("metrics")
    if selected is None:
        selected = ["spring_motion_ratio", "wheel_tangent_rate", "body_stiffness"]
    if not isinstance(selected, (list, tuple)):
        raise ValueError("study.metrics must be a list")
    result = [str(name) for name in selected]
    unknown = [name for name in result if name not in definitions]
    if unknown:
        raise ValueError(f"unknown metric name(s): {', '.join(unknown)}")
    if len(result) != len(set(result)):
        raise ValueError("study.metrics must not contain duplicates")
    return result


def _empty_pose_corner_map(shape: tuple[int, int, int]) -> dict[str, Any]:
    fields = (
        "jounce", "spring_compression", "damper_compression", "camber_chassis",
        "camber_road", "toe", "caster", "spring_motion_ratio", "damper_motion_ratio",
    )
    corner_map: dict[str, Any] = {
        "valid": np.zeros(shape, dtype=bool),
        "reason": np.full(shape, "not_evaluated", dtype=object),
        "jounce": np.full(shape, np.nan, dtype=float),
        "wheel_center": np.full(shape + (3,), np.nan, dtype=float),
        "spring_compression": np.full(shape, np.nan, dtype=float),
        "damper_compression": np.full(shape, np.nan, dtype=float),
        "camber_chassis": np.full(shape, np.nan, dtype=float),
        "camber_road": np.full(shape, np.nan, dtype=float),
        "toe": np.full(shape, np.nan, dtype=float),
        "caster": np.full(shape, np.nan, dtype=float),
        "spring_motion_ratio": np.full(shape, np.nan, dtype=float),
        "damper_motion_ratio": np.full(shape, np.nan, dtype=float),
    }
    corner_map["validity"] = {field: np.zeros(shape, dtype=bool) for field in fields}
    corner_map["reasons"] = {field: np.full(shape, "not_evaluated", dtype=object) for field in fields}
    return corner_map


def _raw_corner_map(
    geometry: Mapping[str, Any],
    name: str,
    jounce_axis: np.ndarray,
    *,
    solver: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded, exact jounce-indexed geometry/actuation table.

    This table deliberately contains no spring laws, tyre laws, or finite-pose
    derivatives.  It is a reusable mechanism map: invalid stroke actuation
    masks only the actuation fields while a valid rigid corner still exports
    its wheel centre, upright points, and alignment angles.
    """
    corner = geometry["corners"][name]
    max_nfev = int(solver.get("max_nfev", 100))
    tolerance = float(solver.get("residual_tolerance", 1e-9))
    jounce_step = float(solver.get("jounce_step", 1e-4))
    axis = np.asarray(jounce_axis, dtype=float).copy()
    states = solve_corner_sweep(corner, axis, max_nfev=max_nfev, tolerance=tolerance)
    if len(states) != axis.size:
        raise RuntimeError(f"corner {name} travel map returned an unexpected state count")

    fields = (
        "wheel_center", "wheel_center_x", "wheel_center_y", "wheel_center_z",
        "wheel_migration_x", "wheel_migration_y", "upper_ball_joint",
        "lower_ball_joint", "tie_outboard", "camber_chassis", "camber_road",
        "toe", "caster", "spring_compression", "damper_compression",
        "rocker_angle", "spring_stroke_margin", "damper_stroke_margin",
        "spring_motion_ratio", "damper_motion_ratio",
    )
    metrics: dict[str, np.ndarray] = {
        "wheel_center": np.full((axis.size, 3), np.nan),
        "wheel_center_x": np.full(axis.size, np.nan),
        "wheel_center_y": np.full(axis.size, np.nan),
        "wheel_center_z": np.full(axis.size, np.nan),
        "wheel_migration_x": np.full(axis.size, np.nan),
        "wheel_migration_y": np.full(axis.size, np.nan),
        "upper_ball_joint": np.full((axis.size, 3), np.nan),
        "lower_ball_joint": np.full((axis.size, 3), np.nan),
        "tie_outboard": np.full((axis.size, 3), np.nan),
        "camber_chassis": np.full(axis.size, np.nan),
        "camber_road": np.full(axis.size, np.nan),
        "toe": np.full(axis.size, np.nan),
        "caster": np.full(axis.size, np.nan),
        "spring_compression": np.full(axis.size, np.nan),
        "damper_compression": np.full(axis.size, np.nan),
        "rocker_angle": np.full(axis.size, np.nan),
        "spring_stroke_margin": np.full((axis.size, 2), np.nan),
        "damper_stroke_margin": np.full((axis.size, 2), np.nan),
        "spring_motion_ratio": np.full(axis.size, np.nan),
        "damper_motion_ratio": np.full(axis.size, np.nan),
    }
    validity = {field: np.zeros(value.shape, dtype=bool) for field, value in metrics.items()}
    reasons = {field: np.full(value.shape, "not_evaluated", dtype=object) for field, value in metrics.items()}
    state_valid = np.zeros(axis.size, dtype=bool)
    state_reasons = np.full(axis.size, "not_evaluated", dtype=object)
    actuation_valid = np.zeros(axis.size, dtype=bool)
    actuation_reasons = np.full(axis.size, "not_evaluated", dtype=object)
    origin = np.asarray(geometry.get("reference_origin_world", [0.0, 0.0, 0.0]), dtype=float)
    nominal = np.asarray(corner["wheel_center"], dtype=float)

    for sample, state in enumerate(states):
        state_valid[sample] = bool(state.valid)
        state_reasons[sample] = state.reason
        if not state.valid:
            for field in fields:
                reasons[field][sample] = state.reason
            actuation_reasons[sample] = "invalid_corner"
            continue

        # The raw travel table is at the nominal body pose, so its world frame
        # is a pure translation by reference_origin_world.
        for field, value in (
            ("wheel_center", origin + state.wheel_center),
            ("upper_ball_joint", origin + state.upper_ball_joint),
            ("lower_ball_joint", origin + state.lower_ball_joint),
            ("tie_outboard", origin + state.tie_outboard),
        ):
            metrics[field][sample] = value
            field_valid = bool(np.isfinite(value).all())
            validity[field][sample] = field_valid
            reasons[field][sample] = "ok" if field_valid else "nonfinite_geometry"
        center = metrics["wheel_center"][sample]
        metrics["wheel_center_x"][sample] = center[0]
        metrics["wheel_center_y"][sample] = center[1]
        metrics["wheel_center_z"][sample] = center[2]
        migration = state.wheel_center - nominal
        metrics["wheel_migration_x"][sample], metrics["wheel_migration_y"][sample] = migration[:2]
        validity["wheel_center_x"][sample] = validity["wheel_center_y"][sample] = validity["wheel_center_z"][sample] = True
        validity["wheel_migration_x"][sample] = validity["wheel_migration_y"][sample] = True
        for field in ("wheel_center_x", "wheel_center_y", "wheel_center_z", "wheel_migration_x", "wheel_migration_y"):
            reasons[field][sample] = "ok"

        attitude = _corner_attitude(name, corner, state, np.eye(3))
        for field in ("camber_chassis", "camber_road", "toe", "caster"):
            metrics[field][sample] = attitude[field]
            validity[field][sample] = bool(np.isfinite(attitude[field]))
            reasons[field][sample] = "ok" if validity[field][sample] else "invalid_orientation"

        motion = actuation_state(corner, state)
        actuation_valid[sample] = bool(motion["valid"])
        actuation_reasons[sample] = motion["reason"]
        for field, source in (
            ("spring_compression", "spring_compression"),
            ("damper_compression", "damper_compression"),
            ("rocker_angle", "rocker_angle"),
            ("spring_stroke_margin", "spring_stroke_margin"),
            ("damper_stroke_margin", "damper_stroke_margin"),
        ):
            value = np.asarray(motion[source], dtype=float)
            if value.shape == metrics[field][sample].shape:
                metrics[field][sample] = value
            else:
                metrics[field][sample] = float(value)
            valid_value = bool(motion["valid"] and np.isfinite(value).all())
            validity[field][sample] = valid_value
            reasons[field][sample] = "ok" if valid_value else motion["reason"]

        if motion["valid"]:
            for field, component in (("spring_motion_ratio", "spring"), ("damper_motion_ratio", "damper")):
                value, valid_value, reason = _local_compression_derivative(
                    geometry, name, float(axis[sample]), component,
                    jounce_step, max_nfev=max_nfev, tolerance=tolerance,
                )
                metrics[field][sample] = value
                validity[field][sample] = bool(valid_value and np.isfinite(value))
                reasons[field][sample] = "ok" if validity[field][sample] else reason
        else:
            for field in ("spring_motion_ratio", "damper_motion_ratio"):
                reasons[field][sample] = motion["reason"]

    raw = {
        "axes": {"jounce_m": axis.copy()},
        "axis": axis.copy(),
        "jounce": axis.copy(),
        "corner": name,
        "valid": state_valid,
        "reason": state_reasons,
        "metrics": metrics,
        "validity": validity,
        "reasons": reasons,
        "actuation_valid": actuation_valid,
        "actuation_reason": actuation_reasons,
        "ball_joint_labels": {
            "upper": "upper_ball_joint",
            "lower": "lower_ball_joint",
            "tie": "tie_outboard",
        },
        "point_labels": ("wheel_center", "upper_ball_joint", "lower_ball_joint", "tie_outboard"),
        "sample_count": int(axis.size),
        # Direct aliases keep this useful to HDF5/CSV writers without making
        # them understand the nested metric mapping.
    }
    raw.update({field: value.copy() for field, value in metrics.items()})
    return raw


def _build_raw_corner_maps(
    geometry: Mapping[str, Any],
    *,
    solver: Mapping[str, Any],
) -> tuple[dict[str, Any], float, int]:
    requested_cap = int(solver.get("corner_map_max_samples", 36))
    if requested_cap < len(CORNER_ORDER):
        raise ValueError("solver.corner_map_max_samples must allow at least one sample per corner")
    cap = min(requested_cap, 36)
    per_corner = min(9, cap // len(CORNER_ORDER))
    start = perf_counter()
    result: dict[str, Any] = {}
    for name in CORNER_ORDER:
        low, high = map(float, geometry["corners"][name]["jounce_limits"])
        axis = np.linspace(low, high, per_corner)
        result[name] = _raw_corner_map(geometry, name, axis, solver=solver)
    return result, perf_counter() - start, int(per_corner * len(CORNER_ORDER))


def _metric_reason_value(raw: Mapping[str, Any], metric: str, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_validity = raw.get("validity", {}).get(metric)
    raw_reasons = raw.get("reasons", {}).get(metric)
    if raw_validity is None:
        valid = np.isfinite(value)
    else:
        valid = np.asarray(raw_validity, dtype=bool)
    if raw_reasons is None:
        reasons = np.full(value.shape, "ok", dtype=object)
        reasons[~valid] = "invalid"
    else:
        reasons = np.asarray(raw_reasons, dtype=object)
        if reasons.shape == () and value.shape != ():
            reasons = np.full(value.shape, str(reasons.item()), dtype=object)
    if valid.shape != value.shape:
        try:
            valid = np.broadcast_to(valid, value.shape).copy()
        except ValueError:
            valid = np.zeros(value.shape, dtype=bool)
    if reasons.shape != value.shape:
        try:
            reasons = np.broadcast_to(reasons, value.shape).copy()
        except ValueError:
            reasons = np.full(value.shape, "reason_shape_mismatch", dtype=object)
    return valid, reasons


def run_study(
    geometry: Mapping[str, Any] | str | Path,
    setup: Mapping[str, Any] | None = None,
    study: Mapping[str, Any] | None = None,
) -> MapResult:
    """Run a bounded ``(heave, roll, pitch)`` pose map.

    The function accepts either resolved dictionaries or a study path.  In
    both forms the numeric axes are allocated only after metric and sample
    limits are checked.  Each pose is evaluated independently so a failed
    corner leaves neighboring geometry fields available.
    """
    if isinstance(geometry, (str, Path)):
        from .config import resolve_study

        resolved_geometry, resolved_setup, resolved_study = resolve_study(geometry)
        geometry, setup, study = resolved_geometry, resolved_setup, resolved_study
    if setup is None or study is None:
        raise ValueError("run_study requires geometry, setup, and study")
    definitions = metric_definitions()
    selected = _selected_metrics(study, definitions)
    axes = {
        "heave_m": _axis_value(study, "heave_m", "heave"),
        "roll_rad": _axis_value(study, "roll_rad", "roll"),
        "pitch_rad": _axis_value(study, "pitch_rad", "pitch"),
    }
    shape = tuple(len(axes[key]) for key in ("heave_m", "roll_rad", "pitch_rad"))
    sample_count = int(np.prod(shape, dtype=np.int64))
    solver = study.get("solver", {}) if isinstance(study, Mapping) else {}
    max_samples = int(solver.get("max_samples", 2000)) if isinstance(solver, Mapping) else 2000
    if sample_count > max_samples:
        raise ValueError(f"study grid has {sample_count} samples, exceeding max_samples {max_samples}")

    # A tyre-inclusive rate is only useful with the solved operating point and
    # the base constrained response that produced it.  Store those companions
    # explicitly while retaining the user's requested list in metadata.
    tyre_metrics = {
        "ride_rate", "ride_stiffness", "body_ride_stiffness",
        "tyre_equilibrium_hub_displacement", "tyre_equilibrium_jounce",
        "tyre_contact_force", "tyre_effective_reaction",
    }
    tyre_companions = {
        "ride_rate", "ride_stiffness", "body_ride_stiffness",
        "tyre_equilibrium_hub_displacement", "tyre_equilibrium_jounce",
        "tyre_contact_force", "tyre_effective_reaction",
        "wheel_stiffness", "wheel_force", "wheel_material_rate",
        "wheel_tangent_rate", "wheel_energy", "body_gradient",
        "body_restoring_reaction", "body_stiffness",
    }
    stored = list(selected)
    if set(selected) & tyre_metrics:
        stored.extend(name for name in tyre_companions if name not in stored)

    metrics: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    reasons: dict[str, np.ndarray] = {}
    for name in stored:
        tail_shape = tuple(definitions[name].get("tail_shape", ()))
        full_shape = shape + tail_shape
        metrics[name] = np.full(full_shape, np.nan, dtype=float)
        validity[name] = np.zeros(full_shape, dtype=bool)
        reasons[name] = np.full(full_shape, "not_evaluated", dtype=object)

    pose_corner_maps = {name: _empty_pose_corner_map(shape) for name in CORNER_ORDER}
    need_tyres = bool(set(selected) & tyre_metrics)
    start = perf_counter()
    valid_pose_count = 0
    for h_index, heave in enumerate(axes["heave_m"]):
        for r_index, roll in enumerate(axes["roll_rad"]):
            for p_index, pitch in enumerate(axes["pitch_rad"]):
                index = (h_index, r_index, p_index)
                result = evaluate_pose(
                    geometry,
                    setup,
                    np.array([heave, roll, pitch], dtype=float),
                    solver=solver,
                    include_tyres=need_tyres,
                )
                if result["valid"]:
                    valid_pose_count += 1
                raw_metrics = result.get("metrics", {})
                for name in stored:
                    if name not in raw_metrics:
                        continue
                    value = np.asarray(raw_metrics[name], dtype=float)
                    tail_shape = tuple(definitions[name].get("tail_shape", ()))
                    if value.shape != tail_shape:
                        # A scalar metric is allowed to be returned as a
                        # numpy scalar; all non-scalar mismatches are an
                        # implementation error and become an invalid cell.
                        if value.shape != () or tail_shape != ():
                            reasons[name][index] = "metric_shape_mismatch"
                            continue
                    metrics[name][index] = value
                    valid_value, reason_value = _metric_reason_value(result, name, value)
                    validity[name][index] = valid_value
                    reasons[name][index] = reason_value

                for corner_index, corner_name in enumerate(CORNER_ORDER):
                    corner = pose_corner_maps[corner_name]
                    per_corner_valid = bool(result.get("corner_valid", np.zeros(4, dtype=bool))[corner_index])
                    corner["valid"][index] = per_corner_valid
                    corner["reason"][index] = result.get("corner_reason", {}).get(corner_name, "invalid_pose")
                    for field in (
                        "jounce", "spring_compression", "damper_compression",
                        "camber_chassis", "camber_road", "toe", "caster",
                        "spring_motion_ratio", "damper_motion_ratio",
                    ):
                        value = raw_metrics.get(field)
                        if value is not None and np.asarray(value).shape == (4,):
                            corner[field][index] = value[corner_index]
                            field_validity = result.get("validity", {}).get(field)
                            field_reasons = result.get("reasons", {}).get(field)
                            if field_validity is None:
                                corner["validity"][field][index] = bool(np.isfinite(value[corner_index]))
                            else:
                                field_validity = np.asarray(field_validity, dtype=bool)
                                corner["validity"][field][index] = bool(field_validity[corner_index])
                            if field_reasons is None:
                                corner["reasons"][field][index] = "ok" if corner["validity"][field][index] else "invalid"
                            else:
                                corner["reasons"][field][index] = str(np.asarray(field_reasons, dtype=object)[corner_index])
                    world_state = result.get("world_corner_states", {}).get(corner_name)
                    if world_state is not None and world_state.get("valid", False):
                        corner["wheel_center"][index] = np.asarray(world_state["wheel_center"], dtype=float)
    pose_runtime = perf_counter() - start
    raw_corner_maps, raw_corner_runtime, raw_corner_sample_count = _build_raw_corner_maps(
        geometry, solver=solver if isinstance(solver, Mapping) else {}
    )
    for name in CORNER_ORDER:
        raw_corner_maps[name]["pose_grid"] = pose_corner_maps[name]
    runtime = pose_runtime + raw_corner_runtime
    metadata = {
        "schema_version": "1.0.0",
        "id": study.get("id", "study") if isinstance(study, Mapping) else "study",
        "synthetic": bool(geometry.get("synthetic", False)),
        "geometry_id": geometry.get("id"),
        "setup_id": setup.get("id"),
        "study_id": study.get("id"),
        "sample_count": sample_count,
        "valid_pose_count": valid_pose_count,
        "invalid_pose_count": sample_count - valid_pose_count,
        "runtime_seconds": runtime,
        "actual_runtime_seconds": runtime,
        "pose_runtime_seconds": pose_runtime,
        "corner_map_runtime_seconds": raw_corner_runtime,
        "corner_map_sample_count": raw_corner_sample_count,
        "corner_map_resource_cap": min(
            int(solver.get("corner_map_max_samples", 36)), 36
        ) if isinstance(solver, Mapping) else 36,
        "axes_order": ("heave_m", "roll_rad", "pitch_rad"),
        "corner_order": CORNER_ORDER,
        "requested_metrics": selected,
        "selected_metrics": selected,
        "stored_metrics": stored,
        "metric_dependencies": {
            "tyre_rate": sorted(tyre_companions) if set(selected) & tyre_metrics else [],
        },
        "solver": dict(solver) if isinstance(solver, Mapping) else {},
        "road_support": "fixed nominal world hub heights with free x/y migration",
        "synthetic_notice": "Synthetic geometry is not validation of an actual vehicle." if geometry.get("synthetic", False) else "",
        "provenance": {
            "geometry": deepcopy(geometry.get("source", {})),
            "setup": deepcopy(setup.get("source", {})),
            "study": deepcopy(study.get("source", {})),
        },
        # Detached resolved inputs make a MapResult self-contained for a later
        # writer or model consumer; source paths remain provenance only.
        "resolved_geometry": deepcopy(geometry),
        "resolved_setup": deepcopy(setup),
        "resolved_study": deepcopy(study),
        "valid_counts": {name: int(np.count_nonzero(validity[name])) for name in stored},
    }
    return MapResult(
        axes=axes,
        metrics=metrics,
        validity=validity,
        metadata=metadata,
        corner_maps=raw_corner_maps,
        reasons=reasons,
        definitions={name: deepcopy(definitions[name]) for name in stored},
    )


__all__ = ["MapResult", "run_study"]
