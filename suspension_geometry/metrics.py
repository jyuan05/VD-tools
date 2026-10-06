"""Named engineering metrics and their stable array contracts.

The numerical composition lives in :mod:`suspension_geometry.vehicle`; this
module only owns names, units, tail axes, and the small adapter that turns a
raw pose evaluation into the selected metric dictionary.  Keeping the
registry here gives the export layer one authoritative source for shapes and
per-entry units.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np


CORNER_ORDER = ("FL", "FR", "RL", "RR")
Q_LABELS = ("heave_m", "roll_rad", "pitch_rad")


def _corner_definition(
    unit: str,
    meaning: str,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    return {
        "unit": unit,
        "units": unit,
        "axes": ("heave_m", "roll_rad", "pitch_rad", "corner"),
        "tail_axes": ("corner",),
        "tail_shape": (4,),
        "tail_labels": ("FL", "FR", "RL", "RR"),
        "meaning": meaning,
        "definition": meaning,
        **({"name": name} if name is not None else {}),
    }


def _scalar_definition(unit: str, meaning: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "units": unit,
        "axes": ("heave_m", "roll_rad", "pitch_rad"),
        "tail_axes": (),
        "tail_shape": (),
        "tail_labels": (),
        "meaning": meaning,
        "definition": meaning,
    }


def _matrix_definition(
    meaning: str,
    entry_units: tuple[tuple[str, ...], ...],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "unit": "mixed",
        "units": "mixed",
        "axes": ("heave_m", "roll_rad", "pitch_rad"),
        "tail_axes": ("q", "q"),
        "tail_shape": (3, 3),
        "tail_labels": ("q", "q"),
        "entry_units": entry_units,
        "entry_labels": (labels, labels),
        "meaning": meaning,
        "definition": meaning,
    }


def _gain_definition(meaning: str) -> dict[str, Any]:
    entry_units = (("m/m", "m/rad", "m/rad"),) * 4
    return {
        "unit": "mixed",
        "units": "mixed",
        "axes": ("heave_m", "roll_rad", "pitch_rad", "corner", "q"),
        "tail_axes": ("corner", "q"),
        "tail_shape": (4, 3),
        "tail_labels": (("FL", "FR", "RL", "RR"), ("heave", "roll", "pitch")),
        "entry_units": entry_units,
        "entry_labels": (("FL", "FR", "RL", "RR"), ("heave", "roll", "pitch")),
        "meaning": meaning,
        "definition": meaning,
    }


def _definitions() -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    definitions["jounce"] = _corner_definition("m", "Wheel-centre jounce relative to the configured chassis reference.")
    definitions["wheel_center_x"] = _corner_definition("m", "World wheel-centre x coordinate.")
    definitions["wheel_center_y"] = _corner_definition("m", "World wheel-centre y coordinate.")
    definitions["wheel_center_z"] = _corner_definition("m", "World wheel-centre z coordinate.")
    definitions["wheel_migration_x"] = _corner_definition("m", "World wheel-centre longitudinal migration from nominal.")
    definitions["wheel_migration_y"] = _corner_definition("m", "World wheel-centre lateral migration from nominal.")
    definitions["wheel_migration"] = {
        **_corner_definition("m", "World wheel-centre migration [x, y] from nominal."),
        "tail_axes": ("corner", "migration_axis"),
        "tail_shape": (4, 2),
        "tail_labels": (("FL", "FR", "RL", "RR"), ("x", "y")),
    }
    definitions["camber_chassis"] = _corner_definition("rad", "Camber in the chassis frame; positive means the top of the wheel leans outward.")
    definitions["camber_road"] = _corner_definition("rad", "Camber from the world spindle normal relative to the declared flat road frame.")
    definitions["toe"] = _corner_definition("rad", "Toe-in angle; positive means toe-in.")
    definitions["caster"] = _corner_definition("rad", "Caster from the upper and lower kingpin points; positive upper point rearward.")
    definitions["spring_compression"] = _corner_definition("m", "Spring compression relative to its configured reference length.")
    definitions["damper_compression"] = _corner_definition("m", "Damper compression relative to its configured reference length.")
    definitions["rocker_angle"] = _corner_definition("rad", "Rocker rotation relative to its reference assembly.")
    definitions["spring_stroke_margin"] = {
        **_corner_definition("m", "Spring remaining compression and extension stroke."),
        "tail_axes": ("corner", "stroke_side"),
        "tail_shape": (4, 2),
        "tail_labels": (("FL", "FR", "RL", "RR"), ("compression", "extension")),
    }
    definitions["damper_stroke_margin"] = {
        **_corner_definition("m", "Damper remaining compression and extension stroke."),
        "tail_axes": ("corner", "stroke_side"),
        "tail_shape": (4, 2),
        "tail_labels": (("FL", "FR", "RL", "RR"), ("compression", "extension")),
    }
    definitions["spring_motion_ratio"] = _corner_definition("m/m", "Intrinsic local spring compression change per wheel jounce.")
    definitions["damper_motion_ratio"] = _corner_definition("m/m", "Intrinsic local damper compression change per wheel jounce.")
    definitions["wheel_force"] = _corner_definition("N", "Suspension generalized force with respect to each wheel jounce.")
    definitions["wheel_material_rate"] = _corner_definition("N/m", "Material wheel rate excluding force-times-geometry curvature.")
    definitions["wheel_tangent_rate"] = _corner_definition("N/m", "Full local wheel tangent rate including geometry curvature.")
    definitions["wheel_energy"] = _scalar_definition("J", "Total suspension stored-energy change at the prescribed pose.")
    definitions["body_gradient"] = {
        **_scalar_definition("mixed", "dU/dq for q=[heave_m, roll_rad, pitch_rad]."),
        "tail_axes": ("q",),
        "tail_shape": (3,),
        "tail_labels": ("heave", "roll", "pitch"),
        "entry_units": ("N", "N*m/rad", "N*m/rad"),
        "entry_labels": ("heave", "roll", "pitch"),
    }
    definitions["body_restoring_reaction"] = {
        **definitions["body_gradient"],
        "meaning": "Generalized restoring reaction -dU/dq for q=[heave_m, roll_rad, pitch_rad].",
        "definition": "Generalized restoring reaction -dU/dq for q=[heave_m, roll_rad, pitch_rad].",
    }
    body_units = (
        ("N/m", "N/rad", "N/rad"),
        ("N/rad", "N*m/rad", "N*m/rad"),
        ("N/rad", "N*m/rad", "N*m/rad"),
    )
    definitions["body_stiffness"] = _matrix_definition(
        "Full constrained body Hessian d2U/dq2 including geometric and ARB coupling.",
        body_units,
        ("heave", "roll", "pitch"),
    )
    definitions["wheel_stiffness"] = {
        "unit": "N/m",
        "units": "N/m",
        "axes": ("heave_m", "roll_rad", "pitch_rad", "wheel", "wheel"),
        "tail_axes": ("corner", "corner"),
        "tail_shape": (4, 4),
        "tail_labels": (("FL", "FR", "RL", "RR"), ("FL", "FR", "RL", "RR")),
        "meaning": "Full coupled suspension wheel-coordinate Hessian.",
        "definition": "Full coupled suspension wheel-coordinate Hessian.",
    }
    definitions["spring_actuation_gains"] = _gain_definition("dc_spring/dq for q=[heave_m, roll_rad, pitch_rad].")
    definitions["damper_actuation_gains"] = _gain_definition("dc_damper/dq for q=[heave_m, roll_rad, pitch_rad].")
    definitions["axle_heave_stiffness"] = {
        **_scalar_definition("N/m", "Constrained heave stiffness contributed by front and rear axle groups."),
        "tail_axes": ("axle",), "tail_shape": (2,), "tail_labels": ("front", "rear"),
    }
    definitions["axle_roll_stiffness"] = {
        **_scalar_definition("N*m/rad", "Constrained roll stiffness contributed by front and rear axle groups."),
        "tail_axes": ("axle",), "tail_shape": (2,), "tail_labels": ("front", "rear"),
    }
    definitions["axle_pitch_stiffness"] = {
        **_scalar_definition("N*m/rad", "Constrained pitch stiffness contributed by front and rear axle groups."),
        "tail_axes": ("axle",), "tail_shape": (2,), "tail_labels": ("front", "rear"),
    }
    definitions["heave_stiffness"] = _scalar_definition("N/m", "Constrained body heave stiffness with roll and pitch held.")
    definitions["pitch_stiffness"] = _scalar_definition("N*m/rad", "Constrained body pitch stiffness with heave and roll held.")
    definitions["front_roll_stiffness"] = _scalar_definition("N*m/rad", "Front axle contribution to constrained elastic roll stiffness.")
    definitions["rear_roll_stiffness"] = _scalar_definition("N*m/rad", "Rear axle contribution to constrained elastic roll stiffness.")
    definitions["total_roll_stiffness"] = _scalar_definition("N*m/rad", "Total constrained elastic roll stiffness.")
    definitions["front_elastic_roll_fraction"] = _scalar_definition("1", "Front contribution divided by total elastic roll stiffness.")
    definitions["ride_rate"] = _corner_definition("N/m", "Diagonal of the coupled local tyre-inclusive wheel support stiffness.")
    definitions["ride_stiffness"] = {
        **definitions["wheel_stiffness"],
        "meaning": "Coupled local tyre-inclusive wheel support stiffness at the solved internal equilibrium.",
        "definition": "Coupled local tyre-inclusive wheel support stiffness at the solved internal equilibrium.",
    }
    definitions["body_ride_stiffness"] = _matrix_definition(
        "Tyre-inclusive body Hessian after condensing solved internal hub displacements.",
        body_units,
        ("heave", "roll", "pitch"),
    )
    definitions["tyre_equilibrium_hub_displacement"] = _corner_definition("m", "Solved internal world hub rise relative to the fixed-support reference.")
    definitions["tyre_equilibrium_jounce"] = _corner_definition("m", "Wheel jounce associated with the solved tyre equilibrium.")
    definitions["tyre_contact_force"] = _corner_definition("N", "Tyre contact force at the solved internal equilibrium.")
    definitions["tyre_effective_reaction"] = {
        **definitions["body_gradient"],
        "meaning": "Generalized suspension reaction evaluated at the solved tyre equilibrium.",
        "definition": "Generalized suspension reaction evaluated at the solved tyre equilibrium.",
    }
    return definitions


_METRIC_DEFINITIONS = _definitions()


def metric_definitions() -> dict[str, dict[str, Any]]:
    """Return a detached copy of the authoritative metric registry."""
    return deepcopy(_METRIC_DEFINITIONS)


def known_metrics() -> tuple[str, ...]:
    return tuple(_METRIC_DEFINITIONS)


def _copy_metric(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (float, int, bool, np.generic)):
        return value.item() if isinstance(value, np.generic) else value
    return deepcopy(value)


def metrics_from_pose(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the metric dictionary emitted by ``evaluate_pose``.

    This adapter also accepts a raw result that only has the intermediate
    fields, which is useful to callers composing a custom solver.
    """
    if "metrics" in raw:
        return {name: _copy_metric(value) for name, value in raw["metrics"].items()}
    return {}


__all__ = ["CORNER_ORDER", "metric_definitions", "known_metrics", "metrics_from_pose"]
