"""Strict YAML parsing and canonical SI configuration for suspension studies."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "1.0.0"
CORNER_ORDER = ("FL", "FR", "RL", "RR")
_LENGTH_SCALE = {"m": 1.0, "mm": 1e-3}
_ANGLE_SCALE = {"rad": 1.0, "deg": math.pi / 180.0}
_FORCE_SCALE = {"N": 1.0, "kN": 1e3}
_EPS = 1e-12


class ConfigError(ValueError):
    """A configuration error with a field path in its message."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{path}: mapping keys must be strings")
    return dict(value)


def _fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{path}.{unknown[0]}: unknown field")
    missing = sorted(required - set(value))
    if missing:
        raise ConfigError(f"{path}.{missing[0]}: missing required field")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected a non-empty string")
    return value.strip()


def _finite(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path}: expected a finite number")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{path}: must be at least {minimum:g}")
    return result


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer")
    if value < minimum:
        raise ConfigError(f"{path}: must be at least {minimum}")
    return value


def _vector(value: Any, path: str, scale: float = 1.0, *, normalize: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{path}: expected three finite values")
    result = [_finite(item, f"{path}[{index}]") * scale for index, item in enumerate(value)]
    if normalize:
        length = math.sqrt(sum(component * component for component in result))
        if length <= _EPS:
            raise ConfigError(f"{path}: direction vector must have non-zero length")
        result = [component / length for component in result]
    return result


def _pair(
    value: Any,
    path: str,
    scale: float = 1.0,
    *,
    increasing: bool = True,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{path}: expected a pair of finite values")
    result = [_finite(item, f"{path}[{index}]") * scale for index, item in enumerate(value)]
    if positive and result[0] <= 0.0:
        raise ConfigError(f"{path}[0]: must be positive")
    if increasing and result[0] >= result[1]:
        raise ConfigError(f"{path}: values must be strictly increasing")
    return result


def _unit_scale(
    units: Any,
    path: str,
    expected: Mapping[str, Mapping[str, float]],
    *,
    required: set[str],
) -> dict[str, float]:
    value = _mapping(units, path)
    _fields(value, allowed=required, required=required, path=path)
    result: dict[str, float] = {}
    for dimension in sorted(required):
        name = _text(value[dimension], f"{path}.{dimension}")
        if name not in expected[dimension]:
            allowed = ", ".join(expected[dimension])
            raise ConfigError(f"{path}.{dimension}: unsupported unit {name!r}; expected one of {allowed}")
        result[dimension] = expected[dimension][name]
    return result


def _schema_version(value: Any, path: str) -> str:
    version = _text(value, path)
    if version != SCHEMA_VERSION:
        raise ConfigError(f"{path}: unsupported schema version {version!r}; supported version is {SCHEMA_VERSION}")
    return version


def _yaml(path: str | Path, kind: str) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"{kind} file {file_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"{kind} file {file_path}: invalid YAML: {error}") from error
    if value is None:
        raise ConfigError(f"{kind} file {file_path}: document is empty")
    return file_path, _mapping(value, f"{kind}")


def _attachment(
    value: Any,
    path: str,
    length_scale: float,
    *,
    allow_rocker: bool,
) -> dict[str, Any]:
    attachment = _mapping(value, path)
    _fields(attachment, allowed={"body", "point"}, required={"body", "point"}, path=path)
    body = _text(attachment["body"], f"{path}.body")
    bodies = {"chassis", "upright", "lower", "upper"}
    if allow_rocker:
        bodies.add("rocker")
    elif body == "rocker":
        raise ConfigError(f"{path}.body: rocker attachment is only allowed when the corner has a rocker")
    if body not in bodies:
        raise ConfigError(f"{path}.body: expected one of {', '.join(sorted(bodies))}")
    return {"body": body, "point": _vector(attachment["point"], f"{path}.point", length_scale)}


def _component_geometry(
    value: Any,
    path: str,
    length_scale: float,
    *,
    has_rocker: bool,
) -> dict[str, Any]:
    component = _mapping(value, path)
    _fields(
        component,
        allowed={"type", "fixed", "moving", "length_limits"},
        required={"type", "fixed", "moving", "length_limits"},
        path=path,
    )
    component_type = _text(component["type"], f"{path}.type")
    if component_type not in {"direct", "rocker"}:
        raise ConfigError(f"{path}.type: expected 'direct' or 'rocker'")
    fixed = _vector(component["fixed"], f"{path}.fixed", length_scale)
    moving = _attachment(
        component["moving"],
        f"{path}.moving",
        length_scale,
        allow_rocker=has_rocker,
    )
    if moving["body"] == "rocker" and not has_rocker:
        raise ConfigError(f"{path}.moving.body: rocker attachment requires a rocker on this corner")
    if component_type == "rocker" and not has_rocker:
        raise ConfigError(f"{path}.type: rocker component requires a rocker on this corner")
    if component_type == "rocker" and moving["body"] != "rocker":
        raise ConfigError(f"{path}.moving.body: rocker component must attach to the rocker")
    limits = _pair(component["length_limits"], f"{path}.length_limits", length_scale, positive=True)
    reference_length = math.dist(fixed, moving["point"])
    if reference_length <= _EPS:
        raise ConfigError(f"{path}: fixed and moving points have zero reference length")
    if not limits[0] <= reference_length <= limits[1]:
        raise ConfigError(f"{path}.length_limits: reference length {reference_length:g} m is outside the configured limits")
    return {
        "type": component_type,
        "fixed": fixed,
        "moving": moving,
        "length_limits": limits,
    }


def _wishbone(value: Any, path: str, length_scale: float) -> dict[str, list[float]]:
    arm = _mapping(value, path)
    _fields(
        arm,
        allowed={"inboard_a", "inboard_b", "outboard"},
        required={"inboard_a", "inboard_b", "outboard"},
        path=path,
    )
    result = {
        key: _vector(arm[key], f"{path}.{key}", length_scale)
        for key in ("inboard_a", "inboard_b", "outboard")
    }
    if math.dist(result["inboard_a"], result["inboard_b"]) <= _EPS:
        raise ConfigError(f"{path}.inboard_a/inboard_b: wishbone pivot axis has zero length")
    for name in ("inboard_a", "inboard_b"):
        if math.dist(result[name], result["outboard"]) <= _EPS:
            raise ConfigError(f"{path}.{name}/outboard: wishbone arm has zero length")
    return result


def _rocker(value: Any, path: str, length_scale: float, angle_scale: float) -> dict[str, Any]:
    rocker = _mapping(value, path)
    _fields(
        rocker,
        allowed={"pivot", "axis", "rod_point", "rod_mount", "angle_limits"},
        required={"pivot", "axis", "rod_point", "rod_mount", "angle_limits"},
        path=path,
    )
    pivot = _vector(rocker["pivot"], f"{path}.pivot", length_scale)
    axis = _vector(rocker["axis"], f"{path}.axis", normalize=True)
    rod_point = _vector(rocker["rod_point"], f"{path}.rod_point", length_scale)
    rod_mount = _attachment(rocker["rod_mount"], f"{path}.rod_mount", length_scale, allow_rocker=False)
    limits = _pair(rocker["angle_limits"], f"{path}.angle_limits", angle_scale)
    if math.dist(pivot, rod_point) <= _EPS:
        raise ConfigError(f"{path}.rod_point: rocker rod point has zero lever arm from pivot")
    if math.dist(rod_point, rod_mount["point"]) <= _EPS:
        raise ConfigError(f"{path}.rod_mount: rod has zero reference length")
    return {
        "pivot": pivot,
        "axis": axis,
        "rod_point": rod_point,
        "rod_mount": rod_mount,
        "angle_limits": limits,
    }


def _parse_corner(
    value: Any,
    name: str,
    length_scale: float,
    angle_scale: float,
) -> dict[str, Any]:
    path = f"geometry.corners.{name}"
    corner = _mapping(value, path)
    _fields(
        corner,
        allowed={
            "wheel_center", "spindle_axis", "radius", "lower", "upper", "tie",
            "jounce_limits", "spring", "damper", "rocker",
        },
        required={
            "wheel_center", "spindle_axis", "radius", "lower", "upper", "tie",
            "jounce_limits", "spring",
        },
        path=path,
    )
    has_rocker = "rocker" in corner
    result: dict[str, Any] = {
        "wheel_center": _vector(corner["wheel_center"], f"{path}.wheel_center", length_scale),
        # The spindle is an axial direction, oriented by the right-hand rule.
        "spindle_axis": _vector(corner["spindle_axis"], f"{path}.spindle_axis", normalize=True),
        "radius": _finite(corner["radius"], f"{path}.radius", minimum=0.0) * length_scale,
        "lower": _wishbone(corner["lower"], f"{path}.lower", length_scale),
        "upper": _wishbone(corner["upper"], f"{path}.upper", length_scale),
        "jounce_limits": _pair(corner["jounce_limits"], f"{path}.jounce_limits", length_scale),
        "spring": _component_geometry(corner["spring"], f"{path}.spring", length_scale, has_rocker=has_rocker),
    }
    if result["radius"] <= 0.0:
        raise ConfigError(f"{path}.radius: must be positive")
    tie = _mapping(corner["tie"], f"{path}.tie")
    _fields(tie, allowed={"inboard", "outboard"}, required={"inboard", "outboard"}, path=f"{path}.tie")
    result["tie"] = {
        key: _vector(tie[key], f"{path}.tie.{key}", length_scale)
        for key in ("inboard", "outboard")
    }
    if math.dist(result["tie"]["inboard"], result["tie"]["outboard"]) <= _EPS:
        raise ConfigError(f"{path}.tie: toe link has zero reference length")
    if "damper" in corner:
        result["damper"] = _component_geometry(
            corner["damper"], f"{path}.damper", length_scale, has_rocker=has_rocker
        )
    else:
        result["damper"] = copy.deepcopy(result["spring"])
    if has_rocker:
        result["rocker"] = _rocker(corner["rocker"], f"{path}.rocker", length_scale, angle_scale)
    return result


def _mirror_point(point: list[float]) -> list[float]:
    return [point[0], -point[1], point[2]]


def _mirror_axial(vector: list[float]) -> list[float]:
    # Reflection S=diag(1,-1,1), axial vectors transform by det(S) S.
    return [-vector[0], vector[1], -vector[2]]


def _mirror_attachment(value: dict[str, Any]) -> dict[str, Any]:
    return {"body": value["body"], "point": _mirror_point(value["point"])}


def _mirror_corner(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["wheel_center"] = _mirror_point(value["wheel_center"])
    result["spindle_axis"] = _mirror_axial(value["spindle_axis"])
    for wishbone in ("lower", "upper"):
        for point in ("inboard_a", "inboard_b", "outboard"):
            result[wishbone][point] = _mirror_point(value[wishbone][point])
    for point in ("inboard", "outboard"):
        result["tie"][point] = _mirror_point(value["tie"][point])
    for component in ("spring", "damper"):
        result[component]["fixed"] = _mirror_point(value[component]["fixed"])
        result[component]["moving"] = _mirror_attachment(value[component]["moving"])
    if "rocker" in value:
        result["rocker"]["pivot"] = _mirror_point(value["rocker"]["pivot"])
        result["rocker"]["axis"] = _mirror_axial(value["rocker"]["axis"])
        result["rocker"]["rod_point"] = _mirror_point(value["rocker"]["rod_point"])
        result["rocker"]["rod_mount"] = _mirror_attachment(value["rocker"]["rod_mount"])
    return result


def _parse_lever(value: Any, path: str, length_scale: float, angle_scale: float) -> dict[str, Any]:
    lever = _mapping(value, path)
    _fields(
        lever,
        allowed={"pivot", "axis", "tip", "pickup", "angle_limits"},
        required={"pivot", "axis", "tip", "pickup", "angle_limits"},
        path=path,
    )
    pivot = _vector(lever["pivot"], f"{path}.pivot", length_scale)
    tip = _vector(lever["tip"], f"{path}.tip", length_scale)
    if math.dist(pivot, tip) <= _EPS:
        raise ConfigError(f"{path}.tip: lever tip has zero arm length from pivot")
    return {
        "pivot": pivot,
        "axis": _vector(lever["axis"], f"{path}.axis", normalize=True),
        "tip": tip,
        "pickup": _attachment(lever["pickup"], f"{path}.pickup", length_scale, allow_rocker=False),
        "angle_limits": _pair(lever["angle_limits"], f"{path}.angle_limits", angle_scale),
    }


def _parse_geometry_arbs(
    value: Any,
    length_scale: float,
    angle_scale: float,
    *,
    mirrored: bool,
) -> dict[str, Any]:
    arbs = _mapping(value, "geometry.arbs")
    _fields(arbs, allowed={"front", "rear"}, required=set(), path="geometry.arbs")
    result: dict[str, Any] = {}
    for axle, axle_raw in arbs.items():
        path = f"geometry.arbs.{axle}"
        axle_data = _mapping(axle_raw, path)
        _fields(
            axle_data,
            allowed={"left", "right"},
            required={"left"} if mirrored else {"left", "right"},
            path=path,
        )
        if mirrored and "right" in axle_data:
            raise ConfigError(f"{path}.right: right lever is generated from the mirrored left lever")
        left = _parse_lever(axle_data["left"], f"{path}.left", length_scale, angle_scale)
        if mirrored:
            right = copy.deepcopy(left)
            right["pivot"] = _mirror_point(left["pivot"])
            right["axis"] = _mirror_axial(left["axis"])
            right["tip"] = _mirror_point(left["tip"])
            right["pickup"] = _mirror_attachment(left["pickup"])
        else:
            right = _parse_lever(axle_data["right"], f"{path}.right", length_scale, angle_scale)
        result[axle] = {"left": left, "right": right}
    return result


def _parse_geometry_mapping(
    raw: Mapping[str, Any],
    *,
    source: Any = None,
    canonical: bool = False,
) -> dict[str, Any]:
    path = "geometry"
    allowed = {
        "schema_version", "id", "synthetic", "units", "reference_origin_world",
        "corners", "arbs",
    }
    required = {
        "schema_version", "id", "synthetic", "units", "reference_origin_world", "corners",
    }
    if not canonical:
        allowed.add("mirror_right")
    _fields(raw, allowed=allowed, required=required, path=path)
    _schema_version(raw["schema_version"], f"{path}.schema_version")
    config_id = _text(raw["id"], f"{path}.id")
    if not isinstance(raw["synthetic"], bool):
        raise ConfigError(f"{path}.synthetic: expected true or false")
    units = _unit_scale(
        raw["units"],
        f"{path}.units",
        {"length": _LENGTH_SCALE, "angle": _ANGLE_SCALE},
        required={"length", "angle"},
    )
    mirror_right = raw.get("mirror_right", False)
    if not isinstance(mirror_right, bool):
        raise ConfigError(f"{path}.mirror_right: expected true or false")
    if canonical and mirror_right:
        raise ConfigError(f"{path}.mirror_right: canonical geometry must already contain all four corners")
    corners_raw = _mapping(raw["corners"], f"{path}.corners")
    expected_input = {"FL", "RL"} if mirror_right else set(CORNER_ORDER)
    if set(corners_raw) != expected_input:
        expected = ", ".join(name for name in CORNER_ORDER if name in expected_input)
        raise ConfigError(f"{path}.corners: expected exactly {expected}")
    corners: dict[str, Any] = {}
    for name in ("FL", "RL") if mirror_right else CORNER_ORDER:
        corners[name] = _parse_corner(corners_raw[name], name, units["length"], units["angle"])
    if mirror_right:
        corners["FR"] = _mirror_corner(corners["FL"])
        corners["RR"] = _mirror_corner(corners["RL"])
    reference_origin = _vector(
        raw["reference_origin_world"],
        f"{path}.reference_origin_world",
        units["length"],
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": config_id,
        "synthetic": raw["synthetic"],
        "units": {"length": "m", "angle": "rad"},
        "reference_origin_world": reference_origin,
        "corners": {name: corners[name] for name in CORNER_ORDER},
        "arbs": _parse_geometry_arbs(
            raw.get("arbs", {}),
            units["length"],
            units["angle"],
            mirrored=mirror_right,
        ),
        "source": source if source is not None else raw.get("source", {}),
    }
    return result


def load_geometry(path: str | Path) -> dict[str, Any]:
    """Load a geometry YAML file and return a detached canonical SI dictionary."""
    file_path, raw = _yaml(path, "geometry")
    source = {"path": str(file_path), "mirrored_right": raw.get("mirror_right", False)}
    return _parse_geometry_mapping(raw, source=source)


def _require_unit_direction(value: Any, path: str) -> None:
    direction = _vector(value, path)
    norm = math.sqrt(sum(component * component for component in direction))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ConfigError(f"{path}: canonical direction vector must already be normalized to unit length")


def validate_geometry(geometry: Mapping[str, Any]) -> None:
    """Validate a canonical SI geometry dictionary without running the solver."""
    value = _mapping(geometry, "geometry")
    source = value.pop("source", {})
    units = _mapping(value.get("units"), "geometry.units")
    if units != {"length": "m", "angle": "rad"}:
        raise ConfigError("geometry.units: validate_geometry accepts canonical SI units only (m and rad)")
    corners = _mapping(value.get("corners"), "geometry.corners")
    for name in CORNER_ORDER:
        if name not in corners:
            continue
        corner = _mapping(corners[name], f"geometry.corners.{name}")
        if "damper" not in corner:
            raise ConfigError(
                f"geometry.corners.{name}.damper: canonical geometry must include resolved damper geometry"
            )
        _require_unit_direction(corner.get("spindle_axis"), f"geometry.corners.{name}.spindle_axis")
        if "rocker" in corner:
            rocker = _mapping(corner["rocker"], f"geometry.corners.{name}.rocker")
            _require_unit_direction(rocker.get("axis"), f"geometry.corners.{name}.rocker.axis")
    arbs = _mapping(value.get("arbs", {}), "geometry.arbs")
    for axle, axle_value in arbs.items():
        axle_data = _mapping(axle_value, f"geometry.arbs.{axle}")
        for side, lever_value in axle_data.items():
            lever = _mapping(lever_value, f"geometry.arbs.{axle}.{side}")
            _require_unit_direction(lever.get("axis"), f"geometry.arbs.{axle}.{side}.axis")
    _parse_geometry_mapping(value, source=source, canonical=True)
    return None


def _force_curve(
    value: Any,
    path: str,
    length_scale: float,
    force_scale: float,
) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ConfigError(f"{path}: expected at least two [compression, force] points")
    result: list[list[float]] = []
    for index, pair in enumerate(value):
        point_path = f"{path}[{index}]"
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"{point_path}: expected [compression, force]")
        x = _finite(pair[0], f"{point_path}[0]") * length_scale
        y = _finite(pair[1], f"{point_path}[1]") * force_scale
        if result and x <= result[-1][0]:
            raise ConfigError(f"{point_path}[0]: curve abscissae must be strictly increasing")
        if result and y < result[-1][1] - _EPS:
            raise ConfigError(f"{point_path}[1]: force curve must have a passive non-negative tangent")
        result.append([x, y])
    return result


def _spring(value: Any, path: str, length_scale: float, force_scale: float) -> dict[str, Any]:
    spring = _mapping(value, path)
    if "curve" in spring:
        _fields(spring, allowed={"curve"}, required={"curve"}, path=path)
        return {"curve": _force_curve(spring["curve"], f"{path}.curve", length_scale, force_scale)}
    _fields(
        spring,
        allowed={"rate", "preload_force", "preload_compression"},
        required={"rate"},
        path=path,
    )
    if "preload_force" in spring and "preload_compression" in spring:
        raise ConfigError(f"{path}.preload_force: conflicts with preload_compression; specify only one preload")
    rate = _finite(spring["rate"], f"{path}.rate", minimum=0.0) * force_scale / length_scale
    if rate <= 0.0:
        raise ConfigError(f"{path}.rate: must be positive")
    if "preload_compression" in spring:
        compression = _finite(spring["preload_compression"], f"{path}.preload_compression", minimum=0.0) * length_scale
        preload_force = rate * compression
    else:
        preload_force = _finite(spring.get("preload_force", 0.0), f"{path}.preload_force", minimum=0.0) * force_scale
    return {"rate": rate, "preload_force": preload_force}


def _bump_stop(value: Any, path: str, length_scale: float, force_scale: float) -> dict[str, Any]:
    stop = _mapping(value, path)
    if "curve" in stop:
        _fields(stop, allowed={"engagement", "curve"}, required={"engagement", "curve"}, path=path)
        return {
            "engagement": _finite(stop["engagement"], f"{path}.engagement") * length_scale,
            "curve": _force_curve(stop["curve"], f"{path}.curve", length_scale, force_scale),
        }
    _fields(stop, allowed={"engagement", "rate"}, required={"engagement", "rate"}, path=path)
    rate = _finite(stop["rate"], f"{path}.rate", minimum=0.0) * force_scale / length_scale
    if rate <= 0.0:
        raise ConfigError(f"{path}.rate: must be positive")
    return {
        "engagement": _finite(stop["engagement"], f"{path}.engagement") * length_scale,
        "rate": rate,
    }


def _tyre(value: Any, path: str, length_scale: float, force_scale: float) -> dict[str, Any]:
    tyre = _mapping(value, path)
    if "curve" in tyre:
        _fields(tyre, allowed={"curve"}, required={"curve"}, path=path)
        return {"curve": _force_curve(tyre["curve"], f"{path}.curve", length_scale, force_scale)}
    _fields(tyre, allowed={"rate", "reference_force"}, required={"rate"}, path=path)
    rate = _finite(tyre["rate"], f"{path}.rate", minimum=0.0) * force_scale / length_scale
    if rate <= 0.0:
        raise ConfigError(f"{path}.rate: must be positive")
    return {
        "rate": rate,
        "reference_force": _finite(tyre.get("reference_force", 0.0), f"{path}.reference_force", minimum=0.0)
        * force_scale,
    }


def _arb_law(value: Any, path: str, length_scale: float, angle_scale: float, force_scale: float) -> dict[str, Any]:
    law = _mapping(value, path)
    torque_scale = length_scale * force_scale
    if "curve" in law:
        _fields(law, allowed={"curve"}, required={"curve"}, path=path)
        if not isinstance(law["curve"], (list, tuple)) or len(law["curve"]) < 2:
            raise ConfigError(f"{path}.curve: expected at least two [twist, torque] points")
        curve: list[list[float]] = []
        for index, pair in enumerate(law["curve"]):
            point_path = f"{path}.curve[{index}]"
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ConfigError(f"{point_path}: expected [twist, torque]")
            twist = _finite(pair[0], f"{point_path}[0]") * angle_scale
            torque = _finite(pair[1], f"{point_path}[1]") * torque_scale
            if curve and twist <= curve[-1][0]:
                raise ConfigError(f"{point_path}[0]: curve abscissae must be strictly increasing")
            if curve and torque < curve[-1][1] - _EPS:
                raise ConfigError(f"{point_path}[1]: torque curve must have a passive non-negative tangent")
            curve.append([twist, torque])
        return {"curve": curve}
    _fields(law, allowed={"rate", "reference_twist"}, required={"rate", "reference_twist"}, path=path)
    rate = _finite(law["rate"], f"{path}.rate", minimum=0.0) * torque_scale / angle_scale
    return {
        "rate": rate,
        "reference_twist": _finite(law["reference_twist"], f"{path}.reference_twist") * angle_scale,
    }


def _parse_setup_mapping(
    raw: Mapping[str, Any],
    *,
    source: Any,
    geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _fields(
        raw,
        allowed={"schema_version", "id", "units", "corners", "arbs"},
        required={"schema_version", "id", "units", "corners"},
        path="setup",
    )
    _schema_version(raw["schema_version"], "setup.schema_version")
    setup_id = _text(raw["id"], "setup.id")
    units = _unit_scale(
        raw["units"],
        "setup.units",
        {"length": _LENGTH_SCALE, "force": _FORCE_SCALE, "angle": _ANGLE_SCALE},
        required={"length", "force", "angle"},
    )
    corners_raw = _mapping(raw["corners"], "setup.corners")
    if set(corners_raw) != set(CORNER_ORDER):
        raise ConfigError("setup.corners: expected exactly FL, FR, RL and RR")
    corners: dict[str, Any] = {}
    for name in CORNER_ORDER:
        path = f"setup.corners.{name}"
        corner = _mapping(corners_raw[name], path)
        _fields(
            corner,
            allowed={"spring", "bump_stop", "tyre"},
            required={"spring"},
            path=path,
        )
        item: dict[str, Any] = {
            "spring": _spring(corner["spring"], f"{path}.spring", units["length"], units["force"])
        }
        if "bump_stop" in corner:
            item["bump_stop"] = _bump_stop(
                corner["bump_stop"], f"{path}.bump_stop", units["length"], units["force"]
            )
        if "tyre" in corner:
            item["tyre"] = _tyre(corner["tyre"], f"{path}.tyre", units["length"], units["force"])
        corners[name] = item

    arbs_raw = _mapping(raw.get("arbs", {}), "setup.arbs")
    _fields(arbs_raw, allowed={"front", "rear"}, required=set(), path="setup.arbs")
    geometry_arbs = geometry.get("arbs", {}) if geometry is not None else None
    arbs: dict[str, Any] = {"front": None, "rear": None}
    for axle, law_raw in arbs_raw.items():
        law = _arb_law(
            law_raw,
            f"setup.arbs.{axle}",
            units["length"],
            units["angle"],
            units["force"],
        )
        active = False
        if "rate" in law:
            active = law["rate"] > 0.0
        else:
            active = any(abs(point[1]) > _EPS for point in law["curve"]) or any(
                law["curve"][index + 1][1] > law["curve"][index][1] + _EPS
                for index in range(len(law["curve"]) - 1)
            )
        if active:
            if geometry is None:
                raise ConfigError(f"setup.arbs.{axle}: active ARB requires matching geometry")
            axle_geometry = geometry_arbs.get(axle) if isinstance(geometry_arbs, Mapping) else None
            if not axle_geometry or set(axle_geometry) != {"left", "right"}:
                raise ConfigError(f"setup.arbs.{axle}: active ARB requires matching geometry")
        arbs[axle] = law
    return {
        "schema_version": SCHEMA_VERSION,
        "id": setup_id,
        "units": {"length": "m", "force": "N", "angle": "rad"},
        "corners": corners,
        "arbs": arbs,
        "source": source,
    }


def load_setup(path: str | Path, geometry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a setup and optionally validate active anti-roll bars against geometry."""
    file_path, raw = _yaml(path, "setup")
    if geometry is not None:
        validate_geometry(geometry)
    return _parse_setup_mapping(raw, source={"path": str(file_path)}, geometry=geometry)


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected a list of names")
    result = [_text(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ConfigError(f"{path}: names must not contain duplicates")
    return result


def _axis_count(value: Any, path: str) -> int:
    if isinstance(value, Mapping):
        axis = _mapping(value, path)
        _fields(axis, allowed={"min", "max", "count"}, required={"min", "max", "count"}, path=path)
        minimum = _finite(axis["min"], f"{path}.min")
        maximum = _finite(axis["max"], f"{path}.max")
        count = _integer(axis["count"], f"{path}.count", minimum=1)
        if maximum < minimum:
            raise ConfigError(f"{path}: max must be at least min")
        if count > 1 and maximum == minimum:
            raise ConfigError(f"{path}: range endpoints must differ when count is greater than one")
        return count
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{path}: expected a non-empty list or min/max/count mapping")
    return len(value)


def _axis(value: Any, path: str, scale: float, *, max_count: int | None = None) -> list[float]:
    if isinstance(value, Mapping):
        axis = _mapping(value, path)
        _fields(axis, allowed={"min", "max", "count"}, required={"min", "max", "count"}, path=path)
        minimum = _finite(axis["min"], f"{path}.min") * scale
        maximum = _finite(axis["max"], f"{path}.max") * scale
        count = _integer(axis["count"], f"{path}.count", minimum=1)
        if max_count is not None and count > max_count:
            raise ConfigError(f"{path}.count: generated axis exceeds max_samples limit {max_count}")
        if maximum < minimum:
            raise ConfigError(f"{path}: max must be at least min")
        if count > 1 and maximum == minimum:
            raise ConfigError(f"{path}: range endpoints must differ when count is greater than one")
        if count == 1:
            return [minimum]
        step = (maximum - minimum) / (count - 1)
        return [minimum + index * step for index in range(count)]
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{path}: expected a non-empty list or min/max/count mapping")
    result = [_finite(item, f"{path}[{index}]") * scale for index, item in enumerate(value)]
    if any(result[index] >= result[index + 1] for index in range(len(result) - 1)):
        raise ConfigError(f"{path}: explicit values must be strictly increasing")
    return result


def _solver(value: Any, length_scale: float, angle_scale: float) -> dict[str, Any]:
    defaults = {
        "max_samples": 2000,
        "max_nfev": 100,
        "residual_tolerance": 1e-9,
        "derivative_steps": [1e-4, 1e-4, 1e-4],
        "jounce_step": 1e-4,
    }
    if value is None:
        value = {}
    solver = _mapping(value, "study.solver")
    _fields(
        solver,
        allowed=set(defaults),
        required=set(),
        path="study.solver",
    )
    result = dict(defaults)
    result.update(solver)
    result["max_samples"] = _integer(result["max_samples"], "study.solver.max_samples", minimum=1)
    result["max_nfev"] = _integer(result["max_nfev"], "study.solver.max_nfev", minimum=1)
    residual_tolerance = _finite(
        result["residual_tolerance"], "study.solver.residual_tolerance", minimum=0.0
    )
    if "residual_tolerance" in solver:
        residual_tolerance *= length_scale
    result["residual_tolerance"] = residual_tolerance
    if result["residual_tolerance"] <= 0.0:
        raise ConfigError("study.solver.residual_tolerance: must be positive")
    steps = result["derivative_steps"]
    if not isinstance(steps, (list, tuple)) or len(steps) != 3:
        raise ConfigError("study.solver.derivative_steps: expected three positive values")
    step_scales = (
        [length_scale, angle_scale, angle_scale]
        if "derivative_steps" in solver
        else [1.0, 1.0, 1.0]
    )
    result["derivative_steps"] = [
        _finite(steps[0], "study.solver.derivative_steps[0]", minimum=0.0) * step_scales[0],
        _finite(steps[1], "study.solver.derivative_steps[1]", minimum=0.0) * step_scales[1],
        _finite(steps[2], "study.solver.derivative_steps[2]", minimum=0.0) * step_scales[2],
    ]
    if any(step <= 0.0 for step in result["derivative_steps"]):
        raise ConfigError("study.solver.derivative_steps: values must be positive")
    jounce_step = _finite(result["jounce_step"], "study.solver.jounce_step", minimum=0.0)
    if "jounce_step" in solver:
        jounce_step *= length_scale
    result["jounce_step"] = jounce_step
    if result["jounce_step"] <= 0.0:
        raise ConfigError("study.solver.jounce_step: must be positive")
    return result


def _parse_study_mapping(raw: Mapping[str, Any], *, file_path: Path, source: Any) -> dict[str, Any]:
    _fields(
        raw,
        allowed={
            "schema_version", "id", "geometry", "setup", "units", "reference",
            "axes", "metrics", "plots", "solver",
        },
        required={
            "schema_version", "id", "geometry", "setup", "units", "reference",
            "axes", "metrics", "plots",
        },
        path="study",
    )
    _schema_version(raw["schema_version"], "study.schema_version")
    study_id = _text(raw["id"], "study.id")
    geometry_path = (file_path.parent / _text(raw["geometry"], "study.geometry")).resolve()
    setup_path = (file_path.parent / _text(raw["setup"], "study.setup")).resolve()
    units = _unit_scale(
        raw["units"],
        "study.units",
        {"length": _LENGTH_SCALE, "angle": _ANGLE_SCALE},
        required={"length", "angle"},
    )
    reference_raw = raw["reference"]
    if not isinstance(reference_raw, (list, tuple)) or len(reference_raw) != 3:
        raise ConfigError("study.reference: expected [heave, roll, pitch]")
    reference = [
        _finite(reference_raw[0], "study.reference[0]") * units["length"],
        _finite(reference_raw[1], "study.reference[1]") * units["angle"],
        _finite(reference_raw[2], "study.reference[2]") * units["angle"],
    ]
    axes_raw = _mapping(raw["axes"], "study.axes")
    _fields(
        axes_raw,
        allowed={"heave", "roll", "pitch"},
        required={"heave", "roll", "pitch"},
        path="study.axes",
    )
    solver = _solver(raw.get("solver"), units["length"], units["angle"])
    axis_counts = {
        "heave": _axis_count(axes_raw["heave"], "study.axes.heave"),
        "roll": _axis_count(axes_raw["roll"], "study.axes.roll"),
        "pitch": _axis_count(axes_raw["pitch"], "study.axes.pitch"),
    }
    sample_count = axis_counts["heave"] * axis_counts["roll"] * axis_counts["pitch"]
    if sample_count > solver["max_samples"]:
        raise ConfigError(
            f"study.solver.max_samples: grid has {sample_count} samples, exceeding limit {solver['max_samples']}"
        )
    axes = {
        "heave_m": _axis(
            axes_raw["heave"], "study.axes.heave", units["length"], max_count=solver["max_samples"]
        ),
        "roll_rad": _axis(
            axes_raw["roll"], "study.axes.roll", units["angle"], max_count=solver["max_samples"]
        ),
        "pitch_rad": _axis(
            axes_raw["pitch"], "study.axes.pitch", units["angle"], max_count=solver["max_samples"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "id": study_id,
        "geometry": str(geometry_path),
        "setup": str(setup_path),
        "units": {"length": "m", "angle": "rad"},
        "reference": reference,
        "axes": axes,
        # Metric membership belongs to the run/CLI layer, which owns the metric registry.
        "metrics": _string_list(raw["metrics"], "study.metrics"),
        "plots": _string_list(raw["plots"], "study.plots"),
        "solver": solver,
        "source": source,
    }


def load_study(path: str | Path) -> dict[str, Any]:
    """Load a study, resolving geometry/setup paths relative to the study file."""
    file_path, raw = _yaml(path, "study")
    return _parse_study_mapping(raw, file_path=file_path, source={"path": str(file_path)})


def resolve_study(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a study and its referenced geometry and setup configurations."""
    study = load_study(path)
    geometry = load_geometry(study["geometry"])
    setup = load_setup(study["setup"], geometry=geometry)
    return geometry, setup, study
