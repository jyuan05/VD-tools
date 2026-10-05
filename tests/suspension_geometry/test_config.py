"""Behavioral tests for the suspension geometry configuration contract."""

from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path

import pytest
import yaml


try:
    config = importlib.import_module("suspension_geometry.config")
except ModuleNotFoundError as error:
    if error.name not in {"suspension_geometry", "suspension_geometry.config"}:
        raise
    config = None


def _api(name: str):
    assert config is not None, "suspension_geometry.config is not implemented yet"
    assert hasattr(config, name), f"suspension_geometry.config.{name} is missing"
    return getattr(config, name)


def _attachment(body: str, point: list[float]) -> dict:
    return {"body": body, "point": point}


def _corner(x: float, side: float, *, rocker: bool = False) -> dict:
    result = {
        "wheel_center": [x, side * 0.75, 0.32],
        "spindle_axis": [0.0, 1.0, 0.0],
        "radius": 0.32,
        "lower": {
            "inboard_rearward": [x - 0.22, side * 0.35, 0.20],
            "inboard_forward": [x + 0.22, side * 0.35, 0.20],
            "lower_ball_joint": [x, side * 0.75, 0.20],
        },
        "upper": {
            "inboard_rearward": [x - 0.20, side * 0.40, 0.49],
            "inboard_forward": [x + 0.20, side * 0.40, 0.49],
            "upper_ball_joint": [x, side * 0.75, 0.49],
        },
        "tie": {
            "inboard": [x - 0.48, side * 0.35, 0.45],
            "outboard": [x - 0.28, side * 0.70, 0.43],
        },
        "jounce_limits": [-0.08, 0.08],
        "spring": {
            "type": "direct",
            "fixed": [x, side * 0.30, 0.70],
            "moving": _attachment("lower", [x, side * 0.71, 0.20]),
            "length_limits": [0.35, 0.65],
        },
    }
    if rocker:
        result["spring"] = {
            "type": "rocker",
            "fixed": [x + 0.30, side * 0.38, 0.20],
            "moving": _attachment("rocker", [x + 0.30, side * 0.56, 0.61]),
            "length_limits": [0.35, 0.70],
        }
        result["rocker"] = {
            "pivot": [x + 0.25, side * 0.40, 0.55],
            "axis": [1.0, 0.0, 0.0],
            "rod_point": [x + 0.12, side * 0.31, 0.51],
            "rod_mount": _attachment("lower", [x + 0.12, side * 0.68, 0.22]),
            "angle_limits": [-0.5, 0.5],
        }
    return result


def _geometry(*, units: dict | None = None, mirrored: bool = False, rocker: bool = False) -> dict:
    corners = {
        "FL": _corner(1.3, 1.0, rocker=rocker),
        "FR": _corner(1.3, -1.0, rocker=rocker),
        "RL": _corner(-1.2, 1.0, rocker=rocker),
        "RR": _corner(-1.2, -1.0, rocker=rocker),
    }
    value = {
        "schema_version": "1.0.0",
        "id": "test-geometry",
        "synthetic": True,
        "units": units or {"length": "m", "angle": "rad"},
        "reference_origin_world": [0.0, 0.0, 0.0],
        "corners": {"FL": corners["FL"], "RL": corners["RL"]} if mirrored else corners,
    }
    if mirrored:
        value["mirror_right"] = True
    return value


def _setup(*, units: dict | None = None) -> dict:
    return {
        "schema_version": "1.0.0",
        "id": "test-setup",
        "units": units or {"length": "m", "force": "N", "angle": "rad"},
        "corners": {
            name: {
                "spring": {"rate": 30_000.0, "preload_force": 400.0},
                "tyre": {"rate": 180_000.0, "reference_force": 2_800.0},
            }
            for name in ("FL", "FR", "RL", "RR")
        },
    }


def _write_yaml(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _scale_geometry_lengths(value: dict, scale: float) -> dict:
    """Express the test geometry in another length unit without changing directions."""
    result = yaml.safe_load(yaml.safe_dump(value))

    def visit(item, key=None):
        if isinstance(item, dict):
            return {k: visit(v, k) for k, v in item.items()}
        if isinstance(item, list):
            if key in {
                "reference_origin_world", "wheel_center", "point", "fixed", "pivot", "tip",
                "rod_point", "inboard_forward", "inboard_rearward", "upper_ball_joint",
                "lower_ball_joint", "outboard", "inboard",
            }:
                return [x * scale for x in item]
            if key in {"jounce_limits", "length_limits"}:
                return [x * scale for x in item]
            return [visit(x) for x in item]
        if isinstance(item, (int, float)) and key == "radius":
            return item * scale
        return item

    return visit(result)


def _study(*, axes: dict | None = None) -> dict:
    return {
        "schema_version": "1.0.0",
        "id": "test-study",
        "geometry": "geometry.yaml",
        "setup": "setup.yaml",
        "units": {"length": "m", "angle": "rad"},
        "reference": [0.0, 0.0, 0.0],
        "axes": axes or {"heave": [0.0, 0.01, 0.03], "roll": [0.0], "pitch": [-0.01, 0.01]},
        "metrics": ["spring_motion_ratio", "body_stiffness", "heave_stiffness"],
        "plots": ["spring_motion_ratio"],
    }


def test_geometry_length_units_convert_mm_to_si(tmp_path):
    metric = _geometry()
    millimetres = _scale_geometry_lengths(metric, 1000.0)
    millimetres["units"] = {"length": "mm", "angle": "rad"}

    metric_path = _write_yaml(tmp_path, "metric.yaml", metric)
    mm_path = _write_yaml(tmp_path, "millimetres.yaml", millimetres)
    metric_result = _api("load_geometry")(metric_path)
    mm_result = _api("load_geometry")(mm_path)

    assert mm_result["corners"]["FL"]["wheel_center"] == pytest.approx(
        metric_result["corners"]["FL"]["wheel_center"]
    )
    assert mm_result["corners"]["FL"]["radius"] == pytest.approx(0.32)
    assert mm_result["corners"]["FL"]["spring"]["length_limits"] == pytest.approx([0.35, 0.65])


def test_geometry_angle_units_convert_degrees_to_radians(tmp_path):
    radians = _geometry(mirrored=True, rocker=True)
    degrees = _geometry(units={"length": "m", "angle": "deg"}, mirrored=True, rocker=True)
    for item in radians["corners"].values():
        item["rocker"]["angle_limits"] = [-math.pi / 6.0, math.pi / 6.0]
    for item in degrees["corners"].values():
        item["rocker"]["angle_limits"] = [-30.0, 30.0]

    rad_path = _write_yaml(tmp_path, "radians.yaml", radians)
    deg_path = _write_yaml(tmp_path, "degrees.yaml", degrees)
    rad_result = _api("load_geometry")(rad_path)
    deg_result = _api("load_geometry")(deg_path)

    assert deg_result["corners"]["FL"]["rocker"]["angle_limits"] == pytest.approx(
        rad_result["corners"]["FL"]["rocker"]["angle_limits"]
    )


def test_explicit_four_corners_preserve_asymmetric_geometry(tmp_path):
    raw = _geometry()
    raw["corners"]["FR"]["wheel_center"][0] += 0.017
    raw["corners"]["RR"]["upper"]["upper_ball_joint"][2] += 0.009
    loaded = _api("load_geometry")(_write_yaml(tmp_path, "asymmetric.yaml", raw))

    assert loaded["corners"]["FR"]["wheel_center"][0] == pytest.approx(1.317)
    assert loaded["corners"]["RR"]["upper"]["upper_ball_joint"][2] == pytest.approx(0.499)
    assert loaded["corners"]["FL"]["wheel_center"][0] == pytest.approx(1.3)
    assert loaded["corners"]["RL"]["upper"]["inboard_rearward"] == pytest.approx(
        [-1.4, 0.4, 0.49]
    )
    assert loaded["corners"]["RR"]["upper"]["inboard_forward"] == pytest.approx(
        [-1.0, -0.4, 0.49]
    )


def test_mirror_reflects_points_and_axial_axes_physically(tmp_path):
    loaded = _api("load_geometry")(_write_yaml(tmp_path, "mirrored.yaml", _geometry(mirrored=True, rocker=True)))
    left = loaded["corners"]["FL"]
    right = loaded["corners"]["FR"]

    assert right["wheel_center"] == pytest.approx([1.3, -0.75, 0.32])
    assert right["lower"]["inboard_rearward"] == pytest.approx([1.08, -0.35, 0.20])
    assert right["lower"]["lower_ball_joint"] == pytest.approx([1.3, -0.75, 0.20])
    assert right["spring"]["moving"]["point"] == pytest.approx([1.6, -0.56, 0.61])
    assert right["rocker"]["axis"] == pytest.approx([-1.0, 0.0, 0.0])
    assert right["spindle_axis"] == pytest.approx([0.0, 1.0, 0.0])
    assert left["rocker"]["axis"] == pytest.approx([1.0, 0.0, 0.0])


def test_wishbone_pivot_names_and_zero_length_error_are_actionable(tmp_path):
    raw = _geometry()
    legacy = copy.deepcopy(raw)
    for corner in legacy["corners"].values():
        for name in ("lower", "upper"):
            arm = corner[name]
            arm["inboard_a"] = arm.pop("inboard_rearward")
            arm["inboard_b"] = arm.pop("inboard_forward")

    with pytest.raises(ValueError, match=r"FL\.lower\.inboard_a.*inboard_forward.*inboard_rearward"):
        _api("load_geometry")(_write_yaml(tmp_path, "legacy-pivots.yaml", legacy))

    raw = _geometry()
    raw["corners"]["FL"]["lower"]["inboard_forward"] = raw["corners"]["FL"]["lower"]["inboard_rearward"]

    with pytest.raises(ValueError, match=r"FL.*lower\.inboard"):
        _api("load_geometry")(_write_yaml(tmp_path, "bad-pivot.yaml", raw))


def test_missing_toe_link_identifies_corner_and_field(tmp_path):
    raw = _geometry()
    del raw["corners"]["RR"]["tie"]

    with pytest.raises(ValueError, match=r"RR.*tie"):
        _api("load_geometry")(_write_yaml(tmp_path, "missing-tie.yaml", raw))


def test_nonfinite_geometry_is_rejected_with_field_path(tmp_path):
    raw = _geometry()
    raw["corners"]["RL"]["wheel_center"][2] = float("nan")

    with pytest.raises(ValueError, match=r"RL.*wheel_center"):
        _api("load_geometry")(_write_yaml(tmp_path, "nan.yaml", raw))


def test_setup_rate_and_preload_are_converted_from_n_per_mm(tmp_path):
    raw = _setup(units={"length": "mm", "force": "N", "angle": "deg"})
    raw["corners"]["FL"]["spring"]["rate"] = 30.0
    path = _write_yaml(tmp_path, "setup-n-per-mm.yaml", raw)
    loaded = _api("load_setup")(path)

    assert loaded["corners"]["FL"]["spring"]["rate"] == pytest.approx(30_000.0)
    assert loaded["corners"]["FL"]["spring"]["preload_force"] == pytest.approx(400.0)
    assert loaded["corners"]["FL"]["tyre"]["reference_force"] == pytest.approx(2_800.0)


def test_conflicting_spring_preload_specifications_are_rejected(tmp_path):
    raw = _setup()
    raw["corners"]["FL"]["spring"]["preload_compression"] = 0.01

    with pytest.raises(ValueError, match=r"FL.*spring.*preload"):
        _api("load_setup")(_write_yaml(tmp_path, "conflicting-preload.yaml", raw))


def test_active_setup_arb_requires_matching_geometry(tmp_path):
    geometry_path = _write_yaml(tmp_path, "geometry.yaml", _geometry())
    geometry = _api("load_geometry")(geometry_path)
    raw = _setup()
    raw["arbs"] = {"front": {"rate": 900.0, "reference_twist": 0.0}}

    with pytest.raises(ValueError, match=r"front.*ARB.*geometry"):
        _api("load_setup")(_write_yaml(tmp_path, "active-arb.yaml", raw), geometry=geometry)


def test_active_setup_arb_is_rejected_when_geometry_is_not_supplied(tmp_path):
    raw = _setup()
    raw["arbs"] = {"rear": {"rate": 700.0, "reference_twist": 0.0}}

    with pytest.raises(ValueError, match=r"rear.*ARB.*geometry"):
        _api("load_setup")(_write_yaml(tmp_path, "arb-without-geometry.yaml", raw))


def test_study_accepts_irregular_and_one_point_axes_and_resolves_relative_paths(tmp_path):
    geometry_path = _write_yaml(tmp_path, "geometry.yaml", _geometry())
    setup_path = _write_yaml(tmp_path, "setup.yaml", _setup())
    raw = _study(axes={"heave": [0.0, 0.01, 0.03], "roll": [0.0], "pitch": [0.0, 0.02]})
    study_path = _write_yaml(tmp_path, "study.yaml", raw)

    study = _api("load_study")(study_path)
    geometry, setup, resolved_study = _api("resolve_study")(study_path)

    assert study["axes"]["heave_m"] == pytest.approx([0.0, 0.01, 0.03])
    assert study["axes"]["roll_rad"] == pytest.approx([0.0])
    assert Path(resolved_study["geometry"]) == geometry_path.resolve()
    assert Path(resolved_study["setup"]) == setup_path.resolve()
    assert geometry["id"] == "test-geometry"
    assert setup["id"] == "test-setup"


def test_study_rejects_grid_larger_than_default_sample_limit(tmp_path):
    raw = _study(axes={"heave": [i / 1000.0 for i in range(2001)], "roll": [0.0], "pitch": [0.0]})

    with pytest.raises(ValueError, match=r"max_samples.*2000|2000.*max_samples"):
        _api("load_study")(_write_yaml(tmp_path, "too-many.yaml", raw))


def test_omitted_solver_defaults_remain_canonical_si_across_input_units(tmp_path):
    si_raw = _study()
    mixed_raw = _study()
    mixed_raw["units"] = {"length": "mm", "angle": "deg"}

    si_study = _api("load_study")(_write_yaml(tmp_path, "si-defaults.yaml", si_raw))
    mixed_study = _api("load_study")(_write_yaml(tmp_path, "mixed-defaults.yaml", mixed_raw))

    assert mixed_study["solver"] == si_study["solver"]
    assert mixed_study["solver"]["residual_tolerance"] == pytest.approx(1e-9)
    assert mixed_study["solver"]["derivative_steps"] == pytest.approx([1e-4, 1e-4, 1e-4])
    assert mixed_study["solver"]["jounce_step"] == pytest.approx(1e-4)


def test_study_preserves_metric_names_for_run_layer_validation(tmp_path):
    raw = _study()
    raw["metrics"] = ["wishbone_magic"]

    loaded = _api("load_study")(_write_yaml(tmp_path, "unknown-metric.yaml", raw))

    assert loaded["metrics"] == ["wishbone_magic"]


def test_geometry_rejects_unknown_fields_instead_of_ignoring_typos(tmp_path):
    raw = _geometry()
    raw["corners"]["FL"]["sprng"] = raw["corners"]["FL"].pop("spring")

    with pytest.raises(ValueError, match=r"FL.*sprng"):
        _api("load_geometry")(_write_yaml(tmp_path, "typo.yaml", raw))

    raw = _geometry()
    raw["corners"]["FL"]["upper"]["outboard"] = raw["corners"]["FL"]["upper"].pop("upper_ball_joint")
    with pytest.raises(ValueError, match=r"FL.*upper\.outboard.*upper_ball_joint"):
        _api("load_geometry")(_write_yaml(tmp_path, "legacy-ball-joint.yaml", raw))


def test_rocker_attachment_is_allowed_only_on_rocker_corner(tmp_path):
    raw = _geometry()
    raw["corners"]["FL"]["spring"]["moving"] = _attachment("rocker", [1.6, 0.48, 0.62])

    with pytest.raises(ValueError, match=r"FL.*rocker"):
        _api("load_geometry")(_write_yaml(tmp_path, "orphan-rocker-attachment.yaml", raw))


def test_validate_geometry_rejects_noncanonical_units(tmp_path):
    raw = _geometry(units={"length": "mm", "angle": "deg"})

    with pytest.raises(ValueError, match=r"canonical.*SI|units.*m.*rad"):
        _api("validate_geometry")(raw)


def test_validate_geometry_requires_resolved_damper_geometry(tmp_path):
    geometry = _api("load_geometry")(_write_yaml(tmp_path, "resolved.yaml", _geometry()))
    del geometry["corners"]["FL"]["damper"]

    with pytest.raises(ValueError, match=r"FL.*damper"):
        _api("validate_geometry")(geometry)


def test_validate_geometry_rejects_unnormalized_directions_without_mutating_input(tmp_path):
    raw = _geometry(mirrored=True, rocker=True)
    raw["arbs"] = {
        "front": {
            "left": {
                "pivot": [1.2, 0.42, 0.43],
                "axis": [1.0, 0.0, 0.0],
                "tip": [1.2, 0.55, 0.43],
                "pickup": _attachment("lower", [1.3, 0.65, 0.20]),
                "angle_limits": [-0.4, 0.4],
            }
        }
    }
    geometry = _api("load_geometry")(_write_yaml(tmp_path, "directions.yaml", raw))
    direction_locations = (
        (("corners", "FL", "spindle_axis"), [0.0, 2.0, 0.0]),
        (("corners", "FL", "rocker", "axis"), [2.0, 0.0, 0.0]),
        (("arbs", "front", "left", "axis"), [2.0, 0.0, 0.0]),
    )

    for path, unnormalized in direction_locations:
        candidate = copy.deepcopy(geometry)
        target = candidate
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = unnormalized
        before = copy.deepcopy(candidate)
        with pytest.raises(ValueError, match=r"normalized|unit direction"):
            _api("validate_geometry")(candidate)
        assert candidate == before


def test_oversized_generated_axis_is_rejected_before_axis_materialization(tmp_path, monkeypatch):
    raw = _study(
        axes={
            "heave": {"min": 0.0, "max": 1.0, "count": 10**12},
            "roll": [0.0],
            "pitch": [0.0],
        }
    )
    study_path = _write_yaml(tmp_path, "enormous-axis.yaml", raw)
    materialize_called = False

    def forbidden_materialization(*args, **kwargs):
        nonlocal materialize_called
        materialize_called = True
        raise AssertionError("generated axis materialization ran before the sample cap")

    monkeypatch.setattr(config, "_axis", forbidden_materialization)
    with pytest.raises(ValueError, match=r"max_samples"):
        _api("load_study")(study_path)
    assert not materialize_called


def test_bundled_synthetic_studies_resolve_to_valid_inputs():
    root = Path(__file__).resolve().parents[2]
    config_root = root / "configs" / "suspension_geometry"
    resolve_study = _api("resolve_study")

    for study_name in ("direct_heave.yaml", "rocker_heave.yaml", "synthetic_3d.yaml"):
        geometry, setup, study = resolve_study(config_root / "studies" / study_name)
        assert geometry["synthetic"] is True
        assert geometry["units"] == {"length": "m", "angle": "rad"}
        assert setup["units"] == {"length": "m", "force": "N", "angle": "rad"}
        assert all(
            setup["corners"][name]["tyre"]["reference_force"] > 0.0
            for name in ("FL", "FR", "RL", "RR")
        )
        assert math.dist(geometry["corners"]["FL"]["tie"]["inboard"], geometry["corners"]["FL"]["tie"]["outboard"]) > 0.0

    _, _, grid_study = resolve_study(config_root / "studies" / "synthetic_3d.yaml")
    assert [len(grid_study["axes"][axis]) for axis in ("heave_m", "roll_rad", "pitch_rad")] == [7, 5, 3]
