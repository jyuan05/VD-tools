"""Closure checks for spring, rocker and anti-roll-bar actuation geometry."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from suspension_geometry.config import load_geometry


try:
    kinematics = importlib.import_module("suspension_geometry.kinematics")
except ModuleNotFoundError as error:
    if error.name != "suspension_geometry.kinematics":
        raise
    kinematics = None

try:
    actuation = importlib.import_module("suspension_geometry.actuation")
except ModuleNotFoundError as error:
    if error.name != "suspension_geometry.actuation":
        raise
    actuation = None


ROOT = Path(__file__).resolve().parents[2]
DIRECT_GEOMETRY = ROOT / "configs/suspension_geometry/geometries/synthetic_direct.yaml"
ROCKER_GEOMETRY = ROOT / "configs/suspension_geometry/geometries/synthetic_rocker.yaml"


def _api(module, module_name: str, name: str):
    assert module is not None, f"suspension_geometry.{module_name} is not implemented yet"
    assert hasattr(module, name), f"suspension_geometry.{module_name}.{name} is missing"
    return getattr(module, name)


def _geometry(path: Path) -> dict:
    return load_geometry(path)


def _rotate_about(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return origin + Rotation.from_rotvec(axis * angle).apply(point - origin)


def _arm_angle(arm: dict, actual_outboard: np.ndarray) -> float:
    first = np.asarray(arm["inboard_rearward"], dtype=float)
    axis = np.asarray(arm["inboard_forward"], dtype=float) - first
    axis /= np.linalg.norm(axis)
    reference = np.asarray(arm["outboard"], dtype=float)
    r0 = reference - first - axis * np.dot(reference - first, axis)
    r1 = actual_outboard - first - axis * np.dot(actual_outboard - first, axis)
    return float(np.arctan2(np.dot(axis, np.cross(r0, r1)), np.dot(r0, r1)))


def _expected_lower_point(corner: dict, state, point) -> np.ndarray:
    angle = _arm_angle(corner["lower"], np.asarray(state.lower_outboard, dtype=float))
    arm = corner["lower"]
    axis = np.asarray(arm["inboard_forward"], dtype=float) - arm["inboard_rearward"]
    axis /= np.linalg.norm(axis)
    return _rotate_about(
        np.asarray(point, dtype=float), np.asarray(arm["inboard_rearward"], dtype=float), axis, angle
    )


def test_direct_spring_compression_comes_from_full_endpoint_geometry():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    actuation_state = _api(actuation, "actuation", "actuation_state")
    corner = _geometry(DIRECT_GEOMETRY)["corners"]["FL"]
    state = solve_corner(corner, 0.05)
    assert state.valid

    result = actuation_state(corner, state)

    spring = corner["spring"]
    moving = _expected_lower_point(corner, state, spring["moving"]["point"])
    expected_length = np.linalg.norm(moving - np.asarray(spring["fixed"], dtype=float))
    reference_length = np.linalg.norm(
        np.asarray(spring["moving"]["point"], dtype=float) - np.asarray(spring["fixed"], dtype=float)
    )
    assert result["valid"]
    assert result["spring_length"] == pytest.approx(expected_length, abs=2e-9)
    assert result["spring_compression"] == pytest.approx(reference_length - expected_length, abs=2e-9)
    assert result["spring_compression"] > 0.0
    assert result["damper_length"] == pytest.approx(result["spring_length"], abs=1e-12)
    assert len(result["spring_stroke_margin"]) == 2
    assert result["spring_stroke_margin"][0] >= 0.0
    assert len(result["damper_stroke_margin"]) == 2


def test_attachment_positions_follow_upright_hinge_and_chassis_bodies():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    attachment_position = _api(actuation, "actuation", "attachment_position")
    corner = _geometry(DIRECT_GEOMETRY)["corners"]["FL"]
    state = solve_corner(corner, 0.04)
    assert state.valid
    nominal_center = np.asarray(corner["wheel_center"], dtype=float)
    upright_point = nominal_center + np.asarray([0.04, -0.03, 0.11])

    assert attachment_position(corner, state, {"body": "upright", "point": upright_point.tolist()}) == pytest.approx(
        state.wheel_center + state.rotation @ (upright_point - nominal_center), abs=1e-10
    )
    assert attachment_position(corner, state, {"body": "chassis", "point": [0.2, 0.3, 0.4]}) == pytest.approx(
        [0.2, 0.3, 0.4], abs=1e-12
    )


def test_rocker_closes_rod_and_has_usable_continuous_bump_travel():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    solve_corner_sweep = _api(kinematics, "kinematics", "solve_corner_sweep")
    actuation_state = _api(actuation, "actuation", "actuation_state")
    attachment_position = _api(actuation, "actuation", "attachment_position")
    corner = _geometry(ROCKER_GEOMETRY)["corners"]["FL"]
    jounces = [-0.04, -0.02, 0.0, 0.02, 0.04]
    states = solve_corner_sweep(corner, jounces)

    assert all(state.valid for state in states)
    values = [actuation_state(corner, state) for state in states]
    assert all(value["valid"] for value in values)
    angles = np.asarray([value["rocker_angle"] for value in values])
    assert np.all(np.abs(np.diff(angles)) < 0.20)
    assert angles[2] == pytest.approx(0.0, abs=2e-8)
    bump = values[-1]
    assert bump["spring_compression"] > 0.0
    rocker = corner["rocker"]
    moving_rod = attachment_position(corner, states[-1], rocker["rod_mount"])
    rotated_rod_point = _rotate_about(
        np.asarray(rocker["rod_point"], dtype=float),
        np.asarray(rocker["pivot"], dtype=float),
        np.asarray(rocker["axis"], dtype=float),
        bump["rocker_angle"],
    )
    reference_rod_length = np.linalg.norm(
        np.asarray(rocker["rod_point"], dtype=float)
        - np.asarray(rocker["rod_mount"]["point"], dtype=float)
    )
    assert np.linalg.norm(rotated_rod_point - moving_rod) == pytest.approx(reference_rod_length, abs=2e-9)
    assert abs(bump["rod_residual"]) <= 2e-9
    spring = corner["spring"]
    spring_moving = _rotate_about(
        np.asarray(spring["moving"]["point"], dtype=float),
        np.asarray(rocker["pivot"], dtype=float),
        np.asarray(rocker["axis"], dtype=float),
        bump["rocker_angle"],
    )
    expected_spring_length = np.linalg.norm(spring_moving - np.asarray(spring["fixed"], dtype=float))
    reference_spring_length = np.linalg.norm(
        np.asarray(spring["moving"]["point"], dtype=float) - np.asarray(spring["fixed"], dtype=float)
    )
    assert bump["spring_length"] == pytest.approx(expected_spring_length, abs=2e-9)
    assert bump["spring_compression"] == pytest.approx(
        reference_spring_length - expected_spring_length, abs=2e-9
    )
    assert np.linalg.norm(spring_moving - np.asarray(rocker["pivot"])) == pytest.approx(
        np.linalg.norm(np.asarray(spring["moving"]["point"]) - np.asarray(rocker["pivot"])), abs=2e-9
    )


def test_rocker_and_component_limits_invalidate_actuation():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    actuation_state = _api(actuation, "actuation", "actuation_state")
    original = _geometry(ROCKER_GEOMETRY)["corners"]["FL"]
    state = solve_corner(original, 0.04)
    assert state.valid

    rocker_limited = copy.deepcopy(original)
    rocker_limited["rocker"]["angle_limits"] = [-1e-5, 1e-5]
    rocker_result = actuation_state(rocker_limited, state)
    assert not rocker_result["valid"]
    assert rocker_result["reason"] == "rocker_limit"
    assert np.isnan(rocker_result["spring_compression"])

    stroke_limited = copy.deepcopy(original)
    reference_length = np.linalg.norm(
        np.asarray(stroke_limited["spring"]["moving"]["point"])
        - np.asarray(stroke_limited["spring"]["fixed"])
    )
    stroke_limited["spring"]["length_limits"] = [reference_length - 1e-5, reference_length + 0.1]
    stroke_result = actuation_state(stroke_limited, state)
    assert not stroke_result["valid"]
    assert stroke_result["reason"] == "stroke_limit"
    assert np.isnan(stroke_result["spring_length"])


def test_arb_tie_rod_closure_and_signed_mirrored_twist():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    arb_angles = _api(actuation, "actuation", "arb_angles")
    geometry = _geometry(DIRECT_GEOMETRY)

    attachment_position = _api(actuation, "actuation", "attachment_position")
    equal_states = {
        "FL": solve_corner(geometry["corners"]["FL"], 0.035),
        "FR": solve_corner(geometry["corners"]["FR"], 0.035),
        "RL": solve_corner(geometry["corners"]["RL"], 0.0),
        "RR": solve_corner(geometry["corners"]["RR"], 0.0),
    }
    equal = arb_angles(geometry, equal_states)
    assert equal["front"]["valid"]
    assert equal["front"]["left_angle"] == pytest.approx(equal["front"]["right_angle"], abs=2e-8)
    assert equal["front"]["twist"] == pytest.approx(0.0, abs=2e-8)
    for side in ("left", "right"):
        lever = geometry["arbs"]["front"][side]
        axle_states = equal_states
        pickup = attachment_position(
            geometry["corners"]["F" + ("L" if side == "left" else "R")],
            axle_states["F" + ("L" if side == "left" else "R")],
            lever["pickup"],
        )
        axis = np.asarray(lever["axis"], dtype=float)
        raw_angle = equal["front"][f"{side}_angle"]
        rotated_tip = _rotate_about(
            np.asarray(lever["tip"], dtype=float),
            np.asarray(lever["pivot"], dtype=float),
            axis,
            raw_angle,
        )
        reference_length = np.linalg.norm(
            np.asarray(lever["tip"], dtype=float) - np.asarray(lever["pickup"]["point"], dtype=float)
        )
        assert np.linalg.norm(rotated_tip - pickup) == pytest.approx(reference_length, abs=2e-9)

    opposite_states = {
        "FL": solve_corner(geometry["corners"]["FL"], 0.035),
        "FR": solve_corner(geometry["corners"]["FR"], -0.035),
        "RL": equal_states["RL"],
        "RR": equal_states["RR"],
    }
    opposite = arb_angles(geometry, opposite_states)
    assert opposite["front"]["valid"]
    assert abs(opposite["front"]["twist"]) > 1e-3


def test_arb_twist_is_invariant_to_axis_reversal_with_reversed_limits():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    arb_angles = _api(actuation, "actuation", "arb_angles")
    geometry = _geometry(DIRECT_GEOMETRY)
    states = {
        "FL": solve_corner(geometry["corners"]["FL"], 0.035),
        "FR": solve_corner(geometry["corners"]["FR"], -0.035),
        "RL": solve_corner(geometry["corners"]["RL"], 0.0),
        "RR": solve_corner(geometry["corners"]["RR"], 0.0),
    }
    original = arb_angles(geometry, states)["front"]

    reversed_geometry = copy.deepcopy(geometry)
    for side in ("left", "right"):
        lever = reversed_geometry["arbs"]["front"][side]
        lever["axis"] = (-np.asarray(lever["axis"], dtype=float)).tolist()
        low, high = lever["angle_limits"]
        lever["angle_limits"] = [-high, -low]
    reversed_result = arb_angles(reversed_geometry, states)["front"]

    assert original["valid"] and reversed_result["valid"]
    assert reversed_result["left_angle"] == pytest.approx(original["left_angle"], abs=2e-8)
    assert reversed_result["right_angle"] == pytest.approx(original["right_angle"], abs=2e-8)
    assert reversed_result["twist"] == pytest.approx(original["twist"], abs=2e-8)


def test_arb_rejects_lever_axis_outside_shared_shaft_line():
    solve_corner = _api(kinematics, "kinematics", "solve_corner")
    arb_angles = _api(actuation, "actuation", "arb_angles")
    geometry = _geometry(DIRECT_GEOMETRY)
    states = {
        name: solve_corner(geometry["corners"][name], 0.0)
        for name in ("FL", "FR", "RL", "RR")
    }
    geometry["arbs"]["front"]["right"]["axis"] = [1.0, 0.0, 0.0]

    result = arb_angles(geometry, states)["front"]

    assert not result["valid"]
    assert result["reason"] == "bar_axis_invalid"
    assert np.isnan(result["twist"])
