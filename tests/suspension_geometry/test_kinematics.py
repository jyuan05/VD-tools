"""Independent closure and continuation checks for rigid corner kinematics."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest

from suspension_geometry.config import load_geometry


try:
    kinematics = importlib.import_module("suspension_geometry.kinematics")
except ModuleNotFoundError as error:
    if error.name not in {"suspension_geometry.kinematics"}:
        raise
    kinematics = None


ROOT = Path(__file__).resolve().parents[2]
DIRECT_GEOMETRY = ROOT / "configs/suspension_geometry/geometries/synthetic_direct.yaml"


def _api(name: str):
    assert kinematics is not None, "suspension_geometry.kinematics is not implemented yet"
    assert hasattr(kinematics, name), f"suspension_geometry.kinematics.{name} is missing"
    return getattr(kinematics, name)


def _geometry() -> dict:
    return load_geometry(DIRECT_GEOMETRY)


def _scale_corner(corner: dict, scale: float) -> dict:
    result = copy.deepcopy(corner)
    for field in ("wheel_center",):
        result[field] = (np.asarray(result[field], dtype=float) * scale).tolist()
    result["radius"] *= scale
    result["jounce_limits"] = (np.asarray(result["jounce_limits"], dtype=float) * scale).tolist()
    for arm_name in ("lower", "upper"):
        for field in ("inboard_rearward", "inboard_forward", f"{arm_name}_ball_joint"):
            result[arm_name][field] = (np.asarray(result[arm_name][field], dtype=float) * scale).tolist()
    for field in ("inboard", "outboard"):
        result["tie"][field] = (np.asarray(result["tie"][field], dtype=float) * scale).tolist()
    for component_name in ("spring", "damper"):
        component = result[component_name]
        component["fixed"] = (np.asarray(component["fixed"], dtype=float) * scale).tolist()
        component["moving"]["point"] = (np.asarray(component["moving"]["point"], dtype=float) * scale).tolist()
        component["length_limits"] = (np.asarray(component["length_limits"], dtype=float) * scale).tolist()
    return result


def _point_map(corner: dict, state) -> dict[str, np.ndarray]:
    center = np.asarray(corner["wheel_center"], dtype=float)
    return {
        "wheel_center": np.asarray(state.wheel_center, dtype=float),
        "lower": np.asarray(state.lower_ball_joint, dtype=float),
        "upper": np.asarray(state.upper_ball_joint, dtype=float),
        "tie": np.asarray(state.tie_outboard, dtype=float),
    }


def _assert_nominal_link_lengths(corner: dict, state) -> None:
    for name in ("lower", "upper"):
        arm = corner[name]
        point = np.asarray(getattr(state, f"{name}_ball_joint"), dtype=float)
        for pivot in ("inboard_rearward", "inboard_forward"):
            expected = np.linalg.norm(np.asarray(arm[f"{name}_ball_joint"]) - arm[pivot])
            actual = np.linalg.norm(point - np.asarray(arm[pivot]))
            assert actual == pytest.approx(expected, abs=2e-9)
    tie = corner["tie"]
    tie_length = np.linalg.norm(np.asarray(tie["outboard"]) - tie["inboard"])
    actual_tie_length = np.linalg.norm(np.asarray(state.tie_outboard) - tie["inboard"])
    assert actual_tie_length == pytest.approx(tie_length, abs=2e-9)


def test_zero_pose_is_exact_reference_and_closes_every_link():
    solve_corner = _api("solve_corner")
    corner = _geometry()["corners"]["FL"]

    state = solve_corner(corner, 0.0)

    assert state.valid
    assert state.reason == "ok"
    assert state.solution == pytest.approx(np.zeros(6), abs=1e-12)
    assert state.rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert state.wheel_center == pytest.approx(corner["wheel_center"], abs=1e-12)
    assert state.residual <= 1e-9
    assert np.isfinite(state.condition) and state.condition < 1e10
    _assert_nominal_link_lengths(corner, state)


def test_constraint_condition_is_invariant_to_uniform_geometry_scale():
    solve_corner = _api("solve_corner")
    base_corner = _geometry()["corners"]["FL"]

    states = [solve_corner(_scale_corner(base_corner, scale), 0.0) for scale in (0.01, 1.0, 100.0)]

    assert all(state.valid for state in states)
    conditions = np.asarray([state.condition for state in states])
    assert np.all(np.isfinite(conditions))
    assert np.max(conditions) / np.min(conditions) < 1.01


@pytest.mark.parametrize("jounce", [-0.06, -0.03, 0.03, 0.06])
def test_solved_pose_preserves_all_upright_distances_and_sets_wheel_height(jounce):
    solve_corner = _api("solve_corner")
    corner = _geometry()["corners"]["FL"]

    state = solve_corner(corner, jounce)

    assert state.valid, state.reason
    assert state.wheel_center[2] == pytest.approx(corner["wheel_center"][2] + jounce, abs=1e-9)
    _assert_nominal_link_lengths(corner, state)
    nominal = {
        "wheel_center": np.asarray(corner["wheel_center"], dtype=float),
        "lower": np.asarray(corner["lower"]["lower_ball_joint"], dtype=float),
        "upper": np.asarray(corner["upper"]["upper_ball_joint"], dtype=float),
        "tie": np.asarray(corner["tie"]["outboard"], dtype=float),
    }
    actual = _point_map(corner, state)
    for first in nominal:
        for second in nominal:
            want = np.linalg.norm(nominal[first] - nominal[second])
            got = np.linalg.norm(actual[first] - actual[second])
            assert got == pytest.approx(want, abs=2e-9)


def test_corner_sweeps_continue_outward_and_preserve_caller_order():
    solve_corner_sweep = _api("solve_corner_sweep")
    corner = _geometry()["corners"]["FL"]
    requested = np.asarray([-0.06, -0.03, 0.0, 0.03, 0.06])

    forward = solve_corner_sweep(corner, requested)
    repeated = solve_corner_sweep(corner, requested)
    reverse = solve_corner_sweep(corner, requested[::-1])

    assert [state.jounce for state in forward] == pytest.approx(requested)
    assert all(state.valid for state in forward)
    for first, second in zip(forward, repeated, strict=True):
        assert first.solution == pytest.approx(second.solution, abs=1e-10)
    for index, state in enumerate(forward):
        assert state.solution == pytest.approx(reverse[-index - 1].solution, abs=2e-8)


def test_mirrored_corner_solutions_are_reflections_with_axial_rotation_signs():
    solve_corner = _api("solve_corner")
    geometry = _geometry()
    left = geometry["corners"]["FL"]
    right = geometry["corners"]["FR"]
    reflection = np.diag([1.0, -1.0, 1.0])

    for jounce in (-0.05, 0.0, 0.05):
        left_state = solve_corner(left, jounce)
        right_state = solve_corner(right, jounce)
        assert left_state.valid and right_state.valid
        assert right_state.wheel_center == pytest.approx(reflection @ left_state.wheel_center, abs=2e-8)
        assert right_state.rotation == pytest.approx(
            reflection @ left_state.rotation @ reflection, abs=2e-8
        )
        assert right_state.lower_ball_joint == pytest.approx(
            reflection @ left_state.lower_ball_joint, abs=2e-8
        )
        assert right_state.upper_ball_joint == pytest.approx(
            reflection @ left_state.upper_ball_joint, abs=2e-8
        )
        assert right_state.tie_outboard == pytest.approx(
            reflection @ left_state.tie_outboard, abs=2e-8
        )


def test_impossible_or_nonfinite_jounce_is_invalid_without_poisoning_neighbors():
    solve_corner_sweep = _api("solve_corner_sweep")
    corner = _geometry()["corners"]["FL"]

    states = solve_corner_sweep(corner, [-0.02, 0.0, 0.09, 0.02, float("nan")])

    assert states[0].valid and states[1].valid and states[3].valid
    assert not states[2].valid
    assert states[2].reason == "jounce_out_of_range"
    assert np.isnan(states[2].wheel_center).all()
    assert not states[4].valid
    assert states[4].reason == "nonfinite_jounce"


def test_geometrically_impossible_in_range_jounce_keeps_neighbor_solutions_valid():
    solve_corner_sweep = _api("solve_corner_sweep")
    corner = _geometry()["corners"]["FL"]
    corner["jounce_limits"] = [-0.7, 0.7]

    states = solve_corner_sweep(corner, [0.0, 0.6, 0.02])

    assert states[0].valid and states[2].valid
    assert not states[1].valid
    assert states[1].reason == "no_convergence"
    assert states[1].residual > 1e-3
    assert np.isfinite(states[1].condition)
    assert np.isnan(states[1].wheel_center).all()
