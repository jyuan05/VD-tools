"""Finite body pose, wheel composition, and generalized-rate contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from suspension_geometry.config import load_geometry, load_setup
from suspension_geometry.elasticity import wheel_energy
from suspension_geometry.kinematics import CornerState


ROOT = Path(__file__).resolve().parents[2]
DIRECT_GEOMETRY = ROOT / "configs/suspension_geometry/geometries/synthetic_direct.yaml"
DIRECT_SETUP = ROOT / "configs/suspension_geometry/setups/synthetic_direct_loaded.yaml"


def _api(module_name: str, name: str):
    import importlib

    module = importlib.import_module(f"suspension_geometry.{module_name}")
    assert hasattr(module, name), f"suspension_geometry.{module_name}.{name} is missing"
    return getattr(module, name)


def _geometry():
    geometry = load_geometry(DIRECT_GEOMETRY)
    for corner in geometry["corners"].values():
        corner["spring"]["length_limits"] = [0.2, 1.2]
        corner["damper"]["length_limits"] = [0.2, 1.2]
    return geometry


def _setup():
    return load_setup(DIRECT_SETUP, geometry=_geometry())


def test_body_rotation_is_finite_ry_then_rx_with_declared_signs():
    body_rotation = _api("vehicle", "body_rotation")

    q = np.array([0.0, 0.17, -0.23])
    rotation = body_rotation(q)
    expected = np.array(
        [
            [np.cos(-0.23), np.sin(-0.23) * np.sin(0.17), np.sin(-0.23) * np.cos(0.17)],
            [0.0, np.cos(0.17), -np.sin(0.17)],
            [-np.sin(-0.23), np.cos(-0.23) * np.sin(0.17), np.cos(-0.23) * np.cos(0.17)],
        ]
    )
    np.testing.assert_allclose(rotation, expected, atol=1e-13)


def test_pose_jounces_keeps_world_hub_heights_and_has_small_pose_signs():
    pose_jounces = _api("vehicle", "pose_jounces")
    geometry = _geometry()

    q = np.array([0.004, 0.003, -0.002])
    result = pose_jounces(geometry, q)

    assert result["valid"], result["reason"]
    np.testing.assert_allclose(result["world_wheel_centers"][:, 2], [0.32] * 4, atol=2e-9)
    # j = h - y*phi + x*theta, evaluated with the nominal x/y hard points.
    expected = np.array(
        [
            0.004 - 0.75 * 0.003 + 1.3 * -0.002,
            0.004 + 0.75 * 0.003 + 1.3 * -0.002,
            0.004 - 0.75 * 0.003 - 1.2 * -0.002,
            0.004 + 0.75 * 0.003 - 1.2 * -0.002,
        ]
    )
    np.testing.assert_allclose(result["jounce"], expected, atol=5e-5)


def test_pose_metrics_retain_preload_geometric_curvature_and_arb_coupling():
    evaluate_pose = _api("vehicle", "evaluate_pose")
    geometry = _geometry()
    setup = _setup()
    result = evaluate_pose(geometry, setup, np.zeros(3), solver={"derivative_steps": [2e-4] * 3})

    assert result["valid"], result["reason"]
    body_stiffness = np.asarray(result["metrics"]["body_stiffness"])
    wheel_stiffness = np.asarray(result["metrics"]["wheel_stiffness"])
    assert body_stiffness.shape == (3, 3)
    assert wheel_stiffness.shape == (4, 4)
    np.testing.assert_allclose(body_stiffness, body_stiffness.T, atol=2e-5)
    assert result["metrics"]["body_gradient"].shape == (3,)
    assert result["metrics"]["body_restoring_reaction"].shape == (3,)
    assert result["validity"]["body_stiffness"].shape == (3, 3)
    assert result["validity"]["body_stiffness"].all()
    assert result["reasons"]["body_stiffness"].shape == (3, 3)


def test_geometry_metric_masks_keep_good_corners_when_one_corner_is_out_of_range():
    evaluate_pose = _api("vehicle", "evaluate_pose")
    geometry = _geometry()
    geometry["corners"]["FL"]["jounce_limits"] = [-1e-6, 1e-6]
    result = evaluate_pose(geometry, _setup(), np.array([0.02, 0.0, 0.0]))

    assert not result["valid"]
    assert not result["validity"]["jounce"][0]
    assert result["validity"]["jounce"][1:].all()
    assert np.isfinite(result["metrics"]["camber_chassis"][1:]).all()


def test_metric_definitions_record_mixed_body_units_and_tail_labels():
    definitions = _api("metrics", "metric_definitions")()

    body = definitions["body_stiffness"]
    assert body["tail_shape"] == (3, 3)
    assert body["tail_labels"] == ("q", "q")
    assert body["entry_units"][0][0] == "N/m"
    assert body["entry_units"][1][1] == "N*m/rad"
    assert body["entry_units"][0][1] == "N/rad"
    gains = definitions["spring_actuation_gains"]
    assert gains["tail_shape"] == (4, 3)
    assert gains["entry_units"][0] == ("m/m", "m/rad", "m/rad")


def test_reference_origin_translates_world_points_without_changing_local_pose_or_rates():
    evaluate_pose = _api("vehicle", "evaluate_pose")
    base_geometry = _geometry()
    shifted_geometry = copy.deepcopy(base_geometry)
    shifted_geometry["reference_origin_world"] = [7.0, -3.0, 5.0]
    q = np.array([0.002, 0.004, -0.003])
    base = evaluate_pose(base_geometry, _setup(), q)
    shifted = evaluate_pose(shifted_geometry, _setup(), q)

    np.testing.assert_allclose(base["jounce"], shifted["jounce"], atol=2e-9)
    np.testing.assert_allclose(base["metrics"]["body_stiffness"], shifted["metrics"]["body_stiffness"], atol=2e-3)
    np.testing.assert_allclose(base["metrics"]["camber_chassis"], shifted["metrics"]["camber_chassis"], atol=2e-9)
    np.testing.assert_allclose(
        shifted["pose"]["world_wheel_centers"] - base["pose"]["world_wheel_centers"],
        np.tile(np.array([7.0, -3.0, 5.0]), (4, 1)),
        atol=2e-9,
    )


def test_loaded_nonzero_body_hessian_matches_independent_energy_perturbations():
    evaluate_pose = _api("vehicle", "evaluate_pose")
    pose_jounces = _api("vehicle", "pose_jounces")
    geometry = _geometry()
    setup = _setup()
    q = np.array([0.001, 0.007, -0.004])
    step = 4e-4
    result = evaluate_pose(geometry, setup, q, solver={"derivative_steps": [step] * 3})
    assert result["valid"], result["reason"]

    def energy(point):
        pose = pose_jounces(geometry, point)
        assert pose["valid"], pose["reason"]
        energy_result = wheel_energy(geometry, setup, pose["jounce"])
        assert energy_result["valid"], energy_result["reason"]
        return energy_result["energy"]

    expected = np.full((3, 3), np.nan)
    for first in range(3):
        for second in range(3):
            e_first = np.zeros(3)
            e_second = np.zeros(3)
            e_first[first] = step
            e_second[second] = step
            expected[first, second] = (
                energy(q + e_first + e_second)
                - energy(q + e_first - e_second)
                - energy(q - e_first + e_second)
                + energy(q - e_first - e_second)
            ) / (4.0 * step**2)
    np.testing.assert_allclose(result["metrics"]["body_stiffness"], expected, rtol=3e-3, atol=1.0)


def test_straight_wheel_path_retains_preload_pose_curvature(monkeypatch):
    """The exact fixed-height root keeps the F*jounce curvature term.

    This fixture has a straight local wheel path and a prescribed linear
    spring force.  At zero pose the analytic root has
    ``j_phi_phi=j_theta_theta=z0``; accepting the linearized pose estimate as
    an exact root would incorrectly return zero for both terms.
    """
    vehicle = __import__("suspension_geometry.vehicle", fromlist=["vehicle"])
    corners = {}
    for name, x, y, z0 in (
        ("FL", 1.3, 0.75, 0.32),
        ("FR", 1.3, -0.75, 0.32),
        ("RL", -1.2, 0.75, 0.30),
        ("RR", -1.2, -0.75, 0.30),
    ):
        corners[name] = {
            "wheel_center": [x, y, z0],
            "jounce_limits": [-0.2, 0.2],
        }
    geometry = {"reference_origin_world": [0.0, 0.0, 0.0], "corners": corners}

    def straight_solver(corner, jounce, *, max_nfev, tolerance):
        center = np.asarray(corner["wheel_center"], dtype=float).copy()
        center[2] += float(jounce)
        return CornerState(
            valid=True,
            reason="ok",
            jounce=float(jounce),
            wheel_center=center,
            rotation=np.eye(3),
            upper_ball_joint=center + [0.0, 0.0, 0.1],
            lower_ball_joint=center - [0.0, 0.0, 0.1],
            tie_outboard=center.copy(),
            residual=0.0,
            condition=1.0,
            solution=np.zeros(6),
        )

    monkeypatch.setattr(vehicle, "solve_corner", straight_solver)
    pose_map_derivatives = vehicle._pose_map_derivatives
    derivatives = pose_map_derivatives(
        geometry,
        np.zeros(3),
        np.zeros(4),
        np.full(3, 2e-4),
        max_nfev=20,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(derivatives["jacobian"][:, 1], [-0.75, 0.75, -0.75, 0.75], atol=2e-5)
    np.testing.assert_allclose(derivatives["jacobian"][:, 2], [1.3, 1.3, -1.2, -1.2], atol=2e-5)
    np.testing.assert_allclose(derivatives["hessian"][:, 1, 1], [0.32, 0.32, 0.30, 0.30], atol=3e-4)
    np.testing.assert_allclose(derivatives["hessian"][:, 2, 2], [0.32, 0.32, 0.30, 0.30], atol=3e-4)
