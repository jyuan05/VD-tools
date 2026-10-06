"""Tyre equilibrium and local condensed ride-rate checks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from suspension_geometry.config import load_geometry
from suspension_geometry.elasticity import wheel_response


ROOT = Path(__file__).resolve().parents[2]
DIRECT_GEOMETRY = ROOT / "configs/suspension_geometry/geometries/synthetic_direct.yaml"


def _api(name: str):
    from suspension_geometry import vehicle

    assert hasattr(vehicle, name), f"suspension_geometry.vehicle.{name} is missing"
    return getattr(vehicle, name)


def _geometry():
    geometry = load_geometry(DIRECT_GEOMETRY)
    for corner in geometry["corners"].values():
        corner["spring"]["length_limits"] = [0.2, 1.2]
        corner["damper"]["length_limits"] = [0.2, 1.2]
    return geometry


def _setup(
    rate_spring=12000.0,
    rate_tyre=60000.0,
    reference_force=700.0,
    *,
    spring_preload=700.0,
    front_arb_rate=None,
):
    front_arb = None
    if front_arb_rate is not None:
        front_arb = {"rate": float(front_arb_rate), "reference_twist": 0.0}
    return {
        "corners": {
            name: {
                "spring": {"rate": rate_spring, "preload_force": spring_preload},
                "tyre": {"rate": rate_tyre, "reference_force": reference_force},
            }
            for name in ("FL", "FR", "RL", "RR")
        },
        "arbs": {"front": front_arb, "rear": None},
    }


def test_scalar_tyres_condense_to_series_rate():
    tyre_effective_response = _api("tyre_effective_response")
    result = tyre_effective_response(_geometry(), _setup(), np.zeros(3))

    assert result["valid"], result["reason"]
    # Derive A independently from the accepted elasticity API at the returned
    # equilibrium jounce.  Using the result's own suspension_hub_hessian here
    # would reproduce a wrong condensation implementation by construction.
    accepted = wheel_response(_geometry(), _setup(), result["tyre_equilibrium_jounce"])
    assert accepted["valid"], accepted["reason"]
    suspension_tangent = np.asarray(accepted["stiffness"], dtype=float)
    tyre_tangent = np.full(4, 60000.0)
    expected_matrix = suspension_tangent - suspension_tangent @ np.linalg.solve(
        suspension_tangent + np.diag(tyre_tangent), suspension_tangent
    )
    np.testing.assert_allclose(result["ride_stiffness"], expected_matrix, rtol=3e-3, atol=2e-2)
    np.testing.assert_allclose(result["ride_rate"], np.diag(expected_matrix), rtol=3e-3, atol=2e-2)
    assert np.isfinite(result["tyre_equilibrium_hub_displacement"]).all()
    assert np.asarray(result["tyre_contact_force"]).min() > 0.0


def test_coupled_quadratic_schur_fixture_has_off_diagonal_series_terms():
    """A unit-motion-ratio quadratic energy has the exact coupled Schur form."""
    condense = _api("_condense_internal_hessian")
    # Fixture energy: Us=.5*u.T*A*u + q.T*B*u + .5*q.T*C*q and
    # Ut=.5*u.T*T*u.  The unit-MR boundary map makes these Hessian blocks
    # exact, so no numerical pose interpolation is involved in the oracle.
    A = np.array([[12000.0, 1800.0, 0.0, 0.0], [1800.0, 12500.0, 0.0, 0.0],
                  [0.0, 0.0, 9000.0, -1300.0], [0.0, 0.0, -1300.0, 9400.0]])
    T = np.array([60000.0, 62000.0, 58000.0, 59000.0])
    Hqq = np.diag([2500.0, 1700.0, 2100.0])
    Hqu = np.array([[100.0, 70.0, 50.0, 40.0], [300.0, -300.0, 100.0, -100.0],
                    [200.0, 180.0, -220.0, -200.0]])
    ride, body, total = condense(A, T, Hqq, Hqu)
    expected = A - A @ np.linalg.solve(A + np.diag(T), A)
    expected_body = Hqq - Hqu @ np.linalg.solve(A + np.diag(T), Hqu.T)
    np.testing.assert_allclose(ride, expected, rtol=1e-13, atol=1e-10)
    np.testing.assert_allclose(body, expected_body, rtol=1e-13, atol=1e-10)
    assert ride[0, 1] != pytest.approx(0.0)
    assert body is not None and total.shape == (4, 4)


def test_absent_tyres_are_explicitly_unavailable_nan():
    tyre_effective_response = _api("tyre_effective_response")
    setup = _setup()
    for corner in setup["corners"].values():
        corner.pop("tyre")
    result = tyre_effective_response(_geometry(), setup, np.zeros(3))

    assert not result["valid"]
    assert result["reason"] == "tyres_unavailable"
    assert np.isnan(result["ride_rate"]).all()
    assert not np.asarray(result["validity"]["ride_rate"]).any()


def test_tyre_export_separates_equilibrium_jounce_from_base_pose():
    tyre_effective_response = _api("tyre_effective_response")
    setup = _setup(reference_force=1600.0)
    result = tyre_effective_response(_geometry(), setup, np.zeros(3))

    assert result["valid"], result["reason"]
    assert np.asarray(result["tyre_equilibrium_hub_displacement"])[0] != pytest.approx(0.0)
    assert np.asarray(result["tyre_equilibrium_jounce"])[0] != pytest.approx(0.0)
    assert "base_pose_reaction" in result
    assert "tyre_effective_reaction" in result


@pytest.mark.parametrize("reference_force", [0.0, -100.0])
def test_zero_or_negative_contact_is_unavailable(reference_force):
    tyre_effective_response = _api("tyre_effective_response")
    result = tyre_effective_response(
        _geometry(), _setup(reference_force=reference_force, spring_preload=0.0), np.zeros(3)
    )

    assert not result["valid"]
    assert result["reason"] == "nonpositive_contact"
    assert not np.asarray(result["validity"]["tyre_contact_force"]).any()
    assert np.isnan(result["ride_rate"]).all()


def test_singular_internal_hessian_is_rejected_by_mock_boundary(monkeypatch):
    from suspension_geometry import vehicle

    def zero_force(*args, **kwargs):
        return np.zeros(4), {}, True, "ok"

    def flat_energy(*args, **kwargs):
        return 0.0, {}, True, "ok", np.zeros(4)

    def zero_hessian(*args, **kwargs):
        return np.zeros((3, 3)), np.zeros((3, 4)), np.zeros((4, 4)), True, "ok"

    def positive_flat_tyre(spec, compression):
        return 0.0, 700.0, 0.0

    def valid_response(*args, **kwargs):
        return {
            "valid": True,
            "gradient": np.zeros(4),
            "gradient_valid": np.ones(4, dtype=bool),
            "stiffness": np.zeros((4, 4)),
            "stiffness_valid": np.ones((4, 4), dtype=bool),
        }

    monkeypatch.setattr(vehicle, "_internal_force", zero_force)
    monkeypatch.setattr(vehicle, "_energy_at_internal", flat_energy)
    monkeypatch.setattr(vehicle, "_energy_hessian", zero_hessian)
    monkeypatch.setattr(vehicle, "_tyre_law", positive_flat_tyre)
    monkeypatch.setattr(vehicle, "wheel_response", valid_response)
    result = vehicle.tyre_effective_response(_geometry(), _setup(), np.zeros(3))

    assert not result["valid"]
    assert result["reason"] == "unstable_internal_hessian"


def test_final_suspension_kink_invalidates_tyre_response(monkeypatch):
    from suspension_geometry import vehicle

    original = vehicle.wheel_response

    def kinked_response(*args, **kwargs):
        result = original(*args, **kwargs)
        result["stiffness_valid"][:] = False
        result["stiffness_reason"][:] = "law_kink"
        # The equilibrium residual uses the force gradient; the final tangent
        # gate must independently reject this kinked local response.
        return result

    monkeypatch.setattr(vehicle, "wheel_response", kinked_response)
    result = vehicle.tyre_effective_response(_geometry(), _setup(), np.zeros(3))

    assert not result["valid"]
    assert result["reason"] == "suspension_tangent_invalid"
