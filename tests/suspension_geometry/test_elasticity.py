"""Analytic checks for elastic energy and loaded suspension rates."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from suspension_geometry.config import load_geometry


ROOT = Path(__file__).resolve().parents[2]
DIRECT_GEOMETRY = ROOT / "configs/suspension_geometry/geometries/synthetic_direct.yaml"


def _api(name):
    from suspension_geometry import elasticity

    assert hasattr(elasticity, name), f"suspension_geometry.elasticity.{name} is missing"
    return getattr(elasticity, name)


def _setup(*, spring_rate=12000.0, preload=700.0, stop=None, stop_corners=None, arb_rate=0.0):
    stop_corners = set(stop_corners or ("FL", "FR", "RL", "RR"))
    return {
        "corners": {
            name: {
                "spring": {"rate": spring_rate, "preload_force": preload},
                **({"bump_stop": dict(stop)} if stop is not None and name in stop_corners else {}),
            }
            for name in ("FL", "FR", "RL", "RR")
        },
        "arbs": {
            "front": {"rate": arb_rate, "reference_twist": 0.0} if arb_rate else None,
            "rear": None,
        },
    }


def test_linear_component_law_has_preload_energy_force_and_tangent():
    law = _api("component_law")
    energy, force, tangent = law({"rate": 1200.0, "preload_force": 250.0}, 0.03)

    assert energy == pytest.approx(250.0 * 0.03 + 0.5 * 1200.0 * 0.03**2)
    assert force == pytest.approx(250.0 + 1200.0 * 0.03)
    assert tangent == pytest.approx(1200.0)


def test_force_curve_energy_integral_tangent_and_kink_are_declared():
    law = _api("component_law")
    curve = {"curve": [[-0.1, 20.0], [0.0, 40.0], [0.2, 60.0]]}

    energy, force, tangent = law(curve, 0.1)
    assert energy == pytest.approx(4.5)
    assert force == pytest.approx(50.0)
    assert tangent == pytest.approx(100.0)

    kink_energy, kink_force, kink_tangent = law(curve, 0.0)
    assert kink_energy == pytest.approx(0.0)
    assert kink_force == pytest.approx(40.0)
    assert np.isnan(kink_tangent)
    assert _api("CURVE_INTERPOLATION") == "piecewise_linear_force"


def test_curve_requires_reference_coverage_and_rejects_extrapolation():
    law = _api("component_law")
    domain_error = _api("LawDomainError")

    with pytest.raises(domain_error, match="zero"):
        law({"curve": [[0.1, 2.0], [0.2, 3.0]]}, 0.15)
    with pytest.raises(domain_error, match="domain"):
        law({"curve": [[-0.1, 0.0], [0.0, 10.0], [0.1, 20.0]]}, 0.11)


def test_bump_stop_has_zero_inactive_force_and_masks_its_engagement_kink():
    law = _api("component_law")
    stop = {"engagement": 0.02, "rate": 40000.0}

    assert law(stop, 0.01) == pytest.approx((0.0, 0.0, 0.0))
    assert law(stop, 0.05) == pytest.approx((18.0, 1200.0, 40000.0))
    energy, force, tangent = law(stop, 0.02)
    assert energy == pytest.approx(0.0)
    assert force == pytest.approx(0.0)
    assert np.isnan(tangent)


def test_bump_stop_energy_is_relative_to_reference_when_already_engaged():
    law = _api("component_law")
    stop = {"engagement": -0.01, "rate": 1000.0}

    assert law(stop, 0.0) == pytest.approx((0.0, 10.0, 1000.0))
    assert law(stop, -0.02) == pytest.approx((-0.05, 0.0, 0.0))
    assert law(stop, 0.01) == pytest.approx((0.15, 20.0, 1000.0))


def test_bump_stop_curve_rejects_force_discontinuity_for_every_query():
    law = _api("component_law")
    stop = {"engagement": 0.02, "curve": [[0.0, 10.0], [0.1, 100.0]]}

    with pytest.raises(ValueError, match="zero force at engagement"):
        law(stop, 0.05)


def test_chain_rule_keeps_preload_times_geometric_curvature():
    compose = _api("_compose_scalar_law")
    rate = 1800.0
    preload = 350.0
    a = 0.72
    b = 1.8
    jounce = 0.04
    compression = a * jounce + b * jounce**2
    motion_ratio = a + 2.0 * b * jounce
    curvature = 2.0 * b

    result = compose(
        (0.5 * rate * compression**2 + preload * compression, preload + rate * compression, rate),
        np.array([motion_ratio]),
        np.array([[curvature]]),
    )

    expected = rate * motion_ratio**2 + (preload + rate * compression) * curvature
    material_only = rate * motion_ratio**2
    assert result["gradient"][0] == pytest.approx((preload + rate * compression) * motion_ratio)
    assert result["stiffness"][0, 0] == pytest.approx(expected)
    assert abs(result["stiffness"][0, 0] - material_only) > 1000.0


def test_real_arb_energy_cancels_in_equal_heave_and_couples_opposed_wheels():
    wheel_energy = _api("wheel_energy")
    geometry = load_geometry(DIRECT_GEOMETRY)
    geometry = copy.deepcopy(geometry)
    for corner in geometry["corners"].values():
        corner["spring"]["length_limits"] = [0.2, 1.2]
        corner["damper"]["length_limits"] = [0.2, 1.2]
    setup = _setup(arb_rate=900.0)

    equal = wheel_energy(geometry, setup, np.array([0.035, 0.035, 0.0, 0.0]))
    opposed = wheel_energy(geometry, setup, np.array([0.035, -0.035, 0.0, 0.0]))

    assert equal["valid"] and opposed["valid"]
    assert equal["component_energies"]["arb_front"] == pytest.approx(0.0, abs=1e-12)
    assert opposed["component_energies"]["arb_front"] > 1e-4

    response = _api("wheel_response")(geometry, setup, np.zeros(4))
    assert response["valid"]
    assert response["stiffness_valid"][0, 1]
    assert response["stiffness"][0, 1] < 0.0


def test_arb_reference_twist_sets_nominal_torque_but_zeroes_reference_energy():
    geometry = load_geometry(DIRECT_GEOMETRY)
    setup = _setup(arb_rate=900.0)
    setup["arbs"]["front"]["reference_twist"] = 0.008

    energy = _api("wheel_energy")(geometry, setup, np.zeros(4))
    response = _api("wheel_response")(geometry, setup, np.zeros(4))

    assert energy["valid"] and response["valid"]
    assert energy["arb_state"]["front"]["twist"] == pytest.approx(0.0, abs=2e-8)
    assert energy["component_energies"]["arb_front"] == pytest.approx(0.0, abs=1e-14)
    assert response["energy"] == pytest.approx(energy["energy"], abs=1e-14)
    assert response["component_contributions"]["arb_front_torque"] == pytest.approx(7.2)


def test_bump_stop_at_engagement_keeps_center_force_and_invalidates_only_its_tangent():
    wheel_response = _api("wheel_response")
    geometry = load_geometry(DIRECT_GEOMETRY)
    response = wheel_response(
        geometry,
        _setup(stop={"engagement": 0.0, "rate": 40000.0}, stop_corners=("FL",)),
        np.zeros(4),
    )

    assert response["valid"]
    assert response["component_contributions"]["bump_stop_force"][0] == pytest.approx(0.0, abs=1e-12)
    assert not response["stiffness_valid"][0, 0]
    assert response["stiffness_reason"][0, 0] == "law_kink"
    assert np.isnan(response["material_rates"][0])
    expected_valid = np.ones((4, 4), dtype=bool)
    expected_valid[0, 0] = False
    np.testing.assert_array_equal(response["stiffness_valid"], expected_valid)


def test_wheel_response_matches_an_independent_energy_difference():
    wheel_energy = _api("wheel_energy")
    wheel_response = _api("wheel_response")
    geometry = load_geometry(DIRECT_GEOMETRY)
    setup = _setup()
    jounce = np.array([0.02, 0.0, 0.0, 0.0])
    response = wheel_response(geometry, setup, jounce)
    h = 5e-4
    plus = jounce.copy()
    minus = jounce.copy()
    plus[0] += h
    minus[0] -= h
    e_plus = wheel_energy(geometry, setup, plus)["energy"]
    e_minus = wheel_energy(geometry, setup, minus)["energy"]
    center = wheel_energy(geometry, setup, jounce)["energy"]
    independent_force = (e_plus - e_minus) / (2.0 * h)
    independent_tangent = (e_plus - 2.0 * center + e_minus) / h**2

    assert response["gradient"][0] == pytest.approx(independent_force, rel=2e-4, abs=0.05)
    assert response["stiffness"][0, 0] == pytest.approx(independent_tangent, rel=1e-2, abs=10.0)


def test_active_arb_missing_geometry_is_nan_with_component_masks():
    geometry = copy.deepcopy(load_geometry(DIRECT_GEOMETRY))
    geometry["arbs"].pop("front")
    setup = _setup(arb_rate=900.0)
    wheel_energy = _api("wheel_energy")
    wheel_response = _api("wheel_response")

    energy = wheel_energy(geometry, setup, np.zeros(4))
    assert not energy["valid"]
    assert np.isfinite(energy["component_energies"]["spring"]).all()
    assert np.isnan(energy["component_energies"]["arb_front"])
    assert not energy["component_energy_validity"]["arb_front"]
    assert energy["component_energy_reason"]["arb_front"] == "arb_geometry_missing"

    response = wheel_response(geometry, setup, np.zeros(4))
    assert not response["valid"]
    assert np.isnan(response["component_contributions"]["arb_front_energy"])
    assert not response["component_contribution_validity"]["arb_front"]["energy"]
    np.testing.assert_array_equal(
        np.isnan(response["component_contributions"]["arb_front_force"]),
        [True, True, False, False],
    )
    assert not response["component_contribution_validity"]["arb_front"]["force"][:2].any()
    assert response["component_contribution_validity"]["arb_front"]["force"][2:].all()


def test_configured_bump_stop_law_failure_preserves_known_energy_and_masks_stop():
    geometry = load_geometry(DIRECT_GEOMETRY)
    setup = _setup(
        stop={"engagement": -0.1, "curve": [[0.0, 0.0], [0.01, 10.0]]},
        stop_corners=("FL",),
    )
    wheel_energy = _api("wheel_energy")
    wheel_response = _api("wheel_response")

    energy = wheel_energy(geometry, setup, np.zeros(4))
    assert not energy["valid"]
    assert np.isfinite(energy["component_energies"]["spring"]).all()
    assert np.isnan(energy["component_energies"]["bump_stop"][0])
    np.testing.assert_array_equal(energy["component_energies"]["bump_stop"][1:], 0.0)
    np.testing.assert_array_equal(
        energy["component_energy_validity"]["bump_stop"], [False, True, True, True]
    )
    assert energy["component_energy_reason"]["bump_stop"][0] == "curve_domain"
    assert np.all(energy["component_energy_reason"]["bump_stop"][1:] == "inactive")

    response = wheel_response(geometry, setup, np.zeros(4))
    assert not response["valid"]
    assert np.isfinite(response["component_contributions"]["spring_energy"]).all()
    assert np.isnan(response["component_contributions"]["bump_stop_energy"][0])
    np.testing.assert_array_equal(response["component_contributions"]["bump_stop_force"][1:], 0.0)
    assert np.isnan(response["component_contributions"]["bump_stop_force"][0])
    assert not response["component_contribution_validity"]["bump_stop"]["force"][0]
    assert response["component_contribution_validity"]["bump_stop"]["force"][1:].all()


def test_corner_solve_failure_masks_configured_stop_without_erasing_other_corners():
    geometry = load_geometry(DIRECT_GEOMETRY)
    setup = _setup(stop={"engagement": 0.02, "rate": 40000.0}, stop_corners=("FL",))
    wheel_energy = _api("wheel_energy")

    energy = wheel_energy(geometry, setup, np.array([5.0, 0.0, 0.0, 0.0]))
    assert not energy["valid"]
    assert not energy["corner_valid"][0]
    assert np.isnan(energy["component_energies"]["spring"][0])
    assert np.isnan(energy["component_energies"]["bump_stop"][0])
    assert np.isfinite(energy["component_energies"]["spring"][1:]).all()
    np.testing.assert_array_equal(energy["component_energies"]["bump_stop"][1:], 0.0)
    assert not energy["component_energy_validity"]["spring"][0]
    assert not energy["component_energy_validity"]["bump_stop"][0]


def test_omitted_inactive_arbs_remain_valid_zero_with_masks():
    geometry = load_geometry(DIRECT_GEOMETRY)
    setup = _setup(arb_rate=0.0)
    wheel_energy = _api("wheel_energy")
    wheel_response = _api("wheel_response")

    energy = wheel_energy(geometry, setup, np.zeros(4))
    assert energy["valid"]
    assert energy["component_energies"]["arb_front"] == pytest.approx(0.0)
    assert energy["component_energies"]["arb_rear"] == pytest.approx(0.0)
    assert energy["component_energy_validity"]["arb_front"]
    assert energy["component_energy_reason"]["arb_front"] == "inactive"

    response = wheel_response(geometry, setup, np.zeros(4))
    assert response["valid"]
    assert response["component_contributions"]["arb_front_energy"] == pytest.approx(0.0)
    assert response["component_contributions"]["arb_front_torque"] == pytest.approx(0.0)
    assert response["component_contribution_validity"]["arb_front"]["energy"]
    assert response["component_contribution_validity"]["arb_front"]["force"].all()
