"""Analytic regressions for validity-aware scaled finite differences."""

from __future__ import annotations

import numpy as np
import pytest


def _api():
    from suspension_geometry.numerics import finite_difference_derivatives

    return finite_difference_derivatives


def test_scaled_stencils_recover_polynomial_gradient_and_mixed_hessian():
    differentiate = _api()

    def polynomial(point):
        x, y = point
        return 3.0 + 2.0 * x - 5.0 * y + 4.0 * x * x + 3.0 * x * y + 7.0 * y * y

    point = np.array([0.4, -0.3])
    result = differentiate(polynomial, point, steps=[1e-3, 1e-3], scales=[2.0, 0.5])

    assert result["gradient"] == pytest.approx([2.0 + 8.0 * point[0] + 3.0 * point[1], -5.0 + 3.0 * point[0] + 14.0 * point[1]], abs=1e-8)
    np.testing.assert_allclose(result["hessian"], [[8.0, 3.0], [3.0, 14.0]], atol=2e-6)
    assert result["gradient_valid"].all()
    assert result["hessian_valid"].all()
    assert result["steps"] == pytest.approx([2e-3, 5e-4])


def test_scaled_stencils_use_one_sided_valid_samples_at_a_boundary():
    differentiate = _api()

    def polynomial(point):
        x, y = point
        return x * x + 2.0 * x * y + 3.0 * y * y

    result = differentiate(
        polynomial,
        [0.0, 0.25],
        steps=[1e-3, 1e-3],
        valid=lambda point: point[0] >= -1e-14,
    )

    assert result["gradient"] == pytest.approx([0.5, 1.5], abs=2e-6)
    np.testing.assert_allclose(result["hessian"], [[2.0, 2.0], [2.0, 6.0]], atol=2e-5)
    assert result["gradient_method"][0] == "forward_second_order"
    assert result["hessian_method"][0, 1] == "forward_second_order"
    assert result["gradient_valid"].all()
    assert result["hessian_valid"].all()


def test_unavailable_stencils_are_nan_and_invalid_with_a_reason():
    differentiate = _api()
    result = differentiate(
        lambda point: float(np.dot(point, point)),
        [0.0],
        steps=[1e-3],
        valid=lambda point: np.array_equal(point, np.array([0.0])),
    )

    assert not result["gradient_valid"][0]
    assert not result["hessian_valid"][0, 0]
    assert np.isnan(result["gradient"][0])
    assert np.isnan(result["hessian"][0, 0])
    assert result["gradient_reason"][0] == "invalid_stencil"
    assert result["hessian_reason"][0, 0] == "invalid_stencil"
