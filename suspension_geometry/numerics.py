"""Validity-aware scaled finite differences for scalar energy functions."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np


_TOL = 1e-14


def finite_difference_derivatives(
    function: Callable[[np.ndarray], Any],
    point: Any,
    *,
    steps: Any,
    scales: Any | None = None,
    valid: Callable[[np.ndarray], bool] | None = None,
) -> dict[str, Any]:
    """Differentiate a scalar function with validity-aware scaled stencils.

    ``steps`` are dimensionless increments and ``scales`` set their coordinate
    units, so a coordinate's actual step is ``steps[i] * scales[i]``. Central
    second-order stencils are preferred. Where those leave a valid domain,
    one-sided second-order stencils are used; first-order fallbacks are marked
    in the returned method arrays. Missing or invalid samples produce NaNs,
    false masks, and explicit reason strings, never zero derivatives.

    A callback may return either a scalar or ``(value, is_valid, reason)``.
    The optional ``valid`` predicate is checked before evaluating the function.
    """
    x = np.asarray(point, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("point must be a non-empty finite vector")
    n = x.size
    requested_steps = np.asarray(steps, dtype=float)
    if requested_steps.ndim == 0:
        requested_steps = np.full(n, float(requested_steps), dtype=float)
    if requested_steps.shape != (n,) or not np.all(np.isfinite(requested_steps)) or np.any(requested_steps <= 0.0):
        raise ValueError("steps must be positive finite scalar or vector matching point")
    if scales is None:
        coordinate_scales = np.ones(n, dtype=float)
    else:
        coordinate_scales = np.asarray(scales, dtype=float)
        if coordinate_scales.ndim == 0:
            coordinate_scales = np.full(n, float(coordinate_scales), dtype=float)
        if coordinate_scales.shape != (n,) or not np.all(np.isfinite(coordinate_scales)) or np.any(coordinate_scales <= 0.0):
            raise ValueError("scales must be positive finite scalar or vector matching point")
    h = requested_steps * coordinate_scales

    cache: dict[tuple[float, ...], tuple[float, bool, str]] = {}

    def sample(offsets: tuple[int, ...] | np.ndarray) -> tuple[float, bool, str]:
        key = tuple(float(value) for value in offsets)
        if key in cache:
            return cache[key]
        candidate = x + np.asarray(offsets, dtype=float) * h
        if valid is not None and not bool(valid(candidate.copy())):
            result = (float("nan"), False, "outside_valid_domain")
            cache[key] = result
            return result
        raw = function(candidate.copy())
        if isinstance(raw, tuple) and len(raw) == 3:
            value, sample_valid, reason = raw
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
                sample_valid = False
                reason = "non_scalar_sample"
            result = (
                value,
                bool(sample_valid) and np.isfinite(value),
                str(reason) if not sample_valid or not np.isfinite(value) else "ok",
            )
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
                result = (value, False, "non_scalar_sample")
            else:
                result = (value, bool(np.isfinite(value)), "ok" if np.isfinite(value) else "nonfinite_sample")
        cache[key] = result
        return result

    zero = (0,) * n
    center, center_valid, center_reason = sample(zero)
    gradient = np.full(n, np.nan, dtype=float)
    hessian = np.full((n, n), np.nan, dtype=float)
    gradient_valid = np.zeros(n, dtype=bool)
    hessian_valid = np.zeros((n, n), dtype=bool)
    gradient_reason = np.full(n, "invalid_stencil", dtype=object)
    hessian_reason = np.full((n, n), "invalid_stencil", dtype=object)
    gradient_method = np.full(n, "unavailable", dtype=object)
    hessian_method = np.full((n, n), "unavailable", dtype=object)

    if not center_valid:
        gradient_reason[:] = "invalid_center:" + center_reason
        hessian_reason[:, :] = "invalid_center:" + center_reason
        return {
            "valid": False,
            "center_valid": False,
            "center_value": float("nan"),
            "gradient": gradient,
            "hessian": hessian,
            "gradient_valid": gradient_valid,
            "hessian_valid": hessian_valid,
            "gradient_reason": gradient_reason,
            "hessian_reason": hessian_reason,
            "gradient_method": gradient_method,
            "hessian_method": hessian_method,
            "steps": h.copy(),
            "sample_count": len(cache),
        }

    def offset(index: int, amount: int) -> tuple[int, ...]:
        values = [0] * n
        values[index] = amount
        return tuple(values)

    def first_candidates(index: int) -> list[tuple[str, list[tuple[int, float]]]]:
        step = h[index]
        candidates: list[tuple[str, list[tuple[int, float]]]] = []
        minus_ok = sample(offset(index, -1))[1]
        plus_ok = sample(offset(index, 1))[1]
        if minus_ok and plus_ok:
            candidates.append(("central", [(-1, -0.5 / step), (1, 0.5 / step)]))
        if plus_ok and sample(offset(index, 2))[1]:
            candidates.append(("forward_second_order", [(0, -1.5 / step), (1, 2.0 / step), (2, -0.5 / step)]))
        if minus_ok and sample(offset(index, -2))[1]:
            candidates.append(("backward_second_order", [(0, 1.5 / step), (-1, -2.0 / step), (-2, 0.5 / step)]))
        if plus_ok:
            candidates.append(("forward_first_order", [(0, -1.0 / step), (1, 1.0 / step)]))
        if minus_ok:
            candidates.append(("backward_first_order", [(0, 1.0 / step), (-1, -1.0 / step)]))
        return candidates

    first_stencils = [first_candidates(index) for index in range(n)]
    for index, candidates in enumerate(first_stencils):
        for method, terms in candidates:
            values = [(sample(offset(index, multiple))[0], coefficient, sample(offset(index, multiple))[1]) for multiple, coefficient in terms]
            if all(is_valid_sample for _, _, is_valid_sample in values):
                gradient[index] = sum(value * coefficient for value, coefficient, _ in values)
                gradient_valid[index] = True
                gradient_reason[index] = "ok"
                gradient_method[index] = method
                break

    def second_candidates(index: int) -> list[tuple[str, list[tuple[int, float]]]]:
        step = h[index]
        candidates: list[tuple[str, list[tuple[int, float]]]] = []
        minus_ok = sample(offset(index, -1))[1]
        plus_ok = sample(offset(index, 1))[1]
        if minus_ok and plus_ok:
            candidates.append(("central", [(-1, 1.0 / step**2), (0, -2.0 / step**2), (1, 1.0 / step**2)]))
        if all(sample(offset(index, amount))[1] for amount in (1, 2, 3)):
            candidates.append(("forward_second_order", [(0, 2.0 / step**2), (1, -5.0 / step**2), (2, 4.0 / step**2), (3, -1.0 / step**2)]))
        if all(sample(offset(index, amount))[1] for amount in (-1, -2, -3)):
            candidates.append(("backward_second_order", [(0, 2.0 / step**2), (-1, -5.0 / step**2), (-2, 4.0 / step**2), (-3, -1.0 / step**2)]))
        if plus_ok and sample(offset(index, 2))[1]:
            candidates.append(("forward_first_order", [(0, 1.0 / step**2), (1, -2.0 / step**2), (2, 1.0 / step**2)]))
        if minus_ok and sample(offset(index, -2))[1]:
            candidates.append(("backward_first_order", [(0, 1.0 / step**2), (-1, -2.0 / step**2), (-2, 1.0 / step**2)]))
        return candidates

    for index in range(n):
        for method, terms in second_candidates(index):
            values = [(sample(offset(index, multiple))[0], coefficient, sample(offset(index, multiple))[1]) for multiple, coefficient in terms]
            if all(is_valid_sample for _, _, is_valid_sample in values):
                hessian[index, index] = sum(value * coefficient for value, coefficient, _ in values)
                hessian_valid[index, index] = True
                hessian_reason[index, index] = "ok"
                hessian_method[index, index] = method
                break

    def mixed_terms(
        first_index: int,
        first_multiple: int,
        second_index: int,
        second_multiple: int,
        coefficient: float,
    ) -> tuple[tuple[int, ...], float]:
        offsets = [0] * n
        offsets[first_index] = first_multiple
        offsets[second_index] = second_multiple
        return tuple(offsets), coefficient

    for first_index in range(n):
        for second_index in range(first_index + 1, n):
            found = False
            for (first_method, first_terms), (second_method, second_terms) in product(
                first_stencils[first_index], first_stencils[second_index]
            ):
                stencil = [
                    mixed_terms(first_index, first_multiple, second_index, second_multiple, first_coefficient * second_coefficient)
                    for first_multiple, first_coefficient in first_terms
                    for second_multiple, second_coefficient in second_terms
                ]
                values = [(sample(offsets)[0], coefficient, sample(offsets)[1]) for offsets, coefficient in stencil]
                if not all(is_valid_sample for _, _, is_valid_sample in values):
                    continue
                value = sum(sample_value * coefficient for sample_value, coefficient, _ in values)
                method = first_method if first_method != "central" else second_method
                hessian[first_index, second_index] = value
                hessian[second_index, first_index] = value
                hessian_valid[first_index, second_index] = True
                hessian_valid[second_index, first_index] = True
                hessian_reason[first_index, second_index] = "ok"
                hessian_reason[second_index, first_index] = "ok"
                hessian_method[first_index, second_index] = method
                hessian_method[second_index, first_index] = method
                found = True
                break
            if not found:
                hessian_reason[first_index, second_index] = "invalid_stencil"
                hessian_reason[second_index, first_index] = "invalid_stencil"

    return {
        "valid": bool(gradient_valid.all() and hessian_valid.all()),
        "center_valid": True,
        "center_value": center,
        "gradient": gradient,
        "hessian": hessian,
        "gradient_valid": gradient_valid,
        "hessian_valid": hessian_valid,
        "gradient_reason": gradient_reason,
        "hessian_reason": hessian_reason,
        "gradient_method": gradient_method,
        "hessian_method": hessian_method,
        "steps": h.copy(),
        "sample_count": len(cache),
    }
