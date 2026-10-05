"""Rigid double-wishbone corner closure in the canonical corner frame.

``CornerState.solution[:3]`` is wheel-centre translation from nominal in
metres; ``solution[3:]`` is the incremental upright rotation vector in
radians.  The nominal state is therefore six zeros.  ``rotation`` applies
that incremental rotation to all nominal upright point offsets about the
reference wheel centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


_NAN3 = np.full(3, np.nan, dtype=float)
_NAN33 = np.full((3, 3), np.nan, dtype=float)
_NAN6 = np.full(6, np.nan, dtype=float)
_MAX_CONDITION = 1.0e10


@dataclass(frozen=True)
class CornerState:
    """Solved or invalid rigid corner pose with SI diagnostics."""

    valid: bool
    reason: str
    jounce: float
    wheel_center: np.ndarray
    rotation: np.ndarray
    upper_outboard: np.ndarray
    lower_outboard: np.ndarray
    tie_outboard: np.ndarray
    residual: float
    condition: float
    solution: np.ndarray


def _invalid_state(
    jounce: float,
    reason: str,
    *,
    residual: float = float("nan"),
    condition: float = float("nan"),
    solution: np.ndarray | None = None,
) -> CornerState:
    return CornerState(
        valid=False,
        reason=reason,
        jounce=jounce,
        wheel_center=_NAN3.copy(),
        rotation=_NAN33.copy(),
        upper_outboard=_NAN3.copy(),
        lower_outboard=_NAN3.copy(),
        tie_outboard=_NAN3.copy(),
        residual=float(residual),
        condition=float(condition),
        solution=_NAN6.copy() if solution is None else np.asarray(solution, dtype=float).copy(),
    )


def _pose(corner: dict, solution: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    reference_center = np.asarray(corner["wheel_center"], dtype=float)
    center = reference_center + solution[:3]
    rotation = Rotation.from_rotvec(solution[3:]).as_matrix()

    def upright_point(point: Iterable[float]) -> np.ndarray:
        return center + rotation @ (np.asarray(point, dtype=float) - reference_center)

    points = {
        "lower": upright_point(corner["lower"]["outboard"]),
        "upper": upright_point(corner["upper"]["outboard"]),
        "tie": upright_point(corner["tie"]["outboard"]),
    }
    return center, rotation, points


def _constraints(solution: np.ndarray, corner: dict, jounce: float) -> np.ndarray:
    reference_center = np.asarray(corner["wheel_center"], dtype=float)
    center, _, points = _pose(corner, solution)
    constraints: list[float] = []
    for name in ("lower", "upper"):
        arm = corner[name]
        nominal = np.asarray(arm["outboard"], dtype=float)
        current = points[name]
        for pivot_name in ("inboard_rearward", "inboard_forward"):
            pivot = np.asarray(arm[pivot_name], dtype=float)
            constraints.append(float(np.linalg.norm(current - pivot) - np.linalg.norm(nominal - pivot)))
    tie = corner["tie"]
    tie_inboard = np.asarray(tie["inboard"], dtype=float)
    tie_nominal = np.asarray(tie["outboard"], dtype=float)
    constraints.append(
        float(np.linalg.norm(points["tie"] - tie_inboard) - np.linalg.norm(tie_nominal - tie_inboard))
    )
    constraints.append(float(center[2] - reference_center[2] - jounce))
    return np.asarray(constraints, dtype=float)


def _condition_number(jacobian: np.ndarray, corner: dict) -> float:
    # Normalize angular columns to an equivalent linear displacement using
    # the corner's upright-to-wheel-centre lever arm.
    center = np.asarray(corner["wheel_center"], dtype=float)
    lever_arms = [
        np.linalg.norm(np.asarray(corner[name]["outboard"], dtype=float) - center)
        for name in ("lower", "upper")
    ]
    characteristic_length = max(float(np.mean(lever_arms)), 0.1)
    scale = np.diag([1.0, 1.0, 1.0, characteristic_length, characteristic_length, characteristic_length])
    singular_values = np.linalg.svd(np.asarray(jacobian, dtype=float) @ scale, compute_uv=False)
    if not singular_values.size or singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _seed_solution(seed: CornerState | np.ndarray | Iterable[float] | None) -> tuple[np.ndarray, float | None] | None:
    if seed is None:
        return None
    if isinstance(seed, CornerState):
        if not seed.valid:
            return None
        value = np.asarray(seed.solution, dtype=float)
        seed_jounce = float(seed.jounce)
    else:
        value = np.asarray(seed, dtype=float)
        seed_jounce = None
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("seed must be a finite CornerState or six-element solution")
    return value.copy(), seed_jounce


def _branch_is_continuous(
    solution: np.ndarray,
    seed_solution: np.ndarray,
    jounce_delta: float,
) -> bool:
    translation_delta = float(np.linalg.norm(solution[:3] - seed_solution[:3]))
    rotation_delta = float(
        np.linalg.norm(
            Rotation.from_rotvec(seed_solution[3:])
            .inv()
            .__mul__(Rotation.from_rotvec(solution[3:]))
            .as_rotvec()
        )
    )
    translation_limit = 0.10 + 4.0 * jounce_delta
    rotation_limit = 0.25 + 12.0 * jounce_delta
    return translation_delta <= translation_limit and rotation_delta <= rotation_limit


def solve_corner(
    corner: dict,
    jounce: float,
    seed: CornerState | np.ndarray | Iterable[float] | None = None,
    max_nfev: int = 100,
    tolerance: float = 1e-9,
) -> CornerState:
    """Solve a rigid upright pose from four arm lengths, tie length and z.

    Length residuals are expressed in metres.  Invalid poses keep finite
    residual/conditioning/solution diagnostics when the optimizer produced
    them, while all physical point outputs are NaN.
    """
    try:
        requested_jounce = float(jounce)
    except (TypeError, ValueError, OverflowError):
        return _invalid_state(float("nan"), "nonfinite_jounce")
    if not np.isfinite(requested_jounce):
        return _invalid_state(requested_jounce, "nonfinite_jounce")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    limits = np.asarray(corner["jounce_limits"], dtype=float)
    if requested_jounce < limits[0] - 1e-12 or requested_jounce > limits[1] + 1e-12:
        return _invalid_state(requested_jounce, "jounce_out_of_range")

    parsed_seed = _seed_solution(seed)
    if seed is not None and isinstance(seed, CornerState) and not seed.valid:
        parsed_seed = None
    if parsed_seed is None:
        x0 = np.zeros(6, dtype=float)
        seed_jounce = None
    else:
        x0, seed_jounce = parsed_seed

    lower_bounds = np.asarray([-0.5, -0.5, -0.5, -np.pi, -np.pi, -np.pi], dtype=float)
    upper_bounds = -lower_bounds
    if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
        return _invalid_state(requested_jounce, "invalid_seed")

    try:
        fit = least_squares(
            _constraints,
            x0,
            args=(corner, requested_jounce),
            method="trf",
            bounds=(lower_bounds, upper_bounds),
            x_scale=np.asarray([0.1, 0.1, 0.1, 0.25, 0.25, 0.25]),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=int(max_nfev),
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return _invalid_state(requested_jounce, "solver_error")

    solution = np.asarray(fit.x, dtype=float)
    residuals = _constraints(solution, corner, requested_jounce)
    residual = float(np.max(np.abs(residuals)))
    condition = _condition_number(fit.jac, corner)
    if not np.isfinite(solution).all() or not np.isfinite(residual):
        return _invalid_state(requested_jounce, "solver_error", residual=residual, condition=condition, solution=solution)
    if residual > tolerance:
        return _invalid_state(requested_jounce, "no_convergence", residual=residual, condition=condition, solution=solution)
    if condition > _MAX_CONDITION:
        return _invalid_state(requested_jounce, "singular_constraints", residual=residual, condition=condition, solution=solution)

    if parsed_seed is not None:
        delta_jounce = (
            abs(requested_jounce - seed_jounce) if seed_jounce is not None else abs(requested_jounce)
        )
        if not _branch_is_continuous(solution, parsed_seed[0], delta_jounce):
            return _invalid_state(
                requested_jounce,
                "discontinuous_branch",
                residual=residual,
                condition=condition,
                solution=solution,
            )
    elif not _branch_is_continuous(solution, np.zeros(6, dtype=float), abs(requested_jounce)):
        return _invalid_state(
            requested_jounce,
            "discontinuous_branch",
            residual=residual,
            condition=condition,
            solution=solution,
        )

    center, rotation, points = _pose(corner, solution)
    return CornerState(
        valid=True,
        reason="ok",
        jounce=requested_jounce,
        wheel_center=center,
        rotation=rotation,
        upper_outboard=points["upper"],
        lower_outboard=points["lower"],
        tie_outboard=points["tie"],
        residual=residual,
        condition=condition,
        solution=solution,
    )


def solve_corner_sweep(
    corner: dict,
    jounces: Iterable[float],
    max_nfev: int = 100,
    tolerance: float = 1e-9,
) -> list[CornerState]:
    """Solve both travel directions outward from zero, preserving input order."""
    requested = np.asarray(list(jounces), dtype=float)
    if requested.ndim != 1:
        raise ValueError("jounces must be a one-dimensional sequence")
    if requested.size == 0:
        return []

    states: list[CornerState | None] = [None] * requested.size
    reference = solve_corner(corner, 0.0, max_nfev=max_nfev, tolerance=tolerance)
    for index, value in enumerate(requested):
        if not np.isfinite(value):
            states[index] = _invalid_state(float(value), "nonfinite_jounce")
        elif value == 0.0:
            states[index] = reference

    for indices in (
        sorted((i for i, value in enumerate(requested) if np.isfinite(value) and value > 0.0), key=lambda i: requested[i]),
        sorted((i for i, value in enumerate(requested) if np.isfinite(value) and value < 0.0), key=lambda i: abs(requested[i])),
    ):
        previous: CornerState | None = reference if reference.valid else None
        for index in indices:
            state = solve_corner(
                corner,
                float(requested[index]),
                seed=previous,
                max_nfev=max_nfev,
                tolerance=tolerance,
            )
            states[index] = state
            if state.valid:
                previous = state

    return [state for state in states if state is not None]
