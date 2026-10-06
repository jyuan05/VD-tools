"""Bounded study allocation and partial-corner map contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from suspension_geometry.config import resolve_study


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "configs/suspension_geometry/studies/direct_heave.yaml"


def _api(name: str):
    from suspension_geometry import sweeps

    assert hasattr(sweeps, name), f"suspension_geometry.sweeps.{name} is missing"
    return getattr(sweeps, name)


def test_run_study_uses_n_r_p_leading_shape_and_exposes_reasons():
    geometry, setup, study = resolve_study(STUDY)
    study = dict(study)
    study["axes"] = {"heave_m": [0.0, 0.01], "roll_rad": [0.0], "pitch_rad": [0.0, 0.01]}
    study["metrics"] = ["spring_motion_ratio", "body_stiffness"]
    run_study = _api("run_study")
    result = run_study(geometry, setup, study)

    assert result.axes.keys() == {"heave_m", "roll_rad", "pitch_rad"}
    assert result.metrics["spring_motion_ratio"].shape == (2, 1, 2, 4)
    assert result.metrics["body_stiffness"].shape == (2, 1, 2, 3, 3)
    assert result.validity["spring_motion_ratio"].shape == (2, 1, 2, 4)
    assert result.reasons["body_stiffness"].shape == (2, 1, 2, 3, 3)
    assert set(result.corner_maps) == {"FL", "FR", "RL", "RR"}
    assert result.metadata["sample_count"] == 4
    assert result.metadata["synthetic"] is True


def test_run_study_keeps_independent_corner_geometry_when_foreign_corner_fails():
    geometry, setup, study = resolve_study(STUDY)
    geometry["corners"]["FL"]["jounce_limits"] = [-1e-6, 1e-6]
    study = dict(study)
    study["axes"] = {"heave_m": [0.0, 0.02], "roll_rad": [0.0], "pitch_rad": [0.0]}
    study["metrics"] = ["camber_chassis", "spring_motion_ratio"]

    result = _api("run_study")(geometry, setup, study)

    assert not result.validity["camber_chassis"][1, 0, 0, 0]
    assert result.validity["camber_chassis"][1, 0, 0, 1:].all()
    assert np.isfinite(result.metrics["camber_chassis"][1, 0, 0, 1:]).all()


def test_run_study_rejects_unknown_metric_before_allocating_grid():
    geometry, setup, study = resolve_study(STUDY)
    study = dict(study)
    study["metrics"] = ["not_a_metric"]

    with pytest.raises(ValueError, match="not_a_metric"):
        _api("run_study")(geometry, setup, study)
