"""Optional setup-level vehicle inputs with stable JSON field names."""

from __future__ import annotations

import json
import math
from typing import Any

from .validation import ValidationError


SETUP_CHOICES = {
    "front_wing_height": ("1", "2", "3", "4", "5"),
    "front_spring_rate": ("200", "250", "300", "350"),
    "rw_setting": ("1", "2", "3", "LD"),
    "rear_spring_rate": ("200", "250", "300", "350"),
    "rear_arb_blade_setting": ("1", "2", "3", "4", "5", "6", "OFF"),
    "rear_arb_motion_ratio_setting": ("MR1", "MR2"),
}
NUMERIC_SETUP_FIELDS = (
    "front_damping_ratio",
    "rear_damping_ratio",
    "diff_ramp_angle",
    "diff_preload",
)
CORNERS = ("FL", "FR", "RL", "RR")
CORNER_FIELDS = ("camber", "toe", "pressure", "corner_weight")

_TEXT_SETUP_FIELDS = {"sprocket_size"}
_SETUP_FIELDS = (
    set(SETUP_CHOICES)
    | set(NUMERIC_SETUP_FIELDS)
    | _TEXT_SETUP_FIELDS
    | {"corners"}
)


def _is_blank(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def _optional_number(value: Any, field: str) -> int | float | None:
    if value is None or _is_blank(value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(field, "Enter a finite number or leave this field blank.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(field, "Enter a finite number or leave this field blank.")
    return value


def _normalise_corner_values(value: Any, corner: str) -> dict[str, int | float | None]:
    if not isinstance(value, dict):
        raise ValidationError("corners", "Enter corner settings as an object.")
    unknown_fields = set(value) - set(CORNER_FIELDS)
    if unknown_fields:
        raise ValidationError("corners", f"Unknown corner setting: {sorted(unknown_fields)[0]}.")
    return {
        field: _optional_number(value[field], f"{corner}_{field}")
        for field in CORNER_FIELDS
        if field in value
    }


def normalise_setup_settings_json(value: str) -> str:
    """Validate and canonically serialize optional setup settings JSON."""
    if not isinstance(value, str):
        raise ValidationError(
            "structured_settings_json",
            "Enter structured setup settings as JSON text.",
        )
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValidationError(
            "structured_settings_json",
            "Enter valid JSON for structured setup settings.",
        ) from error
    if not isinstance(parsed, dict):
        raise ValidationError(
            "structured_settings_json",
            "Structured setup settings must be a JSON object.",
        )

    unknown_keys = set(parsed) - _SETUP_FIELDS
    if unknown_keys:
        field = sorted(unknown_keys)[0]
        raise ValidationError(field, f"Unknown setup setting: {field}.")

    normalized: dict[str, Any] = {}
    for field, choices in SETUP_CHOICES.items():
        if field not in parsed:
            continue
        value_at_field = parsed[field]
        if value_at_field is None or _is_blank(value_at_field):
            normalized[field] = None
        elif isinstance(value_at_field, str) and value_at_field in choices:
            normalized[field] = value_at_field
        else:
            raise ValidationError(field, "Choose one of the listed setup values.")

    for field in NUMERIC_SETUP_FIELDS:
        if field in parsed:
            normalized[field] = _optional_number(parsed[field], field)

    if "sprocket_size" in parsed:
        sprocket_size = parsed["sprocket_size"]
        if sprocket_size is None or _is_blank(sprocket_size):
            normalized["sprocket_size"] = None
        elif isinstance(sprocket_size, str):
            normalized["sprocket_size"] = sprocket_size.strip()
        else:
            raise ValidationError("sprocket_size", "Enter text or leave this field blank.")

    if "corners" in parsed:
        corners = parsed["corners"]
        if not isinstance(corners, dict):
            raise ValidationError("corners", "Enter corner settings as an object.")
        unknown_corners = set(corners) - set(CORNERS)
        if unknown_corners:
            raise ValidationError("corners", f"Unknown corner: {sorted(unknown_corners)[0]}.")
        normalized["corners"] = {
            corner: _normalise_corner_values(corners[corner], corner)
            for corner in CORNERS
            if corner in corners
        }

    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
