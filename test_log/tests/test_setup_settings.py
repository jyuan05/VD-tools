from __future__ import annotations

import json
import unittest

from vd_test_log.validation import ValidationError

try:
    from vd_test_log.setup_settings import (
        CORNERS,
        CORNER_FIELDS,
        NUMERIC_SETUP_FIELDS,
        SETUP_CHOICES,
        normalise_setup_settings_json,
    )
except (ImportError, AttributeError):
    CORNERS = None
    CORNER_FIELDS = None
    NUMERIC_SETUP_FIELDS = None
    SETUP_CHOICES = None
    normalise_setup_settings_json = None


class SetupSettingsTests(unittest.TestCase):
    def require_normaliser(self):
        self.assertIsNotNone(
            normalise_setup_settings_json,
            "the setup settings API is not implemented yet",
        )
        return normalise_setup_settings_json

    def test_exports_frozen_field_choices_and_order(self):
        self.assertEqual(
            SETUP_CHOICES,
            {
                "front_wing_height": ("1", "2", "3", "4", "5"),
                "front_spring_rate": ("200", "250", "300", "350"),
                "rw_setting": ("1", "2", "3", "LD"),
                "rear_spring_rate": ("200", "250", "300", "350"),
                "rear_arb_blade_setting": ("1", "2", "3", "4", "5", "6", "OFF"),
                "rear_arb_motion_ratio_setting": ("MR1", "MR2"),
            },
        )
        self.assertEqual(
            NUMERIC_SETUP_FIELDS,
            (
                "front_damping_ratio",
                "rear_damping_ratio",
                "diff_ramp_angle",
                "diff_preload",
            ),
        )
        self.assertEqual(CORNERS, ("FL", "FR", "RL", "RR"))
        self.assertEqual(CORNER_FIELDS, ("camber", "toe", "pressure", "corner_weight"))

    def test_normalises_every_enum_choice_and_preserves_canonical_values(self):
        normalise = self.require_normaliser()
        for field, choices in SETUP_CHOICES.items():
            for choice in choices:
                with self.subTest(field=field, choice=choice):
                    actual = json.loads(normalise(json.dumps({field: choice})))
                    self.assertEqual(actual, {field: choice})

        settings = {
            "front_damping_ratio": 0,
            "rear_damping_ratio": 1.25,
            "diff_ramp_angle": -35.0,
            "diff_preload": 0.0,
            "sprocket_size": "  42  ",
            "corners": {
                "FL": {"camber": -1.5, "toe": 0.0, "pressure": 0, "corner_weight": 55.25},
                "FR": {"camber": None},
                "RL": {},
                "RR": {"toe": -0.2},
            },
        }
        actual = json.loads(normalise(json.dumps(settings)))
        self.assertEqual(actual["front_damping_ratio"], 0)
        self.assertEqual(actual["diff_preload"], 0.0)
        self.assertEqual(actual["sprocket_size"], "42")
        self.assertEqual(actual["corners"]["FL"], settings["corners"]["FL"])
        self.assertEqual(actual["corners"]["FR"], {"camber": None})
        self.assertEqual(actual["corners"]["RR"], {"toe": -0.2})

    def test_blank_structured_values_normalise_to_null_and_empty_object(self):
        normalise = self.require_normaliser()
        settings = {
            "front_wing_height": "",
            "front_damping_ratio": "",
            "sprocket_size": "  ",
            "corners": {"FL": {"camber": "", "toe": None}},
        }
        actual = json.loads(normalise(json.dumps(settings)))
        self.assertEqual(
            actual,
            {
                "front_wing_height": None,
                "front_damping_ratio": None,
                "sprocket_size": None,
                "corners": {"FL": {"camber": None, "toe": None}},
            },
        )
        self.assertEqual(normalise("{}"), "{}")

    def test_normalises_optional_engine_tune_text_and_omits_blank_values(self):
        normalise = self.require_normaliser()
        self.assertEqual(
            normalise('{"engine_tune":"  Honda K v3  "}'),
            '{"engine_tune":"Honda K v3"}',
        )
        self.assertEqual(normalise('{"engine_tune":"  "}'), "{}")
        self.assertEqual(normalise("{}"), "{}")

    def test_rejects_non_text_engine_tune_values(self):
        normalise = self.require_normaliser()
        for value in (None, 42, True, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError) as raised:
                    normalise(json.dumps({"engine_tune": value}))
                self.assertEqual(raised.exception.field, "engine_tune")

    def test_rejects_nonfinite_and_non_numeric_values_with_stable_fields(self):
        normalise = self.require_normaliser()
        invalid_values = (float("nan"), float("inf"), float("-inf"), True, "1.5")
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError) as raised:
                    normalise(json.dumps({"front_damping_ratio": value}))
                self.assertEqual(raised.exception.field, "front_damping_ratio")

        with self.assertRaises(ValidationError) as raised:
            normalise(json.dumps({"corners": {"FL": {"camber": float("nan")}}}))
        self.assertEqual(raised.exception.field, "FL_camber")

    def test_rejects_unknown_keys_invalid_choices_and_wrong_shapes(self):
        normalise = self.require_normaliser()
        cases = (
            ({"unknown": 1}, "unknown"),
            ({"front_wing_height": "6"}, "front_wing_height"),
            ({"corners": "FL"}, "corners"),
            ({"corners": {"XX": {"camber": 1}},}, "corners"),
            ({"corners": {"FL": {"ride_height": 1}},}, "corners"),
            ({"sprocket_size": 42}, "sprocket_size"),
        )
        for payload, expected_field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError) as raised:
                    normalise(json.dumps(payload))
                self.assertEqual(raised.exception.field, expected_field)


if __name__ == "__main__":
    unittest.main()
