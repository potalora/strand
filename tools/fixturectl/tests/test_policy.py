from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

import fixturectl  # noqa: E402


POLICY_PATH = TOOL_ROOT / "policy.json"


class PolicyTests(unittest.TestCase):
    def _write_policy(self, body: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "policy.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_committed_policy_is_exact_and_contains_no_source_paths(self) -> None:
        policy = fixturectl.load_policy(POLICY_PATH)

        self.assertEqual(policy.schema_version, 1)
        self.assertEqual(
            tuple(policy.datasets),
            ("medtimeline", "medtimeline-fidelity-v2"),
        )
        self.assertEqual(
            policy.datasets["medtimeline"],
            fixturectl.DatasetPolicy(
                dataset_id="medtimeline",
                allowed_top_level=("raw",),
                max_files=10_000,
                max_total_bytes=268_435_456,
                destination="/srv/private-fixtures/medtimeline",
            ),
        )
        self.assertEqual(
            policy.datasets["medtimeline-fidelity-v2"],
            fixturectl.DatasetPolicy(
                dataset_id="medtimeline-fidelity-v2",
                allowed_top_level=("Run-v5-71a2f50",),
                max_files=20_000,
                max_total_bytes=3_221_225_472,
                destination="/srv/private-fixtures/medtimeline-fidelity-v2",
            ),
        )
        self.assertNotIn("/Users/", POLICY_PATH.read_text(encoding="utf-8"))

    def test_duplicate_json_key_is_rejected(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "policy.json"
        path.write_text(
            '{"schema_version":1,"schema_version":1,"datasets":{}}',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(fixturectl.FixturePolicyError, "duplicate key"):
            fixturectl.load_policy(path)

    def test_unknown_policy_field_is_rejected(self) -> None:
        body = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        body["unexpected"] = True

        with self.assertRaisesRegex(fixturectl.FixturePolicyError, "policy fields"):
            fixturectl.load_policy(self._write_policy(body))

    def test_dataset_key_and_identifier_must_be_safe(self) -> None:
        body = {
            "schema_version": 1,
            "datasets": {
                "../private": {
                    "allowed_top_level": ["raw"],
                    "max_files": 1,
                    "max_total_bytes": 1,
                    "destination": "/srv/private-fixtures/private",
                }
            },
        }

        with self.assertRaisesRegex(fixturectl.FixturePolicyError, "dataset id"):
            fixturectl.load_policy(self._write_policy(body))

    def test_allowed_top_level_is_one_normalized_component(self) -> None:
        body = {
            "schema_version": 1,
            "datasets": {
                "synthetic": {
                    "allowed_top_level": ["raw/child"],
                    "max_files": 1,
                    "max_total_bytes": 1,
                    "destination": "/srv/private-fixtures/synthetic",
                }
            },
        }

        with self.assertRaisesRegex(fixturectl.FixturePolicyError, "top-level"):
            fixturectl.load_policy(self._write_policy(body))

    def test_boolean_or_nonpositive_limits_are_rejected(self) -> None:
        body = {
            "schema_version": 1,
            "datasets": {
                "synthetic": {
                    "allowed_top_level": ["raw"],
                    "max_files": True,
                    "max_total_bytes": 0,
                    "destination": "/srv/private-fixtures/synthetic",
                }
            },
        }

        with self.assertRaisesRegex(fixturectl.FixturePolicyError, "limit"):
            fixturectl.load_policy(self._write_policy(body))

    def test_destination_must_be_dataset_specific_absolute_target(self) -> None:
        body = {
            "schema_version": 1,
            "datasets": {
                "synthetic": {
                    "allowed_top_level": ["raw"],
                    "max_files": 1,
                    "max_total_bytes": 1,
                    "destination": "relative/private-fixtures",
                }
            },
        }

        with self.assertRaisesRegex(fixturectl.FixturePolicyError, "destination"):
            fixturectl.load_policy(self._write_policy(body))


if __name__ == "__main__":
    unittest.main()
