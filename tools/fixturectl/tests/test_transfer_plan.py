from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
import stat
import sys
import tempfile
import unittest


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

import fixturectl  # noqa: E402
from test_manifest import make_synthetic_dataset, synthetic_policy  # noqa: E402


FIXED_TIME = datetime(2026, 8, 27, 0, 0, tzinfo=UTC)


class TransferPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.source = make_synthetic_dataset(base)
        self.policy = synthetic_policy(destination="/srv/private-fixtures/synthetic")
        manifest = fixturectl.build_manifest(self.policy, self.source, FIXED_TIME)
        canonical = fixturectl.canonical_json(manifest)
        self.verified = fixturectl.VerifiedManifest(
            manifest=manifest,
            canonical_bytes=canonical,
            sha256=fixturectl.hashlib.sha256(canonical).hexdigest(),
        )
        self.state = base / "state"
        self.state.mkdir(mode=0o700)

    def test_plan_writes_deterministic_nul_file_list_mode_0600(self) -> None:
        plan = fixturectl.plan_transfer(
            self.verified,
            self.policy,
            self.source,
            self.policy.destination,
            self.state,
        )

        self.assertEqual(plan.dataset, "synthetic")
        self.assertEqual(plan.manifest_sha256, self.verified.sha256)
        self.assertEqual(plan.file_count, 2)
        self.assertEqual(plan.total_bytes, self.verified.manifest.total_bytes)
        self.assertEqual(
            plan.files_from.read_bytes(),
            b"raw/empty.txt\0raw/record.json\0",
        )
        self.assertEqual(stat.S_IMODE(plan.files_from.stat().st_mode), 0o600)
        self.assertNotIn(b"\n", plan.files_from.read_bytes())

        second_state = Path(self.directory.name) / "second-state"
        second_state.mkdir(mode=0o700)
        second = fixturectl.plan_transfer(
            self.verified,
            self.policy,
            self.source,
            self.policy.destination,
            second_state,
        )
        self.assertEqual(plan.files_from.read_bytes(), second.files_from.read_bytes())

    def test_plan_rejects_target_mismatch_and_changed_source(self) -> None:
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "destination"):
            fixturectl.plan_transfer(
                self.verified,
                self.policy,
                self.source,
                "/srv/private-fixtures/other",
                self.state,
            )

        changed = self.source / "raw/record.json"
        changed.write_text("changed synthetic bytes\n", encoding="utf-8")
        changed.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mismatch"):
            fixturectl.plan_transfer(
                self.verified,
                self.policy,
                self.source,
                self.policy.destination,
                self.state,
            )

    def test_plan_rejects_missing_and_extra_source_files(self) -> None:
        (self.source / "raw/record.json").unlink()
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mismatch"):
            fixturectl.plan_transfer(
                self.verified,
                self.policy,
                self.source,
                self.policy.destination,
                self.state,
            )

        record = self.source / "raw/record.json"
        record.write_text('{"resourceType":"Synthetic"}\n', encoding="utf-8")
        record.chmod(0o600)
        extra = self.source / "raw/extra.txt"
        extra.write_text("synthetic\n", encoding="utf-8")
        extra.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mismatch"):
            fixturectl.plan_transfer(
                self.verified,
                self.policy,
                self.source,
                self.policy.destination,
                self.state,
            )

    def test_redacted_summary_never_prints_paths_or_filenames(self) -> None:
        plan = fixturectl.plan_transfer(
            self.verified,
            self.policy,
            self.source,
            self.policy.destination,
            self.state,
        )
        summary = fixturectl.render_transfer_summary(
            plan,
            source_alias="owner-mac",
            destination_alias="authorized-test-vps",
            retained_releases=3,
        )

        self.assertIn("dataset: synthetic", summary)
        self.assertIn("source: owner-mac", summary)
        self.assertIn("destination: authorized-test-vps", summary)
        self.assertIn("deletions: 0", summary)
        self.assertNotIn(str(self.source), summary)
        self.assertNotIn("record.json", summary)
        self.assertNotIn("empty.txt", summary)

    def test_transfer_id_is_random_128_bit_lowercase_hex(self) -> None:
        first = fixturectl.new_transfer_id()
        second = fixturectl.new_transfer_id()
        self.assertRegex(first, re.compile(r"^[0-9a-f]{32}$"))
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
