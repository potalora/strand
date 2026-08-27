from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
from datetime import UTC, datetime
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import unittest
from unittest import mock


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

import fixturectl  # noqa: E402


FIXED_TIME = datetime(2026, 8, 27, 0, 0, tzinfo=UTC)
POLICY_PATH = TOOL_ROOT / "policy.json"


def synthetic_policy(**changes: object) -> fixturectl.DatasetPolicy:
    policy = fixturectl.DatasetPolicy(
        dataset_id="synthetic",
        allowed_top_level=("raw",),
        max_files=20,
        max_total_bytes=1_000_000,
        destination="/srv/private-fixtures/synthetic",
    )
    return replace(policy, **changes)


def make_synthetic_dataset(base: Path) -> Path:
    root = base / "dataset"
    raw = root / "raw"
    raw.mkdir(parents=True)
    root.chmod(0o700)
    raw.chmod(0o700)
    record = raw / "record.json"
    record.write_text('{"resourceType":"Synthetic"}\n', encoding="utf-8")
    record.chmod(0o600)
    empty = raw / "empty.txt"
    empty.write_bytes(b"")
    empty.chmod(0o600)
    return root


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = make_synthetic_dataset(Path(self.directory.name))

    def _manifest_bytes(self) -> bytes:
        return fixturectl.canonical_json(
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)
        )

    def _write_manifest(self, path: Path) -> Path:
        path.write_bytes(self._manifest_bytes())
        path.chmod(0o600)
        return path

    def test_manifest_is_deterministic_and_canonical(self) -> None:
        manifest = fixturectl.build_manifest(
            synthetic_policy(),
            self.root,
            FIXED_TIME,
        )
        first = fixturectl.canonical_json(manifest)
        second = fixturectl.canonical_json(
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)
        )

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertEqual(manifest.created_at, "2026-08-27T00:00:00Z")
        self.assertEqual(
            [item.path for item in manifest.files],
            ["raw/empty.txt", "raw/record.json"],
        )
        self.assertEqual(manifest.files[0].size, 0)
        self.assertTrue(all(item.mode == "0600" for item in manifest.files))
        self.assertEqual(manifest.total_bytes, manifest.files[1].size)

    def test_verify_accepts_exact_bytes_and_modes(self) -> None:
        manifest = fixturectl.build_manifest(
            synthetic_policy(),
            self.root,
            FIXED_TIME,
        )

        fixturectl.verify_manifest(manifest, synthetic_policy(), self.root)

    def test_root_directory_must_be_private_and_owned(self) -> None:
        self.root.chmod(0o755)

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "root mode",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_root_directory_must_not_be_a_symlink(self) -> None:
        linked_root = Path(self.directory.name) / "linked-dataset"
        linked_root.symlink_to(self.root, target_is_directory=True)

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "real directory",
        ):
            fixturectl.build_manifest(synthetic_policy(), linked_root, FIXED_TIME)

    def test_nested_directory_must_be_private(self) -> None:
        (self.root / "raw").chmod(0o750)

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "directory mode",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_regular_file_must_be_private(self) -> None:
        (self.root / "raw/record.json").chmod(0o640)

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "file mode",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_symlink_is_rejected(self) -> None:
        (self.root / "raw/link").symlink_to(self.root / "raw/record.json")

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "symlink forbidden",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_hardlink_is_rejected(self) -> None:
        os.link(
            self.root / "raw/record.json",
            self.root / "raw/record-copy.json",
        )

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "hardlink forbidden",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_fifo_is_rejected(self) -> None:
        fifo = self.root / "raw/pipe"
        os.mkfifo(fifo, 0o600)

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "fifo forbidden",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_socket_is_rejected(self) -> None:
        path = self.root / "raw/socket"
        server = socket.socket(socket.AF_UNIX)
        self.addCleanup(server.close)
        server.bind(str(path))

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "socket forbidden",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_unexpected_top_level_path_is_rejected(self) -> None:
        other = self.root / "other"
        other.mkdir(mode=0o700)
        item = other / "record.txt"
        item.write_text("synthetic\n", encoding="utf-8")
        item.chmod(0o600)

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "top-level path forbidden",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_case_collision_is_rejected(self) -> None:
        other = self.root / "raw/RECORD.JSON"
        other.write_text("synthetic\n", encoding="utf-8")
        other.chmod(0o600)
        if other.samefile(self.root / "raw/record.json"):
            self.skipTest(
                "filesystem is case-insensitive; mapping collision test covers this"
            )

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "path collision",
        ):
            fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_file_mutation_during_hashing_is_rejected(self) -> None:
        real_fstat = os.fstat
        calls = 0

        class ChangedMetadata:
            def __init__(self, original: os.stat_result) -> None:
                self._original = original

            def __getattr__(self, name: str) -> object:
                if name == "st_mtime_ns":
                    return self._original.st_mtime_ns + 1
                return getattr(self._original, name)

        def changing_fstat(descriptor: int) -> os.stat_result | ChangedMetadata:
            nonlocal calls
            calls += 1
            metadata = real_fstat(descriptor)
            return metadata if calls == 1 else ChangedMetadata(metadata)

        with mock.patch.object(fixturectl.os, "fstat", side_effect=changing_fstat):
            with self.assertRaisesRegex(
                fixturectl.FixturePolicyError,
                "changed during manifest creation",
            ):
                fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_directory_swap_to_symlink_during_scan_is_rejected(self) -> None:
        nested = self.root / "raw/nested"
        nested.mkdir(mode=0o700)
        payload = nested / "inside.txt"
        payload.write_text("synthetic\n", encoding="utf-8")
        payload.chmod(0o600)
        outside = Path(self.directory.name) / "outside"
        outside.mkdir(mode=0o700)
        outside_payload = outside / "outside.txt"
        outside_payload.write_text("must not be read\n", encoding="utf-8")
        outside_payload.chmod(0o600)
        moved = self.root / "raw/nested-before-swap"
        real_stat = os.stat
        swapped = False

        def swapping_stat(
            path: str | bytes | int,
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            nonlocal swapped
            metadata = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            if path == "nested" and dir_fd is not None and not swapped:
                swapped = True
                nested.rename(moved)
                nested.symlink_to(outside, target_is_directory=True)
            return metadata

        with mock.patch.object(fixturectl.os, "stat", side_effect=swapping_stat):
            with self.assertRaisesRegex(
                fixturectl.FixturePolicyError,
                "opened safely",
            ):
                fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)

    def test_file_and_size_limits_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "file limit",
        ):
            fixturectl.build_manifest(
                synthetic_policy(max_files=1),
                self.root,
                FIXED_TIME,
            )

        with self.assertRaisesRegex(
            fixturectl.FixturePolicyError,
            "byte limit",
        ):
            fixturectl.build_manifest(
                synthetic_policy(max_total_bytes=1),
                self.root,
                FIXED_TIME,
            )

    def test_changed_missing_and_extra_bytes_are_rejected(self) -> None:
        manifest = fixturectl.build_manifest(
            synthetic_policy(),
            self.root,
            FIXED_TIME,
        )
        record = self.root / "raw/record.json"
        record.write_text('{"resourceType":"Changed"}\n', encoding="utf-8")
        record.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mismatch"):
            fixturectl.verify_manifest(manifest, synthetic_policy(), self.root)

        record.unlink()
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mismatch"):
            fixturectl.verify_manifest(manifest, synthetic_policy(), self.root)

        record.write_text('{"resourceType":"Synthetic"}\n', encoding="utf-8")
        record.chmod(0o600)
        extra = self.root / "raw/extra.txt"
        extra.write_text("synthetic\n", encoding="utf-8")
        extra.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mismatch"):
            fixturectl.verify_manifest(manifest, synthetic_policy(), self.root)

    def test_manifest_mapping_rejects_unsafe_paths_and_schema_additions(self) -> None:
        base = json.loads(
            fixturectl.canonical_json(
                fixturectl.build_manifest(
                    synthetic_policy(),
                    self.root,
                    FIXED_TIME,
                )
            )
        )
        for unsafe in (
            "/absolute",
            "../escape",
            ".",
            "raw/../escape",
            "raw/./escape",
            "raw//escape",
            "raw/escape/",
        ):
            changed = json.loads(json.dumps(base))
            changed["files"][0]["path"] = unsafe
            with self.subTest(unsafe=unsafe):
                with self.assertRaisesRegex(
                    fixturectl.FixtureManifestError,
                    "manifest path",
                ):
                    fixturectl.manifest_from_mapping(changed)

        changed = json.loads(json.dumps(base))
        changed["unexpected"] = True
        with self.assertRaisesRegex(
            fixturectl.FixtureManifestError,
            "manifest fields",
        ):
            fixturectl.manifest_from_mapping(changed)

    def test_manifest_mapping_rejects_unicode_and_case_collisions(self) -> None:
        base = json.loads(
            fixturectl.canonical_json(
                fixturectl.build_manifest(
                    synthetic_policy(),
                    self.root,
                    FIXED_TIME,
                )
            )
        )
        template = base["files"][0]
        for paths in (
            ("raw/name.txt", "raw/NAME.txt"),
            ("raw/e\u0301.txt", "raw/é.txt"),
        ):
            changed = json.loads(json.dumps(base))
            changed["files"] = [
                {**template, "path": paths[0]},
                {**template, "path": paths[1]},
            ]
            changed["file_count"] = 2
            changed["total_bytes"] = template["size"] * 2
            with self.subTest(paths=paths):
                with self.assertRaisesRegex(
                    fixturectl.FixtureManifestError,
                    "path collision",
                ):
                    fixturectl.manifest_from_mapping(changed)

    def test_manifest_mapping_rejects_device_metadata_and_count_mismatch(self) -> None:
        base = json.loads(
            fixturectl.canonical_json(
                fixturectl.build_manifest(
                    synthetic_policy(),
                    self.root,
                    FIXED_TIME,
                )
            )
        )
        changed = json.loads(json.dumps(base))
        changed["files"][0]["mode"] = oct(stat.S_IFCHR | 0o600)
        with self.assertRaisesRegex(
            fixturectl.FixtureManifestError,
            "file mode",
        ):
            fixturectl.manifest_from_mapping(changed)

        changed = json.loads(json.dumps(base))
        changed["file_count"] += 1
        with self.assertRaisesRegex(
            fixturectl.FixtureManifestError,
            "file count",
        ):
            fixturectl.manifest_from_mapping(changed)

    def test_naive_creation_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            fixturectl.FixtureManifestError,
            "created_at",
        ):
            fixturectl.build_manifest(
                synthetic_policy(),
                self.root,
                datetime(2026, 8, 27),
            )

    def test_load_manifest_requires_a_stable_private_regular_file(self) -> None:
        manifest = self._write_manifest(Path(self.directory.name) / "manifest.json")
        self.assertEqual(fixturectl.load_manifest(manifest).dataset, "synthetic")

        manifest.chmod(0o640)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "mode"):
            fixturectl.load_manifest(manifest)
        manifest.chmod(0o600)

        linked = Path(self.directory.name) / "linked.json"
        linked.symlink_to(manifest)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "regular file"):
            fixturectl.load_manifest(linked)

        hardlinked = Path(self.directory.name) / "hardlinked.json"
        os.link(manifest, hardlinked)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "hardlink"):
            fixturectl.load_manifest(hardlinked)

    def test_load_manifest_rejects_fifo_socket_wrong_owner_and_mutation(self) -> None:
        fifo = Path(self.directory.name) / "manifest.pipe"
        os.mkfifo(fifo, 0o600)
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "regular file"):
            fixturectl.load_manifest(fifo)

        socket_path = Path(self.directory.name) / "manifest.socket"
        server = socket.socket(socket.AF_UNIX)
        self.addCleanup(server.close)
        server.bind(str(socket_path))
        with self.assertRaisesRegex(fixturectl.FixtureManifestError, "regular file"):
            fixturectl.load_manifest(socket_path)

        manifest = self._write_manifest(Path(self.directory.name) / "owned.json")
        with mock.patch.object(fixturectl.os, "getuid", return_value=os.getuid() + 1):
            with self.assertRaisesRegex(fixturectl.FixtureManifestError, "owner"):
                fixturectl.load_manifest(manifest)

        real_fstat = os.fstat
        calls = 0

        class ChangedMetadata:
            def __init__(self, original: os.stat_result) -> None:
                self._original = original

            def __getattr__(self, name: str) -> object:
                if name == "st_ctime_ns":
                    return self._original.st_ctime_ns + 1
                return getattr(self._original, name)

        def changing_fstat(descriptor: int) -> os.stat_result | ChangedMetadata:
            nonlocal calls
            calls += 1
            metadata = real_fstat(descriptor)
            return metadata if calls == 1 else ChangedMetadata(metadata)

        with mock.patch.object(fixturectl.os, "fstat", side_effect=changing_fstat):
            with self.assertRaisesRegex(fixturectl.FixtureManifestError, "changed"):
                fixturectl.load_manifest(manifest)

    def test_cli_writes_private_output_outside_dataset_and_verifies_it(self) -> None:
        output = Path(self.directory.name) / "manifest.json"
        create_args = [
            "--policy",
            str(POLICY_PATH),
            "manifest",
            "create",
            "--dataset",
            "medtimeline",
            "--root",
            str(self.root),
            "--output",
            str(output),
        ]
        self.assertEqual(fixturectl.main(create_args), 0)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(
            fixturectl.main(
                [
                    "--policy",
                    str(POLICY_PATH),
                    "manifest",
                    "verify",
                    "--dataset",
                    "medtimeline",
                    "--root",
                    str(self.root),
                    "--manifest",
                    str(output),
                ]
            ),
            0,
        )

    def test_cli_refuses_output_inside_dataset_without_path_leakage(self) -> None:
        output = self.root / "raw/sensitive-manifest-name.json"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                fixturectl.main(
                    [
                        "--policy",
                        str(POLICY_PATH),
                        "manifest",
                        "create",
                        "--dataset",
                        "medtimeline",
                        "--root",
                        str(self.root),
                        "--output",
                        str(output),
                    ]
                )
        self.assertFalse(output.exists())
        self.assertEqual(stderr.getvalue(), "fixturectl: validation failed\n")

    def test_cli_containment_uses_the_scanned_root_identity(self) -> None:
        output = self.root / "raw/manifest.json"
        moved_root = Path(self.directory.name) / "dataset-moved"
        real_ancestry_check = fixturectl._directory_is_within

        def swap_root_then_check(candidate_fd: int, root_fd: int) -> bool:
            self.root.rename(moved_root)
            replacement = Path(self.directory.name) / "dataset"
            replacement_raw = replacement / "raw"
            replacement_raw.mkdir(parents=True)
            replacement.chmod(0o700)
            replacement_raw.chmod(0o700)
            return real_ancestry_check(candidate_fd, root_fd)

        stderr = io.StringIO()
        with mock.patch.object(
            fixturectl,
            "_directory_is_within",
            side_effect=swap_root_then_check,
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    fixturectl.main(
                        [
                            "--policy",
                            str(POLICY_PATH),
                            "manifest",
                            "create",
                            "--dataset",
                            "medtimeline",
                            "--root",
                            str(self.root),
                            "--output",
                            str(output),
                        ]
                    )
        self.assertFalse((moved_root / "raw/manifest.json").exists())
        self.assertFalse(output.exists())
        self.assertEqual(stderr.getvalue(), "fixturectl: validation failed\n")

    def test_cli_redacts_root_disappearance_and_ancestry_errors(self) -> None:
        moved_root = Path(self.directory.name) / "dataset-disappeared"
        output = Path(self.directory.name) / "manifest.json"
        real_build = fixturectl._build_manifest

        def disappear_before_build(
            *args: object, **kwargs: object
        ) -> fixturectl.Manifest:
            self.root.rename(moved_root)
            return real_build(*args, **kwargs)

        stderr = io.StringIO()
        with mock.patch.object(
            fixturectl,
            "_build_manifest",
            side_effect=disappear_before_build,
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    fixturectl.main(
                        [
                            "--policy",
                            str(POLICY_PATH),
                            "manifest",
                            "create",
                            "--dataset",
                            "medtimeline",
                            "--root",
                            str(self.root),
                            "--output",
                            str(output),
                        ]
                    )
        self.assertEqual(stderr.getvalue(), "fixturectl: validation failed\n")

        self.root = moved_root
        stderr = io.StringIO()
        output = self.root / "raw/manifest.json"
        with mock.patch.object(
            fixturectl,
            "_directory_is_within",
            side_effect=FileNotFoundError("synthetic-sensitive-path"),
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    fixturectl.main(
                        [
                            "--policy",
                            str(POLICY_PATH),
                            "manifest",
                            "create",
                            "--dataset",
                            "medtimeline",
                            "--root",
                            str(self.root),
                            "--output",
                            str(output),
                        ]
                    )
        self.assertEqual(stderr.getvalue(), "fixturectl: validation failed\n")


if __name__ == "__main__":
    unittest.main()
