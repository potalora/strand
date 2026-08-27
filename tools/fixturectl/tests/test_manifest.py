from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
from datetime import UTC, datetime
import io
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import traceback
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


class SignatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.root = make_synthetic_dataset(base)
        manifest = fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)
        self.manifest = base / "manifest.json"
        self.manifest.write_bytes(fixturectl.canonical_json(manifest))
        self.manifest.chmod(0o600)
        self.secret_key = base / "test-only-signing.key"
        self.secret_key.write_text("synthetic-test-key-material\n", encoding="utf-8")
        self.secret_key.chmod(0o600)
        self.public_key = base / "test-only-signing.pub"
        self.public_key.write_text("synthetic-test-public-key\n", encoding="utf-8")
        self.public_key.chmod(0o644)
        self.signature = base / "manifest.minisig"

    def _successful_sign(
        self, args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.assertNotIn("synthetic-test-key-material", " ".join(args))
        signature_fd = int(args[args.index("-x") + 1].rsplit("/", 1)[1])
        os.ftruncate(signature_fd, 0)
        os.lseek(signature_fd, 0, os.SEEK_SET)
        os.write(signature_fd, b"synthetic detached signature\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    def test_sign_manifest_uses_safe_argv_and_private_key(self) -> None:
        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            side_effect=self._successful_sign,
        ) as run:
            fixturectl.sign_manifest(self.manifest, self.secret_key, self.signature)

        args, kwargs = run.call_args
        command = args[0]
        self.assertEqual(command[:3], ["minisign", "-S", "-s"])
        self.assertTrue(
            all(command[index].startswith("/dev/fd/") for index in (3, 5, 7))
        )
        self.assertNotIn(str(self.secret_key), command)
        self.assertNotIn(str(self.manifest), command)
        self.assertNotIn(str(self.signature), command)
        self.assertEqual(
            set(kwargs["pass_fds"]),
            {int(item.rsplit("/", 1)[1]) for item in command[3::2]},
        )
        self.assertEqual(kwargs["timeout"], 60)
        self.assertIsNone(kwargs["stdin"])
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertTrue(kwargs["text"])
        self.assertEqual(stat.S_IMODE(self.signature.stat().st_mode), 0o600)

    def test_verify_signature_uses_only_public_key_file(self) -> None:
        self.signature.write_text("synthetic detached signature\n", encoding="utf-8")
        self.signature.chmod(0o600)
        with mock.patch.object(fixturectl.subprocess, "run") as run:
            verified = fixturectl.verify_signature(
                self.manifest,
                self.signature,
                self.public_key,
            )

        args, kwargs = run.call_args
        command = args[0]
        self.assertEqual(command[:4], ["minisign", "-V", "-q", "-p"])
        self.assertTrue(
            all(command[index].startswith("/dev/fd/") for index in (4, 6, 8))
        )
        self.assertEqual(
            set(kwargs["pass_fds"]),
            {int(item.rsplit("/", 1)[1]) for item in command[4::2]},
        )
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertEqual(verified.manifest.dataset, "synthetic")
        self.assertEqual(
            verified.sha256,
            fixturectl.hashlib.sha256(verified.canonical_bytes).hexdigest(),
        )

    def test_invalid_signature_wrong_key_and_absent_signature_fail_closed(self) -> None:
        self.signature.write_text("synthetic detached signature\n", encoding="utf-8")
        self.signature.chmod(0o600)
        for failure in (
            subprocess.CalledProcessError(1, ["minisign"]),
            subprocess.CalledProcessError(1, ["minisign"], stderr="wrong key"),
        ):
            with self.subTest(failure=failure):
                with mock.patch.object(
                    fixturectl.subprocess,
                    "run",
                    side_effect=failure,
                ):
                    with self.assertRaisesRegex(
                        fixturectl.FixtureSignatureError,
                        "signature verification failed",
                    ):
                        fixturectl.verify_signature(
                            self.manifest,
                            self.signature,
                            self.public_key,
                        )

        self.signature.unlink()
        with self.assertRaisesRegex(
            fixturectl.FixtureSignatureError,
            "signature file",
        ):
            fixturectl.verify_signature(
                self.manifest,
                self.signature,
                self.public_key,
            )

    def test_signing_rejects_insecure_key_and_timeout_without_leaking(self) -> None:
        self.secret_key.chmod(0o640)
        with self.assertRaisesRegex(
            fixturectl.FixtureSignatureError,
            "secret key mode",
        ):
            fixturectl.sign_manifest(self.manifest, self.secret_key, self.signature)
        self.secret_key.chmod(0o600)

        def partial_timeout(args: list[str], **kwargs: object) -> None:
            signature_fd = int(args[args.index("-x") + 1].rsplit("/", 1)[1])
            os.write(signature_fd, b"partial signature")
            raise subprocess.TimeoutExpired(
                ["minisign"],
                60,
                output="synthetic-test-key-material",
                stderr="synthetic-passphrase",
            )

        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            side_effect=partial_timeout,
        ):
            with self.assertRaises(fixturectl.FixtureSignatureError) as raised:
                fixturectl.sign_manifest(
                    self.manifest,
                    self.secret_key,
                    self.signature,
                )
        rendered = str(raised.exception)
        self.assertNotIn("synthetic-test-key-material", rendered)
        self.assertNotIn("synthetic-passphrase", rendered)
        self.assertEqual(rendered, "signing command timed out")
        formatted = "".join(
            traceback.format_exception(
                type(raised.exception),
                raised.exception,
                raised.exception.__traceback__,
            )
        )
        self.assertNotIn("synthetic-test-key-material", formatted)
        self.assertNotIn("synthetic-passphrase", formatted)
        context = raised.exception.__context__
        if context is not None:
            self.assertIsNone(getattr(context, "output", None))
            self.assertIsNone(getattr(context, "stderr", None))
        self.assertFalse(self.signature.exists())
        self.assertEqual(
            list(self.signature.parent.glob(".fixturectl-signature-*.tmp")), []
        )

    def test_signature_output_races_empty_and_oversize_fail_cleanly(self) -> None:
        protected = Path(self.directory.name) / "protected.txt"
        protected.write_text("unchanged\n", encoding="utf-8")
        protected.chmod(0o600)

        def precreate_symlink(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            self.signature.symlink_to(protected)
            return self._successful_sign(args, **kwargs)

        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            side_effect=precreate_symlink,
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureSignatureError,
                "installation failed",
            ):
                fixturectl.sign_manifest(
                    self.manifest,
                    self.secret_key,
                    self.signature,
                )
        self.assertEqual(protected.read_text(encoding="utf-8"), "unchanged\n")
        self.signature.unlink()
        self.assertEqual(
            list(self.signature.parent.glob(".fixturectl-signature-*.tmp")), []
        )

    def test_signature_temp_setup_failure_is_cleaned(self) -> None:
        with mock.patch.object(
            fixturectl.os,
            "fchmod",
            side_effect=OSError("synthetic setup failure"),
        ):
            with self.assertRaises(fixturectl.FixtureSignatureError):
                fixturectl.sign_manifest(
                    self.manifest,
                    self.secret_key,
                    self.signature,
                )
        self.assertFalse(self.signature.exists())
        self.assertEqual(
            list(self.signature.parent.glob(".fixturectl-signature-*.tmp")), []
        )

        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["minisign"], 0),
        ):
            with self.assertRaisesRegex(fixturectl.FixtureSignatureError, "changed"):
                fixturectl.sign_manifest(
                    self.manifest,
                    self.secret_key,
                    self.signature,
                )
        self.assertFalse(self.signature.exists())

        def oversized_signature(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            signature_fd = int(args[args.index("-x") + 1].rsplit("/", 1)[1])
            os.write(signature_fd, b"x" * (fixturectl._MAX_SIGNATURE_BYTES + 1))
            return subprocess.CompletedProcess(args, 0)

        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            side_effect=oversized_signature,
        ):
            with self.assertRaisesRegex(fixturectl.FixtureSignatureError, "changed"):
                fixturectl.sign_manifest(
                    self.manifest,
                    self.secret_key,
                    self.signature,
                )
        self.assertFalse(self.signature.exists())
        self.assertEqual(
            list(self.signature.parent.glob(".fixturectl-signature-*.tmp")), []
        )

    def test_sign_and_verify_reject_path_swaps_without_installing_output(self) -> None:
        replacement_manifest = self.manifest.with_name("replacement-manifest.json")
        replacement_manifest.write_text("{}\n", encoding="utf-8")
        replacement_manifest.chmod(0o600)

        def swap_during_sign(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            replacement_manifest.replace(self.manifest)
            return self._successful_sign(args, **kwargs)

        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            side_effect=swap_during_sign,
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureSignatureError,
                "manifest changed",
            ):
                fixturectl.sign_manifest(
                    self.manifest,
                    self.secret_key,
                    self.signature,
                )
        self.assertFalse(self.signature.exists())

        self.manifest.write_bytes(
            fixturectl.canonical_json(
                fixturectl.build_manifest(synthetic_policy(), self.root, FIXED_TIME)
            )
        )
        self.manifest.chmod(0o600)
        self.signature.write_text("synthetic detached signature\n", encoding="utf-8")
        self.signature.chmod(0o600)
        replacement_public_key = self.public_key.with_name("replacement.pub")
        replacement_public_key.write_text("replacement public key\n", encoding="utf-8")
        replacement_public_key.chmod(0o644)

        def swap_during_verify(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            replacement_public_key.replace(self.public_key)
            return subprocess.CompletedProcess(args, 0, "", "")

        with mock.patch.object(
            fixturectl.subprocess,
            "run",
            side_effect=swap_during_verify,
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureSignatureError,
                "public key changed",
            ):
                fixturectl.verify_signature(
                    self.manifest,
                    self.signature,
                    self.public_key,
                )


class ManifestRaceAndMinisignIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = make_synthetic_dataset(Path(self.directory.name))

    @unittest.skipUnless(shutil.which("minisign"), "Minisign is not installed")
    def test_temporary_key_round_trip_and_tamper_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = make_synthetic_dataset(base)
            manifest_path = base / "manifest.json"
            manifest_path.write_bytes(
                fixturectl.canonical_json(
                    fixturectl.build_manifest(synthetic_policy(), root, FIXED_TIME)
                )
            )
            manifest_path.chmod(0o600)
            secret_key = base / "test-only.key"
            public_key = base / "test-only.pub"
            signature = base / "manifest.minisig"
            subprocess.run(
                [
                    "minisign",
                    "-G",
                    "-W",
                    "-s",
                    str(secret_key),
                    "-p",
                    str(public_key),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            secret_key.chmod(0o600)
            public_key.chmod(0o644)

            fixturectl.sign_manifest(manifest_path, secret_key, signature)
            fixturectl.verify_signature(manifest_path, signature, public_key)

            changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_manifest["created_at"] = "2026-08-27T00:00:01Z"
            manifest_path.write_bytes(
                (
                    json.dumps(
                        changed_manifest,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            manifest_path.chmod(0o600)
            with self.assertRaises(fixturectl.FixtureSignatureError):
                fixturectl.verify_signature(manifest_path, signature, public_key)

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
