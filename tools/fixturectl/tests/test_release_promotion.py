from __future__ import annotations

from datetime import UTC, datetime
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

import fixturectl  # noqa: E402
from test_manifest import make_synthetic_dataset, synthetic_policy  # noqa: E402


FIXED_TIME = datetime(2026, 8, 27, 0, 0, tzinfo=UTC)


def _process_promote(
    dataset_root: Path,
    staging: Path,
    manifest_sha: str,
    policy: fixturectl.DatasetPolicy,
    public_key: Path,
    verified: fixturectl.VerifiedManifest,
    start: object,
    results: object,
) -> None:
    start.wait()
    try:
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=verified,
        ):
            release = fixturectl.promote(
                dataset_root,
                staging,
                manifest_sha,
                policy,
                public_key,
            )
        results.put(("ok", str(release)))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.source = make_synthetic_dataset(base / "source-parent")
        self.dataset_root = base / "target" / "synthetic"
        (self.dataset_root / "incoming").mkdir(parents=True)
        (self.dataset_root / "releases").mkdir()
        for directory in (
            self.dataset_root,
            self.dataset_root / "incoming",
            self.dataset_root / "releases",
        ):
            directory.chmod(0o700)
        self.policy = synthetic_policy(destination=str(self.dataset_root))
        manifest = fixturectl.build_manifest(self.policy, self.source, FIXED_TIME)
        canonical = fixturectl.canonical_json(manifest)
        self.signature_bytes = b"synthetic signature\n"
        self.verified = fixturectl.VerifiedManifest(
            manifest=manifest,
            canonical_bytes=canonical,
            sha256=fixturectl.hashlib.sha256(canonical).hexdigest(),
            signature_sha256=fixturectl.hashlib.sha256(
                self.signature_bytes
            ).hexdigest(),
        )
        self.public_key = base / "test.pub"
        self.public_key.write_text("synthetic public key\n", encoding="utf-8")
        self.public_key.chmod(0o644)

    def _stage(self, transfer_id: str) -> tuple[Path, Path, Path]:
        staging = fixturectl.create_staging(self.dataset_root, transfer_id)
        shutil.copytree(self.source, staging / "data", copy_function=shutil.copyfile)
        for directory in (staging / "data", staging / "data/raw"):
            directory.chmod(0o700)
        for file in (staging / "data/raw").iterdir():
            file.chmod(0o600)
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(self.verified.canonical_bytes)
        manifest_path.chmod(0o600)
        signature_path = staging / "manifest.minisig"
        signature_path.write_bytes(self.signature_bytes)
        signature_path.chmod(0o600)
        return staging, manifest_path, signature_path

    def _receive(self, transfer_id: str) -> tuple[Path, fixturectl.Receipt]:
        staging, manifest_path, signature_path = self._stage(transfer_id)
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=self.verified,
        ):
            receipt = fixturectl.receive(
                staging,
                manifest_path,
                signature_path,
                self.public_key,
                self.policy,
                target_id="synthetic-target",
                verified_at=FIXED_TIME,
            )
        return staging, receipt

    def _promote(self, staging: Path, manifest_sha: str) -> Path:
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=self.verified,
        ):
            return fixturectl.promote(
                self.dataset_root,
                staging,
                manifest_sha,
                self.policy,
                self.public_key,
            )

    def test_staging_is_exclusive_and_private(self) -> None:
        transfer_id = "a" * 32
        staging = fixturectl.create_staging(self.dataset_root, transfer_id)
        self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o700)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "staging exists"):
            fixturectl.create_staging(self.dataset_root, transfer_id)

    def test_receive_verifies_and_writes_redacted_private_receipt(self) -> None:
        staging, receipt = self._receive("b" * 32)

        self.assertEqual(receipt.dataset, "synthetic")
        self.assertEqual(receipt.manifest_sha256, self.verified.sha256)
        self.assertEqual(receipt.status, "verified")
        receipt_path = staging / "receipt.json"
        self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
        body = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(body),
            {
                "schema_version",
                "dataset",
                "manifest_sha256",
                "file_count",
                "total_bytes",
                "verified_at",
                "target_id",
                "status",
            },
        )
        rendered = receipt_path.read_text(encoding="utf-8")
        self.assertNotIn("record.json", rendered)
        self.assertNotIn(str(self.source), rendered)

    def test_receive_refuses_interrupted_changed_extra_and_bad_signature(self) -> None:
        for mutation in ("missing", "changed", "extra"):
            with self.subTest(mutation=mutation):
                staging, manifest_path, signature_path = self._stage(
                    {"missing": "c", "changed": "d", "extra": "e"}[mutation] * 32
                )
                record = staging / "data/raw/record.json"
                if mutation == "missing":
                    record.unlink()
                elif mutation == "changed":
                    record.write_text("changed\n", encoding="utf-8")
                    record.chmod(0o600)
                else:
                    extra = staging / "data/raw/extra.txt"
                    extra.write_text("extra synthetic\n", encoding="utf-8")
                    extra.chmod(0o600)
                with mock.patch.object(
                    fixturectl,
                    "verify_signature",
                    return_value=self.verified,
                ):
                    with self.assertRaisesRegex(
                        fixturectl.FixtureManifestError,
                        "mismatch",
                    ):
                        fixturectl.receive(
                            staging,
                            manifest_path,
                            signature_path,
                            self.public_key,
                            self.policy,
                            target_id="synthetic-target",
                            verified_at=FIXED_TIME,
                        )
                self.assertFalse((staging / "receipt.json").exists())

        staging, manifest_path, signature_path = self._stage("f" * 32)
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            side_effect=fixturectl.FixtureSignatureError("invalid"),
        ):
            with self.assertRaises(fixturectl.FixtureSignatureError):
                fixturectl.receive(
                    staging,
                    manifest_path,
                    signature_path,
                    self.public_key,
                    self.policy,
                    target_id="synthetic-target",
                    verified_at=FIXED_TIME,
                )

        extra = staging / "unexpected.txt"
        extra.write_text("synthetic\n", encoding="utf-8")
        extra.chmod(0o600)
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=self.verified,
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError, "staging layout"
            ):
                fixturectl.receive(
                    staging,
                    manifest_path,
                    signature_path,
                    self.public_key,
                    self.policy,
                    target_id="synthetic-target",
                    verified_at=FIXED_TIME,
                )
        self.assertFalse((staging / "receipt.json").exists())

    def test_receive_refuses_target_root_and_staging_extras(self) -> None:
        staging, manifest_path, signature_path = self._stage("1" * 32)
        wrong_policy = synthetic_policy(destination=str(self.dataset_root / "other"))
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=self.verified,
        ):
            with self.assertRaisesRegex(fixturectl.FixtureTransferError, "target root"):
                fixturectl.receive(
                    staging,
                    manifest_path,
                    signature_path,
                    self.public_key,
                    wrong_policy,
                    target_id="synthetic-target",
                    verified_at=FIXED_TIME,
                )

    def test_receive_refuses_staging_swap_before_receipt(self) -> None:
        staging, manifest_path, signature_path = self._stage("6" * 32)
        displaced = staging.with_name(f"{staging.name}.displaced")

        def swap_staging(*_args: object, **_kwargs: object) -> None:
            staging.rename(displaced)
            replacement = staging
            replacement.mkdir(mode=0o700)
            (replacement / "data").mkdir(mode=0o700)
            for name, body in (
                ("manifest.json", self.verified.canonical_bytes),
                ("manifest.minisig", self.signature_bytes),
            ):
                path = replacement / name
                path.write_bytes(body)
                path.chmod(0o600)

        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl,
                "verify_manifest",
                side_effect=swap_staging,
            ),
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "staging directory changed",
            ):
                fixturectl.receive(
                    staging,
                    manifest_path,
                    signature_path,
                    self.public_key,
                    self.policy,
                    target_id="synthetic-target",
                    verified_at=FIXED_TIME,
                )
        self.assertFalse((staging / "receipt.json").exists())
        self.assertFalse((displaced / "receipt.json").exists())

    def test_receive_refuses_manifest_mutation_after_data_verification(self) -> None:
        staging, manifest_path, signature_path = self._stage("9" * 32)
        real_verify_manifest = fixturectl.verify_manifest

        def verify_then_tamper(*args: object, **kwargs: object) -> None:
            real_verify_manifest(*args, **kwargs)
            manifest_path.write_text("{}\n", encoding="utf-8")
            manifest_path.chmod(0o600)

        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl,
                "verify_manifest",
                side_effect=verify_then_tamper,
            ),
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "staging manifest changed",
            ):
                fixturectl.receive(
                    staging,
                    manifest_path,
                    signature_path,
                    self.public_key,
                    self.policy,
                    target_id="synthetic-target",
                    verified_at=FIXED_TIME,
                )
        self.assertFalse((staging / "receipt.json").exists())

    def test_receive_cleans_partial_receipt_after_write_failure(self) -> None:
        staging, manifest_path, signature_path = self._stage("d" * 32)
        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl.os,
                "fsync",
                side_effect=OSError("synthetic write failure"),
            ),
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "receipt could not be written",
            ):
                fixturectl.receive(
                    staging,
                    manifest_path,
                    signature_path,
                    self.public_key,
                    self.policy,
                    target_id="synthetic-target",
                    verified_at=FIXED_TIME,
                )
        self.assertFalse((staging / "receipt.json").exists())

    def test_receipt_parser_refuses_duplicate_keys_and_impossible_time(self) -> None:
        staging, _ = self._receive("7" * 32)
        receipt_path = staging / "receipt.json"
        canonical = receipt_path.read_text(encoding="utf-8").rstrip("\n")
        duplicate = canonical.replace(
            '"dataset":"synthetic",',
            '"dataset":"synthetic","dataset":"synthetic",',
        )
        receipt_path.write_text(duplicate + "\n", encoding="utf-8")
        receipt_path.chmod(0o600)
        with self.assertRaisesRegex(
            fixturectl.FixtureTransferError,
            "duplicate key",
        ):
            fixturectl._load_receipt(receipt_path)

        impossible = canonical.replace(
            '"verified_at":"2026-08-27T00:00:00Z"',
            '"verified_at":"2026-02-31T00:00:00Z"',
        )
        receipt_path.write_text(impossible + "\n", encoding="utf-8")
        receipt_path.chmod(0o600)
        with self.assertRaisesRegex(
            fixturectl.FixtureTransferError,
            "receipt values",
        ):
            fixturectl._load_receipt(receipt_path)

    def test_promote_renames_release_and_atomically_swaps_current(self) -> None:
        staging, _ = self._receive("2" * 32)
        release = self._promote(staging, self.verified.sha256)

        self.assertEqual(release, self.dataset_root / "releases" / self.verified.sha256)
        self.assertTrue(release.is_dir())
        self.assertFalse(staging.exists())
        current = self.dataset_root / "current"
        self.assertTrue(current.is_symlink())
        self.assertEqual(os.readlink(current), f"releases/{self.verified.sha256}")

    def test_promotion_refuses_release_name_mismatch_and_different_existing(
        self,
    ) -> None:
        staging, _ = self._receive("3" * 32)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "release id"):
            self._promote(staging, "0" * 64)
        self.assertTrue(staging.exists())

        existing = self.dataset_root / "releases" / self.verified.sha256
        existing.mkdir(mode=0o700)
        wrong = existing / "manifest.json"
        wrong.write_text("{}\n", encoding="utf-8")
        wrong.chmod(0o600)
        with self.assertRaisesRegex(
            fixturectl.FixtureTransferError, "existing release"
        ):
            self._promote(staging, self.verified.sha256)

    def test_two_concurrent_identical_promotions_converge(self) -> None:
        first, _ = self._receive("4" * 32)
        second, _ = self._receive("5" * 32)
        barrier = threading.Barrier(2)
        releases: list[Path] = []
        errors: list[BaseException] = []

        def promote(staging: Path) -> None:
            try:
                barrier.wait()
                releases.append(
                    fixturectl.promote(
                        self.dataset_root,
                        staging,
                        self.verified.sha256,
                        self.policy,
                        self.public_key,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=promote, args=(first,)),
            threading.Thread(target=promote, args=(second,)),
        ]
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=self.verified,
        ):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(releases), 2)
        self.assertEqual(releases[0], releases[1])
        self.assertEqual(
            os.readlink(self.dataset_root / "current"),
            f"releases/{self.verified.sha256}",
        )

    def test_two_process_promotions_use_the_filesystem_lock(self) -> None:
        first, _ = self._receive("a" * 32)
        second, _ = self._receive("b" * 32)
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_process_promote,
                args=(
                    self.dataset_root,
                    staging,
                    self.verified.sha256,
                    self.policy,
                    self.public_key,
                    self.verified,
                    start,
                    results,
                ),
            )
            for staging in (first, second)
        ]
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=10)

        self.assertEqual([process.exitcode for process in processes], [0, 0])
        outcomes = [results.get(timeout=2) for _ in processes]
        self.assertEqual(
            [status for status, _ in outcomes],
            ["ok", "ok"],
            outcomes,
        )
        self.assertEqual(outcomes[0][1], outcomes[1][1])
        release = self.dataset_root / "releases" / self.verified.sha256
        self.assertTrue(release.is_dir())
        self.assertEqual(
            os.readlink(self.dataset_root / "current"),
            f"releases/{self.verified.sha256}",
        )

    def test_atomic_noreplace_refuses_an_empty_destination_race(self) -> None:
        staging, _ = self._receive("c" * 32)
        release = self.dataset_root / "releases" / self.verified.sha256
        real_rename = fixturectl._rename_noreplace
        raced_inode: list[int] = []

        def create_destination_then_rename(*args: object, **kwargs: object) -> None:
            release.mkdir(mode=0o700)
            raced_inode.append(release.stat().st_ino)
            real_rename(*args, **kwargs)

        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl,
                "_rename_noreplace",
                side_effect=create_destination_then_rename,
            ),
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "release already exists",
            ):
                fixturectl.promote(
                    self.dataset_root,
                    staging,
                    self.verified.sha256,
                    self.policy,
                    self.public_key,
                )
        self.assertTrue(staging.exists())
        self.assertEqual(release.stat().st_ino, raced_inode[0])
        self.assertFalse((self.dataset_root / "current").exists())

    def test_promotion_refuses_staging_swap_before_rename(self) -> None:
        staging, _ = self._receive("8" * 32)
        displaced = staging.with_name(f"{staging.name}.displaced")

        def swap_staging(*_args: object, **_kwargs: object) -> None:
            staging.rename(displaced)
            replacement = staging
            replacement.mkdir(mode=0o700)
            (replacement / "data").mkdir(mode=0o700)
            for name, body in (
                ("manifest.json", self.verified.canonical_bytes),
                ("manifest.minisig", self.signature_bytes),
                ("receipt.json", (displaced / "receipt.json").read_bytes()),
            ):
                path = replacement / name
                path.write_bytes(body)
                path.chmod(0o600)

        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl,
                "verify_manifest",
                side_effect=swap_staging,
            ),
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "staging directory changed",
            ):
                fixturectl.promote(
                    self.dataset_root,
                    staging,
                    self.verified.sha256,
                    self.policy,
                    self.public_key,
                )
        self.assertFalse((self.dataset_root / "current").exists())
        self.assertFalse(
            (self.dataset_root / "releases" / self.verified.sha256).exists()
        )

    def test_promotion_refuses_manifest_mutation_after_data_verification(
        self,
    ) -> None:
        staging, _ = self._receive("a" * 31 + "9")
        manifest_path = staging / "manifest.json"
        real_verify_manifest = fixturectl.verify_manifest

        def verify_then_tamper(*args: object, **kwargs: object) -> None:
            real_verify_manifest(*args, **kwargs)
            manifest_path.write_text("{}\n", encoding="utf-8")
            manifest_path.chmod(0o600)

        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl,
                "verify_manifest",
                side_effect=verify_then_tamper,
            ),
        ):
            with self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "staging manifest changed",
            ):
                fixturectl.promote(
                    self.dataset_root,
                    staging,
                    self.verified.sha256,
                    self.policy,
                    self.public_key,
                )
        self.assertTrue(staging.exists())
        self.assertFalse((self.dataset_root / "current").exists())
        self.assertFalse(
            (self.dataset_root / "releases" / self.verified.sha256).exists()
        )

    def test_promotion_refuses_dataset_root_swap_during_final_verification(
        self,
    ) -> None:
        staging, _ = self._receive("e" * 32)
        detached_root = self.dataset_root.with_name("synthetic-detached")
        real_verify_release = fixturectl._verify_release
        swapped = False

        def swap_root_then_verify(*args: object, **kwargs: object) -> None:
            nonlocal swapped
            if not swapped:
                swapped = True
                self.dataset_root.rename(detached_root)
                shutil.copytree(detached_root, self.dataset_root, symlinks=True)
                tampered = (
                    detached_root / "releases" / self.verified.sha256 / "manifest.json"
                )
                tampered.write_text("{}\n", encoding="utf-8")
                tampered.chmod(0o600)
            real_verify_release(*args, **kwargs)

        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            mock.patch.object(
                fixturectl,
                "_verify_release",
                side_effect=swap_root_then_verify,
            ),
        ):
            with self.assertRaises(fixturectl.FixtureTransferError):
                fixturectl.promote(
                    self.dataset_root,
                    staging,
                    self.verified.sha256,
                    self.policy,
                    self.public_key,
                )
        self.assertTrue(swapped)
        self.assertFalse((self.dataset_root / "current").exists())
        self.assertFalse((detached_root / "current").exists())


if __name__ == "__main__":
    unittest.main()
