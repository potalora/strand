from __future__ import annotations

from datetime import UTC, datetime
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


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


class TransferCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.source = make_synthetic_dataset(self.base / "source-parent")
        self.destination = "/srv/private-fixtures/synthetic"
        self.policy = synthetic_policy(destination=self.destination)
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
        self.manifest = self.base / "manifest.json"
        self.manifest.write_bytes(canonical)
        self.manifest.chmod(0o600)
        self.signature = self.base / "manifest.minisig"
        self.signature.write_bytes(self.signature_bytes)
        self.signature.chmod(0o600)
        self.public_key = self.base / "fixture-signing.pub"
        self.public_key.write_text("synthetic public key\n", encoding="utf-8")
        self.public_key.chmod(0o644)
        self.state = self.base / "state"
        self.state.mkdir(mode=0o700)
        self.config = self.base / "transfer-config.json"
        self.config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_alias": "owner-mac",
                    "state_directory": str(self.state),
                    "ssh_executable": "/usr/bin/ssh",
                    "rsync_executable": "/opt/homebrew/opt/rsync/bin/rsync",
                    "rsync_version": "3.5.0",
                    "targets": {
                        "authorized-test-vps": {
                            "ssh_destination": "fixture-receiver@100.100.10.20",
                            "target_id": "authorized-test-vps",
                            "remote_fixturectl": "/usr/local/libexec/fixturectl",
                        }
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.config.chmod(0o600)

    def _run(self, *, apply: bool) -> fixturectl.TransferResult:
        with mock.patch.object(
            fixturectl,
            "verify_signature",
            return_value=self.verified,
        ):
            return fixturectl.run_transfer(
                config_path=self.config,
                policy=self.policy,
                dataset="synthetic",
                target_alias="authorized-test-vps",
                source=self.source,
                manifest_path=self.manifest,
                signature_path=self.signature,
                public_key=self.public_key,
                apply=apply,
                retained_releases=3,
                verified_at=FIXED_TIME,
            )

    def _install_fake_binaries(self, receipt: fixturectl.Receipt) -> Path:
        binary_directory = self.base / "fake-bin"
        binary_directory.mkdir(mode=0o700)
        script = textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys

            log = Path(os.environ["FIXTURECTL_FAKE_LOG"])
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
            if Path(sys.argv[0]).name == "ssh":
                if "preflight" in sys.argv:
                    print(os.environ["FIXTURECTL_FAKE_PREFLIGHT"])
                elif "receipt" in sys.argv and "print" in sys.argv:
                    print(os.environ["FIXTURECTL_FAKE_RECEIPT"])
            elif Path(sys.argv[0]).name == "rsync" and "--version" in sys.argv:
                print("rsync  version 3.5.0  protocol version 32")
            """
        )
        for name in ("ssh", "rsync"):
            path = binary_directory / name
            path.write_text(script, encoding="utf-8")
            path.chmod(0o700)
        self.fake_log = self.base / "fake-subprocesses.jsonl"
        self.fake_receipt = fixturectl._receipt_json(receipt).decode("utf-8").strip()
        self.fake_preflight = json.dumps(
            {
                "dataset": "synthetic",
                "protocol_version": 1,
                "status": "ready",
                "target_id": "authorized-test-vps",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        body = json.loads(self.config.read_text(encoding="utf-8"))
        body["ssh_executable"] = str(binary_directory / "ssh")
        body["rsync_executable"] = str(binary_directory / "rsync")
        self.config.write_text(json.dumps(body), encoding="utf-8")
        self.config.chmod(0o600)
        return binary_directory

    def test_default_dry_run_executes_no_network_process(self) -> None:
        with mock.patch.object(fixturectl.subprocess, "run") as run:
            result = self._run(apply=False)

        run.assert_not_called()
        self.assertFalse(result.applied)
        self.assertIn("deletions: 0", result.summary)
        self.assertNotIn(str(self.source), result.summary)
        self.assertNotIn("record.json", result.summary)
        self.assertEqual(stat.S_IMODE(result.plan.files_from.stat().st_mode), 0o600)

    def test_apply_is_one_way_strict_and_verifies_before_promotion(self) -> None:
        receipt = fixturectl.Receipt(
            schema_version=1,
            dataset="synthetic",
            manifest_sha256=self.verified.sha256,
            file_count=self.verified.manifest.file_count,
            total_bytes=self.verified.manifest.total_bytes,
            verified_at="2026-08-27T00:00:00Z",
            target_id="authorized-test-vps",
            status="verified",
        )
        binary_directory = self._install_fake_binaries(receipt)
        environment = {
            "PATH": f"{binary_directory}{os.pathsep}{os.environ['PATH']}",
            "FIXTURECTL_FAKE_LOG": str(self.fake_log),
            "FIXTURECTL_FAKE_PREFLIGHT": self.fake_preflight,
            "FIXTURECTL_FAKE_RECEIPT": self.fake_receipt,
        }
        with mock.patch.dict(os.environ, environment):
            result = self._run(apply=True)

        calls = [
            json.loads(line)
            for line in self.fake_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [call[0] for call in calls],
            [
                "rsync",
                "ssh",
                "ssh",
                "rsync",
                "rsync",
                "ssh",
                "ssh",
                "ssh",
            ],
        )
        ssh_calls = [call for call in calls if call[0] == "ssh"]
        for call in ssh_calls:
            self.assertIn("BatchMode=yes", call)
            self.assertIn("StrictHostKeyChecking=yes", call)
            self.assertIn("fixture-receiver@100.100.10.20", call)
            self.assertIn("/usr/local/libexec/fixturectl", call)
            self.assertNotIn("shell", call)
        rsync_calls = [
            call for call in calls if call[0] == "rsync" and "--version" not in call
        ]
        data_call = rsync_calls[0]
        self.assertIn("--from0", data_call)
        self.assertTrue(any(arg.startswith("--files-from=") for arg in data_call))
        self.assertNotIn("--protect-args", data_call)
        self.assertNotIn("--secluded-args", data_call)
        self.assertTrue(any(arg.startswith("--partial-dir=.") for arg in data_call))
        for call in rsync_calls:
            self.assertNotIn("--delete", call)
            remote_arguments = [
                index
                for index, argument in enumerate(call)
                if argument.startswith("fixture-receiver@100.100.10.20:")
            ]
            self.assertEqual(remote_arguments, [len(call) - 1])
        command_text = [" ".join(call) for call in ssh_calls]
        verify_index = next(
            index
            for index, value in enumerate(command_text)
            if "receive verify" in value
        )
        promote_index = next(
            index
            for index, value in enumerate(command_text)
            if "receive promote" in value
        )
        self.assertLess(verify_index, promote_index)
        self.assertTrue(result.applied)
        self.assertIsNotNone(result.receipt_path)
        assert result.receipt_path is not None
        self.assertEqual(stat.S_IMODE(result.receipt_path.stat().st_mode), 0o600)
        self.assertEqual(
            fixturectl._load_receipt(result.receipt_path).manifest_sha256,
            self.verified.sha256,
        )

    def test_config_requires_private_exact_safe_target_fields(self) -> None:
        self.config.chmod(0o644)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "config mode"):
            fixturectl.load_transfer_config(self.config)
        self.config.chmod(0o600)

        body = json.loads(self.config.read_text(encoding="utf-8"))
        body["unexpected"] = True
        self.config.write_text(json.dumps(body), encoding="utf-8")
        self.config.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "config fields"):
            fixturectl.load_transfer_config(self.config)

        body.pop("unexpected")
        body["targets"]["authorized-test-vps"]["ssh_destination"] = "target;touch bad"
        self.config.write_text(json.dumps(body), encoding="utf-8")
        self.config.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "target values"):
            fixturectl.load_transfer_config(self.config)

        body["targets"]["authorized-test-vps"]["ssh_destination"] = (
            "fixture-receiver@fixture-target"
        )
        self.config.write_text(json.dumps(body), encoding="utf-8")
        self.config.chmod(0o600)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "target values"):
            fixturectl.load_transfer_config(self.config)

    def test_transfer_requires_canonical_metadata_names_before_network(self) -> None:
        wrong_manifest = self.base / "selected-manifest.json"
        wrong_manifest.write_bytes(self.manifest.read_bytes())
        wrong_manifest.chmod(0o600)
        with (
            mock.patch.object(fixturectl.subprocess, "run") as run,
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ),
            self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "canonical name",
            ),
        ):
            fixturectl.run_transfer(
                config_path=self.config,
                policy=self.policy,
                dataset="synthetic",
                target_alias="authorized-test-vps",
                source=self.source,
                manifest_path=wrong_manifest,
                signature_path=self.signature,
                public_key=self.public_key,
                apply=True,
                retained_releases=3,
                verified_at=FIXED_TIME,
            )
        run.assert_not_called()

    def test_transfer_state_must_be_outside_source_before_state_writes(self) -> None:
        nested_state = self.source / "state"
        nested_state.mkdir(mode=0o700)
        body = json.loads(self.config.read_text(encoding="utf-8"))
        body["state_directory"] = str(nested_state)
        self.config.write_text(json.dumps(body), encoding="utf-8")
        self.config.chmod(0o600)
        with (
            mock.patch.object(
                fixturectl,
                "verify_signature",
                return_value=self.verified,
            ) as verify,
            mock.patch.object(fixturectl.subprocess, "run") as run,
            self.assertRaisesRegex(fixturectl.FixtureTransferError, "outside source"),
        ):
            self._run(apply=True)
        verify.assert_not_called()
        run.assert_not_called()
        self.assertEqual(list(nested_state.iterdir()), [])

    def test_apply_uses_long_bulk_timeout_and_short_control_timeout(self) -> None:
        receipt = fixturectl.Receipt(
            schema_version=1,
            dataset="synthetic",
            manifest_sha256=self.verified.sha256,
            file_count=self.verified.manifest.file_count,
            total_bytes=self.verified.manifest.total_bytes,
            verified_at="2026-08-27T00:00:00Z",
            target_id="authorized-test-vps",
            status="verified",
        )
        preflight = json.dumps(
            {
                "dataset": "synthetic",
                "protocol_version": 1,
                "status": "ready",
                "target_id": "authorized-test-vps",
            }
        ).encode()

        def response(arguments: list[str], **kwargs: object) -> bytes:
            if "--version" in arguments:
                return b"rsync  version 3.5.0  protocol version 32\n"
            if "preflight" in arguments:
                return preflight
            if "receipt" in arguments:
                return fixturectl._receipt_json(receipt)
            return b""

        with (
            mock.patch.object(
                fixturectl,
                "_run_transfer_process",
                side_effect=response,
            ) as process,
        ):
            self._run(apply=True)

        rsync_calls = [
            call
            for call in process.call_args_list
            if call.args[0][0] == "/opt/homebrew/opt/rsync/bin/rsync"
            and "--version" not in call.args[0]
        ]
        self.assertEqual(len(rsync_calls), 2)
        self.assertEqual(rsync_calls[0].kwargs["timeout"], 14_400)
        self.assertEqual(rsync_calls[1].kwargs["timeout"], 600)
        control_calls = [
            call for call in process.call_args_list if call.args[0][0] == "/usr/bin/ssh"
        ]
        self.assertTrue(control_calls)
        self.assertTrue(
            all(call.kwargs.get("timeout", 120) == 120 for call in control_calls)
        )

    def test_incompatible_owner_rsync_refuses_before_remote_staging(self) -> None:
        with (
            mock.patch.object(
                fixturectl,
                "_run_transfer_process",
                return_value=(
                    b"openrsync: protocol version 29\nrsync version 2.6.9 compatible\n"
                ),
            ) as process,
            self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "pinned protocol",
            ),
        ):
            self._run(apply=True)
        process.assert_called_once_with(
            ["/opt/homebrew/opt/rsync/bin/rsync", "--version"],
            capture=True,
            timeout=10,
        )

    def test_post_staging_failure_reports_quarantine_recovery(self) -> None:
        preflight = json.dumps(
            {
                "dataset": "synthetic",
                "protocol_version": 1,
                "status": "ready",
                "target_id": "authorized-test-vps",
            }
        ).encode()
        responses: list[bytes | Exception] = [
            b"rsync  version 3.5.0  protocol version 32\n",
            preflight,
            b"",
            fixturectl.FixtureTransferError("redacted failure"),
        ]

        def response(arguments: list[str], **kwargs: object) -> bytes:
            del arguments, kwargs
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        transfer_id = "a" * 32
        with (
            mock.patch.object(
                fixturectl,
                "_run_transfer_process",
                side_effect=response,
            ),
            mock.patch.object(fixturectl, "new_transfer_id", return_value=transfer_id),
            self.assertRaises(fixturectl.FixtureTransferApplyError) as raised,
        ):
            self._run(apply=True)
        rendered = str(raised.exception)
        self.assertIn(f"transfer_id={transfer_id}", rendered)
        self.assertIn("status=quarantined", rendered)
        self.assertIn(
            "fixturectl transfer-status --dataset synthetic "
            "--target authorized-test-vps ",
            rendered,
        )
        self.assertIn(f"--transfer-id {transfer_id}", rendered)
        self.assertIn(f"--manifest-sha {self.verified.sha256}", rendered)

    def test_post_promotion_failures_require_owner_reconciliation(self) -> None:
        good_receipt = fixturectl.Receipt(
            schema_version=1,
            dataset="synthetic",
            manifest_sha256=self.verified.sha256,
            file_count=self.verified.manifest.file_count,
            total_bytes=self.verified.manifest.total_bytes,
            verified_at="2026-08-27T00:00:00Z",
            target_id="authorized-test-vps",
            status="verified",
        )
        wrong_receipt = fixturectl.Receipt(
            **{
                **good_receipt.__dict__,
                "target_id": "wrong-target",
            }
        )

        for failure in ("timeout", "malformed", "mismatch", "local-write"):
            with self.subTest(failure=failure):

                def response(arguments: list[str], **kwargs: object) -> bytes:
                    del kwargs
                    if "--version" in arguments:
                        return b"rsync  version 3.5.0  protocol version 32\n"
                    if "preflight" in arguments:
                        return json.dumps(
                            {
                                "dataset": "synthetic",
                                "protocol_version": 1,
                                "status": "ready",
                                "target_id": "authorized-test-vps",
                            }
                        ).encode()
                    if "receipt" in arguments:
                        if failure == "timeout":
                            raise fixturectl.FixtureTransferError("timeout")
                        if failure == "malformed":
                            return b"not-json"
                        if failure == "mismatch":
                            return fixturectl._receipt_json(wrong_receipt)
                        return fixturectl._receipt_json(good_receipt)
                    return b""

                write_patch = (
                    mock.patch.object(
                        fixturectl,
                        "_write_local_receipt",
                        side_effect=fixturectl.FixtureTransferError("local write"),
                    )
                    if failure == "local-write"
                    else mock.patch.object(
                        fixturectl,
                        "_write_local_receipt",
                        wraps=fixturectl._write_local_receipt,
                    )
                )
                with (
                    mock.patch.object(
                        fixturectl,
                        "_run_transfer_process",
                        side_effect=response,
                    ),
                    write_patch,
                    self.assertRaises(fixturectl.FixtureTransferApplyError) as raised,
                ):
                    self._run(apply=True)
                rendered = str(raised.exception)
                self.assertIn("status=confirmation-pending", rendered)
                self.assertIn("fixturectl transfer-status", rendered)

    def test_subprocess_failure_is_redacted_and_leaves_no_local_receipt(self) -> None:
        receipt = fixturectl.Receipt(
            schema_version=1,
            dataset="synthetic",
            manifest_sha256=self.verified.sha256,
            file_count=self.verified.manifest.file_count,
            total_bytes=self.verified.manifest.total_bytes,
            verified_at="2026-08-27T00:00:00Z",
            target_id="authorized-test-vps",
            status="verified",
        )
        binary_directory = self._install_fake_binaries(receipt)
        failing = binary_directory / "ssh"
        failing.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
        failing.chmod(0o700)
        with mock.patch.dict(
            os.environ,
            {
                "PATH": f"{binary_directory}{os.pathsep}{os.environ['PATH']}",
                "FIXTURECTL_FAKE_LOG": str(self.fake_log),
                "FIXTURECTL_FAKE_PREFLIGHT": self.fake_preflight,
                "FIXTURECTL_FAKE_RECEIPT": self.fake_receipt,
            },
        ):
            with self.assertRaises(fixturectl.FixtureTransferError) as raised:
                self._run(apply=True)
        rendered = str(raised.exception)
        self.assertNotIn(str(self.source), rendered)
        self.assertNotIn("record.json", rendered)
        self.assertNotIn("synthetic write failure", rendered)
        self.assertFalse(any(self.state.rglob("receipt.json")))

    def test_transfer_cli_defaults_to_dry_run(self) -> None:
        result = fixturectl.TransferResult(
            plan=fixturectl.TransferPlan(
                dataset="synthetic",
                source=self.source,
                destination=self.destination,
                manifest_sha256=self.verified.sha256,
                files_from=self.state / "files-from0",
                file_count=self.verified.manifest.file_count,
                total_bytes=self.verified.manifest.total_bytes,
            ),
            summary="dataset: synthetic\ndeletions: 0\n",
            applied=False,
            transfer_id=None,
            receipt_path=None,
        )
        policy_set = fixturectl.Policy(
            schema_version=1, datasets={"synthetic": self.policy}
        )
        stdout = io.StringIO()
        with (
            mock.patch.object(fixturectl, "load_policy", return_value=policy_set),
            mock.patch.object(
                fixturectl,
                "run_transfer",
                return_value=result,
            ) as run_transfer,
            mock.patch.object(sys, "stdout", stdout),
        ):
            exit_code = fixturectl.main(
                [
                    "--policy",
                    str(self.base / "policy.json"),
                    "transfer",
                    "--dataset",
                    "synthetic",
                    "--target",
                    "authorized-test-vps",
                    "--config",
                    str(self.config),
                    "--source",
                    str(self.source),
                    "--manifest",
                    str(self.manifest),
                    "--signature",
                    str(self.signature),
                    "--public-key",
                    str(self.public_key),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse(run_transfer.call_args.kwargs["apply"])
        self.assertIn("deletions: 0", stdout.getvalue())

    def test_owner_status_reports_quarantine_over_strict_ssh(self) -> None:
        transfer_id = "b" * 32
        response = json.dumps(
            {
                "dataset": "synthetic",
                "status": "quarantined",
                "transfer_id": transfer_id,
            }
        ).encode()
        with mock.patch.object(
            fixturectl,
            "_run_transfer_process",
            return_value=response,
        ) as process:
            status, receipt_path = fixturectl.run_transfer_status(
                config_path=self.config,
                dataset="synthetic",
                target_alias="authorized-test-vps",
                transfer_id=transfer_id,
                manifest_sha=self.verified.sha256,
            )
        self.assertEqual(status, "quarantined")
        self.assertIsNone(receipt_path)
        command = process.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/ssh")
        self.assertIn("BatchMode=yes", command)
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("receive", command)
        self.assertIn("status", command)

    def test_owner_status_recovers_promoted_receipt_locally(self) -> None:
        transfer_id = "c" * 32
        status_response = json.dumps(
            {
                "dataset": "synthetic",
                "status": "not-found",
                "transfer_id": transfer_id,
            }
        ).encode()
        receipt = fixturectl.Receipt(
            schema_version=1,
            dataset="synthetic",
            manifest_sha256=self.verified.sha256,
            file_count=self.verified.manifest.file_count,
            total_bytes=self.verified.manifest.total_bytes,
            verified_at="2026-08-27T00:00:00Z",
            target_id="authorized-test-vps",
            status="verified",
        )
        with mock.patch.object(
            fixturectl,
            "_run_transfer_process",
            side_effect=[status_response, fixturectl._receipt_json(receipt)],
        ) as process:
            status, receipt_path = fixturectl.run_transfer_status(
                config_path=self.config,
                dataset="synthetic",
                target_alias="authorized-test-vps",
                transfer_id=transfer_id,
                manifest_sha=self.verified.sha256,
            )
        self.assertEqual(status, "verified")
        self.assertIsNotNone(receipt_path)
        assert receipt_path is not None
        self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
        self.assertEqual(process.call_count, 2)
        self.assertIn("receipt", process.call_args_list[1].args[0])

    def test_receiver_cli_derives_all_staging_paths_from_policy(self) -> None:
        target_root = self.base / "receiver" / "synthetic"
        (target_root / "incoming").mkdir(parents=True)
        (target_root / "releases").mkdir()
        for directory in (
            target_root,
            target_root / "incoming",
            target_root / "releases",
        ):
            directory.chmod(0o700)
        receiver_policy = synthetic_policy(destination=str(target_root))
        policy_set = fixturectl.Policy(
            schema_version=1, datasets={"synthetic": receiver_policy}
        )
        public_key = self.base / "receiver.pub"
        public_key.write_text("synthetic\n", encoding="utf-8")
        public_key.chmod(0o644)
        receiver_config = fixturectl.ReceiverConfig(
            schema_version=1,
            protocol_version=1,
            target_id="authorized-test-vps",
            policy=self.base / "receiver-policy.json",
            public_key=public_key,
            fixturectl_executable="/usr/local/libexec/fixturectl",
            rsync_executable="/usr/bin/rsync",
            rsync_version="3.5.0",
        )
        transfer_id = "f" * 32
        stdout = io.StringIO()
        with (
            mock.patch.object(
                fixturectl,
                "load_receiver_config",
                return_value=receiver_config,
            ),
            mock.patch.object(fixturectl, "load_policy", return_value=policy_set),
            mock.patch.object(sys, "stdout", stdout),
        ):
            fixturectl.main(
                [
                    "receive",
                    "create-staging",
                    "--dataset",
                    "synthetic",
                    "--transfer-id",
                    transfer_id,
                ]
            )
        staging = target_root / "incoming" / transfer_id
        self.assertTrue((staging / "data").is_dir())
        self.assertEqual(stat.S_IMODE((staging / "data").stat().st_mode), 0o700)
        self.assertNotIn(str(target_root), stdout.getvalue())

        with (
            mock.patch.object(
                fixturectl,
                "load_receiver_config",
                return_value=receiver_config,
            ),
            mock.patch.object(fixturectl, "load_policy", return_value=policy_set),
            mock.patch.object(
                fixturectl,
                "receive",
                return_value=fixturectl.Receipt(
                    schema_version=1,
                    dataset="synthetic",
                    manifest_sha256=self.verified.sha256,
                    file_count=2,
                    total_bytes=10,
                    verified_at="2026-08-27T00:00:00Z",
                    target_id="authorized-test-vps",
                    status="verified",
                ),
            ) as receive,
            mock.patch.object(sys, "stdout", io.StringIO()),
        ):
            fixturectl.main(
                [
                    "receive",
                    "verify",
                    "--dataset",
                    "synthetic",
                    "--transfer-id",
                    transfer_id,
                ]
            )
        self.assertEqual(receive.call_args.args[0], staging)
        self.assertEqual(receive.call_args.args[1], staging / "manifest.json")
        self.assertEqual(receive.call_args.args[2], staging / "manifest.minisig")
        self.assertEqual(receive.call_args.args[3], public_key)
        self.assertEqual(
            receive.call_args.kwargs["target_id"],
            "authorized-test-vps",
        )


class ReceiverAuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.policy = synthetic_policy(destination="/srv/private-fixtures/synthetic")
        self.policy_set = fixturectl.Policy(
            schema_version=1,
            datasets={"synthetic": self.policy},
        )
        self.receiver = fixturectl.ReceiverConfig(
            schema_version=1,
            protocol_version=1,
            target_id="authorized-test-vps",
            policy=Path("/etc/fixturectl/policy.json"),
            public_key=Path("/etc/fixturectl/fixture-signing.pub"),
            fixturectl_executable="/usr/local/libexec/fixturectl",
            rsync_executable="/usr/bin/rsync",
            rsync_version="3.5.0",
        )
        self.transfer_id = "d" * 32

    def test_receiver_config_is_exact_private_and_server_owned(self) -> None:
        config = self.base / "receiver.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol_version": 1,
                    "target_id": "authorized-test-vps",
                    "policy": "/etc/fixturectl/policy.json",
                    "public_key": "/etc/fixturectl/fixture-signing.pub",
                    "fixturectl_executable": "/usr/local/libexec/fixturectl",
                    "rsync_executable": "/usr/bin/rsync",
                    "rsync_version": "3.5.0",
                }
            ),
            encoding="utf-8",
        )
        config.chmod(0o600)
        loaded = fixturectl.load_receiver_config(config)
        self.assertEqual(loaded, self.receiver)

        config.chmod(0o644)
        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "config mode"):
            fixturectl.load_receiver_config(config)

    def test_rsync_dispatcher_accepts_only_exact_upload_destination(self) -> None:
        destination = f"{self.policy.destination}/incoming/{self.transfer_id}/data/"
        server_arguments = [
            "--server",
            "-logDtpRe.LsfxCIvu",
            "--partial-dir",
            f".fixturectl-partial-{self.transfer_id}",
            ".",
            destination,
        ]
        self.assertEqual(
            fixturectl._validate_rsync_upload_command(
                self.receiver,
                self.policy,
                dataset="synthetic",
                transfer_id=self.transfer_id,
                kind="data",
                server_arguments=server_arguments,
            ),
            ["/usr/bin/rsync", *server_arguments],
        )

        for rejected in (
            ["--server", "--sender", ".", destination],
            ["--server", ".", "/srv/private-fixtures/synthetic/releases/"],
            ["--server", "--delete-during", ".", destination],
            [
                *server_arguments[:-2],
                "--log-file=/etc/fixturectl/receiver.json",
                ".",
                destination,
            ],
            [*server_arguments[:-2], "--temp-dir=/tmp", ".", destination],
            [*server_arguments[:-2], "--backup-dir=/etc/fixturectl", ".", destination],
        ):
            with self.assertRaises(fixturectl.FixtureTransferError):
                fixturectl._validate_rsync_upload_command(
                    self.receiver,
                    self.policy,
                    dataset="synthetic",
                    transfer_id=self.transfer_id,
                    kind="data",
                    server_arguments=rejected,
                )

        metadata_destination = f"{self.policy.destination}/incoming/{self.transfer_id}/"
        metadata_arguments = [
            "--server",
            "-logDtpre.iLsfxCIvu",
            ".",
            metadata_destination,
        ]
        self.assertEqual(
            fixturectl._validate_rsync_upload_command(
                self.receiver,
                self.policy,
                dataset="synthetic",
                transfer_id=self.transfer_id,
                kind="metadata",
                server_arguments=metadata_arguments,
            ),
            ["/usr/bin/rsync", *metadata_arguments],
        )

    def test_pinned_rsync_real_remote_argv_reaches_dispatcher(self) -> None:
        candidate = Path(
            os.environ.get(
                "FIXTURECTL_PINNED_RSYNC",
                "/opt/homebrew/opt/rsync/bin/rsync",
            )
        )
        if not candidate.is_file():
            self.skipTest("pinned GNU rsync integration binary is unavailable")
        version = subprocess.run(
            [str(candidate), "--version"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        if not re.search(
            rb"^rsync\s+version\s+3\.5\.0\s+protocol version\s+32$",
            version,
            re.MULTILINE,
        ):
            self.skipTest("integration rsync is not pinned 3.5.0 protocol 32")

        source = make_synthetic_dataset(self.base / "integration-source")
        files_from = self.base / "files-from0"
        files_from.write_bytes(b"raw/empty.txt\0raw/record.json\0")
        files_from.chmod(0o600)
        manifest = self.base / "manifest.json"
        signature = self.base / "manifest.minisig"
        for path, content in (
            (manifest, b"{}\n"),
            (signature, b"synthetic\n"),
        ):
            path.write_bytes(content)
            path.chmod(0o600)
        fake_ssh = self.base / "fake-ssh"
        fake_ssh.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                from pathlib import Path
                import sys
                Path(os.environ["FIXTURECTL_SSH_ARGV_LOG"]).write_text(
                    json.dumps(sys.argv[1:]), encoding="utf-8"
                )
                raise SystemExit(23)
                """
            ),
            encoding="utf-8",
        )
        fake_ssh.chmod(0o700)

        for kind in ("data", "metadata"):
            with self.subTest(kind=kind):
                log = self.base / f"{kind}-ssh-argv.json"
                wrapper = (
                    "/usr/local/libexec/fixturectl ssh-dispatch-rsync "
                    f"--dataset synthetic --transfer-id {self.transfer_id} "
                    f"--kind {kind}"
                )
                destination_suffix = "data/" if kind == "data" else ""
                destination = (
                    f"fixture-receiver@100.100.10.20:{self.policy.destination}/"
                    f"incoming/{self.transfer_id}/{destination_suffix}"
                )
                if kind == "data":
                    arguments = [
                        str(candidate),
                        "--archive",
                        "--from0",
                        f"--files-from={files_from}",
                        "--partial",
                        f"--partial-dir=.fixturectl-partial-{self.transfer_id}",
                        "--chmod=D700,F600",
                        f"--rsh={fake_ssh}",
                        f"--rsync-path={wrapper}",
                        "--",
                        f"{source}{os.sep}",
                        destination,
                    ]
                else:
                    arguments = [
                        str(candidate),
                        "--archive",
                        "--chmod=F600",
                        f"--rsh={fake_ssh}",
                        f"--rsync-path={wrapper}",
                        "--",
                        str(manifest),
                        str(signature),
                        destination,
                    ]
                completed = subprocess.run(
                    arguments,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env={**os.environ, "FIXTURECTL_SSH_ARGV_LOG": str(log)},
                    timeout=10,
                )
                self.assertNotEqual(completed.returncode, 0)
                ssh_arguments = json.loads(log.read_text(encoding="utf-8"))
                wrapper_index = next(
                    index
                    for index, value in enumerate(ssh_arguments)
                    if value.startswith("/usr/local/libexec/fixturectl")
                )
                original_command = " ".join(ssh_arguments[wrapper_index:])
                with (
                    mock.patch.object(
                        fixturectl,
                        "load_policy",
                        return_value=self.policy_set,
                    ),
                    mock.patch.object(fixturectl, "_verify_pinned_rsync"),
                    mock.patch.object(fixturectl.os, "execv") as execv,
                    self.assertRaisesRegex(
                        fixturectl.FixtureTransferError,
                        "returned unexpectedly",
                    ),
                ):
                    fixturectl._dispatch_ssh_original_command(
                        self.receiver,
                        original_command,
                    )
                execv.assert_called_once()

    def test_forced_dispatcher_rejects_arbitrary_commands(self) -> None:
        with mock.patch.object(
            fixturectl,
            "load_policy",
            return_value=self.policy_set,
        ):
            for command in (
                "/bin/sh -c id",
                "/usr/local/libexec/fixturectl manifest create",
                "/usr/local/libexec/fixturectl ssh-dispatch-rsync "
                "--dataset synthetic --transfer-id bad --kind data --sender . /tmp/",
            ):
                with self.assertRaises(fixturectl.FixtureTransferError):
                    fixturectl._dispatch_ssh_original_command(self.receiver, command)

    def test_forced_dispatcher_execs_valid_upload_without_a_shell(self) -> None:
        destination = f"{self.policy.destination}/incoming/{self.transfer_id}/data/"
        command = (
            "/usr/local/libexec/fixturectl ssh-dispatch-rsync "
            f"--dataset synthetic --transfer-id {self.transfer_id} --kind data "
            f"--server -logDtpRe.LsfxCIvu "
            f"--partial-dir .fixturectl-partial-{self.transfer_id} "
            f". {destination}"
        )
        with (
            mock.patch.object(
                fixturectl,
                "load_policy",
                return_value=self.policy_set,
            ),
            mock.patch.object(fixturectl, "_verify_pinned_rsync") as verify_rsync,
            mock.patch.object(fixturectl.os, "execv") as execv,
            self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "returned unexpectedly",
            ),
        ):
            fixturectl._dispatch_ssh_original_command(self.receiver, command)
        execv.assert_called_once_with(
            "/usr/bin/rsync",
            [
                "/usr/bin/rsync",
                "--server",
                "-logDtpRe.LsfxCIvu",
                "--partial-dir",
                f".fixturectl-partial-{self.transfer_id}",
                ".",
                destination,
            ],
        )
        verify_rsync.assert_called_once_with("/usr/bin/rsync", "3.5.0")

    def test_forced_dispatcher_passes_only_receiver_commands_to_cli(self) -> None:
        command = (
            "/usr/local/libexec/fixturectl receive preflight "
            "--dataset synthetic --required-bytes 10"
        )
        with mock.patch.object(fixturectl, "main", return_value=0) as cli:
            self.assertEqual(
                fixturectl._dispatch_ssh_original_command(self.receiver, command),
                0,
            )
        cli.assert_called_once_with(
            ["receive", "preflight", "--dataset", "synthetic", "--required-bytes", "10"]
        )


if __name__ == "__main__":
    unittest.main()
