from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


import sys

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

import fixturectl  # noqa: E402


def _policy_at(root: Path) -> fixturectl.Policy:
    datasets = {
        dataset: fixturectl.DatasetPolicy(
            dataset_id=dataset,
            allowed_top_level=("raw",),
            max_files=100,
            max_total_bytes=10_000,
            destination=str(root / dataset),
        )
        for dataset in ("alpha", "beta")
    }
    return fixturectl.Policy(schema_version=1, datasets=datasets)


def _prepare_root(root: Path) -> None:
    root.mkdir(mode=0o700)
    for dataset in ("alpha", "beta"):
        for relative in ("", "incoming", "releases", ".reservations"):
            path = root / dataset / relative
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700)


def _reservation_worker(
    root_text: str,
    transfer_id: str,
    free_bytes: int,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    root = Path(root_text)
    policy = _policy_at(root)
    start.wait()
    try:
        with mock.patch.object(
            fixturectl.shutil,
            "disk_usage",
            return_value=mock.Mock(free=free_bytes),
        ):
            fixturectl.reserve_capacity(
                policy,
                policy.datasets["alpha"],
                transfer_id=transfer_id,
                manifest_sha="a" * 64,
                required_bytes=100,
                reserved_bytes=1_000,
            )
        results.put("accepted")
    except fixturectl.FixtureTransferError:
        results.put("refused")


class CapacityReservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "private-fixtures"
        _prepare_root(self.root)
        self.policy = _policy_at(self.root)
        self.dataset = self.policy.datasets["alpha"]

    def test_reservation_is_private_exact_and_bound_to_transfer(self) -> None:
        with mock.patch.object(
            fixturectl.shutil,
            "disk_usage",
            return_value=mock.Mock(free=10**9),
        ):
            reservation = fixturectl.reserve_capacity(
                self.policy,
                self.dataset,
                transfer_id="a" * 32,
                manifest_sha="b" * 64,
                required_bytes=321,
                reserved_bytes=1_000,
            )

        path = self.root / "alpha/.reservations" / f"{'a' * 32}.json"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            {
                "dataset": "alpha",
                "manifest_sha256": "b" * 64,
                "required_bytes": 321,
                "schema_version": 1,
                "transfer_id": "a" * 32,
            },
        )
        self.assertEqual(reservation.required_bytes, 321)

        with mock.patch.object(
            fixturectl.shutil,
            "disk_usage",
            return_value=mock.Mock(free=0),
        ):
            repeated = fixturectl.reserve_capacity(
                self.policy,
                self.dataset,
                transfer_id="a" * 32,
                manifest_sha="b" * 64,
                required_bytes=321,
                reserved_bytes=1_000,
            )
        self.assertEqual(repeated, reservation)
        self.assertEqual(
            len(list((self.root / "alpha/.reservations").glob("*.json"))),
            1,
        )

        with self.assertRaisesRegex(fixturectl.FixtureTransferError, "reservation"):
            fixturectl.create_reserved_staging(
                self.policy,
                self.dataset,
                transfer_id="a" * 32,
                manifest_sha="b" * 64,
                required_bytes=320,
            )

    def test_two_process_admission_preserves_the_static_reserve(self) -> None:
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        results = context.Queue()
        free_bytes = 1_000 + (2 * 100) + fixturectl._CAPACITY_TRANSFER_OVERHEAD_BYTES
        processes = [
            context.Process(
                target=_reservation_worker,
                args=(str(self.root), transfer_id, free_bytes, start, results),
            )
            for transfer_id in ("c" * 32, "d" * 32)
        ]
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)

        self.assertEqual(
            sorted(results.get(timeout=2) for _ in processes),
            ["accepted", "refused"],
        )
        self.assertEqual(
            len(list((self.root / "alpha/.reservations").glob("*.json"))),
            1,
        )

    def test_malformed_reservation_or_unreserved_staging_blocks_admission(self) -> None:
        bad = self.root / "alpha/.reservations/not-a-reservation"
        bad.write_text("unsafe\n", encoding="utf-8")
        bad.chmod(0o600)
        with (
            mock.patch.object(
                fixturectl.shutil,
                "disk_usage",
                return_value=mock.Mock(free=10**9),
            ),
            self.assertRaisesRegex(fixturectl.FixtureTransferError, "reservation"),
        ):
            fixturectl.reserve_capacity(
                self.policy,
                self.dataset,
                transfer_id="e" * 32,
                manifest_sha="f" * 64,
                required_bytes=100,
                reserved_bytes=1_000,
            )

        bad.unlink()
        (self.root / "beta/incoming" / ("1" * 32)).mkdir(mode=0o700)
        with (
            mock.patch.object(
                fixturectl.shutil,
                "disk_usage",
                return_value=mock.Mock(free=10**9),
            ),
            self.assertRaisesRegex(fixturectl.FixtureTransferError, "reservation"),
        ):
            fixturectl.reserve_capacity(
                self.policy,
                self.dataset,
                transfer_id="e" * 32,
                manifest_sha="f" * 64,
                required_bytes=100,
                reserved_bytes=1_000,
            )

    def test_dataset_mount_must_share_the_common_capacity_device(self) -> None:
        real_open = fixturectl._open_private_child_directory

        def different_device(
            parent_fd: int,
            name: str,
            label: str,
        ) -> tuple[int, object]:
            descriptor, metadata = real_open(parent_fd, name, label)
            if name == "beta":
                metadata = mock.Mock(st_dev=metadata.st_dev + 1)
            return descriptor, metadata

        with (
            mock.patch.object(
                fixturectl,
                "_open_private_child_directory",
                side_effect=different_device,
            ),
            mock.patch.object(
                fixturectl.shutil,
                "disk_usage",
                return_value=mock.Mock(free=10**9),
            ),
            self.assertRaisesRegex(
                fixturectl.FixtureTransferError,
                "filesystem boundary",
            ),
        ):
            fixturectl.reserve_capacity(
                self.policy,
                self.dataset,
                transfer_id="2" * 32,
                manifest_sha="3" * 64,
                required_bytes=100,
                reserved_bytes=1_000,
            )


if __name__ == "__main__":
    unittest.main()
