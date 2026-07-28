"""Adversarial tests for owner-only local-AI plaintext scratch."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest
from fastapi import FastAPI

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.scratch import ScratchJob, sweep_stale_scratch


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _abandon_job(root: Path, job_id: str) -> Path:
    """Simulate a crash while retaining the marker created by ScratchJob."""
    scratch = ScratchJob(root, job_id)
    scratch.__enter__()
    scratch.create_file("plain.txt", b"PHI")
    assert scratch._job_fd is not None
    os.close(scratch._job_fd)
    scratch._job_fd = None
    scratch._cleaned = True
    return scratch.path


def test_scratch_enforces_owner_only_modes_under_permissive_umask(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scratch"
    previous = os.umask(0)
    try:
        with ScratchJob(root, "job-1") as scratch:
            output = scratch.create_file("page-0001.png", b"PHI-CANARY")
            assert _mode(root) == 0o700
            assert _mode(scratch.path) == 0o700
            assert _mode(output) == 0o600
    finally:
        os.umask(previous)


@pytest.mark.parametrize(
    "job_id",
    ["", ".", "..", "../escape", "a/b", r"a\b", "/absolute", "x" * 129],
)
def test_scratch_rejects_unsafe_job_identifiers(tmp_path: Path, job_id: str) -> None:
    with pytest.raises(LocalValidationError, match="job identifier"):
        ScratchJob(tmp_path / "scratch", job_id)


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "../escape", "a/b", r"a\b", "/absolute", "x" * 256],
)
def test_scratch_rejects_unsafe_file_names(tmp_path: Path, filename: str) -> None:
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="file name"):
            scratch.reserve_file(filename)


def test_scratch_refuses_symlink_and_non_directory_roots(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    symlink_root = tmp_path / "linked"
    symlink_root.symlink_to(real, target_is_directory=True)
    with pytest.raises(LocalValidationError, match="scratch root"):
        ScratchJob(symlink_root, "job-1").__enter__()

    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory")
    with pytest.raises(LocalValidationError, match="scratch root"):
        ScratchJob(file_root, "job-1").__enter__()


def test_scratch_refuses_symlink_job_and_file_targets(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "job-1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(LocalValidationError, match="job directory"):
        ScratchJob(root, "job-1").__enter__()

    (root / "job-1").unlink()
    with ScratchJob(root, "job-1") as scratch:
        outside_file = tmp_path / "outside.txt"
        outside_file.write_bytes(b"outside")
        (scratch.path / "page.png").symlink_to(outside_file)
        with pytest.raises(LocalValidationError, match="scratch file"):
            scratch.reserve_file("page.png")
        assert outside_file.read_bytes() == b"outside"


def test_existing_real_job_collision_preserves_preexisting_canary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scratch"
    existing = root / "job-1"
    existing.mkdir(parents=True, mode=0o700)
    canary = existing / "keep.txt"
    canary.write_bytes(b"KEEP")

    with pytest.raises(LocalValidationError, match="already exists"):
        ScratchJob(root, "job-1").__enter__()

    assert canary.read_bytes() == b"KEEP"


def test_decrypt_to_file_uses_streaming_helper_and_validated_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    encrypted = tmp_path / "source.PDF"
    encrypted.write_bytes(b"ciphertext")
    calls: list[bool] = []

    def fake_decrypt_file_to(source, destination) -> None:
        calls.append(True)
        assert source.read() == b"ciphertext"
        destination.write(b"%PDF-1.7\n")

    monkeypatch.setattr(
        scratch_module,
        "decrypt_file_stream_to",
        fake_decrypt_file_to,
    )

    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        destination = scratch.decrypt_to_file(encrypted)
        assert calls == [True]
        assert destination.suffix == ".pdf"
        assert destination.read_bytes() == b"%PDF-1.7\n"
        assert _mode(destination) == 0o600

        with pytest.raises(LocalValidationError, match="source type"):
            scratch.decrypt_to_file(tmp_path / "source.exe")


def test_decrypt_destination_swap_never_writes_through_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    encrypted = tmp_path / "source.pdf"
    encrypted.write_bytes(b"ciphertext")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"KEEP")

    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:

        def swap_destination(_source, destination) -> None:
            reserved = next(scratch.path.glob("source-*.pdf"))
            reserved.rename(scratch.path / "moved-original.pdf")
            reserved.symlink_to(outside)
            destination.write(b"%PDF-1.7\n")

        monkeypatch.setattr(
            scratch_module,
            "decrypt_file_stream_to",
            swap_destination,
        )
        with pytest.raises(LocalValidationError, match="changed unexpectedly"):
            scratch.decrypt_to_file(encrypted)

    assert outside.read_bytes() == b"KEEP"


def test_corrupt_ciphertext_error_is_stable_and_path_free(tmp_path: Path) -> None:
    from app.utils.file_utils import ENC_MAGIC

    encrypted = tmp_path / "Jane-Public-MRN-123.pdf"
    encrypted.write_bytes(ENC_MAGIC + b"\x00")

    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError) as exc:
            scratch.decrypt_to_file(encrypted)

    assert str(encrypted) not in str(exc.value)
    assert exc.value.__cause__ is None


def test_scratch_context_removes_plaintext_after_exception(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    with pytest.raises(RuntimeError):
        with ScratchJob(root, "job-1") as scratch:
            scratch.create_file("page-0001.png", b"PHI-CANARY")
            raise RuntimeError("boom")
    assert not (root / "job-1").exists()


def test_cleanup_is_idempotent_and_does_not_follow_malicious_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scratch"
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "keep.txt"
    canary.write_bytes(b"KEEP")

    scratch = ScratchJob(root, "job-1")
    scratch.__enter__()
    scratch.create_file("plain.txt", b"PHI")
    (scratch.path / "escape").symlink_to(outside, target_is_directory=True)

    scratch.cleanup()
    scratch.cleanup()

    assert canary.read_bytes() == b"KEEP"
    assert not scratch.path.exists()


def test_cleanup_rejects_nested_directory_replacement_after_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    scratch = ScratchJob(tmp_path / "scratch", "job-1")
    scratch.__enter__()
    nested = scratch.path / "nested"
    nested.mkdir()
    (nested / "phi.txt").write_bytes(b"PHI")
    real_open = os.open
    swapped = False

    def replace_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "nested" and kwargs.get("dir_fd") == scratch._job_fd and not swapped:
            swapped = True
            nested.rename(scratch.path / "moved-nested")
            nested.mkdir()
            (nested / "keep.txt").write_bytes(b"KEEP")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(scratch_module.os, "open", replace_before_open)

    with pytest.raises(LocalValidationError, match="cleanup"):
        scratch.cleanup()

    assert swapped
    assert (nested / "keep.txt").read_bytes() == b"KEEP"
    scratch.cleanup()


def test_cleanup_revalidates_regular_file_identity_before_unlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    scratch = ScratchJob(tmp_path / "scratch", "job-1")
    scratch.__enter__()
    original = scratch.create_file("plain.txt", b"PHI")
    real_stat = os.stat
    swapped = False

    def replace_before_final_stat(path, *args, **kwargs):
        nonlocal swapped
        if (
            path == "plain.txt"
            and kwargs.get("dir_fd") == scratch._job_fd
            and not swapped
        ):
            swapped = True
            original.rename(scratch.path / "moved-original.txt")
            original.write_bytes(b"KEEP")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(scratch_module.os, "stat", replace_before_final_stat)

    with pytest.raises(LocalValidationError, match="cleanup"):
        scratch.cleanup()

    assert swapped
    assert original.read_bytes() == b"KEEP"
    scratch.cleanup()


def test_cleanup_incomplete_result_is_retryable_and_keeps_descriptor_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    scratch = ScratchJob(tmp_path / "scratch", "job-1")
    scratch.__enter__()
    plaintext = scratch.create_file("plain.txt", b"PHI")
    original_remove = scratch_module._remove_tree_contents
    calls = 0

    def incomplete_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        return original_remove(*args, **kwargs)

    monkeypatch.setattr(scratch_module, "_remove_tree_contents", incomplete_once)

    with pytest.raises(LocalValidationError, match="cleanup"):
        scratch.cleanup()

    assert plaintext.read_bytes() == b"PHI"
    assert scratch._job_fd is not None
    os.fstat(scratch._job_fd)

    scratch.cleanup()
    assert scratch._job_fd is None
    assert not scratch.path.exists()


def test_cleanup_rmdir_race_keeps_descriptor_and_retry_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    scratch = ScratchJob(tmp_path / "scratch", "job-1")
    scratch.__enter__()
    scratch.create_file("plain.txt", b"PHI")
    real_rmdir = os.rmdir
    raced = False

    def add_late_file_then_fail(path, *args, **kwargs):
        nonlocal raced
        if path == scratch.job_id and kwargs.get("dir_fd") is not None and not raced:
            raced = True
            (scratch.path / "late-phi.txt").write_bytes(b"PHI")
            raise OSError("injected directory-not-empty race")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(scratch_module.os, "rmdir", add_late_file_then_fail)

    with pytest.raises(LocalValidationError, match="cleanup"):
        scratch.cleanup()

    assert raced
    assert scratch._job_fd is not None
    os.fstat(scratch._job_fd)
    assert (scratch.path / "late-phi.txt").read_bytes() == b"PHI"

    scratch.cleanup()
    assert scratch._job_fd is None
    assert not scratch.path.exists()


def test_cleanup_contains_root_replacement_race(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    scratch = ScratchJob(root, "job-1")
    scratch.__enter__()
    scratch.create_file("plain.txt", b"PHI")

    moved = tmp_path / "moved-root"
    root.rename(moved)
    root.mkdir(mode=0o700)
    replacement = root / "job-1"
    replacement.mkdir()
    outside_canary = replacement / "keep.txt"
    outside_canary.write_bytes(b"KEEP")

    with pytest.raises(LocalValidationError, match="scratch root changed"):
        scratch.cleanup()

    assert outside_canary.read_bytes() == b"KEEP"
    assert not (moved / "job-1" / "plain.txt").exists()


def test_cleanup_root_disappearance_after_purge_closes_and_becomes_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scratch"
    scratch = ScratchJob(root, "job-1")
    scratch.__enter__()
    scratch.create_file("plain.txt", b"PHI")
    held_fd = scratch._job_fd
    assert held_fd is not None

    moved = tmp_path / "moved-scratch"
    root.rename(moved)

    with pytest.raises(LocalValidationError, match="containment"):
        scratch.cleanup()

    assert not (moved / "job-1" / "plain.txt").exists()
    assert scratch._job_fd is None
    assert scratch._files == {}
    assert scratch._cleaned is True
    with pytest.raises(OSError):
        os.fstat(held_fd)

    scratch.cleanup()


def test_sweep_only_removes_stale_inactive_valid_jobs(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    now = time.time()

    for name in ("stale", "active", "recent"):
        _abandon_job(root, name)
    os.utime(root / "stale", (now - 10_000, now - 10_000))
    os.utime(root / "active", (now - 10_000, now - 10_000))
    os.utime(root / "recent", (now - 5, now - 5))

    unknown = root / "not valid"
    unknown.mkdir()
    os.utime(unknown, (now - 10_000, now - 10_000))
    insecure_unknown = root / "safe-looking-unknown"
    insecure_unknown.mkdir(mode=0o700)
    os.utime(insecure_unknown, (now - 10_000, now - 10_000))
    worker_home = root / "worker-home"
    worker_home.mkdir(mode=0o700)
    (worker_home / "model.sock").write_bytes(b"not scratch")
    os.utime(worker_home, (now - 10_000, now - 10_000))
    regular = root / "regular-file"
    regular.write_bytes(b"leave")
    target = tmp_path / "outside"
    target.mkdir()
    symlink = root / "symlink-job"
    symlink.symlink_to(target, target_is_directory=True)

    removed = sweep_stale_scratch(
        root,
        stale_after_seconds=60,
        active_job_ids={"active"},
        now=now,
    )

    assert removed == 1
    assert not (root / "stale").exists()
    assert (root / "active").exists()
    assert (root / "recent").exists()
    assert unknown.exists()
    assert insecure_unknown.exists()
    assert worker_home.exists()
    assert regular.exists()
    assert symlink.is_symlink()
    assert target.exists()


def test_sweep_tolerates_concurrent_job_disappearance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    root = tmp_path / "scratch"
    job = _abandon_job(root, "stale")
    os.utime(job, (1, 1))

    original = scratch_module._remove_job_at

    def disappear_first(root_fd: int, job_id: str, *args, **kwargs) -> bool:
        job_fd = os.open(job_id, os.O_RDONLY, dir_fd=root_fd)
        scratch_module._remove_tree_contents(job_fd)
        os.close(job_fd)
        os.rmdir(job_id, dir_fd=root_fd)
        return original(root_fd, job_id, *args, **kwargs)

    monkeypatch.setattr(scratch_module, "_remove_job_at", disappear_first)
    assert sweep_stale_scratch(root, stale_after_seconds=1, now=100) == 0


def test_sweep_revalidates_identity_before_touching_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    root = tmp_path / "scratch"
    stale = _abandon_job(root, "stale")
    os.utime(stale, (1, 1))
    original = scratch_module._remove_job_at

    def replace_before_remove(root_fd: int, job_id: str, *args, **kwargs) -> bool:
        os.rename(job_id, "moved-original", src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.mkdir(job_id, mode=0o700, dir_fd=root_fd)
        replacement_fd = os.open(job_id, os.O_RDONLY, dir_fd=root_fd)
        try:
            canary_fd = os.open(
                "keep.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_fd,
            )
            os.write(canary_fd, b"KEEP")
            os.close(canary_fd)
        finally:
            os.close(replacement_fd)
        return original(root_fd, job_id, *args, **kwargs)

    monkeypatch.setattr(scratch_module, "_remove_job_at", replace_before_remove)
    assert sweep_stale_scratch(root, stale_after_seconds=1, now=100) == 0
    assert (root / "stale" / "keep.txt").read_bytes() == b"KEEP"


def test_marker_added_after_entry_scan_cannot_authenticate_unknown_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.scratch as scratch_module

    root = tmp_path / "scratch"
    unknown = root / "worker-home"
    unknown.mkdir(parents=True, mode=0o700)
    (unknown / "keep.txt").write_bytes(b"KEEP")
    os.utime(unknown, (1, 1))
    original = scratch_module._remove_job_at

    def add_marker_after_scan(root_fd: int, job_id: str, *args, **kwargs) -> bool:
        job_fd = os.open(job_id, os.O_RDONLY, dir_fd=root_fd)
        try:
            marker_fd = os.open(
                scratch_module._JOB_MARKER_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=job_fd,
            )
            os.write(marker_fd, scratch_module._JOB_MARKER_CONTENT)
            os.close(marker_fd)
        finally:
            os.close(job_fd)
        return original(root_fd, job_id, *args, **kwargs)

    monkeypatch.setattr(scratch_module, "_remove_job_at", add_marker_after_scan)
    assert sweep_stale_scratch(root, stale_after_seconds=1, now=100) == 0
    assert (unknown / "keep.txt").read_bytes() == b"KEEP"


def test_sweep_node_budget_leaves_unfinished_job_for_later(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    stale = _abandon_job(root, "stale")
    for index in range(5):
        (stale / f"extra-{index}.txt").write_bytes(b"PHI")
    os.utime(stale, (1, 1))

    assert (
        sweep_stale_scratch(
            root,
            stale_after_seconds=1,
            max_nodes=2,
            now=100,
        )
        == 0
    )
    assert stale.exists()

    assert (
        sweep_stale_scratch(
            root,
            stale_after_seconds=1,
            max_nodes=100,
            now=100,
        )
        == 1
    )
    assert not stale.exists()


@pytest.mark.parametrize("active_ids", ["job-1", b"job-1"])
def test_sweep_rejects_scalar_active_job_ids(
    tmp_path: Path,
    active_ids: object,
) -> None:
    with pytest.raises(LocalValidationError, match="active job"):
        sweep_stale_scratch(
            tmp_path / "scratch",
            stale_after_seconds=1,
            active_job_ids=active_ids,
        )


def test_sweep_rejects_invalid_active_job_id(tmp_path: Path) -> None:
    with pytest.raises(LocalValidationError, match="job identifier"):
        sweep_stale_scratch(
            tmp_path / "scratch",
            stale_after_seconds=1,
            active_job_ids=["valid", "../invalid"],
        )


def test_sweep_bounds_active_job_iteration_at_limit_plus_one(tmp_path: Path) -> None:
    consumed = 0

    def active_ids():
        nonlocal consumed
        for value in ("one", "two", "three"):
            consumed += 1
            yield value
        pytest.fail("active job iterator was consumed beyond the rejecting element")

    with pytest.raises(LocalValidationError, match="too many active"):
        sweep_stale_scratch(
            tmp_path / "scratch",
            stale_after_seconds=1,
            active_job_ids=active_ids(),
            max_active_jobs=2,
        )

    assert consumed == 3


def test_sweep_node_budget_is_total_across_all_jobs(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    first = _abandon_job(root, "first")
    second = _abandon_job(root, "second")
    os.utime(first, (1, 1))
    os.utime(second, (1, 1))

    removed = sweep_stale_scratch(
        root,
        stale_after_seconds=1,
        max_nodes=2,
        now=100,
    )

    assert removed <= 1
    assert first.exists() or second.exists()


def test_sweep_depth_bound_leaves_deep_job_for_later(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    stale = _abandon_job(root, "deep")
    nested = stale / "one" / "two" / "three"
    nested.mkdir(parents=True)
    (nested / "phi.txt").write_bytes(b"PHI")
    os.utime(stale, (1, 1))

    assert (
        sweep_stale_scratch(
            root,
            stale_after_seconds=1,
            max_depth=1,
            now=100,
        )
        == 0
    )
    assert stale.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_sweep_rejects_non_finite_age_and_now(tmp_path: Path, value: float) -> None:
    with pytest.raises(LocalValidationError, match="finite"):
        sweep_stale_scratch(tmp_path / "scratch", stale_after_seconds=value)
    with pytest.raises(LocalValidationError, match="finite"):
        sweep_stale_scratch(
            tmp_path / "scratch",
            stale_after_seconds=1,
            now=value,
        )


@pytest.mark.asyncio
async def test_enabled_startup_sweeps_immediately_before_model_manager_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.main as main_module
    import app.services.auth_service as auth_service
    import app.services.extraction.terminology as terminology
    from app.api import upload

    events: list[str] = []

    class FakeManager:
        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

    class FakeResult:
        rowcount = 0

    class FakeSession:
        async def execute(self, _query: object) -> FakeResult:
            return FakeResult()

        async def commit(self) -> None:
            return None

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def no_purge(_db: object) -> int:
        return 0

    async def reconcile(_db: object, _root: Path) -> None:
        events.append("reconcile")

    monkeypatch.setattr(main_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        main_module.settings, "local_ai_scratch_dir", str(tmp_path / "scratch")
    )
    monkeypatch.setattr(main_module.settings, "phi_ner_enabled", False)
    monkeypatch.setattr(main_module.settings, "extraction_engine", "gemini")
    monkeypatch.setattr(main_module, "local_model_manager", FakeManager())
    monkeypatch.setattr(main_module, "reconcile_zip_child_sets", reconcile)
    monkeypatch.setattr(
        main_module,
        "sweep_stale_scratch",
        lambda *_args, **_kwargs: events.append("sweep"),
    )
    monkeypatch.setattr(main_module, "async_session_factory", FakeSessionContext)
    monkeypatch.setattr(terminology, "schedule_medication_refresh", lambda: None)
    monkeypatch.setattr(auth_service, "purge_expired_revoked_tokens", no_purge)
    monkeypatch.setattr(
        upload,
        "start_extraction_worker",
        lambda: events.append("worker"),
    )

    async with main_module.lifespan(FastAPI()):
        assert events == ["reconcile", "sweep", "start", "worker"]

    assert events == ["reconcile", "sweep", "start", "worker", "stop"]


@pytest.mark.asyncio
async def test_disabled_startup_does_not_touch_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module
    import app.services.auth_service as auth_service
    import app.services.extraction.terminology as terminology
    from app.api import upload

    class FakeResult:
        rowcount = 0

    class FakeSession:
        async def execute(self, _query: object) -> FakeResult:
            return FakeResult()

        async def commit(self) -> None:
            return None

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def no_purge(_db: object) -> int:
        return 0

    async def no_reconcile(_db: object, _root: Path) -> None:
        return None

    monkeypatch.setattr(main_module.settings, "local_ai_enabled", False)
    monkeypatch.setattr(main_module.settings, "phi_ner_enabled", False)
    monkeypatch.setattr(main_module.settings, "extraction_engine", "gemini")
    monkeypatch.setattr(main_module, "reconcile_zip_child_sets", no_reconcile)
    monkeypatch.setattr(
        main_module,
        "sweep_stale_scratch",
        lambda *_args, **_kwargs: pytest.fail("scratch sweep must stay lazy"),
    )
    monkeypatch.setattr(main_module, "async_session_factory", FakeSessionContext)
    monkeypatch.setattr(terminology, "schedule_medication_refresh", lambda: None)
    monkeypatch.setattr(auth_service, "purge_expired_revoked_tokens", no_purge)
    monkeypatch.setattr(upload, "start_extraction_worker", lambda: None)

    async with main_module.lifespan(FastAPI()):
        pass
