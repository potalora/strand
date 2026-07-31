"""Crash-safe two-phase lifecycle tests for mixed-ZIP unstructured children."""

from __future__ import annotations

import os
import io
import zipfile
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.uploaded_file import UploadedFile
from app.services.ingestion.zip_child_sets import (
    STAGING_EXTRACTION_STATUS,
    ZipChildSet,
    reconcile_zip_child_sets,
)
from app.utils.file_utils import EncryptedFileWriter
from tests.conftest import auth_headers


def _set_names() -> tuple[str, str]:
    final_name = f"medtimeline-zip-set-{uuid4().hex}"
    return final_name, f".{final_name}.pending"


def _write_encrypted(path: Path, payload: bytes = b"%PDF-1.7\nPHI") -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("wb") as destination:
        writer = EncryptedFileWriter(destination)
        writer.write_chunk(payload)
        writer.finalize()
    path.chmod(0o600)


def _make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _open_fd_count() -> int:
    return len(os.listdir("/dev/fd"))


async def _add_staging_row(
    db: AsyncSession,
    user_id: UUID,
    storage_path: Path,
    *,
    filename: str = "private-clinical-name.pdf",
) -> UploadedFile:
    row = UploadedFile(
        user_id=user_id,
        filename=filename,
        mime_type="application/pdf",
        file_size_bytes=8,
        file_hash="a" * 64,
        storage_path=str(storage_path),
        ingestion_status=STAGING_EXTRACTION_STATUS,
        file_category="unstructured",
        manual_extraction_required=True,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


def test_zip_child_set_fsync_order_and_atomic_directory_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.zip_child_sets as sets_module

    events: list[str] = []
    real_rename = os.rename

    monkeypatch.setattr(
        sets_module,
        "_fsync",
        lambda _descriptor, event: events.append(event),
    )

    source_a = tmp_path / "a.pdf"
    source_b = tmp_path / "b.pdf"
    source_a.write_bytes(b"%PDF-1.7\nA")
    source_b.write_bytes(b"%PDF-1.7\nB")
    upload_root = tmp_path / "uploads"
    child_set = ZipChildSet(upload_root)
    child_set.encrypt(source_a, "a.pdf")
    child_set.encrypt(source_b, "b.pdf")
    child_set.seal()

    def observe_atomic_publish(source, destination, *args, **kwargs):
        assert source == child_set.pending_name
        assert destination == child_set.final_name
        assert set(child_set.pending_path.iterdir()) == {
            child_set.pending_path / "a.pdf",
            child_set.pending_path / "b.pdf",
        }
        assert not child_set.final_path.exists()
        result = real_rename(source, destination, *args, **kwargs)
        assert not child_set.pending_path.exists()
        assert set(child_set.final_path.iterdir()) == {
            child_set.final_path / "a.pdf",
            child_set.final_path / "b.pdf",
        }
        return result

    monkeypatch.setattr(sets_module.os, "rename", observe_atomic_publish)
    child_set.publish()

    assert events == [
        "upload_root_after_pending_create",
        "ciphertext_file",
        "ciphertext_file",
        "pending_directory_after_files",
        "upload_root_after_set_publish",
    ]
    child_set.close()


def test_zip_child_set_cleanup_fsyncs_affected_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.zip_child_sets as sets_module

    events: list[str] = []
    monkeypatch.setattr(
        sets_module,
        "_fsync",
        lambda _descriptor, event: events.append(event),
    )
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\nPHI")
    child_set = ZipChildSet(tmp_path / "uploads")
    pending = child_set.pending_path
    child_set.encrypt(source, "child.pdf")

    child_set.close()

    assert not pending.exists()
    assert events == [
        "upload_root_after_pending_create",
        "ciphertext_file",
        "pending_directory_after_cleanup",
        "upload_root_after_pending_cleanup",
    ]


def test_zip_child_set_constructor_validation_failure_cleans_fd_and_pending_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.zip_child_sets as sets_module

    upload_root = tmp_path / "uploads"
    upload_root.mkdir(mode=0o700)
    canary = upload_root / "keep.txt"
    canary.write_bytes(b"KEEP")
    real_open = os.open
    real_fstat = os.fstat
    pending_descriptors: set[int] = set()
    fail_next_pending_validation = False

    def tracking_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".medtimeline-zip-set-"):
            pending_descriptors.add(descriptor)
        return descriptor

    def fail_pending_validation_once(descriptor):
        nonlocal fail_next_pending_validation
        info = real_fstat(descriptor)
        if descriptor in pending_descriptors and fail_next_pending_validation:
            fail_next_pending_validation = False
            values = list(info)
            values[1] += 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(sets_module.os, "open", tracking_open)
    monkeypatch.setattr(sets_module.os, "fstat", fail_pending_validation_once)
    baseline = _open_fd_count()

    for _ in range(3):
        fail_next_pending_validation = True
        with pytest.raises(ValueError, match="changed during validation"):
            ZipChildSet(upload_root)
        assert _open_fd_count() == baseline
        assert list(upload_root.glob(".medtimeline-zip-set-*.pending")) == []
        assert canary.read_bytes() == b"KEEP"


def test_zip_child_set_constructor_fsync_failure_cleans_fd_and_pending_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.zip_child_sets as sets_module

    upload_root = tmp_path / "uploads"
    upload_root.mkdir(mode=0o700)
    canary = upload_root / "keep.txt"
    canary.write_bytes(b"KEEP")
    real_fsync = sets_module._fsync

    def fail_pending_create_sync(descriptor: int, event: str) -> None:
        if event == "upload_root_after_pending_create":
            raise OSError("injected pending-create fsync failure")
        real_fsync(descriptor, event)

    monkeypatch.setattr(sets_module, "_fsync", fail_pending_create_sync)
    baseline = _open_fd_count()

    for _ in range(3):
        with pytest.raises(OSError, match="fsync failure"):
            ZipChildSet(upload_root)
        assert _open_fd_count() == baseline
        assert list(upload_root.glob(".medtimeline-zip-set-*.pending")) == []
        assert canary.read_bytes() == b"KEEP"


def test_zip_child_set_constructor_persistent_fsync_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.zip_child_sets as sets_module

    upload_root = tmp_path / "uploads"
    upload_root.mkdir(mode=0o700)
    canary = upload_root / "keep.txt"
    canary.write_bytes(b"KEEP")
    first_failure = OSError("sanitized constructor durability failure")
    cleanup_failure = OSError("cleanup durability failure")
    constructor_sync_pending = True

    def fail_every_sync(_descriptor: int, _event: str) -> None:
        nonlocal constructor_sync_pending
        if constructor_sync_pending:
            constructor_sync_pending = False
            raise first_failure
        raise cleanup_failure

    monkeypatch.setattr(sets_module, "_fsync", fail_every_sync)
    baseline = _open_fd_count()

    for _ in range(3):
        constructor_sync_pending = True
        with pytest.raises(OSError) as exc:
            ZipChildSet(upload_root)
        assert exc.value is first_failure
        assert _open_fd_count() == baseline
        assert list(upload_root.glob(".medtimeline-zip-set-*.pending")) == []
        assert canary.read_bytes() == b"KEEP"


@pytest.mark.asyncio
async def test_reconcile_crash_before_set_rename(
    client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    _, user_text = await auth_headers(client, "zip-recover-before@example.com")
    user_id = UUID(user_text)
    upload_root = tmp_path / "uploads"
    final_name, pending_name = _set_names()
    pending = upload_root / pending_name
    final = upload_root / final_name
    pending.mkdir(mode=0o700, parents=True)
    _write_encrypted(pending / "child.pdf")
    row = await _add_staging_row(db_session, user_id, final / "child.pdf")

    result = await reconcile_zip_child_sets(db_session, upload_root)
    await db_session.refresh(row)

    assert result.recovered_groups == 1
    assert row.ingestion_status == "pending_extraction"
    assert row.manual_extraction_required is True
    assert final.is_dir()
    assert not pending.exists()
    assert (final / "child.pdf").is_file()


@pytest.mark.asyncio
async def test_reconcile_crash_after_set_rename_before_status_commit_is_idempotent(
    client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    _, user_text = await auth_headers(client, "zip-recover-after@example.com")
    user_id = UUID(user_text)
    upload_root = tmp_path / "uploads"
    final_name, _pending_name = _set_names()
    final = upload_root / final_name
    final.mkdir(mode=0o700, parents=True)
    _write_encrypted(final / "child.pdf")
    row = await _add_staging_row(db_session, user_id, final / "child.pdf")

    first = await reconcile_zip_child_sets(db_session, upload_root)
    second = await reconcile_zip_child_sets(db_session, upload_root)
    await db_session.refresh(row)

    assert first.recovered_groups == 1
    assert second.recovered_groups == 0
    assert second.failed_groups == 0
    assert row.ingestion_status == "pending_extraction"
    assert row.manual_extraction_required is True
    assert (final / "child.pdf").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["missing", "malformed", "symlink"])
async def test_reconcile_invalid_group_fails_all_rows_without_phi_in_error(
    failure_kind: str,
    client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    _, user_text = await auth_headers(client, f"zip-recover-{failure_kind}@example.com")
    user_id = UUID(user_text)
    upload_root = tmp_path / "uploads"
    final_name, pending_name = _set_names()
    final = upload_root / final_name
    pending = upload_root / pending_name

    if failure_kind == "missing":
        pending.mkdir(mode=0o700, parents=True)
        _write_encrypted(pending / "one.pdf")
    elif failure_kind == "malformed":
        pending.mkdir(mode=0o700, parents=True)
        (pending / "one.pdf").write_bytes(b"PLAINTEXT-PHI")
        (pending / "one.pdf").chmod(0o600)
        _write_encrypted(pending / "two.pdf")
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_bytes(b"KEEP")
        upload_root.mkdir(mode=0o700)
        pending.symlink_to(outside, target_is_directory=True)

    first = await _add_staging_row(
        db_session,
        user_id,
        final / "one.pdf",
        filename="Jane-Public-one.pdf",
    )
    second = await _add_staging_row(
        db_session,
        user_id,
        final / "two.pdf",
        filename="Jane-Public-two.pdf",
    )

    result = await reconcile_zip_child_sets(db_session, upload_root)
    await db_session.refresh(first)
    await db_session.refresh(second)

    assert result.failed_groups == 1
    assert first.ingestion_status == second.ingestion_status == "failed"
    assert first.ingestion_errors == second.ingestion_errors
    assert "Jane" not in str(first.ingestion_errors)
    assert "one.pdf" not in str(first.ingestion_errors)
    assert "two.pdf" not in str(first.ingestion_errors)
    if failure_kind == "symlink":
        assert (tmp_path / "outside" / "keep.txt").read_bytes() == b"KEEP"


@pytest.mark.asyncio
@pytest.mark.parametrize("directory_kind", ["pending", "final"])
async def test_reconcile_removes_valid_orphan_set(
    directory_kind: str,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    final_name, pending_name = _set_names()
    orphan = upload_root / (pending_name if directory_kind == "pending" else final_name)
    orphan.mkdir(mode=0o700, parents=True)
    _write_encrypted(orphan / "orphan.pdf")

    result = await reconcile_zip_child_sets(db_session, upload_root)

    assert result.removed_orphans == 1
    assert not orphan.exists()


@pytest.mark.asyncio
async def test_reconcile_row_bound_stops_without_partial_group_updates(
    client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    _, user_text = await auth_headers(client, "zip-recover-bound@example.com")
    user_id = UUID(user_text)
    upload_root = tmp_path / "uploads"
    rows: list[UploadedFile] = []
    for index in range(2):
        final_name, pending_name = _set_names()
        pending = upload_root / pending_name
        pending.mkdir(mode=0o700, parents=True)
        _write_encrypted(pending / f"child-{index}.pdf")
        rows.append(
            await _add_staging_row(
                db_session,
                user_id,
                upload_root / final_name / f"child-{index}.pdf",
            )
        )

    result = await reconcile_zip_child_sets(db_session, upload_root, max_rows=1)
    for row in rows:
        await db_session.refresh(row)

    assert result.bounded is True
    assert {row.ingestion_status for row in rows} == {STAGING_EXTRACTION_STATUS}
    assert len(list(upload_root.glob("*.pending"))) == 2


@pytest.mark.asyncio
async def test_reconcile_total_root_entry_bound_counts_unrelated_entries(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.zip_child_sets as sets_module

    upload_root = tmp_path / "uploads"
    upload_root.mkdir(mode=0o700)
    for index in range(2_500):
        (upload_root / f"unrelated-{index:04d}.txt").write_bytes(b"KEEP")
    real_scandir = os.scandir
    yielded = 0

    class CountingScandir:
        def __init__(self, directory) -> None:
            self._iterator = real_scandir(directory)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal yielded
            yielded += 1
            return next(self._iterator)

    monkeypatch.setattr(sets_module.os, "scandir", CountingScandir)

    result = await reconcile_zip_child_sets(
        db_session,
        upload_root,
        max_root_entries=1,
    )

    assert result.bounded is True
    assert yielded == 2
    assert len(list(upload_root.iterdir())) == 2_500


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
async def test_reconcile_rejects_invalid_total_root_entry_bound(
    bound: object,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="bounds"):
        await reconcile_zip_child_sets(
            db_session,
            tmp_path / "uploads",
            max_root_entries=bound,
        )


@pytest.mark.asyncio
async def test_mixed_zip_rows_are_nonclaimable_until_atomic_set_publish(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    _, user_text = await auth_headers(client, "zip-two-phase@example.com")
    user_id = UUID(user_text)
    patient = await coordinator.get_or_create_patient(db_session, user_id)
    parent = UploadedFile(
        user_id=user_id,
        filename="parent.zip",
        mime_type="application/zip",
        file_size_bytes=1,
        file_hash="b" * 64,
        storage_path=str(tmp_path / "parent.zip"),
        ingestion_status="processing",
    )
    db_session.add(parent)
    await db_session.commit()
    await db_session.refresh(parent)

    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(coordinator.settings, "upload_dir", str(upload_root))
    monkeypatch.setattr(
        coordinator.settings,
        "temp_extract_dir",
        str(tmp_path / "extract"),
    )
    archive = tmp_path / "children.zip"
    archive.write_bytes(
        _make_zip(
            {
                "one.pdf": b"%PDF-1.7\nONE",
                "two.pdf": b"%PDF-1.7\nTWO",
            }
        )
    )

    original_commit = db_session.commit
    snapshots: list[tuple[set[str], set[bool], bool, bool]] = []

    async def observing_commit() -> None:
        await original_commit()
        rows = (
            (
                await db_session.execute(
                    select(UploadedFile).where(
                        UploadedFile.user_id == user_id,
                        UploadedFile.file_category == "unstructured",
                    )
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return
        final_parent = Path(rows[0].storage_path).parent
        pending = upload_root / f".{final_parent.name}.pending"
        snapshots.append(
            (
                {row.ingestion_status for row in rows},
                {row.manual_extraction_required for row in rows},
                pending.is_dir(),
                final_parent.is_dir(),
            )
        )

    monkeypatch.setattr(db_session, "commit", observing_commit)

    result = await coordinator._ingest_zip(
        db_session,
        user_id,
        patient.id,
        parent.id,
        archive,
    )

    assert len(result["unstructured_files"]) == 2
    assert all(
        child["manual_extraction_required"] is True
        for child in result["unstructured_files"]
    )
    assert snapshots == [
        ({STAGING_EXTRACTION_STATUS}, {True}, True, False),
        ({"pending_extraction"}, {True}, False, True),
    ]
    rows = (
        (
            await db_session.execute(
                select(UploadedFile).where(
                    UploadedFile.user_id == user_id,
                    UploadedFile.file_category == "unstructured",
                )
            )
        )
        .scalars()
        .all()
    )
    assert len({Path(row.storage_path).parent for row in rows}) == 1
    assert all(Path(row.storage_path).is_file() for row in rows)
    assert all(row.manual_extraction_required is True for row in rows)


@pytest.mark.asyncio
async def test_publication_failure_leaves_whole_group_staging_and_parent_writable(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    _, user_text = await auth_headers(client, "zip-publish-fail@example.com")
    user_id = UUID(user_text)
    patient = await coordinator.get_or_create_patient(db_session, user_id)
    parent = UploadedFile(
        user_id=user_id,
        filename="parent.zip",
        mime_type="application/zip",
        file_size_bytes=1,
        file_hash="c" * 64,
        storage_path=str(tmp_path / "parent.zip"),
        ingestion_status="processing",
    )
    db_session.add(parent)
    await db_session.commit()
    await db_session.refresh(parent)
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(coordinator.settings, "upload_dir", str(upload_root))
    monkeypatch.setattr(
        coordinator.settings,
        "temp_extract_dir",
        str(tmp_path / "extract"),
    )
    archive = tmp_path / "children.zip"
    archive.write_bytes(_make_zip({"one.pdf": b"%PDF-1.7\nONE"}))

    def fail_publish(_self: ZipChildSet) -> None:
        raise OSError("injected set publication failure")

    monkeypatch.setattr(ZipChildSet, "publish", fail_publish)

    with pytest.raises(OSError, match="publication failure"):
        await coordinator._ingest_zip(
            db_session,
            user_id,
            patient.id,
            parent.id,
            archive,
        )

    rows = (
        (
            await db_session.execute(
                select(UploadedFile).where(
                    UploadedFile.user_id == user_id,
                    UploadedFile.file_category == "unstructured",
                )
            )
        )
        .scalars()
        .all()
    )
    assert rows
    assert {row.ingestion_status for row in rows} == {STAGING_EXTRACTION_STATUS}
    assert not Path(rows[0].storage_path).parent.exists()
    assert (upload_root / f".{Path(rows[0].storage_path).parent.name}.pending").is_dir()

    parent.ingestion_status = "failed"
    await db_session.commit()
    await db_session.refresh(parent)
    assert parent.ingestion_status == "failed"


@pytest.mark.asyncio
async def test_status_commit_failure_leaves_published_group_recoverable(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    _, user_text = await auth_headers(client, "zip-status-fail@example.com")
    user_id = UUID(user_text)
    patient = await coordinator.get_or_create_patient(db_session, user_id)
    parent = UploadedFile(
        user_id=user_id,
        filename="parent.zip",
        mime_type="application/zip",
        file_size_bytes=1,
        file_hash="d" * 64,
        storage_path=str(tmp_path / "parent.zip"),
        ingestion_status="processing",
    )
    db_session.add(parent)
    await db_session.commit()
    await db_session.refresh(parent)
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(coordinator.settings, "upload_dir", str(upload_root))
    monkeypatch.setattr(
        coordinator.settings,
        "temp_extract_dir",
        str(tmp_path / "extract"),
    )
    archive = tmp_path / "children.zip"
    archive.write_bytes(_make_zip({"one.pdf": b"%PDF-1.7\nONE"}))

    original_commit = db_session.commit
    commit_count = 0

    async def fail_second_commit() -> None:
        nonlocal commit_count
        commit_count += 1
        if commit_count == 2:
            raise RuntimeError("injected status commit failure")
        await original_commit()

    monkeypatch.setattr(db_session, "commit", fail_second_commit)
    with pytest.raises(RuntimeError, match="status commit failure"):
        await coordinator._ingest_zip(
            db_session,
            user_id,
            patient.id,
            parent.id,
            archive,
        )

    rows = (
        (
            await db_session.execute(
                select(UploadedFile).where(
                    UploadedFile.user_id == user_id,
                    UploadedFile.file_category == "unstructured",
                )
            )
        )
        .scalars()
        .all()
    )
    assert {row.ingestion_status for row in rows} == {STAGING_EXTRACTION_STATUS}
    assert all(Path(row.storage_path).is_file() for row in rows)

    monkeypatch.setattr(db_session, "commit", original_commit)
    recovery = await reconcile_zip_child_sets(db_session, upload_root)
    for row in rows:
        await db_session.refresh(row)
    assert recovery.recovered_groups == 1
    assert {row.ingestion_status for row in rows} == {"pending_extraction"}
    assert {row.manual_extraction_required for row in rows} == {True}
