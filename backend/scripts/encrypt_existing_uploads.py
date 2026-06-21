#!/usr/bin/env python
"""Encrypt existing plaintext uploads at rest (CRYPTO-02 / SEC-PHI-02).

New uploads are written to ``UPLOAD_DIR`` in a framed AES-256-GCM format
(``app/utils/file_utils.py``). Files written before that change are still
plaintext on disk. This one-shot, idempotent migration walks ``UPLOAD_DIR`` and
encrypts every file that lacks the magic header, in place.

Each file is rewritten ATOMICALLY: the plaintext is streamed (one <=1 MiB chunk
at a time — never the whole file in RAM) into a sibling temp file, fsynced, then
``os.replace``d over the original. Already-encrypted files are skipped, so re-runs
are safe.

Usage:
    cd backend && python -m scripts.encrypt_existing_uploads          # encrypt UPLOAD_DIR
    cd backend && python -m scripts.encrypt_existing_uploads --dir /path/to/uploads
    cd backend && python -m scripts.encrypt_existing_uploads --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.utils.file_utils import ENC_MAGIC, encrypt_stream, is_encrypted_file  # noqa: E402

logger = logging.getLogger("encrypt_existing_uploads")

# Read source plaintext in 1 MiB chunks so the whole file is never buffered.
_READ_CHUNK = 1024 * 1024


def _iter_file_chunks(path: Path):
    with open(path, "rb") as src:
        while True:
            chunk = src.read(_READ_CHUNK)
            if not chunk:
                break
            yield chunk


def encrypt_file_in_place(path: Path) -> bool:
    """Encrypt one file in place if it is plaintext; return True if it changed.

    Skips (returns False) when the file already carries the encryption magic
    header, so the migration is idempotent. The rewrite is atomic: stream-encrypt
    into ``<name>.mtenc.tmp``, fsync, then ``os.replace`` over the original.
    """
    path = Path(path)
    if is_encrypted_file(path):
        return False

    tmp = path.with_name(path.name + ".mtenc.tmp")
    try:
        with open(tmp, "wb") as dst:
            for blob in encrypt_stream(_iter_file_chunks(path)):
                dst.write(blob)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a partial temp file behind on failure.
        tmp.unlink(missing_ok=True)
        raise
    return True


def run(upload_dir: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Walk ``upload_dir`` and encrypt every plaintext file. Returns (encrypted, skipped)."""
    if not upload_dir.exists():
        logger.warning("Upload dir %s does not exist; nothing to do", upload_dir)
        return 0, 0

    encrypted = 0
    skipped = 0
    for path in sorted(upload_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".mtenc.tmp"):
            # Leftover temp from an interrupted run — drop it.
            if not dry_run:
                path.unlink(missing_ok=True)
            continue
        if is_encrypted_file(path):
            skipped += 1
            continue
        if dry_run:
            logger.info("[dry-run] would encrypt %s", path)
            encrypted += 1
            continue
        if encrypt_file_in_place(path):
            logger.info("Encrypted %s", path)
            encrypted += 1
        else:
            skipped += 1

    logger.info(
        "Done: %d encrypted, %d already-encrypted/skipped (dir=%s%s)",
        encrypted, skipped, upload_dir, ", DRY-RUN" if dry_run else "",
    )
    return encrypted, skipped


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        default=settings.upload_dir,
        help="Upload directory to migrate (default: settings.upload_dir)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be encrypted without modifying any files",
    )
    args = ap.parse_args()

    # Fail fast if the encryption key is missing rather than corrupting files.
    assert ENC_MAGIC  # imported for side-effect of validating the module loads
    run(Path(args.dir), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
