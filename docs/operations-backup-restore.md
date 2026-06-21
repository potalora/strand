# Backup, restore, and data retention

Operator runbook for MedTimeline. Covers what to back up, how to back up and restore a self-hosted instance, how to verify a restore, and the current data-retention behavior.

This maps to HIPAA contingency planning (§164.308(a)(7), data backup and disaster recovery) and the documentation-retention requirement (§164.316(b)(2)(i)). A single-user self-hosted instance is not a covered entity, so treat this as operational guidance, not a compliance mandate. If you ever host this for others, see "Data retention" below.

## Read this first: the encryption key is part of the backup

PHI in this app is encrypted at rest with AES-256-GCM under `DATABASE_ENCRYPTION_KEY`. As of the current schema, the ciphertext columns include:

- `health_records.fhir_resource` (the clinical JSONB)
- `uploaded_files.extracted_text` and the JSONB extraction columns
- `ai_summary_prompts` prompts and responses
- `users.email` (with `users.email_hmac` as a blind index for lookups)
- `patients` demographic identifiers (name, MRN, DOB, contact)
- stored per-user LLM API keys

Uploaded source files on disk are partly encrypted: unstructured uploads (scanned PDFs, TIFFs, RTF) are encrypted at rest with the same key, while structured archives (FHIR bundles, Epic exports, ZIP, CDA) are still written in plaintext because the ingestion path reads them by streaming and a streaming decrypt is a separate change. So treat the uploads directory as a mix of ciphertext and plaintext, and back up `DATABASE_ENCRYPTION_KEY` regardless. Enabling OS full-disk encryption on the host covers the plaintext files until structured-upload encryption lands.

What this means for backups:

- **A database dump and a file backup are ciphertext. Without `DATABASE_ENCRYPTION_KEY` they cannot be decrypted, by you or anyone else.** Lose the key and the backup is permanently unreadable. There is no recovery path and no password reset that brings the data back.
- **Back up the key separately from the data.** Do not put the key inside the database dump, in the same archive, or in the same storage location or bucket. The point of separation is that one stolen artifact is useless on its own: a backup without the key leaks nothing, and a key without a backup leaks nothing.
- **The key is itself a secret to protect.** Anyone who holds both a backup and the key has full plaintext PHI. Store the key in a password manager or a secrets vault, with its own access control, and keep at least one offline copy.

The key lives in `.env` as `DATABASE_ENCRYPTION_KEY` (64 hex characters, 32 bytes). `JWT_SECRET_KEY` and `DB_PASSWORD` live there too. Back up the whole `.env`, but keep it apart from the data backup for the same reason.

## What to back up

Three things, and all three are needed for a working restore:

1. **The Postgres database** (`medtimeline`). Holds every record, upload row, summary, user, and the append-only `audit_log`.
2. **The uploads directory** (native: `data/uploads`, the `UPLOAD_DIR` setting; containerized: the `uploads` named volume mounted at `/data`). Holds the original uploaded files.
3. **The secrets**, above all `DATABASE_ENCRYPTION_KEY`. Stored separately, with its own protection.

The Redis instance does not need backing up. It holds transient job state, not source data.

## Backup procedure

### Native (Homebrew Postgres on the host)

```bash
# 1. Database, custom format (compressed, restores with pg_restore)
pg_dump -Fc medtimeline -f medtimeline-$(date +%Y%m%d).dump

# 2. Uploaded files
tar -czf uploads-$(date +%Y%m%d).tar.gz -C data uploads

# 3. Secrets: copy .env somewhere separate from the two files above
#    (a password manager entry, a vault, an offline drive). Do NOT
#    drop it next to the dump.
```

### Containerized (`docker compose`)

The `db` service has no host port by default, so run `pg_dump` inside the container and stream the output to the host:

```bash
# 1. Database
docker compose exec -T db pg_dump -Fc -U postgres medtimeline > medtimeline-$(date +%Y%m%d).dump

# 2. Uploads: copy the named volume's contents out via a throwaway container
docker run --rm \
  -v test_autonomous_ai_web_records_uploads:/data:ro \
  -v "$PWD":/backup \
  alpine tar -czf /backup/uploads-$(date +%Y%m%d).tar.gz -C /data .

# 3. Secrets: same as native, copy .env to separate, protected storage.
```

Check the volume name with `docker volume ls` if `test_autonomous_ai_web_records_uploads` does not match. Compose prefixes volumes with the project directory name.

### Cadence and storage

- Run a backup on a schedule that matches how often you ingest new records. Weekly is a reasonable floor for an instance that gets occasional uploads; daily if you import frequently.
- The database dump and the uploads archive are ciphertext, but still store them with access control and on encrypted media (FileVault, LUKS, BitLocker, or an encrypted backup target). Defense in depth, not because the contents are plaintext.
- Keep more than one generation. A single rolling backup gives you no recovery point if the latest run captured corruption.
- Test a restore periodically. A backup you have never restored is a guess, not a backup.

## Restore procedure

The same `DATABASE_ENCRYPTION_KEY` that encrypted the data must be in place before the app reads anything. Restore the database and files first, put the key back, then start the app.

### Native

```bash
# 1. Create the target database (skip if restoring over an existing one)
createdb medtimeline
psql medtimeline -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

# 2. Restore the dump
pg_restore -d medtimeline --no-owner medtimeline-YYYYMMDD.dump

# 3. Restore uploaded files
tar -xzf uploads-YYYYMMDD.tar.gz -C data

# 4. Put the SAME DATABASE_ENCRYPTION_KEY (and the rest of .env) back in place.
#    A wrong or missing key makes every encrypted column fail to decrypt.

# 5. Fresh environment only: bring the schema up to date
alembic upgrade head
```

### Containerized

```bash
# 1. Start just the database
docker compose up -d db

# 2. Restore into it
docker compose exec -T db psql -U postgres -c "CREATE DATABASE medtimeline;" 2>/dev/null || true
docker compose exec -T db psql -U postgres -d medtimeline -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
docker compose exec -T db pg_restore -U postgres -d medtimeline --no-owner < medtimeline-YYYYMMDD.dump

# 3. Restore uploads into the named volume
docker run --rm \
  -v test_autonomous_ai_web_records_uploads:/data \
  -v "$PWD":/backup \
  alpine sh -c "tar -xzf /backup/uploads-YYYYMMDD.tar.gz -C /data"

# 4. Confirm .env carries the SAME DATABASE_ENCRYPTION_KEY as the source instance.

# 5. Bring the rest of the stack up. The one-shot `migrate` service runs
#    `alembic upgrade head` before the backend starts.
docker compose up -d
```

Notes:

- `--no-owner` avoids restore failures when the role names differ between the source and target. Drop it if you are restoring onto an identical role setup.
- The init script (`scripts/init-db.sql`) creates `pgcrypto`, `uuid-ossp`, and `pg_trgm` on a container's first boot. A restore into a fresh native database needs `pgcrypto` created by hand, as shown above. The extension is enabled but the app does not use it for encryption; the encryption is done in the application layer.
- If you are restoring a `pg_dump` taken at an older schema version into a newer codebase, run `alembic upgrade head` after the restore.

## Verify the restore

Confirm the data both loaded and decrypts. A row count alone does not prove the key is right, because ciphertext counts the same with the wrong key.

1. **Start the app and sign in.** A failed login here usually means `users.email` is not decrypting, which points at a wrong or missing `DATABASE_ENCRYPTION_KEY`.
2. **Open a record.** If the timeline and a record detail render real clinical content (not blanks or an error), the clinical JSONB is decrypting under the current key.
3. **Spot-check counts** against what you expect:

   ```bash
   # native
   psql medtimeline -c "SELECT count(*) FROM health_records WHERE deleted_at IS NULL;"
   psql medtimeline -c "SELECT count(*) FROM uploaded_files;"

   # containerized
   docker compose exec -T db psql -U postgres -d medtimeline \
     -c "SELECT count(*) FROM health_records WHERE deleted_at IS NULL;"
   ```

If records exist but none render, stop and check the key before doing anything else. Do not re-ingest or "fix" data on a wrong key; restore the correct key instead.

## Data retention

Current behavior, stated plainly:

- **Records are soft-deleted, never hard-deleted.** A delete sets `deleted_at`; the row stays in the table and queries filter it out. Nothing in the app purges soft-deleted records. The same holds for `uploaded_files`.
- **`audit_log` is append-only and is never purged.** A database trigger blocks `UPDATE` and `DELETE` on the table (migration `a1b2c3d4e5f7`, W15 / AUDIT-01), so the access trail cannot be edited or trimmed from the application.
- **`revoked_tokens` is purged of expired rows automatically.** Once a token is past its `expires_at` the JWT layer rejects it regardless, so an expired blacklist row is dead weight. The app deletes expired rows at startup (W23), and an operator can prune by hand at any time:

  ```sql
  DELETE FROM revoked_tokens WHERE expires_at < now();
  ```

What this means against retention obligations: if you ever fall under HIPAA or a similar regime, the rule of thumb is to retain audit records and ePHI-related documentation for at least six years (§164.316(b)(2)(i)). The soft-delete and append-only design already supports that. It keeps history rather than discarding it, so the default posture is over-retention, not loss.

The gap for a hosted or covered-entity deployment is the other direction: a documented disposal policy. That would spell out when soft-deleted records and old audit data are actually destroyed, who authorizes it, and how the media is sanitized. For the single-user self-hosted posture this is guidance rather than a requirement, because there is no covered entity and no business associate agreement in play. Revisit it before you operate the app on anyone else's behalf.
