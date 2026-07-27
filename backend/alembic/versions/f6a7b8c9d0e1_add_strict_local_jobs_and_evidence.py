"""Add strict-local jobs, encrypted page checkpoints, and evidence.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-26
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_uploaded_files_id_user_id",
        "uploaded_files",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uq_ai_summary_prompts_id_user_id",
        "ai_summary_prompts",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uq_health_records_id_source_file_user",
        "health_records",
        ["id", "source_file_id", "user_id"],
    )

    op.add_column(
        "user_llm_preferences",
        sa.Column("processing_mode", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "uploaded_files",
        sa.Column(
            "processing_mode",
            sa.String(length=32),
            server_default="cloud_assisted",
            nullable=False,
        ),
    )
    op.add_column(
        "uploaded_files",
        sa.Column("processing_manifest", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "uploaded_files",
        sa.Column("processing_schema_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "ai_summary_prompts",
        sa.Column(
            "processing_mode",
            sa.String(length=32),
            server_default="cloud_assisted",
            nullable=False,
        ),
    )
    op.add_column(
        "ai_summary_prompts",
        sa.Column("model_provenance", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "ai_summary_prompts",
        sa.Column("typed_response", sa.LargeBinary(), nullable=True),
    )

    op.create_table(
        "local_ai_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("upload_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("summary_prompt_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("processing_mode", sa.String(length=32), nullable=False),
        sa.Column("manifest_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column(
            "progress",
            postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("failure", postgresql.JSONB(), nullable=True),
        sa.Column(
            "audit_metadata",
            postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(kind = 'ingestion' AND upload_id IS NOT NULL "
            "AND summary_prompt_id IS NULL) "
            "OR (kind = 'summary' AND upload_id IS NULL "
            "AND summary_prompt_id IS NOT NULL)",
            name="ck_local_ai_jobs_kind_target",
        ),
        sa.ForeignKeyConstraint(
            ["summary_prompt_id", "user_id"],
            ["ai_summary_prompts.id", "ai_summary_prompts.user_id"],
            name="fk_local_ai_jobs_summary_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["upload_id", "user_id"],
            ["uploaded_files.id", "uploaded_files.user_id"],
            name="fk_local_ai_jobs_upload_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_local_ai_jobs_user_status",
        "local_ai_jobs",
        ["user_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_local_ai_jobs_upload_id",
        "local_ai_jobs",
        ["upload_id"],
        unique=False,
    )
    op.create_index(
        "ix_local_ai_jobs_summary_prompt_id",
        "local_ai_jobs",
        ["summary_prompt_id"],
        unique=False,
    )
    op.execute(
        """
        CREATE FUNCTION local_ai_json_has_exact_keys(
            payload jsonb,
            expected_keys text[]
        )
        RETURNS boolean
        LANGUAGE sql
        IMMUTABLE
        AS $$
            SELECT COALESCE(
                jsonb_typeof(payload) = 'object'
                AND payload ?& expected_keys
                AND (
                    SELECT count(*)
                    FROM jsonb_object_keys(payload)
                ) = cardinality(expected_keys),
                false
            )
        $$
        """
    )
    op.execute(
        """
        CREATE FUNCTION local_ai_manifest_string_is_valid(payload text)
        RETURNS boolean
        LANGUAGE plpgsql
        IMMUTABLE
        STRICT
        AS $$
        DECLARE
            position integer;
        BEGIN
            IF payload = '' OR char_length(payload) > 2048 THEN
                RETURN false;
            END IF;
            FOR position IN 1..char_length(payload) LOOP
                IF ascii(substr(payload, position, 1)) < 32 THEN
                    RETURN false;
                END IF;
            END LOOP;
            RETURN true;
        END;
        $$
        """
    )
    op.execute(
        r"""
        CREATE FUNCTION local_ai_json_ascii_string(payload text)
        RETURNS text
        LANGUAGE plpgsql
        IMMUTABLE
        STRICT
        AS $$
        DECLARE
            result text := '"';
            character text;
            codepoint integer;
            position integer;
            surrogate integer;
        BEGIN
            FOR position IN 1..char_length(payload) LOOP
                character := substr(payload, position, 1);
                codepoint := ascii(character);
                CASE codepoint
                    WHEN 8 THEN result := result || '\b';
                    WHEN 9 THEN result := result || '\t';
                    WHEN 10 THEN result := result || '\n';
                    WHEN 12 THEN result := result || '\f';
                    WHEN 13 THEN result := result || '\r';
                    WHEN 34 THEN result := result || '\"';
                    WHEN 92 THEN result := result || '\\';
                    ELSE
                        IF codepoint < 32 THEN
                            result := result || '\u'
                                || lpad(to_hex(codepoint), 4, '0');
                        ELSIF codepoint <= 127 THEN
                            result := result || character;
                        ELSIF codepoint <= 65535 THEN
                            result := result || '\u'
                                || lpad(to_hex(codepoint), 4, '0');
                        ELSE
                            surrogate := codepoint - 65536;
                            result := result || '\u'
                                || lpad(to_hex(55296 + (surrogate >> 10)), 4, '0')
                                || '\u'
                                || lpad(to_hex(56320 + (surrogate & 1023)), 4, '0');
                        END IF;
                END CASE;
            END LOOP;
            RETURN result || '"';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE FUNCTION local_ai_canonical_json(payload jsonb)
        RETURNS text
        LANGUAGE plpgsql
        IMMUTABLE
        STRICT
        AS $$
        DECLARE
            result text;
        BEGIN
            CASE jsonb_typeof(payload)
                WHEN 'object' THEN
                    SELECT '{' || COALESCE(
                        string_agg(
                            local_ai_json_ascii_string(entry.key)
                            || ':'
                            || local_ai_canonical_json(entry.value),
                            ',' ORDER BY entry.key
                        ),
                        ''
                    ) || '}'
                    INTO result
                    FROM jsonb_each(payload) AS entry;
                WHEN 'array' THEN
                    SELECT '[' || COALESCE(
                        string_agg(
                            local_ai_canonical_json(entry.value),
                            ',' ORDER BY entry.position
                        ),
                        ''
                    ) || ']'
                    INTO result
                    FROM jsonb_array_elements(payload)
                        WITH ORDINALITY AS entry(value, position);
                WHEN 'string' THEN
                    result := local_ai_json_ascii_string(payload #>> '{}');
                ELSE
                    result := payload::text;
            END CASE;
            RETURN result;
        END;
        $$
        """
    )
    op.execute(
        r"""
        CREATE FUNCTION local_ai_manifest_is_valid(payload jsonb)
        RETURNS boolean
        LANGUAGE plpgsql
        IMMUTABLE
        STRICT
        AS $$
        DECLARE
            artifact jsonb;
            manifest_file jsonb;
            role_name text;
            roles text[] := ARRAY[]::text[];
            file_path text;
            file_paths text[];
            path_suffix text;
            file_size numeric;
            total_files integer := 0;
            total_bytes numeric := 0;
        BEGIN
            IF NOT local_ai_json_has_exact_keys(
                payload,
                ARRAY[
                    'schema_version',
                    'pack_revision',
                    'platform',
                    'runtime',
                    'validation_suite_version',
                    'artifacts'
                ]
            ) THEN
                RETURN false;
            END IF;
            IF payload->'schema_version' <> '1'::jsonb
               OR jsonb_typeof(payload->'pack_revision') <> 'string'
               OR NOT local_ai_manifest_string_is_valid(payload->>'pack_revision')
               OR payload->>'pack_revision'
                    !~ '^[a-z0-9][a-z0-9._-]{0,127}$'
               OR payload->'platform' <> '"apple_silicon"'::jsonb
               OR jsonb_typeof(payload->'validation_suite_version') <> 'string'
               OR NOT local_ai_manifest_string_is_valid(
                    payload->>'validation_suite_version'
               ) THEN
                RETURN false;
            END IF;
            IF NOT local_ai_json_has_exact_keys(
                payload->'runtime',
                ARRAY['name', 'version']
            )
               OR jsonb_typeof(payload->'runtime'->'name') <> 'string'
               OR NOT local_ai_manifest_string_is_valid(
                    payload->'runtime'->>'name'
               )
               OR jsonb_typeof(payload->'runtime'->'version') <> 'string'
               OR NOT local_ai_manifest_string_is_valid(
                    payload->'runtime'->>'version'
               ) THEN
                RETURN false;
            END IF;
            IF jsonb_typeof(payload->'artifacts') <> 'array'
               OR jsonb_array_length(payload->'artifacts') <> 3 THEN
                RETURN false;
            END IF;

            FOR artifact IN
                SELECT value FROM jsonb_array_elements(payload->'artifacts')
            LOOP
                IF NOT local_ai_json_has_exact_keys(
                    artifact,
                    ARRAY[
                        'role',
                        'repository',
                        'revision',
                        'quantization',
                        'license',
                        'attribution',
                        'decode_limits',
                        'files'
                    ]
                ) THEN
                    RETURN false;
                END IF;

                role_name := artifact->>'role';
                IF jsonb_typeof(artifact->'role') <> 'string'
                   OR role_name NOT IN ('ocr', 'extraction', 'summary') THEN
                    RETURN false;
                END IF;
                roles := array_append(roles, role_name);

                IF jsonb_typeof(artifact->'repository') <> 'string'
                   OR NOT local_ai_manifest_string_is_valid(
                        artifact->>'repository'
                   )
                   OR artifact->>'repository'
                        !~ '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$'
                   OR jsonb_typeof(artifact->'revision') <> 'string'
                   OR artifact->>'revision' !~ '^[0-9a-f]{40}$'
                   OR jsonb_typeof(artifact->'quantization') <> 'string'
                   OR NOT local_ai_manifest_string_is_valid(
                        artifact->>'quantization'
                   )
                   OR jsonb_typeof(artifact->'license') <> 'string'
                   OR artifact->>'license' NOT IN (
                        'apache-2.0',
                        'bsd-2-clause',
                        'bsd-3-clause',
                        'mit'
                   )
                   OR jsonb_typeof(artifact->'attribution') <> 'string'
                   OR NOT local_ai_manifest_string_is_valid(
                        artifact->>'attribution'
                   ) THEN
                    RETURN false;
                END IF;

                IF NOT local_ai_json_has_exact_keys(
                    artifact->'decode_limits',
                    ARRAY['max_input_tokens', 'max_output_tokens']
                )
                   OR jsonb_typeof(
                        artifact->'decode_limits'->'max_input_tokens'
                   ) <> 'number'
                   OR artifact->'decode_limits'->>'max_input_tokens'
                        !~ '^[1-9][0-9]*$'
                   OR (
                        artifact->'decode_limits'->>'max_input_tokens'
                   )::numeric > 1000000
                   OR jsonb_typeof(
                        artifact->'decode_limits'->'max_output_tokens'
                   ) <> 'number'
                   OR artifact->'decode_limits'->>'max_output_tokens'
                        !~ '^[1-9][0-9]*$'
                   OR (
                        artifact->'decode_limits'->>'max_output_tokens'
                   )::numeric > 1000000 THEN
                    RETURN false;
                END IF;

                IF jsonb_typeof(artifact->'files') <> 'array'
                   OR jsonb_array_length(artifact->'files') = 0 THEN
                    RETURN false;
                END IF;
                file_paths := ARRAY[]::text[];
                FOR manifest_file IN
                    SELECT value FROM jsonb_array_elements(artifact->'files')
                LOOP
                    IF NOT local_ai_json_has_exact_keys(
                        manifest_file,
                        ARRAY['path', 'sha256', 'size']
                    ) THEN
                        RETURN false;
                    END IF;
                    IF jsonb_typeof(manifest_file->'path') <> 'string' THEN
                        RETURN false;
                    END IF;
                    file_path := manifest_file->>'path';
                    path_suffix := substring(file_path FROM '(\.[^./]+)$');
                    IF file_path = ''
                       OR strpos(file_path, '\') > 0
                       OR left(file_path, 1) = '/'
                       OR right(file_path, 1) = '/'
                       OR file_path LIKE '%//%'
                       OR file_path ~ '(^|/)\.{1,2}(/|$)'
                       OR path_suffix IS NULL
                       OR path_suffix <> lower(path_suffix)
                       OR path_suffix NOT IN (
                            '.safetensors',
                            '.json',
                            '.txt',
                            '.model',
                            '.tiktoken',
                            '.jinja',
                            '.md',
                            '.license'
                       )
                       OR file_path = ANY(file_paths)
                       OR jsonb_typeof(manifest_file->'sha256') <> 'string'
                       OR manifest_file->>'sha256' !~ '^[0-9a-f]{64}$'
                       OR jsonb_typeof(manifest_file->'size') <> 'number'
                       OR manifest_file->>'size' !~ '^[1-9][0-9]*$' THEN
                        RETURN false;
                    END IF;
                    file_size := (manifest_file->>'size')::numeric;
                    IF file_size > 8589934592 THEN
                        RETURN false;
                    END IF;
                    file_paths := array_append(file_paths, file_path);
                    total_files := total_files + 1;
                    total_bytes := total_bytes + file_size;
                END LOOP;
            END LOOP;

            IF NOT (
                roles @> ARRAY['ocr', 'extraction', 'summary']
                AND ARRAY['ocr', 'extraction', 'summary'] @> roles
            )
               OR total_files > 64
               OR total_bytes > 21474836480 THEN
                RETURN false;
            END IF;
            RETURN true;
        EXCEPTION
            WHEN data_exception OR numeric_value_out_of_range THEN
                RETURN false;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_local_ai_job_identity()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            canonical_digest text;
        BEGIN
            IF TG_OP = 'UPDATE'
               AND (
                    NEW.user_id IS DISTINCT FROM OLD.user_id
                    OR NEW.kind IS DISTINCT FROM OLD.kind
                    OR NEW.upload_id IS DISTINCT FROM OLD.upload_id
                    OR NEW.summary_prompt_id IS DISTINCT FROM OLD.summary_prompt_id
                    OR NEW.processing_mode IS DISTINCT FROM OLD.processing_mode
                    OR NEW.manifest_snapshot IS DISTINCT FROM OLD.manifest_snapshot
                    OR NEW.manifest_sha256 IS DISTINCT FROM OLD.manifest_sha256
               ) THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'local AI job identity is immutable';
            END IF;
            IF NEW.processing_mode NOT IN (
                'validated_strict_local',
                'custom_local',
                'cloud_assisted',
                'prompt_only'
            ) THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'local AI job processing mode is invalid';
            END IF;
            IF NOT local_ai_manifest_is_valid(NEW.manifest_snapshot) THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'local AI job manifest is invalid';
            END IF;
            canonical_digest := encode(
                sha256(
                    convert_to(
                        local_ai_canonical_json(NEW.manifest_snapshot),
                        'UTF8'
                    )
                ),
                'hex'
            );
            IF NEW.manifest_sha256 !~ '^[0-9a-f]{64}$'
               OR NEW.manifest_sha256 <> canonical_digest THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'local AI job manifest digest is invalid';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_local_ai_jobs_immutable_identity
        BEFORE INSERT OR UPDATE
        ON local_ai_jobs
        FOR EACH ROW
        EXECUTE FUNCTION enforce_local_ai_job_identity()
        """
    )

    op.create_table(
        "local_ai_pages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("checkpoint_key", sa.String(length=64), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("ocr_result", sa.LargeBinary(), nullable=False),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["local_ai_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id",
            "page_number",
            name="uq_local_ai_pages_job_page",
        ),
    )
    op.create_index(
        "ix_local_ai_pages_job_page",
        "local_ai_pages",
        ["job_id", "page_number"],
        unique=False,
    )

    op.create_table(
        "extraction_evidence",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("upload_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("health_record_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("excerpt", sa.LargeBinary(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("field_paths", sa.LargeBinary(), nullable=False),
        sa.Column("source_metadata", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["health_record_id", "upload_id", "user_id"],
            [
                "health_records.id",
                "health_records.source_file_id",
                "health_records.user_id",
            ],
            name="fk_extraction_evidence_record_lineage",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["health_record_id"],
            ["health_records.id"],
            name="fk_extraction_evidence_health_record",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["upload_id", "user_id"],
            ["uploaded_files.id", "uploaded_files.user_id"],
            name="fk_extraction_evidence_upload_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_extraction_evidence_health_record_id",
        "extraction_evidence",
        ["health_record_id"],
        unique=False,
    )
    op.execute(
        """
        CREATE FUNCTION reject_extraction_evidence_scope_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.user_id IS DISTINCT FROM OLD.user_id
               OR NEW.upload_id IS DISTINCT FROM OLD.upload_id THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'extraction evidence scope is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_extraction_evidence_immutable_scope
        BEFORE UPDATE OF user_id, upload_id
        ON extraction_evidence
        FOR EACH ROW
        EXECUTE FUNCTION reject_extraction_evidence_scope_update()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_extraction_evidence_immutable_scope "
        "ON extraction_evidence"
    )
    op.execute("DROP FUNCTION IF EXISTS reject_extraction_evidence_scope_update()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_local_ai_jobs_immutable_identity "
        "ON local_ai_jobs"
    )
    op.execute("DROP FUNCTION IF EXISTS enforce_local_ai_job_identity()")
    op.execute("DROP FUNCTION IF EXISTS reject_local_ai_job_identity_update()")
    op.execute("DROP FUNCTION IF EXISTS local_ai_manifest_is_valid(jsonb)")
    op.execute("DROP FUNCTION IF EXISTS local_ai_canonical_json(jsonb)")
    op.execute("DROP FUNCTION IF EXISTS local_ai_json_ascii_string(text)")
    op.execute("DROP FUNCTION IF EXISTS local_ai_manifest_string_is_valid(text)")
    op.execute("DROP FUNCTION IF EXISTS local_ai_json_has_exact_keys(jsonb, text[])")
    op.drop_table("extraction_evidence")
    op.drop_table("local_ai_pages")
    op.drop_table("local_ai_jobs")

    op.drop_column("ai_summary_prompts", "typed_response")
    op.drop_column("ai_summary_prompts", "model_provenance")
    op.drop_column("ai_summary_prompts", "processing_mode")
    op.drop_column("uploaded_files", "processing_schema_version")
    op.drop_column("uploaded_files", "processing_manifest")
    op.drop_column("uploaded_files", "processing_mode")
    op.drop_column("user_llm_preferences", "processing_mode")
    op.execute(
        "ALTER TABLE health_records "
        "DROP CONSTRAINT IF EXISTS uq_health_records_id_source_file_user"
    )
    op.execute(
        "ALTER TABLE ai_summary_prompts "
        "DROP CONSTRAINT IF EXISTS uq_ai_summary_prompts_id_user_id"
    )
    op.execute(
        "ALTER TABLE uploaded_files "
        "DROP CONSTRAINT IF EXISTS uq_uploaded_files_id_user_id"
    )
