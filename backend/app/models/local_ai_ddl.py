"""PostgreSQL guards installed by SQLAlchemy ``create_all``.

Alembic remains authoritative for deployed databases. These matching DDL
events keep fresh create-all schemas, including the test database, subject
to the same strict-local manifest and immutability constraints.

SQLAlchemy interpolates ``DDL`` strings with the percent operator, so literal
SQL percent signs in these migration-matching statements are doubled.
"""

from __future__ import annotations

from sqlalchemy import DDL

LOCAL_AI_JOB_DATABASE_GUARDS: tuple[DDL, ...] = (
    DDL(
        """
                CREATE OR REPLACE FUNCTION local_ai_json_has_exact_keys(
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
    ).execute_if(dialect="postgresql"),
    DDL(
        """
                CREATE OR REPLACE FUNCTION local_ai_manifest_string_is_valid(payload text)
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
    ).execute_if(dialect="postgresql"),
    DDL(
        r"""
                CREATE OR REPLACE FUNCTION local_ai_json_ascii_string(payload text)
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
    ).execute_if(dialect="postgresql"),
    DDL(
        """
                CREATE OR REPLACE FUNCTION local_ai_canonical_json(payload jsonb)
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
    ).execute_if(dialect="postgresql"),
    DDL(
        r"""
                CREATE OR REPLACE FUNCTION local_ai_manifest_is_valid(payload jsonb)
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
                    IF jsonb_typeof(payload->'schema_version') <> 'number'
                       OR payload->>'schema_version' <> '2'
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
                        ARRAY[
                            'name',
                            'version',
                            'worker_identity_scheme',
                            'worker_bundle_sha256'
                        ]
                    )
                       OR jsonb_typeof(payload->'runtime'->'name') <> 'string'
                       OR NOT local_ai_manifest_string_is_valid(
                            payload->'runtime'->>'name'
                       )
                       OR jsonb_typeof(payload->'runtime'->'version') <> 'string'
                       OR NOT local_ai_manifest_string_is_valid(
                            payload->'runtime'->>'version'
                       )
                       OR jsonb_typeof(
                            payload->'runtime'->'worker_identity_scheme'
                       ) <> 'string'
                       OR payload->'runtime'->>'worker_identity_scheme'
                            <> 'local-ai-worker-bundle.v1'
                       OR jsonb_typeof(
                            payload->'runtime'->'worker_bundle_sha256'
                       ) <> 'string'
                       OR payload->'runtime'->>'worker_bundle_sha256'
                            !~ '^[0-9a-f]{64}$' THEN
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
                               OR file_path LIKE '%%//%%'
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
    ).execute_if(dialect="postgresql"),
    DDL(
        """
                CREATE OR REPLACE FUNCTION local_ai_legacy_manifest_is_valid(
                    payload jsonb
                )
                RETURNS boolean
                LANGUAGE plpgsql
                IMMUTABLE
                STRICT
                AS $$
                DECLARE
                    attested_payload jsonb;
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
                    )
                       OR jsonb_typeof(payload->'schema_version') <> 'number'
                       OR payload->>'schema_version' <> '1'
                       OR NOT local_ai_json_has_exact_keys(
                            payload->'runtime',
                            ARRAY['name', 'version']
                       ) THEN
                        RETURN false;
                    END IF;
                    attested_payload := jsonb_set(
                        jsonb_set(payload, '{schema_version}', '2'::jsonb),
                        '{runtime}',
                        payload->'runtime' || jsonb_build_object(
                            'worker_identity_scheme',
                            'local-ai-worker-bundle.v1',
                            'worker_bundle_sha256',
                            repeat('0', 64)
                        )
                    );
                    RETURN local_ai_manifest_is_valid(attested_payload);
                END;
                $$
                """
    ).execute_if(dialect="postgresql"),
    DDL(
        """
                CREATE OR REPLACE FUNCTION enforce_local_ai_job_identity()
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
                    canonical_digest := encode(
                        sha256(
                            convert_to(
                                local_ai_canonical_json(NEW.manifest_snapshot),
                                'UTF8'
                            )
                        ),
                        'hex'
                    );
                    IF NOT local_ai_manifest_is_valid(NEW.manifest_snapshot) THEN
                        IF NOT (
                            TG_OP = 'UPDATE'
                            AND local_ai_legacy_manifest_is_valid(
                                OLD.manifest_snapshot
                            )
                            AND OLD.manifest_sha256 ~ '^[0-9a-f]{64}$'
                            AND OLD.manifest_sha256 = canonical_digest
                            AND NEW.id IS NOT DISTINCT FROM OLD.id
                            AND NEW.created_at IS NOT DISTINCT FROM OLD.created_at
                            AND NEW.user_id IS NOT DISTINCT FROM OLD.user_id
                            AND NEW.kind IS NOT DISTINCT FROM OLD.kind
                            AND NEW.upload_id IS NOT DISTINCT FROM OLD.upload_id
                            AND NEW.summary_prompt_id
                                IS NOT DISTINCT FROM OLD.summary_prompt_id
                            AND NEW.processing_mode
                                IS NOT DISTINCT FROM OLD.processing_mode
                            AND NEW.manifest_snapshot
                                IS NOT DISTINCT FROM OLD.manifest_snapshot
                            AND NEW.manifest_sha256
                                IS NOT DISTINCT FROM OLD.manifest_sha256
                            AND NEW.audit_metadata
                                IS NOT DISTINCT FROM OLD.audit_metadata
                            AND NEW.cancel_requested
                                IS NOT DISTINCT FROM OLD.cancel_requested
                            AND NEW.started_at
                                IS NOT DISTINCT FROM OLD.started_at
                            AND OLD.status IN ('queued', 'processing')
                            AND NEW.status = 'failed'
                            AND NEW.stage = 'failed'
                            AND NEW.progress = '{"stage": "failed"}'::jsonb
                            AND NEW.failure = '{"stage": "failed", "code": "runtime_identity_required", "retryable": false, "checkpoint_preserved": false, "cloud_fallback_attempted": false}'::jsonb
                            AND NEW.completed_at IS NOT NULL
                        ) THEN
                            RAISE EXCEPTION USING
                                ERRCODE = '23514',
                                MESSAGE = 'local AI job manifest is invalid';
                        END IF;
                        RETURN NEW;
                    END IF;
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
    ).execute_if(dialect="postgresql"),
    DDL(
        """
                CREATE TRIGGER trg_local_ai_jobs_immutable_identity
                BEFORE INSERT OR UPDATE
                ON local_ai_jobs
                FOR EACH ROW
                EXECUTE FUNCTION enforce_local_ai_job_identity()
                """
    ).execute_if(dialect="postgresql"),
)

EXTRACTION_EVIDENCE_DATABASE_GUARDS: tuple[DDL, ...] = (
    DDL(
        """
                CREATE OR REPLACE FUNCTION reject_extraction_evidence_scope_update()
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
    ).execute_if(dialect="postgresql"),
    DDL(
        """
                CREATE TRIGGER trg_extraction_evidence_immutable_scope
                BEFORE UPDATE OF user_id, upload_id
                ON extraction_evidence
                FOR EACH ROW
                EXECUTE FUNCTION reject_extraction_evidence_scope_update()
                """
    ).execute_if(dialect="postgresql"),
)
