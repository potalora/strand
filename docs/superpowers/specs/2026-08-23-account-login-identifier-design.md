# General Account Login Identifier Design

**Date:** 2026-08-23

**Issue:** [#67 — Allow account registration without an email address](https://github.com/potalora/strand/issues/67)

**Status:** Approved by Pedro; implementation plan awaiting root approval

**Scope:** Registration and login identity semantics only

## Problem

Strand requires an email address to register and sign in, but it has no email
delivery, verification, or password-reset flow. The value is only an account
login identifier. Requiring email syntax therefore collects unnecessary
personal information and implies functionality or transmission that does not
exist.

The email assumption is not isolated to the form. It is embedded in the auth
request and response schemas, encrypted user model, unique blind index,
migrations, services, audit details, frontend account views, tests, scripts,
and operator documentation. The change must preserve existing accounts and all
unrelated authentication and authorization behavior.

## Goals

- Let a new account register without an email address.
- Give the value an accurate canonical name: `login_identifier` in code and
  API contracts, and "Account name" in user-facing copy.
- Keep every existing account able to sign in with its current email value.
- Preserve compatibility for clients that still send the legacy `email` key or
  read the legacy `email` response field.
- Preserve encryption at rest, blind-index uniqueness, rate limiting, account
  lockout, JWT and refresh-token behavior, audit coverage, and user scoping.
- Avoid storing raw identifiers or identifier-derived fragments in audit logs,
  application logs, and authored error details.
- Keep the migration small, deterministic, and safe for existing encrypted
  databases.

## Non-goals

- Email delivery, verification, password reset, or account recovery.
- A public username, profile handle, or replacement for `display_name`.
- Login by either of two independent identifiers.
- Account rename or profile-edit support.
- Reworking password rules, JWT claims, lockout thresholds, rate-limit
  architecture, or token storage.
- A general Unicode security redesign beyond the existing case-normalization
  contract.

## Decision

Strand will have one canonical, private account login identifier.

- Backend/API name: `login_identifier`
- User-facing name: **Account name**
- Existing email addresses remain valid values.
- New values do not need to resemble an email address.
- No optional contact-email field will be added.

This is a general login identifier rather than a username because it is private,
need not use handle-style syntax, and is not displayed as a social or public
identity. It is preferable to optional email plus username because Strand has no
mail use case, and a second identifier would add unnecessary PII, indexes,
collision rules, and login ambiguity.

## Identifier validation and normalization

The canonical input rules are:

1. Accept a string.
2. Strip leading and trailing whitespace before storage.
3. Require at least one character and at most 255 Unicode code points after
   trimming.
4. Require `str.isprintable()` to be true for the complete trimmed value,
   thereby rejecting control characters and embedded line breaks.
5. Permit printable Unicode, internal spaces, punctuation, and `@`.
6. Compare identifiers by the exact normalized output of
   `value.strip().lower()` for uniqueness and login.

The blind-index normalization remains exactly `value.strip().lower()`. This is
the current email lookup contract. Keeping it unchanged means existing
`email_hmac` values remain valid after the column rename and avoids a decrypted
backfill. The contract is ASCII-case-insensitive for ordinary `A`–`Z` letters,
but it is not full Unicode caseless matching. Two identifiers are the same only
when Python's `str.lower()` produces the same code-point sequence after outer
whitespace is removed.

Examples make that boundary executable:

- `Alice` and `alice` normalize to the same identifier.
- ` Straße ` normalizes to `straße`, while `STRASSE` normalizes to `strasse`;
  those remain distinct.
- Precomposed `Å` and the canonically equivalent sequence `A` + combining ring
  remain distinct because this design performs no Unicode normalization.

The design deliberately does not introduce NFKC or `casefold()` normalization.
Doing so would require decrypting and reindexing every existing identifier and
could create collisions between values that are distinct today. Visually
confusable Unicode values can therefore remain distinct. That tradeoff is
acceptable for this bounded compatibility change and should be revisited only
as a separate multi-user identity-hardening design.

## API contract

### Registration

Canonical request:

```json
{
  "login_identifier": "pedro",
  "password": "SecurePass123!",
  "display_name": "Pedro"
}
```

The deprecated request spelling remains accepted:

```json
{
  "email": "existing@example.com",
  "password": "SecurePass123!"
}
```

If a caller sends both `login_identifier` and `email`, accept the request only
when the two values normalize to the same identifier. Otherwise return a
content-free validation error. This supports cautious client migrations without
allowing two competing identities in one request.

Registration response:

```json
{
  "id": "uuid",
  "login_identifier": "pedro",
  "email": "pedro",
  "display_name": "Pedro",
  "is_active": true,
  "created_at": "timestamp"
}
```

`email` is a deprecated compatibility alias for `login_identifier`; it is not
an assertion that the value is an email address. New UI and documentation must
not consume or present it as email. Removing the alias requires a separately
approved breaking API change.

### Login

Canonical request:

```json
{
  "login_identifier": "pedro",
  "password": "SecurePass123!"
}
```

The legacy `email` request spelling and the same both-fields rule apply to
login. Token responses do not change.

### Current user

`GET /auth/me` returns the same additive user shape as registration, including
canonical `login_identifier` and deprecated `email` compatibility alias.

### Errors

- Duplicate or concurrently claimed identifier: HTTP 409 with `Account
  identifier is unavailable.`
- Unknown identifier, wrong password, or disabled account: HTTP 401 with
  `Invalid account identifier or password.`
- Active lockout: retain the current temporary-lockout response and timing.
- Invalid auth payload, identifier shape, password shape/complexity, or
  conflicting aliases: HTTP 422 with exactly
  `{"detail":"Invalid authentication request."}`.
- Rate-limit responses and status codes remain unchanged.

The register and login routes will use a route-local validation boundary rather
than FastAPI's default validation-error serializer. A dedicated `APIRoute`
wrapper on the auth router will catch `RequestValidationError` before the
framework renders its usual `detail` list and return the exact generic response
above. The wrapper must not call `exc.errors()`, include `input`, `ctx`, `loc`, or
request-body fragments in the response, or log the exception/request body.
Identifier and alias validation performed after request-model construction will
raise a dedicated internal validation exception that the same boundary maps to
the identical 422 response. Other routers retain their existing validation
behavior.

The complete serialized 422 body—not merely its `detail` message—must contain
none of the submitted identifier, deprecated-email alias, password, normalized
identifier, blind index, or fragments derived from them. Client-side password
guidance remains available in the registration form; the API validation error
is intentionally generic.

## Data model and migration

The user model becomes:

- `login_identifier`: non-null `EncryptedText`
- `login_identifier_hmac`: non-null, unique, indexed `String(64)`
- existing password, status, lockout, timestamps, and relationships unchanged

The Alembic migration will be based on current head `c5d6e7f8a9b0` and will:

1. Rename `users.email` to `users.login_identifier`.
2. Rename `users.email_hmac` to `users.login_identifier_hmac`.
3. Rename the unique index from `ix_users_email_hmac` to
   `ix_users_login_identifier_hmac`.
4. Preserve existing ciphertext and blind-index bytes exactly.
5. Keep both columns non-null and the blind index unique.

No plaintext value needs to be selected, decrypted, or rewritten during the
upgrade. Model metadata and `Base.metadata.create_all()` must match the migrated
schema.

The downgrade is conditionally safe. Before executing any `ALTER TABLE`, index
rename, or other schema mutation, it will read each encrypted identifier and
decrypt it in-process with the application encryption helper and the already
required `DATABASE_ENCRYPTION_KEY`. It will validate the resulting in-memory
string against the legacy `EmailStr`/email-validator contract. The preflight
must never log plaintext, include plaintext in its exception, persist plaintext,
or write decrypted content back to the database.

If the encryption key is unavailable/incorrect, ciphertext cannot be decrypted,
or any identifier is not legacy-email compatible, the downgrade fails before
schema mutation with a generic operator message that does not name the account
or identifier. If every identifier passes, the migration discards the in-memory
plaintext and performs only the inverse column/index renames; ciphertext and
blind-index bytes remain unchanged. Backup/restore documentation must state
that, after an emailless account is created, rollback to an older release
requires restoring the pre-migration database backup.

## Registration and login flow

Registration will:

1. Resolve the canonical field or deprecated alias.
2. Validate and trim the identifier.
3. Compute its keyed HMAC blind index using the existing encryption key.
4. Check availability by blind index.
5. Store the trimmed identifier through `EncryptedText` and the bcrypt password
   hash.
6. Treat a unique-index `IntegrityError` as the same generic 409 as the explicit
   availability check, covering concurrent registration races.
7. Write the existing `user.register` audit event without identifier content.

Login will perform the same field resolution, validation, and blind-index
lookup. Account lockout counters and password verification remain attached to
the resolved user row. Successful authentication will continue issuing access
and refresh JWTs whose `sub` is the immutable user UUID, so changing the login
identifier semantics cannot change authorization or ownership.

## Audit, privacy, and security

- The identifier remains AES-256-GCM encrypted at rest.
- Exact lookup and uniqueness remain backed by keyed HMAC-SHA256, never raw
  hashes or randomized ciphertext queries.
- Successful `user.login` audit events retain their action and client IP but
  remove `email_domain`; no replacement identifier fragment is needed.
- Registration, login, and validation errors do not echo identifier content.
- Existing IP-keyed registration/login rate limiters remain in the same route
  positions.
- The five-attempt, 15-minute per-account lockout remains unchanged.
- JWT access/refresh claims, JTI rotation, revocation, family invalidation,
  expiry, idle timeout, and logout remain unchanged.
- Authenticated endpoints continue deriving user ownership from the JWT UUID,
  not from the login string.

The registration page must make only a narrow truthful claim: the account name
is used to sign in to this Strand instance and no email address is required or
sent for account authentication. It must not imply that every Strand workflow
is network-isolated; the web process can use explicitly selected cloud-assisted
features.

## Frontend behavior

Registration:

- Replace the Email field with a text field labeled **Account name**.
- Use `id="loginIdentifier"`, `autocomplete="username"`, no spellcheck, and no
  automatic capitalization.
- Explain: `Used only to sign in to this Strand instance. It does not need to be
  an email address.`
- Explain comparison behavior without claiming full Unicode case insensitivity:
  `Leading and trailing spaces are ignored. Capitalization of A–Z does not
  matter; visually similar Unicode text can still be different.`
- Submit `login_identifier`.

Login:

- Label the field **Account name or existing email** during the compatibility
  period.
- Use a text input with `autocomplete="username"`.
- Show the same A–Z/Unicode comparison hint used on registration.
- Submit `login_identifier`.

Authenticated UI:

- Use `login_identifier` for avatar fallback initials.
- Replace the Admin account label `Email` with `Account name`.
- Stop using email-specific masking logic. For the Admin account display, show
  only the first two Unicode code points followed by `•••`; for one- or
  two-code-point identifiers, replace every displayed code point with `•`.
- Frontend `UserResponse`, `RegisterRequest`, and `LoginRequest` types use the
  canonical field while retaining the deprecated response alias type.

## Documentation and public wording

Update:

- README data-model and encrypted-at-rest wording from account email to account
  login identifier.
- Backend handoff request/response examples, error descriptions, and status-code
  table.
- Backup/restore encrypted-column inventory, sign-in verification step, and
  downgrade limitation.
- Standalone E2E scripts and helper names that query or report the old columns.
- Any auth-page trust copy affected by the new account-name explanation.

Public prose must be passed through the repository's humanizer workflow before
acceptance. Wording must not claim mail capability, email transmission, global
network isolation, or a recovery path that Strand does not provide.

## Test strategy

Implementation follows test-driven development. Add failing tests before each
behavioral change.

Backend/API coverage:

- Register and log in with a non-email identifier.
- Existing email-shaped identifiers continue to register and authenticate.
- Legacy `email` request payloads remain accepted.
- Matching dual fields are accepted; conflicting fields are rejected without
  echoing either value.
- `Alice`, `alice`, and values differing only in surrounding whitespace resolve
  to one identity.
- `Straße`/`STRASSE` and precomposed/decomposed `Å` remain distinct, proving the
  contract does not overclaim Unicode caseless or canonical matching.
- Blank, over-length, and non-printable identifiers are rejected.
- Printable Unicode and internal spaces work according to the declared rules.
- For invalid identifier type/shape, conflicting aliases, malformed password
  type, and password-complexity failure, assert both
  `response.json() == {"detail": "Invalid authentication request."}` and that
  the response has no keys other than `detail`.
- Use unique identifier, alias, and password sentinel strings in those 422
  cases; assert none of the sentinels, their normalized forms, or their blind
  indexes occurs anywhere in `response.content`. Also assert the serialized body
  contains none of `input`, `ctx`, or `loc`.
- Duplicate normalized identifiers return the generic 409.
- A simulated concurrent uniqueness race also returns the same 409.
- Unknown identifier and wrong password share the generic 401 response.
- Lockout, expired-lockout reset, disabled users, JWT claims, refresh rotation,
  family revocation, logout, and `/auth/me` remain unchanged.
- Raw database values remain ciphertext and login succeeds through the renamed
  blind index.
- Login audit details contain no identifier or legacy email domain.

Migration coverage:

- Upgrade from the previous head with an existing encrypted email account.
- Verify byte-identical ciphertext/HMAC preservation across column renames.
- Verify the migrated account logs in through the new canonical field and the
  legacy alias.
- Verify fresh `create_all` metadata matches the Alembic head.
- Verify downgrade succeeds while all values are valid legacy emails.
- Verify downgrade refuses after a non-email identifier exists, before issuing
  any schema mutation.
- Verify downgrade preflight decrypts only in-process under the configured key,
  never updates identifier data, preserves ciphertext/HMAC bytes, and emits
  neither the plaintext sentinel nor its fragments through the exception or
  captured logs.
- Verify a missing/wrong encryption key fails through the same pre-mutation,
  content-free downgrade path.

Frontend/E2E coverage:

- Account-name registration and login succeed with a non-email value.
- Browser validation no longer requires email syntax.
- Empty identifier, duplicate identifier, wrong password, loading states, and
  navigation remain covered.
- An existing email account can sign in through the relabeled text input.
- Avatar and Admin account display use `login_identifier`.
- API helpers, auth helpers, and representative authenticated flows use the new
  canonical request key.

Verification should run focused backend auth/encryption/migration tests,
frontend unit/type/lint checks, focused auth E2E tests, and then proportionate
broad backend/frontend suites. No networked provider tests or real medical data
are required for this identity-only change.

## Acceptance criteria

The work is complete only when:

1. A user can register and sign in with no email address.
2. An existing pre-migration email account can still sign in without changing
   credentials.
3. Legacy `email` request payloads remain compatible.
4. The canonical UI/API/model contract no longer describes arbitrary values as
   email.
5. Identifier ciphertext, HMAC uniqueness, rate limiting, lockout, JWTs,
   revocation, and user scoping are preserved by tests.
6. The complete auth 422 response body, authored errors, logs, and audit records
   contain no submitted identifier or password content.
7. Migration/create-all parity and conditional downgrade behavior are tested.
8. Public privacy and recovery wording is accurate and humanized.

## Explicit operational boundary

This design authorizes local implementation and verification only after written
spec approval. It does not authorize commits, pushes, pull requests, merges,
deployments, GitHub comments, issue metadata changes, or modifications outside
this isolated worktree.
