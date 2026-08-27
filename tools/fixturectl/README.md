# Private fixture transport

`fixturectl` moves an explicitly selected, signed fixture dataset from its owner
machine to an approved test target. It does not discover private fixtures, read
`REAL_MEDICAL_FIXTURES_DIR`, or put fixture bytes, manifests, signatures, file
lists, receipts, or generated test output in Git.

The safe default is a local dry-run. It verifies the detached signature and all
local bytes, writes a private NUL-delimited rsync file list under the configured
state directory, and prints only aggregate counts and aliases. No SSH or rsync
process starts unless `--apply` is present.

## Local configuration

Keep the configuration owned by the invoking user and mode `0600`. Keep its
parent and state directory mode `0700`.

```json
{
  "schema_version": 1,
  "source_alias": "owner-mac",
  "state_directory": "/private/operator-state/fixturectl",
  "ssh_executable": "/usr/bin/ssh",
  "rsync_executable": "/opt/homebrew/opt/rsync/bin/rsync",
  "rsync_version": "3.5.0",
  "targets": {
    "authorized-test-vps": {
      "ssh_destination": "fixture-receiver@100.100.10.20",
      "target_id": "authorized-test-vps",
      "remote_fixturectl": "/usr/local/libexec/fixturectl"
    }
  }
}
```

Admission holds the configured static reserve plus twice each active payload and
fixed metadata overhead. The two-times payload bound covers rsync's retained
partial basis plus a full resumed temporary file; exact retries reuse the same
reservation instead of spending capacity twice.

The SSH destination must contain a literal Tailscale address, not DNS or an
arbitrary command. The local target ID is checked against the receiver's signed
preflight identity. SSH runs with batch mode and strict host-key checking.

Apply requires the configured owner-side rsync to report exactly version 3.5.0
and protocol 32. This capability check runs before remote staging is created;
macOS's bundled openrsync is intentionally refused. Provision the pinned GNU
rsync through the machine-management layer before enabling apply.

## Owner workflow

Create and sign the manifest outside Git, then preview the transfer:

```bash
fixturectl --policy POLICY transfer \
  --dataset DATASET \
  --target authorized-test-vps \
  --config PRIVATE_CONFIG \
  --source PRIVATE_SOURCE \
  --manifest PRIVATE_MANIFEST \
  --signature PRIVATE_SIGNATURE \
  --public-key PUBLIC_KEY
```

Review the dataset ID, aliases, file count, byte count, release hash, retained
release count, and `deletions: 0`. Add `--apply` only for an approved transfer.

Apply is one-way. It creates a random private staging directory, sends only the
signed file list, never uses rsync deletion, verifies the signature and bytes on
the target, promotes with an atomic no-clobber rename, atomically switches the
relative `current` link, compares the target receipt with the local manifest,
and stores the matching receipt mode `0600` in local state.

Pre-promotion failures leave staging quarantined. Promotion or receipt failures
are reported as `confirmation-pending`, because promotion may already have
succeeded. Neither case falls back to another dataset, downloads fixture data,
prints fixture paths or filenames, or moves `current` past an unverified
release. The error includes a random transfer ID and a runnable owner-side
reconciliation command of this form:

```bash
fixturectl transfer-status \
  --dataset DATASET \
  --target TARGET \
  --config PRIVATE_CONFIG \
  --transfer-id TRANSFER_ID \
  --manifest-sha MANIFEST_SHA
```

The command checks incoming staging first. If staging is absent, it fetches the
verified release receipt by manifest hash and repairs the matching local receipt
mode `0600`. Output remains aggregate-only.

## Receiver boundary

The remote SSH identity is installed by Ansible and is not a general shell or
developer account. Its `authorized_keys` entry forces
`/usr/local/libexec/fixturectl ssh-dispatch`. The dispatcher ignores normal
shell execution, validates the original request as structured arguments, and
allows only preflight, exclusive staging creation, verification, promotion,
aggregate status/receipt output, and upload-only rsync server mode to the exact
fresh staging destination. It rejects sender mode, deletion options, alternate
destinations, arbitrary commands, and shells. The rsync server argv is an exact
version-pinned allowlist; unknown options such as alternate log, temporary, or
backup directories are refused.

The owner command uses rsync 3.5's default protected remote-argument encoding,
not `--protect-args`/`--secluded-args`: secluded arguments hide server options
from forced-command inspection and are therefore incompatible with this
receiver boundary.

Ansible installs the receiver config, policy, and signing public key as
root-owned mode `0444` trust anchors. The receiver identity can read but cannot
rewrite them. A current-user-owned receiver config may be mode `0600` in an
isolated development test. For example:

```json
{
  "schema_version": 1,
  "protocol_version": 1,
  "target_id": "authorized-test-vps",
  "policy": "/etc/private-fixture-target/policy.json",
  "public_key": "/etc/private-fixture-target/fixture-signing.pub",
  "fixturectl_executable": "/usr/local/libexec/fixturectl",
  "rsync_executable": "/usr/local/libexec/fixture-rsync",
  "rsync_version": "3.5.0",
  "reserved_bytes": 21474836480
}
```

The client cannot override any of those fields. The receiver never accepts a
policy, target identity, or public key from a transfer command or received
manifest.

Production signing-key creation remains a separate human gate. Only the public
key belongs in source control. Real fixture scanning, signing, transfer, test
execution, backup restore, and release retention require their own explicit
operator approvals.
