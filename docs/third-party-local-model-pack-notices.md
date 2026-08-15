# Local model pack notices

## Current status

This Track D state prepares the Apple M4 16 GB v2 candidate. Its static
references are `catalog-v2.json` and `apple-m4-16gb-v2.lock.json`. The entries
below record the model and runtime identities named by that candidate.

No verified v2 release is claimed here. The
`apple-m4-16gb-v2.release.json` file is generated only after the separately
authorized benchmark, synthetic fidelity, promotion, and post-promotion
verification gates pass. It is not present or verified in this Track D state.
The v1 catalog, lock, and release evidence remain immutable diagnostic history.

The lock is authoritative for every artifact file's SHA-256 digest and byte
size. After promotion, release evidence binds that lock to the content-free
benchmark and fidelity reports. This notice records attribution; it is not
proof that a local installation matches the lock or that the v2 release gates
passed.

## Candidate runtime

- Runtime: `mlx-vlm`
- Version: `0.5.0`
- Pack: `apple-m4-16gb-v2`
- Fixture suite: `local-ai-fixtures-v1`
- Pack size: 9.02 GiB
- Supported profile: Apple M4 with 16 GB unified memory; 16 GB minimum and
  recommended

## OCR

- Repository: `sahilchachra/ovisocr2-int4-mlx`
- Revision: `1e9cea98871c19b2349a5d2df36fb6c4c38a1237`
- Quantization: `int4`
- Lock-recorded license: Apache-2.0
- Attribution: <https://huggingface.co/sahilchachra/ovisocr2-int4-mlx>

## Clinical extraction

- Quantized repository: `numind/NuExtract3-mlx-4bits`
- Revision: `29c38269f94054282bf9ea97a20dfc6bb8bbefea`
- Quantization: `4bit`
- Lock-recorded license: Apache-2.0
- Attribution: <https://huggingface.co/numind/NuExtract3-mlx-4bits>
- Base license source: `numind/NuExtract3`
- Base license revision: `2e9fca82ee641e6bb6e1f5d905241e994be27a07`
- Base `LICENSE` SHA-256:
  `50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32`
- Base `LICENSE` size: 11,343 bytes
- Base attribution: <https://huggingface.co/numind/NuExtract3>

## Summary

- Repository: `mlx-community/Qwen3.5-9B-MLX-4bit`
- Revision: `938d8919941c6e7efd3c7150eff7fe9d12afa631`
- Quantization: `4bit`
- Lock-recorded license: Apache-2.0
- Attribution: <https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit>

## Validation summary

The historical v1 profile passed immutable file verification, offline role
loading, privacy gates, the committed synthetic fidelity suite, and the
three-run 16 GB M4 benchmark. All six reported fidelity rate metrics were
`1.0`; the three reported error counts were zero. Peak MLX allocations were
approximately 0.86 GB for OCR, 4.73 GB for extraction, and 7.10 GB for
summarization. Those results do not validate the v2 candidate.

The v2 lock also binds the worker through the
`local-ai-worker-bundle.v1` identity scheme. Its portable digest covers the
fixed console entry point, `pyproject.toml`, `uv.lock`, and the effective
worker `.py` tree. Runtime topology is validated separately and is not hashed.
The identity does not attest the operating-system owner or root of trust.

The v2 candidate must pass the full release gates before promotion. The
release fidelity command removes private fixtures from its environment. If a
base or quantization license cannot be independently verified, the affected
artifact must not ship.
