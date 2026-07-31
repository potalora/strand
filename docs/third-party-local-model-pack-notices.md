# Local model pack notices

## Current status

This checkout ships the release-generated lock and release evidence for the
validated Apple M4 16 GB profile. The entries below record the exact model and
runtime identities in `apple-m4-16gb-v1`.

The lock is authoritative for every artifact file's SHA-256 digest and byte
size. The release-evidence file binds that lock to the content-free benchmark.
This notice records attribution; it is not proof that a particular local
installation still matches the lock.

## Shipped runtime

- Runtime: `mlx-vlm`
- Version: `0.5.0`
- Pack: `apple-m4-16gb-v1`
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

The shipped profile passed immutable file verification, offline role loading,
privacy gates, the committed synthetic fidelity suite, and the three-run 16 GB
M4 benchmark. All six reported fidelity rate metrics were `1.0`; the three
reported error counts were zero. Peak MLX allocations were approximately
0.86 GB for OCR, 4.73 GB for extraction, and 7.10 GB for summarization.

A future pack revision must regenerate the lock and release evidence and pass
the same gates. If a base or quantization license cannot be independently
verified, the affected artifact must not ship.
