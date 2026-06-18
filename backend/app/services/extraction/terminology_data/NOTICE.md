# Bundled Terminology — Sources, Licenses & Attribution

These indexes are derived, compact, **offline** subsets built by
`backend/scripts/build_terminology_index.py`. No terminology network call ever
happens at runtime.

## Conditions — ICD-10-CM
- **License/Provenance**: ICD-10-CM is **public domain** (U.S. CMS/CDC).
- **Built from**: the `simple-icd-10-cm` PyPI package (MIT license), which bundles
  the CMS public-domain tabular list offline.
- Codes: 91 curated colloquial aliases
  layered over the full code/description/inclusion-term index.

## Medications — RxNorm
- **License/Provenance**: RxNorm is produced by the U.S. National Library of
  Medicine and is **public domain**.
- **Built from**: the RxNav REST API (`/REST/allconcepts`, no login/API key) —
  the same public RxNorm content as "Current Prescribable Content".
- RxNorm concepts indexed: 27222.

## Labs — LOINC (curated common subset)
This product includes **LOINC** (loinc.org) codes. LOINC is copyright
1995-2024, Regenstrief Institute, Inc. and the LOINC Committee, and is available
at no cost under the LOINC license (https://loinc.org/license/). LOINC® is a
registered trademark of Regenstrief Institute, Inc.

- The full LOINC table requires a free Regenstrief account to download, so we do
  **not** bundle it. We ship only a **curated common-lab subset**
  (97 codes verified via the public NLM Clinical
  Tables LOINC service). **Coverage limitation**: uncommon/esoteric labs are not
  coded and will return `None` (graceful — never a wrong code).

## Procedures — local category markers
CPT (AMA-proprietary) and SNOMED CT (license-restricted) are **not** permissively
licensed and are deliberately excluded. Public-domain HCPCS Level II / ICD-10-PCS
do not cleanly cover common outpatient diagnostic/surgical procedures, so a
curated common-procedure subset is mapped to a **local category marker**
(`https://medtimeline.local/CodeSystem/procedure`) with accurate displays.
**Coverage limitation**: this is not an authoritative standard procedure coding.

## Supplements — local markers
Functional-medicine items exist in no standard vocabulary; mapped to a local
supplement marker (`https://medtimeline.local/CodeSystem/supplement`).

_Regenerate: `pip install simple-icd-10-cm && python scripts/build_terminology_index.py`._
