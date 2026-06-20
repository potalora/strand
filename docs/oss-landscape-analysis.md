# OSS Landscape & Duplication Analysis — Ingestion & Extraction

**Date:** 2026-06-20
**Question:** Is MedTimeline's ingestion/extraction system duplicative with existing open-source projects, or under-leveraging OSS libraries and standards?
**Method:** One agent inventoried the actual codebase (ground truth); five agents researched the current OSS landscape (June 2026) across PHI de-id, clinical NLP/extraction, terminology coding, FHIR infra/CDA/record-linkage, and whole-app PHR overlap.

---

## TL;DR verdict

Your worry is **partially right, and precisely so** — it's not a vague "we reinvented everything." Three specific subsystems are duplicating mature, permissively-licensed OSS that does the job better; the rest is justified custom work, mostly *forced* by your own constraints (local-first, redistributable, permissive-license-only, no-diagnosis, Gemini-only).

**Genuinely duplicative / under-leveraging OSS (worth changing):**
1. **PHI de-id regex layer** (~184 LOC of hand-maintained patterns) ≈ Microsoft **Presidio** (MIT). Presidio is also the cleanest path to fix your *documented* city/location Safe Harbor gap.
2. **Negation / section-parsing / precision-validator** (~1,000 LOC across `section_parser.py` + `entity_validator.py` + extraction guards) ≈ **medspaCy** (MIT) `context` + `section_detection` + `postprocess`. This is the single clearest win.
3. **6.5-min/note extraction latency** is an artifact of per-section sequential Gemini calls. A **local clinical NER fast-path** (scispaCy / GLiNER-biomed, Apache-2.0) would cut typical notes from minutes to seconds *and* eliminate the PHI de-id round-trip for that path.
4. **Homemade Jaccard fuzzy matcher** in dedup + exact-only terminology lookup ≈ **RapidFuzz** (MIT) — a tiny dependency that replaces hand-rolled string similarity in two places.

**Justified custom (keep — not duplicative):**
- **Epic EHI Tables TSV ingestion** (14 mappers) — unique; no OSS does this.
- **Bundled terminology indexes** — a deliberate license-clean (~2.7 MB) choice; the OSS alternatives (QuickUMLS/scispaCy-linker/MedCAT) all drag in a UMLS-license burden inappropriate for a redistributable local app.
- **Postgres-JSONB FHIR store** — HAPI/Medplum are heavyweight multi-tenant REST servers; JSONB-in-Postgres is exactly what they use under the hood anyway.
- **Dedup LLM judge + field merger** — the "duplicate vs update vs related" + semantic FHIR merge question is *outside* what Splink/recordlinkage/dedupe solve.
- **The app as a whole** — no OSS project combines your ingestion breadth + unstructured→FHIR extraction + de-identified no-diagnosis AI layer.

**Two latent bugs/cleanups surfaced by the inventory:**
- `fhir-resources` and `fhirpathpy` are **declared dependencies but never imported** — you're hand-rolling all FHIR parsing with **zero schema validation**. Either delete the dead deps or actually use them (validation is a real, free quality win).
- CLAUDE.md's "SNOMED CT + CPT license-restricted" is imprecise: **CPT is genuinely restricted (AMA, paid); SNOMED is *free* under the UMLS Affiliate License** but carries redistribution/reporting obligations that make *bundling* impractical. Different reasons — worth correcting the wording.

---

## 1. Ground-truth inventory (what's custom vs delegated)

`backend/app/services/` totals **~8,455 LOC**. Largest subtrees:

| Subsystem | Files / LOC | OSS leverage | Custom burden |
|---|---|---|---|
| **Entity extraction** | `entity_to_fhir.py` (890), `entity_validator.py` (407), `clinical_examples.py` (418), `section_parser.py` (227), `entity_extractor.py` (149) ≈ **1,900 LOC** | LangExtract (the LLM call) | **Very high** — prompts, few-shot, section parsing, precision validation, FHIR mapping all hand-rolled |
| **Epic EHI mappers** | `epic_parser.py` (211) + `epic_mappers/` (926) ≈ **1,140 LOC** | stdlib `csv` only | High volume, low complexity — repetitive per-table dict building |
| **FHIR ingestion** | `fhir_parser.py` (677), `bulk_inserter.py` (51) | `ijson` (streaming only) | **High** — ~700 LOC bespoke field extraction; **`fhir.resources`/`fhirpathpy` declared but UNUSED → no validation** |
| **Dedup** | `detector.py` (325), `orchestrator.py` (199), `llm_judge.py` (138), `field_merger.py` (105) | Gemini (LLM judge) | High — hand-tuned weights + **homemade Jaccard** (no fuzzy lib) |
| **Terminology** | `terminology.py` (452) + data + builder (649) | `simple-icd-10-cm`, RxNav API, NLM CTSS (build-time) | Moderate — runtime is offline gzip+json lookup |
| **PHI de-id** | `phi_scrubber.py` (184), `phi_ner.py` (172), `patient_phi.py` (94) ≈ **450 LOC** | spaCy `en_core_web_md` | High — 23 hand-maintained regexes + eponym allowlists |
| **CDA / XDM** | `cda_parser.py` (176), `xdm_parser.py` (205), `cda_dedup.py` (119) | **`python-fhir-converter`** (CDA→FHIR), `lxml` | Moderate — the hard semantic conversion is delegated |
| **Text extraction** | `text_extractor.py` (154) | `pdfplumber`, `striprtf`, `Pillow`, Gemini vision | **Low** — thin orchestration over mature libs |

**Observation:** You already leverage OSS well where the libraries are mature and obvious (text extraction, CDA conversion, JSON streaming). The custom burden concentrates in **extraction post-processing**, **PHI de-id**, and **dedup scoring** — which is exactly where the highest-value OSS adoptions live.

---

## 2. Subsystem analysis & recommendations

### 2.1 PHI de-identification — **HYBRID: adopt Presidio spine, keep your two unique layers**

Your three-layer scrubber: (1) regex for structured identifiers, (2) decrypt-known-patient-identity + string-match, (3) spaCy PERSON NER + eponym allowlist.

- **Layer 1 ≈ Microsoft Presidio (MIT, v2.2.362 Mar 2026, ~9.3k★).** Presidio ships maintained, fuzz-tested recognizers for SSN/phone/email/IP/URL/medical-license/dates with checksum validation and context boosting. Your hand-maintained regexes are the clearest single duplication.
- **Layer 2 (decrypt patient's own name/MRN/DOB → match) is NOT duplicated by anything** — it's a genuine advantage of a single-patient locally-encrypted app, and it's your documented defense against the `name_encrypted`-NULL leak regression. **Keep it.**
- **Layer 3 ≈ Presidio's `SpacyRecognizer`**, but your eponym/clinical-suffix allowlist is bespoke tuning worth preserving.
- **The city/location gap you deferred is fixable** — and Presidio is the clean way to do it. Point Presidio's `TransformersNlpEngine` at a **clinical** de-id model — `StanfordAIMI/stanford-deidentifier-base` (MIT, **F1 98.9 on i2b2 2014**, 1.36M HF downloads) or `obi/deid_roberta_i2b2` (MIT, F1 0.924) — which labels `LOCATION` *without* tagging `Rifaximin`/`Crohn's` as places (the exact failure that made you reject general GPE NER). This is the "clinical-aware model (scispaCy)" your CLAUDE.md already names as the deferred fix.
- **Hard exclusions (your Rule 17):** PhysioNet `deid` is **GPL-2.0**; NLM-Scrubber is closed/proprietary. Don't vendor either.

**Recommendation:** Migrate Layer 1 to Presidio's predefined recognizers (keep address/account/accession as Presidio *custom* recognizers). Re-home Layer 2 as a Presidio deny-list fed your decrypted identifiers. Add a clinical-NER `LOCATION` pass to finally close the Safe Harbor city gap. Bonus: this lets you benchmark PHI **recall** against published i2b2 numbers — today your scrubber has no measured F1, and for de-id recall *is* the metric (a miss is a breach).

### 2.2 Clinical entity extraction — **HYBRID: keep LangExtract for hard cases, adopt medspaCy + local NER**

- **medspaCy (MIT) is the standout adoption.** Its `context` (negation/family/uncertainty), `section_detection`, and `postprocess` modules map almost 1:1 onto code you hand-rolled in `section_parser.py`, `entity_validator.py`, and your extraction negation/family-history guards. Same spaCy framework you already run for PHI NER. This deletes ~hundreds of LOC of maintenance liability and replaces it with validated, standard clinical-NLP components.
- **The 6.5-min latency is network/round-trip bound, not linguistic.** A 35K-char note is ~6–9K tokens; spaCy/medspaCy/**scispaCy** (Apache-2.0) or **GLiNER-biomed** (Apache-2.0, CPU, ~seconds) process that in single-digit seconds. A hybrid that runs common entities locally and escalates only low-confidence/complex sections to Gemini would plausibly cut typical notes from ~6.5 min to **under ~30 s** — directly serving your stated ingestion-speed priority.
- **Privacy bonus:** a local NER path **never leaves the device**, so it needs *no* PHI de-id round-trip at all — strictly better for a local-first HIPAA app, and it shrinks the de-id attack surface.
- **LangExtract stays justified** for messy narrative + the fact that PDF/TIFF OCR already goes through Gemini vision. Reserve it for escalation, not every section.
- **Exclusions:** **MedCAT is Elastic License 2.0 (source-available, NOT permissive) → violates Rule 17.** CLAMP is non-commercial. cTAKES/MetaMapLite are UMLS-gated + JVM (heavy for single-user macOS). Managed APIs (Comprehend Medical etc.) send PHI off-device + add a non-Gemini provider.
- **Note on Rule 3:** adding scispaCy/medspaCy/GLiNER does *not* violate "Gemini-only" — those are local libraries, not cloud AI *providers*; they reduce reliance on the one external provider.

**Recommendation:** Adopt medspaCy now (highest value/effort ratio in the whole report). Pilot a local scispaCy/GLiNER fast-path mapped to your *existing* bundled RxNorm/ICD-10 indexes; escalate only uncertain spans to Gemini.

### 2.3 Terminology coding — **KEEP the bundled-index architecture; add RapidFuzz**

Your instinct here was right, and the OSS alternatives would make it *worse* for your constraints:

- **QuickUMLS / scispaCy UMLS linker / MedCAT all require a UMLS-derived knowledge base** → every deployment gated behind an individual UMLS license + redistribution headaches + 1 GB–multi-GB footprint (vs your 2.7 MB). MedCAT additionally fails Rule 17 (Elastic 2.0).
- Your "never emit a wrong code; unknown stays uncoded" stance is *more* appropriate for a consumer health app than fuzzy linkers that confidently mismap.
- **The one real capability you lack is fuzzy matching.** Close it cheaply with **RapidFuzz (MIT)** or a local Simstring over your existing indexes, gated above a high similarity threshold — QuickUMLS-style behavior with no UMLS burden.
- **Consider OHDSI Athena** as a richer *build-time* regeneration source (normalized ICD-10-CM/LOINC/RxNorm + crosswalks in one free download).
- **License wording fix:** CPT exclusion is correct and permanent (AMA, paid). SNOMED is *free* under the UMLS Affiliate License (US + member countries) but redistribution/reporting obligations + non-member-country gaps make bundling impractical — so keep it out of the default, but if demand arises expose it as an **opt-in, operator-licensed** feature, not a bundled default.

### 2.4 FHIR storage, CDA conversion, record linkage — **KEEP all three**

- **Storage:** HAPI FHIR (Apache-2.0, JVM) and Medplum (Apache-2.0, TS+Postgres+Redis) are **multi-tenant FHIR REST servers**. Their core value (conformant search API, terminology services, SMART/OAuth, multi-org policies) is unneeded by a single-user local app — and they *also* store FHIR in Postgres, so your JSONB approach is mainstream, not a hack. **Revisit Medplum only on a multi-user/hosted pivot.**
- **The real storage gap is validation, not the store:** you declare `fhir.resources` but never use it. Wiring it in (or `fhirpathpy` for the path queries you currently hand-traverse) would add free correctness with no new dependency. Otherwise delete the dead deps.
- **CDA:** `python-fhir-converter` is an MIT, in-process Python port of the **canonical** Microsoft/Firely Liquid templates — right shape for a single process. Only risk is single-maintainer/template-lag. Mitigate: pin templates, keep fidelity tests as the regression guard, treat the official Microsoft/Firely .NET converter as a sidecar fallback. **No rewrite.**
- **Dedup:** every mature OSS linkage tool (Splink/recordlinkage/dedupe) solves **identity resolution on tabular attributes** (patient matching) — *not* "are these two clinical resources duplicate/update/related, and how do I merge their FHIR fields." Your LLM-judge + field-merger is the differentiated part. Zingg is **AGPL (banned by Rule 17)**; dedupe needs human-labeled training (wrong for an unattended pipeline). The only optional, low-priority upgrade is swapping your homemade Jaccard for **recordlinkage (BSD-3)** or RapidFuzz in the *scoring* tier — and even that's overkill at single-user data scale.

### 2.5 Whole-app overlap — **the from-scratch build is largely justified**

No single OSS project does what MedTimeline does end-to-end. The space splits into two camps you straddle:

| Capability | Fasten OnPrem | Mere Medical | OpenHealth | LLMonFHIR | **MedTimeline** |
|---|---|---|---|---|---|
| Self-hosted single-user | ✅ | ✅ | ✅ | iOS-local | ✅ |
| FHIR-bundle ingest | ✅ | via portals | ❌ | consumes | ✅ |
| Epic EHI Tables (TSV) | ❌ | ❌ | ❌ | ❌ | ✅ |
| CDA / IHE-XDM ingest | C-CDA (3rd-party) | ❌ | ❌ | ❌ | ✅ |
| Unstructured → extract → **FHIR** | ❌ | ❌ | ✅ (no FHIR) | ❌ | ✅ |
| Dedup incl. LLM judge | basic | ❌ | ❌ | ❌ | ✅ |
| AI summarization | roadmap | ❌ | ✅ | ✅ | ✅ |
| **De-id before LLM + no-diagnosis guard** | n/a | n/a | ❌ | ❌ | ✅ |
| License | GPL-3.0 | MIT | AGPL-3.0 | MIT | (yours) |

- **Closest "shape":** Fasten OnPrem (~50% — the PHR/FHIR/dashboard half, *zero* AI/extraction/Epic-EHI). **Closest "AI":** OpenHealth (extraction+chat, but not FHIR-native, no de-id/guardrails).
- **Could you have built ON one?** The two closest functional analogues are **license-incompatible** with your own Rule 17 (Fasten GPL-3.0, OpenHealth AGPL-3.0). **Medplum (Apache-2.0) is the one fair critique** — it could have supplied the FHIR store + auth + audit + REST API, saving real plumbing — at the cost of a Python→TypeScript stack mismatch and integration weight, while *all* your differentiated work (Epic EHI, unstructured→FHIR, de-id, dedup, timeline) would still have to be built on top.
- **Low-risk ecosystem moves:** adopt **Synthea (Apache-2.0)** for synthetic test fixtures; keep **Fasten Connect** in mind as the optional paid path *if* live provider sync ever becomes a requirement (breaks local-first/free, so only on demand).

---

## 3. Prioritized recommendations

| # | Action | Effort | Value | Why now |
|---|---|---|---|---|
| 1 | **Adopt medspaCy** for negation/section/precision-validation | M | **High** | Deletes ~hundreds of LOC of maintenance liability; same spaCy stack; validated clinical NLP |
| 2 | **Local NER fast-path** (scispaCy/GLiNER-biomed) mapped to existing indexes; Gemini only for escalation | M–L | **High** | Cuts ~6.5 min → ~seconds (serves perf priority); removes de-id round-trip for local path |
| 3 | **Decide the `fhir.resources`/`fhirpathpy` deps**: wire in validation OR delete dead deps | S | Med | Free correctness win or honest dependency hygiene |
| 4 | **Adopt Presidio** as regex spine; keep Layers 2–3; add clinical-NER `LOCATION` pass | M | **High** | Retires duplicated regex; **fixes documented Safe Harbor city gap**; enables i2b2 recall benchmarking |
| 5 | **Add RapidFuzz** to terminology lookup + dedup scoring | S | Med | Closes the one real gap (fuzzy match) without UMLS burden |
| 6 | **Fix CLAUDE.md license wording** (CPT vs SNOMED distinction) | S | Low | Accuracy; unblocks a future opt-in SNOMED path |
| 7 | **Adopt Synthea** for synthetic test fixtures | S | Low | Better CI data than hand-built synthetic |
| — | Storage (Postgres-JSONB), dedup LLM-judge, CDA converter, Epic mappers, bundled terminology | — | **KEEP** | Justified; not duplicative |

**Sequencing:** #1 and #2 are the same workstream (clinical-NLP adoption) and deliver the most value — start there, and they directly advance the ingestion-speed priority. #3 is a quick hygiene pass. #4 is a self-contained de-id hardening project worth its own cycle (it fixes a real compliance gap). #5–#7 are small, opportunistic.

## 4. What this means for the original worry

- **"Duplicative with other OSS projects"** — *mostly no* at the app level (your combination is unique), but *yes* at the component level for PHI regex (Presidio), clinical assertion/sectioning (medspaCy), and fuzzy matching (RapidFuzz).
- **"Not leveraging OSS/standards enough"** — *true in three specific places* (medspaCy, Presidio, a local NER engine) and *one dependency you pay for but don't use* (`fhir.resources` validation). Elsewhere your custom code is forced by genuine constraints (license-clean redistribution, local-first, no-diagnosis) that the OSS alternatives violate.

The encouraging read: the custom code that *is* duplicative is concentrated in a few well-bounded modules, and the replacements are all MIT/Apache/BSD — so you can shed maintenance burden and *gain* capability (closed PHI gap, faster extraction, FHIR validation) without violating any of your absolute rules.
