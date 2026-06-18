# Extraction Remediation List

Audit of the full test-data load: **2,153 active records / 16 types** (sources: `fhir_r4`, `ai_extracted` (unstructured LangExtract), `cda_r2`, `epic_ehi`). Goal: improve **richness (recall — fields we drop)** and **precision (false entities we invent)**.

## Headline numbers
- **Coded (code_value+system): 36%** of all records. **AI-extracted records are 0% coded** for every type (no SNOMED/LOINC/RxNorm/ICD). Even **structured medications are 0% RxNorm-coded.**
- **Dated: 79%.** display_text: 100%.
- Richness collapses on the unstructured (`ai_extracted`) path vs structured — it keeps the label + sometimes the value, and drops codes, units, providers, and structured dosage.

Field fill-rates (struct vs ai):

| Type | field | struct | ai |
|---|---|---|---|
| encounter | provider (participant) | 94% | **0%** |
| encounter | serviceProvider (facility) | **0%** | 0% |
| observation | unit | 56% | **0%** |
| observation | referenceRange / interpretation | 7% / 0% | 0% / 0% |
| observation | performer | 55% | **0%** |
| medication | code (RxNorm) | **0%** | **0%** |
| medication | structured dose/route/freq | 13% | **0%** |
| condition | code (coding) | 70% | **0%** |
| condition | onset | 22% | 0% |
| procedure | code (coding) | 40% | **0%** |
| procedure | performer | 97% | **0%** |
| diagnostic_report | result refs / performer | 46% / 100% | 0% / 0% |

---

## A. Precision — LangExtract over-extraction (false records)
Confirmed examples: `2mg` (observation), `COLONOSCOPY` (procedure — patient never had one; mentioned in the doc), `"Go"` / `PPI` / `LDN` / `b12` (medications), `5' 9"` (observation), `Exercise: Tennis player` / `Alcohol: avoid alcohol` / `Diet: ...` (observations), `[NAME] (12+ YO)` (procedure). 15 ai-entities are ≤4 chars; 8 ai-observations are recommendation-phrased.

- **A1 — Mentioned-not-performed procedures.** Only emit a procedure with evidence it was *done* (a date, "s/p", "status post", "underwent", "h/o … removed"). Reject recommended/planned/due/differential mentions ("recommend colonoscopy", "consider", "due for", "screening options include"). → `entity_extractor` prompt + `clinical_examples.py` (add negative examples).
- **A2 — Fragment entities.** Reject bare dose/measurement tokens not bound to a named analyte/drug: `2mg`, `5'9"`, lone vitals. Require an analyte/drug name alongside the value. → extraction prompt + a post-extract validator in `entity_to_fhir`/`upload._collect_entities`.
- **A3 — Recommendations / lifestyle as clinical observations.** `Exercise:`, `Diet:`, `Alcohol: avoid…` are counseling / social history, not lab/clinical observations. Route to social-history Observation (category `social-history`) or drop directives. The `Word:` prefix + imperative verbs are reliable signals. → prompt + classifier.
- **A4 — Drug classes / abbreviations / garbage as medications.** `Go`, `PPI`, `LDN`, `b12` — require a recognizable drug name or expand known abbreviations; drop non-drug tokens. → prompt + a medication allowlist/RxNorm check in `entity_to_fhir`.
- **A5 — PHI-scrubber placeholders extracted as content.** `[NAME]` (and `[DATE]` etc.) become entities because extraction runs after scrubbing. Post-filter any entity whose text is/contains a scrubber placeholder. → `upload._collect_entities` or `entity_to_fhir` guard. (Quick, high-value.)
- **A6 — Within-document duplicates.** `Cystectomy` ×9 from one doc — the `(entity_class, text.lower())` de-dup in `upload._process_unstructured` isn't collapsing near-dups (whitespace/parenthetical date variants). Normalize before the `seen` check.

## B. Recall — richness the extraction drops
- **B1 — Terminology coding (biggest lever; 64% uncoded).** Add a coding step in `entity_to_fhir`: condition→ICD-10/SNOMED, medication→RxNorm, lab→LOINC, procedure→CPT/SNOMED. Either prompt LangExtract to emit codes, or post-map via a terminology lookup. Also fixes structured meds (0% RxNorm — a mapper/parser gap, not just AI).
- **B2 — Provider/performer on AI records.** AI encounters/observations/procedures have 0% provider; structured has 94/55/97%. Extract the note's provider/attending and attach as `participant`/`performer`. → `entity_extractor` (capture provider entity) + `entity_to_fhir` (attach).
- **B3 — Lab value + unit + range together.** AI labs are 0% unit / 0% refRange / 0% interpretation. Parse "Glucose 95 mg/dL (70–99) H" into `valueQuantity{value,unit}` + `referenceRange` + `interpretation`. → `entity_to_fhir`.
- **B4 — Structured medication dosage.** dose/route/freq is 13% (struct) / 0% (ai). Parse sig into `dosageInstruction` (doseAndRate/route/timing). → `entity_to_fhir` + Epic `OrderMedMapper`.
- **B5 — Condition onset + codes.** onset 22% (struct) / 0% (ai); codes 0% (ai). Capture onset/since-date and map codes.
- **B6 — serviceProvider/facility (0% on ALL encounters, even structured).** Populate the encounter facility/org from Epic `PAT_ENC` / FHIR / CDA. → Epic mapper + `fhir_parser`.
- **B7 — diagnostic_report linkage.** AI reports have a conclusion but 0% result refs/performer; link contained result observations and the performing lab.
- **B8 — Lab panel orders vs results.** Lone panel tokens (`CBC`, `CMP`, `A1c`, `FIT`) become valueless observations. Capture as `ServiceRequest` (ordered) or bind to the result values; don't create empty observations.

## C. UI surfacing
- **C1 — Show providers.** Even where provider exists (structured encounters 94%), the Timeline/RecordDetail don't surface it. Render `encounter.participant` / `observation.performer` / `procedure.performer`. (Directly addresses "I don't see providers clearly.")

---

## Suggested priority
1. **P0 (trust):** A1, A2, A3, A4, A5 (over-extraction → wrong records); B1 (coding, 64% uncoded).
2. **P1 (richness):** B2 (providers), B3 (lab value/unit/range), B4 (dosage), C1 (provider UI).
3. **P2:** A6 (dup collapse), B5, B6, B7, B8.

Where the work lands: precision → `services/extraction/entity_extractor.py` + `clinical_examples.py` (prompt/few-shot) + a post-extract validator; richness → `services/extraction/entity_to_fhir.py` (+ a terminology map) and the Epic mappers / `fhir_parser` for structured gaps; UI → Timeline + `RecordDetailSheet`.
