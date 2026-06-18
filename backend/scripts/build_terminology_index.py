#!/usr/bin/env python
"""Build the bundled, offline clinical-terminology indexes.

This script derives the compact terminology indexes shipped under
``app/services/extraction/terminology_data/`` from **free, public-domain**
sources. It is run once now (and re-run to refresh) by a developer; the derived
``*.json.gz`` files are committed to the repo and read at runtime — **no network
call ever happens at runtime**.

Sources & provenance
--------------------
* **Conditions -> ICD-10-CM** (public domain, U.S. CMS/CDC). Derived from the
  ``simple-icd-10-cm`` PyPI package (MIT license; bundles the CMS public-domain
  tabular list offline). We index every code's official description plus its
  inclusion terms, then layer a small curated colloquial-alias overlay
  ("diabetes" -> E11.9, "htn" -> I10, ...) since extracted labels are rarely the
  formal description.
* **Medications -> RxNorm** (public domain, U.S. NLM). The "RxNorm Current
  Prescribable Content" bulk file requires a UTS login to download, so instead we
  pull the *same* public RxNorm content via the **RxNav REST API** (no login, no
  API key): ``/REST/allconcepts`` for every ingredient (IN/PIN/MIN) and brand
  (BN) concept -> RXCUI. A small curated alias overlay adds synonyms RxNav's
  concept names miss ("cyanocobalamin" -> vitamin B12, "ldn" -> naltrexone).
* **Labs -> LOINC.** The full LOINC table requires a free Regenstrief login, so
  we DO NOT bundle it. Instead we ship a **curated common-lab subset** (a few
  hundred high-frequency analytes). Each curated LOINC code is *verified* and its
  display fetched from the NLM Clinical Tables LOINC service (public, no login)
  at build time. LOINC attribution is written to ``terminology_data/NOTICE.md``.
* **Procedures -> local category markers.** The common outpatient diagnostic /
  surgical procedures in personal health records are coded almost exclusively by
  CPT (AMA-proprietary) or SNOMED CT (license-restricted) — neither is
  permissively licensed. The public-domain options (HCPCS Level II, ICD-10-PCS)
  do not cleanly cover them. Rather than ship a proprietary/restricted or a
  *fabricated* standard code, we map a curated common-procedure subset to a
  **local category marker** with an accurate display. (We dropped the old SNOMED
  procedure codes.)
* **Supplements -> local markers.** Functional-medicine items (Candibactin,
  Akkermansia, FODZYME, ...) exist in no standard vocabulary; mapped to a local
  supplement marker so they are handled cleanly rather than silently uncoded.

Output format (per category file, gzipped JSON)
-----------------------------------------------
``{"codes": {key: [system, code, display]}, "index": {normalized_alias: key}}``
where ``key`` is the code string. The runtime loader joins them into
``{normalized_alias: Coding}``.

Usage
-----
    pip install simple-icd-10-cm          # MIT, build-time only
    python scripts/build_terminology_index.py            # build all
    python scripts/build_terminology_index.py --offline  # skip network checks

Refresh cadence: ICD-10-CM updates annually (Oct 1); RxNorm monthly; LOINC twice
a year. Re-run this script to pick up new releases, then commit the regenerated
``terminology_data/*.json.gz``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "services" / "extraction" / "terminology_data"

# --- Code system canonical URIs (kept in sync with terminology.py) ----------
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC_SYSTEM = "http://loinc.org"
SUPPLEMENT_SYSTEM = "https://medtimeline.local/CodeSystem/supplement"
PROCEDURE_SYSTEM = "https://medtimeline.local/CodeSystem/procedure"

# Reuse the runtime normalizer so build-time keys match lookup-time keys exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.extraction.terminology import normalize_term  # noqa: E402


# ===========================================================================
# Curated overlays (hand-authored, reviewed in PR; codes verified at build)
# ===========================================================================

# Colloquial / abbreviated condition labels -> ICD-10-CM code. The build
# validates each code against the ICD-10 package and pulls the official display.
CONDITION_ALIASES: dict[str, list[str]] = {
    "E11.9": ["type 2 diabetes", "type 2 diabetes mellitus", "diabetes mellitus type 2",
              "t2dm", "dm2", "type ii diabetes", "diabetes"],
    "E10.9": ["type 1 diabetes", "type 1 diabetes mellitus", "t1dm", "type i diabetes"],
    "R73.03": ["prediabetes", "pre diabetes", "impaired glucose tolerance"],
    "I10": ["hypertension", "essential hypertension", "htn", "high blood pressure"],
    "E78.5": ["hyperlipidemia", "dyslipidemia", "high cholesterol", "hypercholesterolemia"],
    "E03.9": ["hypothyroidism", "underactive thyroid"],
    "E06.3": ["hashimoto thyroiditis", "hashimotos thyroiditis", "hashimoto disease",
              "autoimmune thyroiditis"],
    "K21.9": ["gerd", "gastroesophageal reflux disease", "acid reflux",
              "gastroesophageal reflux"],
    "J45.909": ["asthma"],
    "J44.9": ["copd", "chronic obstructive pulmonary disease"],
    "J30.9": ["allergic rhinitis", "hay fever"],
    "K50.90": ["crohns disease", "crohn disease", "crohns"],
    "K51.90": ["ulcerative colitis"],
    "F41.9": ["anxiety", "anxiety disorder"],
    "F41.1": ["generalized anxiety disorder", "gad"],
    "F32.9": ["depression", "major depressive disorder", "mdd", "major depression"],
    "E66.9": ["obesity"],
    "E55.9": ["vitamin d deficiency", "vit d deficiency"],
    "E53.8": ["vitamin b12 deficiency", "b12 deficiency", "cobalamin deficiency"],
    "D50.9": ["iron deficiency anemia", "iron deficiency anaemia"],
    "D64.9": ["anemia", "anaemia"],
    "G43.909": ["migraine", "migraines", "migraine headache"],
    "N18.9": ["chronic kidney disease", "ckd"],
    "I48.91": ["atrial fibrillation", "afib", "a fib"],
    "I25.10": ["coronary artery disease", "cad", "atherosclerotic heart disease"],
    "M81.0": ["osteoporosis"],
    "E87.6": ["hypokalemia", "low potassium"],
    "C81.90": ["hodgkin lymphoma", "hodgkins lymphoma", "hodgkin disease", "hodgkins disease"],
    "K58.9": ["irritable bowel syndrome", "ibs"],
    "K59.00": ["constipation"],
    "G47.00": ["insomnia"],
    "M79.7": ["fibromyalgia"],
    "M19.90": ["osteoarthritis", "oa", "degenerative joint disease"],
    "M06.9": ["rheumatoid arthritis", "ra"],
    "E78.00": ["hypercholesterolemia pure"],
    "E83.42": ["hypomagnesemia", "low magnesium"],
    "D51.0": ["pernicious anemia"],
}

# Curated common-lab subset -> LOINC. Each code verified + display fetched from
# the NLM Clinical Tables LOINC service at build. ``display`` here is a fallback
# used only when the network is unavailable (--offline / API down).
# (code, fallback_display, [aliases])
LAB_ENTRIES: list[tuple[str, str, list[str]]] = [
    # --- Glycemic ---
    ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood",
     ["hemoglobin a1c", "hba1c", "a1c", "hgba1c", "glycated hemoglobin", "glycohemoglobin"]),
    ("2345-7", "Glucose [Mass/volume] in Serum or Plasma",
     ["glucose", "blood glucose", "serum glucose", "fasting glucose", "glucose serum"]),
    ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", ["fasting blood glucose", "fbg"]),
    ("2339-0", "Glucose [Mass/volume] in Blood", ["glucose blood", "capillary glucose"]),
    ("20448-7", "Insulin [Pmol/volume] in Serum or Plasma", ["insulin"]),
    ("1986-9", "C peptide [Mass/volume] in Serum or Plasma", ["c peptide", "c-peptide"]),
    # --- Renal / BMP ---
    ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", ["creatinine", "serum creatinine"]),
    ("3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma",
     ["bun", "blood urea nitrogen", "urea nitrogen"]),
    ("33914-3", "Glomerular filtration rate/1.73 sq M.predicted",
     ["egfr", "estimated gfr", "gfr", "glomerular filtration rate"]),
    ("2823-3", "Potassium [Moles/volume] in Serum or Plasma",
     ["potassium", "serum potassium"]),
    ("2951-2", "Sodium [Moles/volume] in Serum or Plasma", ["sodium", "serum sodium"]),
    ("2075-0", "Chloride [Moles/volume] in Serum or Plasma", ["chloride"]),
    ("2028-9", "Carbon dioxide, total [Moles/volume] in Serum or Plasma",
     ["co2", "carbon dioxide", "bicarbonate", "total co2"]),
    ("17861-6", "Calcium [Mass/volume] in Serum or Plasma", ["calcium", "serum calcium"]),
    ("1863-0", "Anion gap 4 in Serum or Plasma", ["anion gap"]),
    ("2777-1", "Phosphate [Mass/volume] in Serum or Plasma",
     ["phosphorus", "phosphate", "serum phosphorus"]),
    ("19123-9", "Magnesium [Mass/volume] in Serum or Plasma",
     ["magnesium", "serum magnesium", "mg"]),
    ("3084-1", "Urate [Mass/volume] in Serum or Plasma", ["uric acid", "urate"]),
    # --- Hepatic / CMP ---
    ("2885-2", "Protein [Mass/volume] in Serum or Plasma", ["total protein", "protein total"]),
    ("1751-7", "Albumin [Mass/volume] in Serum or Plasma", ["albumin", "serum albumin"]),
    ("10834-0", "Globulin [Mass/volume] in Serum by calculation", ["globulin"]),
    ("1759-0", "Albumin/Globulin [Mass Ratio] in Serum or Plasma", ["a g ratio", "albumin globulin ratio"]),
    ("1975-2", "Bilirubin.total [Mass/volume] in Serum or Plasma",
     ["bilirubin", "total bilirubin", "bilirubin total"]),
    ("1968-7", "Bilirubin.direct [Mass/volume] in Serum or Plasma",
     ["direct bilirubin", "bilirubin direct", "conjugated bilirubin"]),
    ("6768-6", "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma",
     ["alkaline phosphatase", "alk phos", "alp"]),
    ("1742-6", "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
     ["alt", "sgpt", "alanine aminotransferase"]),
    ("1920-8", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
     ["ast", "sgot", "aspartate aminotransferase"]),
    ("2324-2", "Gamma glutamyl transferase [Enzymatic activity/volume] in Serum or Plasma",
     ["ggt", "gamma glutamyl transferase", "ggtp"]),
    ("14804-9", "Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma by Lactate to pyruvate reaction",
     ["ldh", "lactate dehydrogenase"]),
    # --- Lipids ---
    ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma",
     ["cholesterol", "total cholesterol"]),
    ("2089-1", "Cholesterol in LDL [Mass/volume] in Serum or Plasma",
     ["ldl", "ldl cholesterol", "ldl c"]),
    ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation",
     ["ldl calculated", "ldl chol calc"]),
    ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma",
     ["hdl", "hdl cholesterol", "hdl c"]),
    ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma",
     ["triglycerides", "triglyceride", "tg"]),
    ("13458-5", "Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation",
     ["vldl", "vldl cholesterol"]),
    ("9830-1", "Cholesterol.total/Cholesterol in HDL [Mass Ratio] in Serum or Plasma",
     ["cholesterol hdl ratio", "chol hdl ratio", "total cholesterol hdl ratio"]),
    ("43396-1", "Cholesterol non HDL [Mass/volume] in Serum or Plasma",
     ["non hdl cholesterol", "non hdl"]),
    # --- Thyroid ---
    ("3016-3", "Thyrotropin [Units/volume] in Serum or Plasma",
     ["tsh", "thyroid stimulating hormone", "thyrotropin"]),
    ("3024-7", "Thyroxine (T4) free [Mass/volume] in Serum or Plasma",
     ["free t4", "ft4", "free thyroxine"]),
    ("3051-0", "Triiodothyronine (T3) free [Mass/volume] in Serum or Plasma",
     ["free t3", "ft3"]),
    ("3026-2", "Thyroxine (T4) [Mass/volume] in Serum or Plasma",
     ["t4", "thyroxine", "total t4"]),
    ("3053-6", "Triiodothyronine (T3) [Mass/volume] in Serum or Plasma",
     ["t3", "total t3", "triiodothyronine"]),
    ("8098-6", "Thyroid peroxidase Ab [Units/volume] in Serum",
     ["tpo", "tpo antibody", "thyroid peroxidase antibody", "anti tpo"]),
    # --- CBC ---
    ("6690-2", "Leukocytes [#/volume] in Blood by Automated count",
     ["wbc", "white blood cell count", "white blood cells", "leukocytes"]),
    ("789-8", "Erythrocytes [#/volume] in Blood by Automated count",
     ["rbc", "red blood cell count", "erythrocytes"]),
    ("718-7", "Hemoglobin [Mass/volume] in Blood",
     ["hemoglobin", "hgb", "hb", "haemoglobin"]),
    ("4544-3", "Hematocrit [Volume Fraction] of Blood by Automated count",
     ["hematocrit", "hct", "haematocrit"]),
    ("787-2", "MCV [Entitic volume] by Automated count", ["mcv", "mean corpuscular volume"]),
    ("785-6", "MCH [Entitic mass] by Automated count", ["mch", "mean corpuscular hemoglobin"]),
    ("786-4", "MCHC [Mass/volume] by Automated count",
     ["mchc", "mean corpuscular hemoglobin concentration"]),
    ("788-0", "Erythrocyte distribution width [Ratio] by Automated count",
     ["rdw", "red cell distribution width"]),
    ("777-3", "Platelets [#/volume] in Blood by Automated count",
     ["platelets", "platelet count", "plt"]),
    ("32623-1", "Platelet mean volume [Entitic volume] in Blood by Automated count",
     ["mpv", "mean platelet volume"]),
    ("751-8", "Neutrophils [#/volume] in Blood by Automated count",
     ["neutrophils absolute", "absolute neutrophils", "anc"]),
    ("770-8", "Neutrophils/100 leukocytes in Blood by Automated count",
     ["neutrophils", "neutrophils percent", "neutrophil percentage"]),
    ("731-0", "Lymphocytes [#/volume] in Blood by Automated count",
     ["lymphocytes absolute", "absolute lymphocytes"]),
    ("736-9", "Lymphocytes/100 leukocytes in Blood by Automated count",
     ["lymphocytes", "lymphocytes percent", "lymphocyte percentage"]),
    ("742-7", "Monocytes [#/volume] in Blood by Automated count", ["monocytes absolute"]),
    ("5905-5", "Monocytes/100 leukocytes in Blood by Automated count",
     ["monocytes", "monocytes percent"]),
    ("711-2", "Eosinophils [#/volume] in Blood by Automated count", ["eosinophils absolute"]),
    ("713-8", "Eosinophils/100 leukocytes in Blood by Automated count",
     ["eosinophils", "eosinophils percent", "eos"]),
    ("704-7", "Basophils [#/volume] in Blood by Automated count", ["basophils absolute"]),
    ("706-2", "Basophils/100 leukocytes in Blood by Automated count",
     ["basophils", "basophils percent"]),
    # --- Iron studies ---
    ("2276-4", "Ferritin [Mass/volume] in Serum or Plasma", ["ferritin"]),
    ("2498-4", "Iron [Mass/volume] in Serum or Plasma", ["iron", "serum iron"]),
    ("2500-7", "Iron binding capacity [Mass/volume] in Serum or Plasma",
     ["tibc", "total iron binding capacity", "iron binding capacity"]),
    ("2502-3", "Transferrin [Mass/volume] in Serum or Plasma", ["transferrin"]),
    ("2503-1", "Iron saturation [Mass Fraction] in Serum or Plasma",
     ["transferrin saturation", "iron saturation", "tsat"]),
    # --- Vitamins / nutrition ---
    ("62292-8", "25-Hydroxyvitamin D3+25-Hydroxyvitamin D2 [Mass/volume] in Serum or Plasma",
     ["vitamin d", "25 hydroxyvitamin d", "vitamin d 25 hydroxy", "25 oh vitamin d", "vit d"]),
    ("2132-9", "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma",
     ["vitamin b12", "b12", "cobalamin", "vit b12"]),
    ("2284-8", "Folate [Mass/volume] in Serum or Plasma", ["folate", "folic acid", "serum folate"]),
    # --- Inflammation ---
    ("1988-5", "C reactive protein [Mass/volume] in Serum or Plasma",
     ["crp", "c reactive protein", "c-reactive protein"]),
    ("30522-7", "C reactive protein [Mass/volume] in Serum or Plasma by High sensitivity method",
     ["hs crp", "hscrp", "high sensitivity crp", "cardiac crp"]),
    ("4537-7", "Erythrocyte sedimentation rate by Westergren method",
     ["esr", "sed rate", "erythrocyte sedimentation rate", "sedimentation rate"]),
    # --- Pancreatic / cardiac / misc enzymes ---
    ("3040-3", "Lactate [Mass/volume] in Serum or Plasma", ["lactate", "lactic acid"]),
    ("3074-2", "Lipase [Enzymatic activity/volume] in Serum or Plasma", ["lipase"]),
    ("1798-8", "Amylase [Enzymatic activity/volume] in Serum or Plasma", ["amylase"]),
    ("2157-6", "Creatine kinase [Enzymatic activity/volume] in Serum or Plasma",
     ["creatine kinase", "ck", "cpk"]),
    # --- Coagulation ---
    ("5902-2", "Prothrombin time (PT)", ["pt", "prothrombin time"]),
    ("6301-6", "INR in Platelet poor plasma by Coagulation assay",
     ["inr", "international normalized ratio"]),
    ("14979-9", "aPTT in Platelet poor plasma by Coagulation assay",
     ["aptt", "ptt", "partial thromboplastin time"]),
    ("48065-7", "Fibrin D-dimer FEU [Mass/volume] in Platelet poor plasma",
     ["d dimer", "d-dimer"]),
    # --- Endocrine / hormones ---
    ("2143-6", "Cortisol [Mass/volume] in Serum or Plasma", ["cortisol", "serum cortisol"]),
    ("2986-8", "Testosterone [Mass/volume] in Serum or Plasma", ["testosterone", "total testosterone"]),
    ("2243-4", "Estradiol (E2) [Mass/volume] in Serum or Plasma", ["estradiol", "e2"]),
    ("15067-2", "Follitropin [Units/volume] in Serum or Plasma", ["fsh", "follicle stimulating hormone"]),
    ("10501-5", "Lutropin [Units/volume] in Serum or Plasma", ["lh", "luteinizing hormone"]),
    ("2842-3", "Prolactin [Mass/volume] in Serum or Plasma", ["prolactin"]),
    ("2191-5", "Dehydroepiandrosterone sulfate (DHEA-S) [Mass/volume] in Serum or Plasma",
     ["dhea s", "dhea sulfate", "dheas"]),
    ("2731-8", "Parathyrin.intact [Mass/volume] in Serum or Plasma",
     ["pth", "parathyroid hormone", "intact pth"]),
    ("83107-3", "Hemoglobin A1c/Hemoglobin.total in Blood by IFCC protocol", ["a1c ifcc"]),
    # --- Tumor markers / other ---
    ("2857-1", "Prostate specific Ag [Mass/volume] in Serum or Plasma",
     ["psa", "prostate specific antigen"]),
    ("2532-0", "Lactate dehydrogenase [Enzymatic activity/volume] in Body fluid", ["ldh fluid"]),
    # --- Urinalysis (common) ---
    ("5811-5", "Specific gravity of Urine by Test strip", ["urine specific gravity", "specific gravity"]),
    ("5803-2", "pH of Urine by Test strip", ["urine ph"]),
    ("14959-1", "Microalbumin/Creatinine [Mass Ratio] in Urine",
     ["microalbumin creatinine ratio", "urine albumin creatinine ratio", "acr", "uacr"]),
    ("14958-3", "Microalbumin [Mass/volume] in Urine", ["microalbumin", "urine microalbumin"]),
]

# Curated common-procedure subset -> local category marker (public-domain-safe;
# no CPT/SNOMED). (code_slug, display, category, [aliases])
PROCEDURE_ENTRIES: list[tuple[str, str, list[str]]] = [
    ("colonoscopy", "Colonoscopy", ["colonoscopy"]),
    ("egd", "Upper endoscopy (EGD)",
     ["egd", "upper endoscopy", "esophagogastroduodenoscopy", "gastroscopy"]),
    ("sigmoidoscopy", "Sigmoidoscopy", ["sigmoidoscopy", "flexible sigmoidoscopy"]),
    ("echocardiogram", "Echocardiogram",
     ["echocardiogram", "echocardiography", "echo", "transthoracic echocardiogram", "tte"]),
    ("ecg", "Electrocardiogram (ECG/EKG)",
     ["electrocardiogram", "ecg", "ekg", "electrocardiography"]),
    ("mammogram", "Mammogram", ["mammogram", "mammography"]),
    ("chest_xray", "Chest X-ray",
     ["chest x ray", "chest xray", "cxr", "chest radiograph"]),
    ("xray", "Radiograph (X-ray)", ["x ray", "xray", "radiograph", "plain film"]),
    ("ct_scan", "CT scan", ["ct scan", "ct", "cat scan", "computed tomography"]),
    ("mri", "MRI", ["mri", "magnetic resonance imaging"]),
    ("ultrasound", "Ultrasound",
     ["ultrasound", "sonogram", "us", "ultrasonography"]),
    ("dexa", "DEXA bone density scan",
     ["dexa", "dexa scan", "bone density scan", "bone densitometry", "dxa"]),
    ("biopsy", "Biopsy", ["biopsy"]),
    ("cholecystectomy", "Cholecystectomy", ["cholecystectomy"]),
    ("appendectomy", "Appendectomy", ["appendectomy", "appendicectomy"]),
    ("colectomy", "Colectomy", ["colectomy"]),
    ("endoscopy", "Endoscopy", ["endoscopy"]),
    ("stress_test", "Cardiac stress test",
     ["stress test", "cardiac stress test", "exercise stress test", "treadmill test"]),
    ("pap_smear", "Pap smear", ["pap smear", "pap test", "cervical cytology"]),
    ("vaccination", "Vaccination/immunization", ["vaccination", "immunization", "vaccine"]),
]

# Functional-medicine supplements (no standard vocabulary) -> local marker.
# (code_slug, display, [aliases])
SUPPLEMENT_ENTRIES: list[tuple[str, str, list[str]]] = [
    ("candibactin-ar", "Candibactin-AR (herbal supplement)", ["candibactin ar", "candibactin-ar"]),
    ("candibactin-br", "Candibactin-BR (herbal supplement)", ["candibactin br", "candibactin-br"]),
    ("akkermansia", "Akkermansia (probiotic supplement)",
     ["akkermansia", "akkermansia muciniphila", "pendulum akkermansia"]),
    ("digest-gold", "Digest Gold (digestive enzyme supplement)",
     ["digest gold", "digestgold"]),
    ("fodzyme", "FODZYME (enzyme supplement)", ["fodzyme"]),
    ("probiotic", "Probiotic supplement",
     ["probiotic", "probiotics", "probiotic supplement"]),
    ("megasporebiotic", "MegaSporeBiotic (spore probiotic)", ["megasporebiotic", "mega spore"]),
    ("vsl3", "VSL#3 (probiotic)", ["vsl3", "vsl 3"]),
    ("l-glutamine", "L-Glutamine (supplement)", ["l glutamine", "glutamine supplement"]),
    ("berberine", "Berberine (supplement)", ["berberine"]),
    ("omega-3", "Omega-3 fish oil (supplement)",
     ["omega 3", "fish oil", "omega 3 fish oil"]),
    ("magnesium-glycinate", "Magnesium glycinate (supplement)",
     ["magnesium glycinate", "mag glycinate"]),
    ("methylfolate", "L-Methylfolate (supplement)",
     ["methylfolate", "l methylfolate", "5 mthf"]),
    ("nac", "N-Acetylcysteine (supplement)", ["nac", "n acetylcysteine"]),
    ("zinc-carnosine", "Zinc-carnosine (supplement)", ["zinc carnosine"]),
    ("dgl", "DGL licorice (supplement)", ["dgl", "deglycyrrhizinated licorice"]),
    ("slippery-elm", "Slippery elm (supplement)", ["slippery elm"]),
    ("psyllium", "Psyllium fiber (supplement)", ["psyllium", "psyllium husk", "metamucil"]),
    ("collagen", "Collagen peptides (supplement)", ["collagen", "collagen peptides"]),
    ("curcumin", "Curcumin/turmeric (supplement)", ["curcumin", "turmeric"]),
]

# Medication synonyms RxNav concept-names miss -> RXCUI. Validated at build.
# (rxcui, fallback_display, [aliases])
MEDICATION_ALIASES: list[tuple[str, str, list[str]]] = [
    ("11248", "vitamin B12", ["cyanocobalamin", "methylcobalamin", "vitamin b 12",
                              "b12", "b 12", "b-12"]),
    ("7243", "naltrexone", ["low dose naltrexone", "ldn", "low-dose naltrexone"]),
    ("2418", "cholecalciferol", ["vitamin d3", "cholecalciferol"]),
]

# Network endpoints ---------------------------------------------------------
RXNAV_ALLCONCEPTS = "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN+PIN+MIN+BN"
RXNAV_PROP = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json?propName=RxNorm%20Name"
LOINC_SEARCH = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"


def _http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "medtimeline-terminology-build/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _write_index(path: Path, codes: dict, index: dict, meta: dict) -> None:
    payload = {"_meta": meta, "codes": codes, "index": index}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    size = path.stat().st_size
    print(f"  wrote {path.name}: {len(index)} aliases -> {len(codes)} codes ({size/1024:.0f} KB)")


# ===========================================================================
# Builders
# ===========================================================================

def build_conditions() -> dict:
    import simple_icd_10_cm as cm

    codes: dict[str, list] = {}
    index: dict[str, str] = {}
    all_codes = cm.get_all_codes(with_dots=True)
    derived = 0
    for code in all_codes:
        if not cm.is_valid_item(code) or cm.is_chapter_or_block(code):
            continue
        desc = cm.get_description(code)
        if not desc:
            continue
        codes.setdefault(code, [ICD10_SYSTEM, code, desc])
        for term in (desc, *cm.get_inclusion_term(code)):
            key = normalize_term(term)
            if key:
                if index.setdefault(key, code) == code:
                    derived += 1
    # Curated colloquial overlay (overrides any derived collision).
    overlay = 0
    for code, aliases in CONDITION_ALIASES.items():
        if not cm.is_valid_item(code):
            print(f"  !! CONDITION_ALIASES code {code} is not a valid ICD-10-CM code — skipped")
            continue
        codes.setdefault(code, [ICD10_SYSTEM, code, cm.get_description(code)])
        for alias in aliases:
            key = normalize_term(alias)
            if key:
                index[key] = code  # overlay wins
                overlay += 1
    meta = {"source": "ICD-10-CM via simple-icd-10-cm (MIT; CMS public domain)",
            "system": ICD10_SYSTEM, "derived_aliases": derived, "curated_aliases": overlay}
    print(f"conditions: {len(codes)} codes, {len(index)} aliases "
          f"({overlay} curated colloquial)")
    _write_index(DATA_DIR / "conditions.json.gz", codes, index, meta)
    return meta


def build_medications(offline: bool, out_path: Path | None = None) -> dict:
    out_path = out_path or (DATA_DIR / "medications.json.gz")
    codes: dict[str, list] = {}
    index: dict[str, str] = {}
    rxnorm_count = 0
    if not offline:
        data = _http_json(RXNAV_ALLCONCEPTS)
        concepts = data.get("minConceptGroup", {}).get("minConcept", [])
        priority = {"IN": 0, "PIN": 1, "MIN": 2, "BN": 3}
        concepts.sort(key=lambda c: priority.get(c.get("tty", ""), 9))
        for c in concepts:
            rxcui, name = c["rxcui"], c["name"]
            key = normalize_term(name)
            if not key:
                continue
            codes.setdefault(rxcui, [RXNORM_SYSTEM, rxcui, name])
            index.setdefault(key, rxcui)
        rxnorm_count = len(codes)
        print(f"medications: {rxnorm_count} RxNorm concepts (IN/PIN/MIN/BN)")
    else:
        print("medications: --offline, skipping RxNorm pull (RxNorm index will be empty!)")

    # Curated medication aliases (synonyms/forms RxNav names miss).
    for rxcui, fallback, aliases in MEDICATION_ALIASES:
        display = codes.get(rxcui, [None, None, fallback])[2]
        if not offline and rxcui not in codes:
            try:
                prop = _http_json(RXNAV_PROP.format(rxcui=rxcui))
                pc = prop.get("propConceptGroup", {}).get("propConcept", [])
                if pc:
                    display = pc[0]["propValue"]
            except Exception as exc:  # noqa: BLE001
                print(f"  !! could not verify RXCUI {rxcui}: {exc}")
        codes.setdefault(rxcui, [RXNORM_SYSTEM, rxcui, display])
        for alias in aliases:
            key = normalize_term(alias)
            if key:
                index.setdefault(key, rxcui)

    # Supplement overlay (local markers, looked up via the medication path).
    for slug, display, aliases in SUPPLEMENT_ENTRIES:
        codes[slug] = [SUPPLEMENT_SYSTEM, slug, display]
        for alias in (*aliases, display):
            key = normalize_term(alias)
            if key:
                index.setdefault(key, slug)

    meta = {"source": "RxNorm via RxNav allconcepts API (NLM public domain) + curated",
            "system": RXNORM_SYSTEM, "rxnorm_concepts": rxnorm_count,
            "supplements": len(SUPPLEMENT_ENTRIES)}
    print(f"  total: {len(codes)} codes, {len(index)} aliases")
    _write_index(out_path, codes, index, meta)
    return meta


def build_labs(offline: bool) -> dict:
    codes: dict[str, list] = {}
    index: dict[str, str] = {}
    verified = 0
    skipped: list[str] = []
    for code, fallback, aliases in LAB_ENTRIES:
        if not code or not code[0].isdigit():
            continue  # skip placeholder rows
        display = fallback
        if not offline:
            try:
                q = urllib.parse.urlencode(
                    {"terms": code, "df": "LOINC_NUM,LONG_COMMON_NAME",
                     "sf": "LOINC_NUM", "maxList": 7})
                rows = _http_json(f"{LOINC_SEARCH}?{q}")[3]
                match = [r for r in rows if r[0] == code]
                if match:
                    display = match[0][1]
                    verified += 1
                else:
                    skipped.append(code)
                    print(f"  !! LOINC {code} not found in NLM service — SKIPPING (no wrong codes)")
                    continue
                time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001
                print(f"  .. LOINC {code} verify failed ({exc}); using fallback display")
        codes[code] = [LOINC_SYSTEM, code, display]
        for alias in (*aliases, display):
            key = normalize_term(alias)
            if key:
                index.setdefault(key, code)
    meta = {"source": "Curated common-lab LOINC subset (verified via NLM Clinical Tables)",
            "system": LOINC_SYSTEM, "verified": verified, "skipped": skipped,
            "attribution": "This product includes LOINC codes (loinc.org). LOINC is "
                           "copyright 1995-2024 Regenstrief Institute, Inc. and the "
                           "LOINC Committee and is available at no cost under the LOINC "
                           "license."}
    print(f"labs: {len(codes)} LOINC codes ({verified} verified online), {len(index)} aliases")
    if skipped:
        print(f"  skipped (unverified): {skipped}")
    _write_index(DATA_DIR / "labs.json.gz", codes, index, meta)
    return meta


def build_procedures() -> dict:
    codes: dict[str, list] = {}
    index: dict[str, str] = {}
    for slug, display, aliases in PROCEDURE_ENTRIES:
        codes[slug] = [PROCEDURE_SYSTEM, slug, display]
        for alias in (*aliases, display):
            key = normalize_term(alias)
            if key:
                index.setdefault(key, slug)
    meta = {"source": "Curated common-procedure subset -> local category markers "
                      "(public-domain-safe; CPT/SNOMED deliberately excluded)",
            "system": PROCEDURE_SYSTEM, "count": len(codes)}
    print(f"procedures: {len(codes)} local procedure markers, {len(index)} aliases")
    _write_index(DATA_DIR / "procedures.json.gz", codes, index, meta)
    return meta


def write_notice(metas: dict) -> None:
    notice = f"""# Bundled Terminology — Sources, Licenses & Attribution

These indexes are derived, compact, **offline** subsets built by
`backend/scripts/build_terminology_index.py`. No terminology network call ever
happens at runtime.

## Conditions — ICD-10-CM
- **License/Provenance**: ICD-10-CM is **public domain** (U.S. CMS/CDC).
- **Built from**: the `simple-icd-10-cm` PyPI package (MIT license), which bundles
  the CMS public-domain tabular list offline.
- Codes: {metas['conditions'].get('curated_aliases', '?')} curated colloquial aliases
  layered over the full code/description/inclusion-term index.

## Medications — RxNorm
- **License/Provenance**: RxNorm is produced by the U.S. National Library of
  Medicine and is **public domain**.
- **Built from**: the RxNav REST API (`/REST/allconcepts`, no login/API key) —
  the same public RxNorm content as "Current Prescribable Content".
- RxNorm concepts indexed: {metas['medications'].get('rxnorm_concepts', '?')}.

## Labs — LOINC (curated common subset)
This product includes **LOINC** (loinc.org) codes. LOINC is copyright
1995-2024, Regenstrief Institute, Inc. and the LOINC Committee, and is available
at no cost under the LOINC license (https://loinc.org/license/). LOINC® is a
registered trademark of Regenstrief Institute, Inc.

- The full LOINC table requires a free Regenstrief account to download, so we do
  **not** bundle it. We ship only a **curated common-lab subset**
  ({metas['labs'].get('verified', '?')} codes verified via the public NLM Clinical
  Tables LOINC service). **Coverage limitation**: uncommon/esoteric labs are not
  coded and will return `None` (graceful — never a wrong code).

## Procedures — local category markers
CPT (AMA-proprietary) and SNOMED CT (license-restricted) are **not** permissively
licensed and are deliberately excluded. Public-domain HCPCS Level II / ICD-10-PCS
do not cleanly cover common outpatient diagnostic/surgical procedures, so a
curated common-procedure subset is mapped to a **local category marker**
(`{metas['procedures'].get('system', '?')}`) with accurate displays.
**Coverage limitation**: this is not an authoritative standard procedure coding.

## Supplements — local markers
Functional-medicine items exist in no standard vocabulary; mapped to a local
supplement marker (`https://medtimeline.local/CodeSystem/supplement`).

_Regenerate: `pip install simple-icd-10-cm && python scripts/build_terminology_index.py`._
"""
    (DATA_DIR / "NOTICE.md").write_text(notice, encoding="utf-8")
    print("  wrote NOTICE.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="skip all network calls (RxNorm index empty, LOINC unverified)")
    ap.add_argument("--only", choices=["conditions", "medications", "labs", "procedures"],
                    help="build only one category")
    ap.add_argument("--refresh-live", action="store_true",
                    help="refresh the gitignored LIVE medication cache from RxNorm "
                         "(used at install/startup; never touches the committed baseline)")
    args = ap.parse_args()

    if args.refresh_live:
        # Delegate to the single source of truth for the live refresh (fail-open).
        from app.services.extraction.terminology import (
            medication_live_cache_path,
            refresh_medication_index,
        )
        refreshed = refresh_medication_index(max_age_days=0)
        print(f"live medication cache {'refreshed' if refreshed else 'unchanged/offline'}: "
              f"{medication_live_cache_path()}")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metas: dict[str, dict] = {}
    if args.only in (None, "conditions"):
        metas["conditions"] = build_conditions()
    if args.only in (None, "medications"):
        metas["medications"] = build_medications(args.offline)
    if args.only in (None, "labs"):
        metas["labs"] = build_labs(args.offline)
    if args.only in (None, "procedures"):
        metas["procedures"] = build_procedures()
    if args.only is None:
        write_notice(metas)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
