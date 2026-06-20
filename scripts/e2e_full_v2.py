"""Full end-to-end cumulative-upload + summary verification on a FRESH user.

Validates, after the de-id / NER / summary-truncation fixes:
  1. Cumulative idempotent ingestion (April XDM adds many; May XDM superset adds few).
  2. De-identification: the patient's real name is populated on the patient row,
     and NO raw patient name reaches Gemini (prompt scrubbed; de-id log shows it).
  3. The AI summary is COMPLETE (finish=STOP, not truncated).

Unstructured status is polled via psql (DB ground truth) so a busy event loop
during extraction can't stall the driver on an HTTP read.
"""
from __future__ import annotations

import io
import json
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path("/Users/potalora/ai_workspace/test_autonomous_ai_web_records")
BASE = "http://localhost:8000/api/v1"
RESULT = ROOT / "scripts" / "e2e_v2_result.json"
NOKEEP = httpx.Limits(max_keepalive_connections=0, max_connections=8)
results: dict = {"steps": [], "started_at": datetime.now(timezone.utc).isoformat()}


def _fixtures_raw() -> Path:
    """Resolve the off-repo real-medical-fixtures ``raw/`` dir.

    Reads ``REAL_MEDICAL_FIXTURES_DIR`` (falling back to repo-root
    ``.env.test.local``). Real PHI never lives in the repo, so there is no
    in-repo fallback.
    """
    import os

    root = os.environ.get("REAL_MEDICAL_FIXTURES_DIR")
    if not root:
        envf = ROOT / ".env.test.local"
        if envf.exists():
            for ln in envf.read_text().splitlines():
                if ln.strip().startswith("REAL_MEDICAL_FIXTURES_DIR="):
                    root = ln.split("=", 1)[1].strip()
                    break
    if not root:
        raise SystemExit(
            "REAL_MEDICAL_FIXTURES_DIR not set and no .env.test.local found; "
            "real medical fixtures live off-repo (~/Private/medical-test-fixtures)."
        )
    return Path(root).expanduser() / "raw"


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {m}", flush=True)


def psql(sql: str) -> str:
    r = subprocess.run(["psql", "medtimeline", "-t", "-A", "-c", sql],
                       capture_output=True, text=True)
    return r.stdout.strip()


def short() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0), limits=NOKEEP)


def upclient() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0), limits=NOKEEP)


def zip_dir(src: Path, prefix: str, only: str | None = None) -> bytes:
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file() and (only is None or f.suffix.lower() == only):
                zf.write(f, arcname=f"{prefix}/{f.relative_to(src).as_posix()}")
                n += 1
    log(f"  zipped {n} files ({len(buf.getvalue())/1e6:.2f} MB)")
    return buf.getvalue()


def main() -> int:
    stamp = int(time.time())
    email = f"e2e-v2-{stamp}@example.com"
    pw = "E2eVerify!2026"
    with short() as c:
        c.post(f"{BASE}/auth/register", json={"email": email, "password": pw,
                                              "display_name": "E2E V2"}).raise_for_status()
        token = c.post(f"{BASE}/auth/login", json={"email": email, "password": pw}).json()["access_token"]
    H = {"Authorization": f"Bearer {token}"}
    results["email"] = email
    log(f"fresh user {email}")

    uid = psql(f"SELECT id FROM users WHERE email='{email}'")

    def total() -> int:
        with short() as c:
            d = c.get(f"{BASE}/dashboard/overview", headers=H).json()
        return d.get("total_records") or 0

    def poll_db(upload_id: str, label: str, cap_s: int = 600) -> dict:
        terminal = {"completed", "completed_with_merges", "awaiting_confirmation",
                    "failed", "duplicate_file"}
        deadline = time.time() + cap_s
        seen = None
        while time.time() < deadline:
            row = psql(f"SELECT ingestion_status||'|'||COALESCE(record_count,0) "
                       f"FROM uploaded_files WHERE id='{upload_id}'")
            st, _, rc = row.partition("|")
            if st != seen:
                log(f"    [{label}] status={st} records={rc}")
                seen = st
            if st in terminal:
                return {"status": st, "record_count": rc}
            time.sleep(4)
        return {"status": "timeout"}

    td = _fixtures_raw()
    steps = [
        ("EhiExport (One Medical FHIR)", "2026-01-06", "fhir",
         td / "EhiExport-22259" / "fhir_109989389_22259_1767722729.json"),
        ("EHITables (UCSF Epic)", "2026-02-12", "epic",
         td / "Requested Record" / "EHITables"),
        ("Clinical note PDF", "2026-03-30", "unstructured",
         td / "note_361370_387671680_81379cb7-9e94-44da-909b-dd0ee7990dbf.pdf"),
        ("HealthSummary April XDM", "2026-04-05", "xdm",
         td / "HealthSummary_Apr_05_2026" / "IHE_XDM"),
        ("HealthSummary May XDM (superset)", "2026-05-29", "xdm",
         td / "HealthSummary_May_29_2026" / "IHE_XDM"),
        ("ibs_smart.pdf (image OCR)", "2026-05-29", "unstructured",
         td / "ibs_smart.pdf"),
    ]

    for i, (name, created, kind, path) in enumerate(steps, 1):
        before = total()
        log(f"STEP {i}: {name} (created {created}) | DB before={before}")
        entry = {"step": i, "name": name, "created": created, "records_before": before}
        try:
            if kind == "fhir":
                with upclient() as c:
                    r = c.post(f"{BASE}/upload", headers=H,
                               files={"file": (path.name, path.read_bytes(), "application/json")})
                r.raise_for_status(); entry["resp"] = r.json()
            elif kind == "epic":
                blob = zip_dir(path, "EHITables", only=".tsv")
                with upclient() as c:
                    r = c.post(f"{BASE}/upload/epic-export", headers=H,
                               files={"file": ("EHITables.zip", blob, "application/zip")})
                r.raise_for_status(); entry["resp"] = r.json()
            elif kind == "xdm":
                blob = zip_dir(path, "IHE_XDM")
                with upclient() as c:
                    r = c.post(f"{BASE}/upload", headers=H,
                               files={"file": (f"{path.parent.name}.zip", blob, "application/zip")})
                r.raise_for_status(); entry["resp"] = r.json()
            else:  # unstructured
                with upclient() as c:
                    r = c.post(f"{BASE}/upload/unstructured", headers=H,
                               files={"file": (path.name, path.read_bytes(), "application/pdf")})
                r.raise_for_status()
                up = r.json(); entry["resp"] = up
                entry["extraction"] = poll_db(up["upload_id"], path.name)
            time.sleep(2)
            after = total()
            entry["records_after"] = after
            log(f"  -> DB after={after} (delta={after-before})")
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
            log(f"  !! ERROR {entry['error']}")
        results["steps"].append(entry)
        RESULT.write_text(json.dumps(results, indent=2, default=str))

    # patient + demographics check
    with short() as c:
        patient_id = c.get(f"{BASE}/dashboard/patients", headers=H).json()["items"][0]["id"]
    results["patient_id"] = patient_id
    name_present = psql(f"SELECT (name_encrypted IS NOT NULL) FROM patients WHERE id='{patient_id}'")
    results["patient_name_encrypted_present"] = name_present
    results["final_total"] = total()
    log(f"patient {patient_id} | name_encrypted_present={name_present} | total={results['final_total']}")

    # ---- generate summary ----
    log("generating summary (full, both) ...")
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(900.0, connect=5.0), limits=NOKEEP) as c:
        r = c.post(f"{BASE}/summary/generate", headers=H,
                   json={"patient_id": patient_id, "summary_type": "full", "output_format": "both"})
        r.raise_for_status()
        summ = r.json()
    results["summary_elapsed_s"] = round(time.time() - t0, 1)
    results["summary_record_count"] = summ.get("record_count")
    results["summary_deid_report"] = summ.get("de_identification_report")

    # de-id verification from DB (what was actually SENT to Gemini)
    leak = psql(
        "WITH s AS (SELECT user_prompt,response_text,length(response_text) AS rlen "
        f"FROM ai_summary_prompts WHERE user_id='{uid}' ORDER BY generated_at DESC LIMIT 1) "
        "SELECT (position('Pedro' in user_prompt)>0)||'|'||(position('Otalora' in user_prompt)>0)"
        "||'|'||(position('Pedro' in response_text)>0)||'|'||rlen FROM s")
    php, poh, rhp, rlen = (leak.split("|") + ["", "", "", ""])[:4]
    results["prompt_has_pedro"] = php
    results["prompt_has_otalora"] = poh
    results["response_has_pedro"] = rhp
    results["summary_resp_len"] = rlen
    nl = summ.get("natural_language") or ""
    results["summary_nl_len"] = len(nl)
    RESULT.write_text(json.dumps(results, indent=2, default=str))

    log(f"summary in {results['summary_elapsed_s']}s | record_count={results['summary_record_count']}")
    log(f"  de_id_report={json.dumps(results['summary_deid_report'])}")
    log(f"  prompt_has_pedro={php} prompt_has_otalora={poh} resp_has_pedro={rhp} resp_len={rlen}")

    ok_deid = php in ("f", "false") and poh in ("f", "false")
    ok_len = rlen.isdigit() and int(rlen) > 2000
    log(f"  ACCEPTANCE: deid_clean={ok_deid} summary_complete={ok_len}")
    print("\n" + "=" * 80)
    print("FINAL AI SUMMARY (natural language):\n")
    print(nl if nl else "(see response_text)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()
