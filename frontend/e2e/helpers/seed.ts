import * as fs from "fs";
import * as path from "path";
import { authHeaders, type AuthContext } from "./auth";

const API_BASE = "http://localhost:8000/api/v1";

export interface SeededData {
  uploadId: string;
  recordsByType: Record<string, string[]>;
}

export async function seedTestData(auth: AuthContext): Promise<SeededData> {
  const headers = authHeaders(auth.accessToken);

  // Upload the expanded fixture bundle
  const fixturePath = path.resolve(
    __dirname,
    "../../../backend/tests/fixtures/sample_fhir_bundle.json"
  );
  const fixtureContent = fs.readFileSync(fixturePath);

  const formData = new FormData();
  formData.append(
    "file",
    new Blob([fixtureContent], { type: "application/json" }),
    "sample_fhir_bundle.json"
  );

  const uploadRes = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${auth.accessToken}` },
    body: formData,
  });

  if (!uploadRes.ok) {
    throw new Error(`Upload failed: ${uploadRes.status} ${await uploadRes.text()}`);
  }

  const uploadData = await uploadRes.json();
  const uploadId = uploadData.upload_id;

  // Poll for ingestion completion
  for (let i = 0; i < 30; i++) {
    const statusRes = await fetch(`${API_BASE}/upload/${uploadId}/status`, { headers });
    const statusData = await statusRes.json();
    if (statusData.status === "completed" || statusData.status === "completed_with_errors") {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }

  // Fetch all records grouped by type
  const recordsByType: Record<string, string[]> = {};
  const recordsRes = await fetch(`${API_BASE}/records?page_size=100`, { headers });
  const recordsData = await recordsRes.json();

  for (const item of recordsData.items) {
    const type = item.record_type;
    if (!recordsByType[type]) recordsByType[type] = [];
    recordsByType[type].push(item.id);
  }

  return { uploadId, recordsByType };
}
