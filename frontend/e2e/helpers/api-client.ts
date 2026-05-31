import * as fs from "fs";

const API_BASE = "http://localhost:8000/api/v1";

export class ApiClient {
  private token: string = "";

  async register(email: string, password: string): Promise<void> {
    const res = await this._withRateLimitRetry(() =>
      fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          display_name: "E2E Test User",
        }),
      })
    );
    // 409 = already exists, that's fine
    if (!res.ok && res.status !== 409) {
      throw new Error(`Register failed: ${res.status} ${await res.text()}`);
    }
  }

  async login(email: string, password: string): Promise<void> {
    const res = await this._withRateLimitRetry(() =>
      fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      })
    );
    if (!res.ok) {
      throw new Error(`Login failed: ${res.status} ${await res.text()}`);
    }
    const data = await res.json();
    this.token = data.access_token;
  }

  private async _withRateLimitRetry(
    fn: () => Promise<Response>,
    maxRetries = 5
  ): Promise<Response> {
    for (let i = 0; i < maxRetries; i++) {
      const res = await fn();
      if (res.status !== 429) return res;
      // Wait for rate limit window to pass (exponential backoff)
      const waitMs = Math.min(2000 * Math.pow(2, i), 30_000);
      await new Promise((r) => setTimeout(r, waitMs));
    }
    return fn();
  }

  private headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.token}`,
      "Content-Type": "application/json",
    };
  }

  async uploadStructured(
    filePath: string,
    filename: string
  ): Promise<{ upload_id: string; status: string; records_inserted: number }> {
    const fileContent = fs.readFileSync(filePath);
    const formData = new FormData();
    formData.append(
      "file",
      new Blob([fileContent], { type: "application/json" }),
      filename
    );

    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.token}` },
      body: formData,
    });
    if (!res.ok) {
      throw new Error(`Upload failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  async uploadUnstructuredBatch(
    files: { path: string; name: string; mime: string }[]
  ): Promise<{ uploads: { upload_id: string }[] }> {
    const formData = new FormData();
    for (const file of files) {
      const content = fs.readFileSync(file.path);
      formData.append("files", new Blob([content], { type: file.mime }), file.name);
    }

    const res = await fetch(`${API_BASE}/upload/unstructured-batch`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.token}` },
      body: formData,
    });
    if (!res.ok) {
      throw new Error(
        `Unstructured batch upload failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async pollUploadStatus(
    uploadId: string,
    timeoutMs: number = 60_000,
    excludeTerminal: string[] = [],
  ): Promise<any> {
    const start = Date.now();
    const terminalStatuses = [
      "completed",
      "completed_with_errors",
      "completed_with_merges",
      "failed",
      "awaiting_confirmation",
      "awaiting_review",
      "dedup_scanning",
      "duplicate_file", // idempotent re-upload of identical content (Phase 2a)
    ].filter((s) => !excludeTerminal.includes(s));

    while (Date.now() - start < timeoutMs) {
      const res = await fetch(`${API_BASE}/upload/${uploadId}/status`, {
        headers: this.headers(),
      });
      if (!res.ok) {
        throw new Error(
          `Poll status failed: ${res.status} ${await res.text()}`
        );
      }
      const data = await res.json();
      const st = data.ingestion_status ?? data.status;
      if (terminalStatuses.includes(st)) {
        return data;
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    throw new Error(
      `Upload ${uploadId} did not complete within ${timeoutMs}ms`
    );
  }

  async getExtractionProgress(): Promise<{
    total: number;
    completed: number;
    processing: number;
    failed: number;
    pending: number;
  }> {
    const res = await fetch(`${API_BASE}/upload/extraction-progress`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(
        `Extraction progress failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async getUploadHistory(): Promise<any> {
    const res = await fetch(`${API_BASE}/upload/history`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(
        `Upload history failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async getRecords(params?: {
    record_type?: string;
    page?: number;
    page_size?: number;
  }): Promise<any> {
    const query = new URLSearchParams();
    if (params?.record_type) query.set("record_type", params.record_type);
    if (params?.page) query.set("page", String(params.page));
    if (params?.page_size) query.set("page_size", String(params.page_size));
    const qs = query.toString();

    const res = await fetch(`${API_BASE}/records${qs ? `?${qs}` : ""}`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(`Get records failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  async getUploadReview(uploadId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/upload/${uploadId}/review`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(
        `Get upload review failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async resolveDedup(
    uploadId: string,
    resolutions: { candidate_id: string; action: "merge" | "dismiss" }[]
  ): Promise<any> {
    const res = await fetch(`${API_BASE}/upload/${uploadId}/review/resolve`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ resolutions }),
    });
    if (!res.ok) {
      throw new Error(
        `Resolve dedup failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async getMe(): Promise<any> {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(`Get me failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  async scanDedup(): Promise<any> {
    const res = await fetch(`${API_BASE}/dedup/scan`, {
      method: "POST",
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(`Scan dedup failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  async getDedupCandidates(page = 1, limit = 20): Promise<any> {
    const res = await fetch(
      `${API_BASE}/dedup/candidates?page=${page}&limit=${limit}`,
      { headers: this.headers() }
    );
    if (!res.ok) {
      throw new Error(
        `Get dedup candidates failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async mergeDedup(candidateId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/dedup/merge`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ candidate_id: candidateId }),
    });
    if (!res.ok) {
      throw new Error(`Merge dedup failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  async dismissDedup(candidateId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/dedup/dismiss`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ candidate_id: candidateId }),
    });
    if (!res.ok) {
      throw new Error(
        `Dismiss dedup failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async getDashboardOverview(): Promise<any> {
    const res = await fetch(`${API_BASE}/dashboard/overview`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(
        `Get dashboard overview failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async getTimeline(params?: { record_type?: string; limit?: number }): Promise<any> {
    const query = new URLSearchParams();
    if (params?.record_type) query.set("record_type", params.record_type);
    if (params?.limit) query.set("limit", String(params.limit));
    const qs = query.toString();

    const res = await fetch(`${API_BASE}/timeline${qs ? `?${qs}` : ""}`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(
        `Get timeline failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }
}
