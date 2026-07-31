import { useAuthStore } from "@/stores/useAuthStore";
import type { ExtractionProgressResponse } from "@/types/api";
import type {
  GenerateSummaryApiResponse,
  GenerateSummaryRequest,
} from "@/types/api";
import type {
  CancelExtractionResponse,
  ExtractionFileStatus,
  UploadHistoryResponse,
} from "@/types/upload";
import type {
  LocalModelRole,
  LocalAIJobResponse,
  LocalPackOperation,
  LocalPackOperationCreated,
  LocalPackStatus,
  ProcessingMode,
  RecordExtractionProvenance,
} from "@/types/local-ai";
import type { TriggerExtractionResponse } from "@/types/api";
import {
  parseLocalRunInfo,
  parseRecordExtractionProvenance,
} from "@/types/local-ai";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

function readAuthState(): { accessToken?: string; refreshToken?: string } | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem("medtimeline-auth");
    if (stored) {
      return JSON.parse(stored)?.state ?? null;
    }
  } catch {
    // ignore
  }
  return null;
}

function getToken(): string | null {
  return readAuthState()?.accessToken ?? null;
}

function getRefreshToken(): string | null {
  return readAuthState()?.refreshToken ?? null;
}

// --- Transparent access-token refresh ---
// The access token is short-lived (15 min). On a 401 we exchange the stored
// refresh token for a new pair and retry the original request once. A single
// in-flight refresh is shared across concurrent 401s because refresh tokens
// rotate (the backend revokes the old one on use), so racing refreshes fail.
let refreshPromise: Promise<string | null> | null = null;

async function refreshAccessToken(): Promise<string | null> {
  if (refreshPromise) return refreshPromise;

  const refreshToken = getRefreshToken();
  if (!refreshToken) return null;

  refreshPromise = (async () => {
    try {
      const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      if (!data?.access_token || !data?.refresh_token) return null;
      // A logout (clearTokens) during the in-flight refresh removes the stored
      // refresh token. Do NOT resurrect a session that was deliberately ended:
      // discard the rotated tokens if the session was cleared while we waited.
      if (!getRefreshToken()) return null;
      // Keep the zustand store (and its localStorage mirror) in sync so that
      // components reading accessToken (e.g. logout) use the rotated token.
      useAuthStore.getState().setTokens(data.access_token, data.refresh_token);
      return data.access_token as string;
    } catch {
      return null;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

function endSessionAndRedirect(): void {
  try {
    useAuthStore.getState().clearTokens();
  } catch {
    // ignore
  }
  if (
    typeof window !== "undefined" &&
    !window.location.pathname.startsWith("/login")
  ) {
    window.location.href = "/login";
  }
}

// --- Idle timeout for HIPAA compliance (30-min session timeout) ---
let idleTimer: ReturnType<typeof setTimeout> | null = null;

function resetIdleTimer() {
  if (typeof window === "undefined") return;
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    // Revoke the session server-side (best-effort) before clearing local state,
    // so an idle logout doesn't leave the refresh token usable for 7 days.
    void api
      .logout()
      .catch(() => {})
      .finally(() => {
        localStorage.removeItem("medtimeline-auth");
        window.location.href = "/login";
      });
  }, IDLE_TIMEOUT_MS);
}

if (typeof window !== "undefined") {
  const events = ["mousedown", "mousemove", "keypress", "scroll", "touchstart"];
  events.forEach((event) => window.addEventListener(event, resetIdleTimer, { passive: true }));
  resetIdleTimer();
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit & { token?: string; _retry?: boolean } = {}
  ): Promise<T> {
    const { token, _retry, ...fetchOptions } = options;
    const authToken = token || getToken();
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string>),
    };

    if (authToken) {
      headers["Authorization"] = `Bearer ${authToken}`;
    }

    // Only set Content-Type for non-FormData
    if (!(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...fetchOptions,
      headers,
    });

    // Transparently refresh an expired access token once, then retry. Most auth
    // endpoints are excluded (their 401s are real credential/refresh failures),
    // but /auth/me carries the access token like any data endpoint — a 401 there
    // is an expired-token race that SHOULD be refreshed + retried, otherwise the
    // current-user fetch silently fails and the account name renders blank.
    const isRefreshableAuthEndpoint = endpoint.startsWith("/auth/me");
    if (
      response.status === 401 &&
      !_retry &&
      (!endpoint.startsWith("/auth/") || isRefreshableAuthEndpoint)
    ) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        return this.request<T>(endpoint, {
          ...options,
          token: newToken,
          _retry: true,
        });
      }
      // No usable refresh token / refresh failed → the session is over.
      endSessionAndRedirect();
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Request failed" }));
      const detail =
        typeof error?.detail === "string" && error.detail.trim()
          ? error.detail
          : "Request failed";
      throw new ApiError(response.status, detail);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  async get<T>(endpoint: string, token?: string): Promise<T> {
    return this.request<T>(endpoint, { method: "GET", token });
  }

  async post<T>(endpoint: string, body?: unknown, token?: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
      token,
    });
  }

  /**
   * Log out, revoking BOTH tokens server-side. The refresh token is sent in the
   * body so the server can blacklist it — otherwise the 7-day refresh token would
   * stay valid after "logout" (SEC-AUTH-02). Best-effort: callers still clear
   * local state if this throws.
   */
  async logout(): Promise<void> {
    const refreshToken = getRefreshToken();
    const accessToken = getToken();
    await this.post<void>(
      "/auth/logout",
      refreshToken ? { refresh_token: refreshToken } : undefined,
      accessToken ?? undefined
    );
  }

  async put<T>(endpoint: string, body?: unknown, token?: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: "PUT",
      body: body ? JSON.stringify(body) : undefined,
      token,
    });
  }

  async postForm<T>(endpoint: string, formData: FormData, token?: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: "POST",
      body: formData,
      token,
    });
  }

  async delete<T>(endpoint: string, token?: string): Promise<T> {
    return this.request<T>(endpoint, { method: "DELETE", token });
  }

  // --- Extraction progress / control (session §2a iii–iv) ---

  /**
   * Scoped extraction progress. Pass the current batch's upload IDs to read
   * "1 of 1" instead of the user-global "84 of 85". With no IDs the endpoint
   * falls back to counting all unstructured files (legacy behavior).
   */
  async getExtractionProgress(
    uploadIds?: string[]
  ): Promise<ExtractionProgressResponse> {
    const qs =
      uploadIds && uploadIds.length
        ? `?ids=${encodeURIComponent(uploadIds.join(","))}`
        : "";
    return this.get<ExtractionProgressResponse>(
      `/upload/extraction-progress${qs}`
    );
  }

  /**
   * Per-file status for a batch (filtered client-side to the batch IDs). Carries
   * optional `progress_stage` / `progress_detail` for section-level progress.
   */
  async getExtractionFileStatuses(
    statuses: string[]
  ): Promise<{ files: ExtractionFileStatus[]; total: number }> {
    const qs = statuses.length
      ? `?statuses=${encodeURIComponent(statuses.join(","))}`
      : "";
    const payload = await this.get<{
      files: ExtractionFileStatus[];
      total: number;
    }>(
      `/upload/pending-extraction${qs}`
    );
    return {
      ...payload,
      files: payload.files.map((file) => ({
        ...file,
        local_run:
          file.local_run == null
            ? file.local_run
            : parseLocalRunInfo(file.local_run),
      })),
    };
  }

  /** Cancel in-flight extractions; the worker stops and marks them `cancelled`. */
  async cancelExtraction(
    uploadIds: string[]
  ): Promise<CancelExtractionResponse> {
    return this.post<CancelExtractionResponse>("/upload/cancel", {
      upload_ids: uploadIds,
    });
  }

  async getLocalAIJobs(
    activeOnly = false
  ): Promise<LocalAIJobResponse[]> {
    return this.get<LocalAIJobResponse[]>(
      `/local-ai/jobs?active_only=${activeOnly ? "true" : "false"}`
    );
  }

  async getLocalAIJob(id: string): Promise<LocalAIJobResponse> {
    return this.get<LocalAIJobResponse>(
      `/local-ai/jobs/${encodeURIComponent(id)}`
    );
  }

  async cancelLocalAIJob(id: string): Promise<LocalAIJobResponse> {
    return this.post<LocalAIJobResponse>(
      `/local-ai/jobs/${encodeURIComponent(id)}/cancel`
    );
  }

  async retryLocalAIJob(id: string): Promise<LocalAIJobResponse> {
    return this.post<LocalAIJobResponse>(
      `/local-ai/jobs/${encodeURIComponent(id)}/retry`
    );
  }

  async getUploadHistory(): Promise<UploadHistoryResponse> {
    return this.get<UploadHistoryResponse>("/upload/history");
  }

  async triggerExtraction(
    uploadIds: string[]
  ): Promise<TriggerExtractionResponse> {
    return this.post<TriggerExtractionResponse>("/upload/trigger-extraction", {
      upload_ids: uploadIds,
    });
  }

  // --- Validated local model pack / evidence ----------------------------

  async getLocalPackStatus(): Promise<LocalPackStatus> {
    return this.get<LocalPackStatus>("/local-ai/status");
  }

  async installLocalPack(): Promise<LocalPackOperationCreated> {
    return this.post<LocalPackOperationCreated>("/local-ai/install");
  }

  async getLocalPackOperation(id: string): Promise<LocalPackOperation> {
    return this.get<LocalPackOperation>(
      `/local-ai/operations/${encodeURIComponent(id)}`
    );
  }

  async resumeLocalPackOperation(id: string): Promise<LocalPackOperation> {
    return this.post<LocalPackOperation>(
      `/local-ai/operations/${encodeURIComponent(id)}/resume`
    );
  }

  async retryLocalPackOperation(id: string): Promise<LocalPackOperation> {
    return this.post<LocalPackOperation>(
      `/local-ai/operations/${encodeURIComponent(id)}/retry`
    );
  }

  async verifyLocalPack(): Promise<LocalPackOperationCreated> {
    return this.post<LocalPackOperationCreated>("/local-ai/verify");
  }

  async updateLocalPack(): Promise<LocalPackOperationCreated> {
    return this.post<LocalPackOperationCreated>("/local-ai/update");
  }

  async rollbackLocalPack(): Promise<LocalPackOperationCreated> {
    return this.post<LocalPackOperationCreated>("/local-ai/rollback");
  }

  async removeLocalModel(role: LocalModelRole): Promise<void> {
    return this.delete<void>(
      `/local-ai/models/${encodeURIComponent(role)}`
    );
  }

  async removeLocalPack(): Promise<void> {
    return this.delete<void>("/local-ai");
  }

  async getRecordEvidence(
    recordId: string
  ): Promise<RecordExtractionProvenance> {
    const payload = await this.get<unknown>(
      `/records/${encodeURIComponent(recordId)}/evidence`
    );
    return parseRecordExtractionProvenance(payload);
  }

  async generateSummary(
    body: GenerateSummaryRequest
  ): Promise<GenerateSummaryApiResponse> {
    return this.post<GenerateSummaryApiResponse>("/summary/generate", body);
  }
}

// --- OCR provider notices -------------------------------------------------
// A durable, per-file notice surfaced when OCR fell back across vision
// providers (one provider refused/failed but a later one read the document) or
// when no provider could read it. Rides along on each uploaded-file status
// object (`notices`, default []). `detail` is available but not shown by
// default. See docs/superpowers/specs/2026-06-21-ocr-provider-notices-design.md.
export interface OcrNotice {
  type: "ocr_fallback" | "ocr_unreadable";
  level: "info" | "warning";
  message: string;
  detail?: {
    used?: string | null;
    refused?: string[];
    attempts?: { provider: string; status: string }[];
  };
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export const api = new ApiClient(API_BASE);

// --- LLM provider settings (Admin → System: "AI providers" card) ---
// Keys are write-only from the UI: the GET only ever returns a masked
// preview (`key_masked`), never the plaintext key.

export interface LlmProviderInfo {
  name: string;
  is_local: boolean;
  supports_vision: boolean;
  configured: boolean;
  has_key: boolean;
  key_masked: string | null;
  base_url: string | null;
  model: string | null;
  enabled: boolean;
  source: string;
}

export interface LlmRouting {
  default: string;
  summary: string;
  section: string;
  dedup: string;
  extraction: string;
  vision: string;
  extraction_engine: string;
  processing_mode: ProcessingMode;
}

export interface LlmSettings {
  providers: LlmProviderInfo[];
  routing: LlmRouting;
}

export interface ProviderUpdate {
  api_key?: string;
  base_url?: string;
  model?: string;
  enabled?: boolean;
}

export interface RoutingUpdate {
  default?: string;
  summary?: string;
  section?: string;
  dedup?: string;
  extraction?: string;
  vision?: string;
  extraction_engine?: string;
  processing_mode?: ProcessingMode;
}

export interface ProviderTestResult {
  ok: boolean;
  model?: string;
  error_type?: string;
}

export function getLlmSettings(): Promise<LlmSettings> {
  return api.get<LlmSettings>("/settings/llm");
}

export function saveProvider(name: string, body: ProviderUpdate): Promise<void> {
  return api.put<void>(`/settings/llm/providers/${name}`, body);
}

export function clearProvider(name: string): Promise<void> {
  return api.delete<void>(`/settings/llm/providers/${name}`);
}

export function saveRouting(body: RoutingUpdate): Promise<void> {
  return api.put<void>("/settings/llm/routing", body);
}

export function testProvider(name: string): Promise<ProviderTestResult> {
  return api.post<ProviderTestResult>(`/settings/llm/providers/${name}/test`);
}
