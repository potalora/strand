import { useAuthStore } from "@/stores/useAuthStore";

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
    localStorage.removeItem("medtimeline-auth");
    window.location.href = "/login";
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
      throw new ApiError(response.status, error.detail || "Request failed");
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
