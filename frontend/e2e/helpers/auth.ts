const API_BASE = "http://localhost:8000/api/v1";

const TEST_LOGIN_IDENTIFIER = "test renderer";
const TEST_PASSWORD = "TestPass123!";
const TEST_DISPLAY_NAME = "Test Renderer";

export interface AuthContext {
  accessToken: string;
  refreshToken: string;
}

export async function getTestAuth(): Promise<AuthContext> {
  // Try to register first (idempotent — if account exists, fall through to login)
  try {
    const regRes = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        login_identifier: TEST_LOGIN_IDENTIFIER,
        password: TEST_PASSWORD,
        display_name: TEST_DISPLAY_NAME,
      }),
    });
    if (regRes.ok) {
      const data = await regRes.json();
      return { accessToken: data.access_token, refreshToken: data.refresh_token };
    }
  } catch {
    // Registration may fail if account already exists — that's fine
  }

  // Login
  const loginRes = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      login_identifier: TEST_LOGIN_IDENTIFIER,
      password: TEST_PASSWORD,
    }),
  });

  if (!loginRes.ok) {
    throw new Error(`Login failed: ${loginRes.status} ${await loginRes.text()}`);
  }

  const data = await loginRes.json();
  return { accessToken: data.access_token, refreshToken: data.refresh_token };
}

export function authHeaders(token: string): Record<string, string> {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}
