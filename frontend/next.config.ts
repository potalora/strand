import type { NextConfig } from "next";

// The Next.js app documents (port 3000) actually execute JS in the browser, so
// the security headers / CSP must live here — the backend's CSP (port 8000)
// only covers API responses, which don't run scripts. (SEC-FE-03, SEC-API-03.)
const isDev = process.env.NODE_ENV !== "production";

/**
 * Origin (scheme://host[:port]) of the backend API the browser is allowed to
 * call. `NEXT_PUBLIC_API_URL` usually carries a path (…/api/v1); connect-src
 * only needs the origin, so normalize it. Falls back to the dev default that
 * `lib/api.ts` also uses.
 */
function apiOrigin(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  try {
    return new URL(raw).origin;
  } catch {
    return "http://localhost:8000";
  }
}

/**
 * Build the Content-Security-Policy. The policy is conditional on the build
 * mode because Next.js dev tooling has looser runtime needs than production:
 *
 *  - script-src: production serves Next.js App Router inline bootstrap / RSC
 *    flight scripts (`self.__next_f.push(...)`) which require 'unsafe-inline'
 *    absent a per-request nonce. Dev/HMR + React Fast Refresh additionally
 *    compile with eval, so dev also needs 'unsafe-eval'. A nonce-based policy
 *    would need request-time middleware; this config-only policy keeps external
 *    script loading and (in prod) eval blocked, which is the high-value win.
 *  - connect-src: same-origin + the backend API. Dev also opens an HMR
 *    websocket to the dev server, so ws:/wss: are allowed in dev only.
 *  - style-src: Tailwind, next/font, and inline style attributes inject inline
 *    <style>/style="" — these need 'unsafe-inline' in every mode.
 */
function buildCsp(): string {
  const api = apiOrigin();
  const scriptSrc = isDev
    ? "'self' 'unsafe-eval' 'unsafe-inline'"
    : "'self' 'unsafe-inline'";
  const connectSrc = isDev ? `'self' ${api} ws: wss:` : `'self' ${api}`;

  return [
    "default-src 'self'",
    `script-src ${scriptSrc}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    `connect-src ${connectSrc}`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

const SECURITY_HEADERS = [
  { key: "Content-Security-Policy", value: buildCsp() },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=()",
  },
];

const nextConfig: NextConfig = {
  // Produce a self-contained server bundle (.next/standalone/server.js) for
  // the Docker runtime image. See frontend/Dockerfile.
  output: "standalone",

  // Emit security headers on every route. The app documents are what execute
  // JS in the browser, so the CSP belongs here (the backend only guards API
  // responses).
  async headers() {
    return [
      {
        source: "/:path*",
        headers: SECURITY_HEADERS,
      },
    ];
  },
};

export default nextConfig;
