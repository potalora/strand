import { isIP } from "node:net";

const TEST_DATABASE_NAME = /(^|[_-])(test|e2e)([_-]|$)/i;
const DATABASE_ROUTE_OVERRIDES = [
  "database",
  "host",
  "port",
  "service",
  "servicefile",
] as const;
export const LOCAL_ONLY_PROFILE_INITIALIZED =
  "MEDTIMELINE_E2E_LOCAL_PROFILE_INITIALIZED";

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  if (host === "localhost") return true;
  if (isIP(host) === 4) return host.startsWith("127.");
  if (isIP(host) === 6) return host === "::1";
  return false;
}

function canonicalDatabaseUrl(rawUrl: string): string {
  try {
    return new URL(rawUrl).href;
  } catch {
    return rawUrl;
  }
}

export function isLocalOnlyProfileReentry(
  rawE2eDatabaseUrl: string,
  currentDatabaseUrl: string | undefined,
  profileInitialized: string | undefined
): boolean {
  return (
    profileInitialized === "1" &&
    currentDatabaseUrl !== undefined &&
    canonicalDatabaseUrl(currentDatabaseUrl) === canonicalDatabaseUrl(rawE2eDatabaseUrl)
  );
}

export function requireDedicatedLoopbackDatabase(
  rawUrl: string,
  developmentDatabaseUrl?: string,
  localOnlyProfileReentry = false
): string {
  let databaseUrl: URL;
  try {
    databaseUrl = new URL(rawUrl);
  } catch {
    throw new Error("E2E_DATABASE_URL must be a valid PostgreSQL URL.");
  }

  if (databaseUrl.protocol !== "postgresql+asyncpg:") {
    throw new Error(
      "E2E_DATABASE_URL must use the postgresql+asyncpg driver."
    );
  }
  if (!isLoopbackHost(databaseUrl.hostname)) {
    throw new Error("E2E_DATABASE_URL must use a loopback database host.");
  }
  if (
    DATABASE_ROUTE_OVERRIDES.some((parameter) =>
      databaseUrl.searchParams.has(parameter)
    )
  ) {
    throw new Error(
      "E2E_DATABASE_URL must not contain a database host override."
    );
  }

  const databaseName = decodeURIComponent(databaseUrl.pathname.replace(/^\/+/, ""));
  if (!databaseName || databaseName.includes("/") || !TEST_DATABASE_NAME.test(databaseName)) {
    throw new Error(
      "E2E_DATABASE_URL database name must contain a test or e2e segment."
    );
  }

  if (
    developmentDatabaseUrl &&
    canonicalDatabaseUrl(developmentDatabaseUrl) === canonicalDatabaseUrl(rawUrl) &&
    !localOnlyProfileReentry
  ) {
    throw new Error(
      "E2E_DATABASE_URL must not be the configured development database."
    );
  }

  return rawUrl;
}
