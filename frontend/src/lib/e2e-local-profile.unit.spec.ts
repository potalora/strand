import { expect, test } from "@playwright/test";

import {
  isLocalOnlyProfileReentry,
  LOCAL_ONLY_PROFILE_INITIALIZED,
  requireDedicatedLoopbackDatabase,
} from "../../e2e/helpers/local-only-profile";

test.describe("requireDedicatedLoopbackDatabase", () => {
  test("accepts dedicated test databases on loopback interfaces", () => {
    expect(
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://localhost:5432/medtimeline_e2e_local"
      )
    ).toBe("postgresql+asyncpg://localhost:5432/medtimeline_e2e_local");
    expect(
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://127.0.0.1:5432/medtimeline_test"
      )
    ).toBe("postgresql+asyncpg://127.0.0.1:5432/medtimeline_test");
    expect(
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://[::1]:5432/e2e_medtimeline"
      )
    ).toBe("postgresql+asyncpg://[::1]:5432/e2e_medtimeline");
  });

  test("rejects remote database hosts", () => {
    expect(() =>
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://db.example.com:5432/medtimeline_e2e"
      )
    ).toThrow(/loopback/i);
    expect(() =>
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://192.0.2.10:5432/medtimeline_test"
      )
    ).toThrow(/loopback/i);
    expect(() =>
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://localhost:5432/medtimeline_test?host=db.example.com"
      )
    ).toThrow(/host override/i);
  });

  test("rejects non-test database names and unsupported drivers", () => {
    expect(() =>
      requireDedicatedLoopbackDatabase(
        "postgresql+asyncpg://localhost:5432/medtimeline"
      )
    ).toThrow(/test or e2e/i);
    expect(() =>
      requireDedicatedLoopbackDatabase(
        "postgresql://localhost:5432/medtimeline_e2e"
      )
    ).toThrow(/postgresql\+asyncpg/i);
  });

  test("rejects the configured development database even if its name looks safe", () => {
    const url = "postgresql+asyncpg://localhost:5432/medtimeline_test";
    expect(() => requireDedicatedLoopbackDatabase(url, url)).toThrow(
      /development database/i
    );
  });

  test("permits parent-initialized local-only profile re-entry", () => {
    const url = "postgresql+asyncpg://localhost:5432/medtimeline_e2e_local";
    const localOnlyProfileReentry = isLocalOnlyProfileReentry(
      url,
      url,
      "1"
    );

    expect(
      requireDedicatedLoopbackDatabase(url, url, localOnlyProfileReentry)
    ).toBe(url);
    expect(
      isLocalOnlyProfileReentry(url, url, undefined)
    ).toBe(false);
    expect(LOCAL_ONLY_PROFILE_INITIALIZED).toBe(
      "MEDTIMELINE_E2E_LOCAL_PROFILE_INITIALIZED"
    );
  });
});
