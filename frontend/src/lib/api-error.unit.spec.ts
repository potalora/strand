import { expect, test } from "@playwright/test";

import * as apiModule from "@/lib/api";
import { api, ApiError } from "@/lib/api";

test.afterEach(() => {
  delete (globalThis as { fetch?: typeof fetch }).fetch;
});

test("exports one API detail formatter", () => {
  const formatter = (apiModule as Record<string, unknown>)[
    "formatApiErrorDetail"
  ];

  expect(typeof formatter).toBe("function");
});

test("FastAPI 422 detail lists become a readable ApiError", async () => {
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        detail: [
          { loc: ["body", "date_from"], msg: "Input should be a valid date" },
          { loc: ["body", "patient_id"], msg: "Field required" },
        ],
      }),
      {
        status: 422,
        headers: { "Content-Type": "application/json" },
      }
    );

  await expect(api.get("/unit-test-validation-error")).rejects.toMatchObject({
    name: "ApiError",
    status: 422,
    message: "Input should be a valid date; Field required",
  } satisfies Partial<ApiError>);
});

test("object detail messages are normalized without exposing raw objects", async () => {
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: { msg: "Pack is unavailable" } }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });

  await expect(api.get("/unit-test-object-error")).rejects.toMatchObject({
    name: "ApiError",
    status: 503,
    message: "Pack is unavailable",
  } satisfies Partial<ApiError>);
});

test("nonblank string details pass through unchanged", async () => {
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: "  Explicit failure  " }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });

  await expect(api.get("/unit-test-string-error")).rejects.toMatchObject({
    message: "  Explicit failure  ",
  });
});

test("blank, malformed, non-string, and null details use the safe fallback", async () => {
  for (const detail of [
    "   ",
    [{ unexpected: "shape" }],
    { msg: null },
    { msg: {} },
    [{ msg: null }, { msg: {} }],
    null,
  ]) {
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ detail }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });

    await expect(api.get("/unit-test-fallback-error")).rejects.toMatchObject({
      message: "Request failed",
    });
  }
});
