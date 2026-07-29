import { expect, test } from "@playwright/test";
import { api } from "@/lib/api";
import type {
  GenerateSummaryApiResponse,
  GenerateSummaryRequest,
} from "@/types/api";
import type {
  LocalPackOperation,
  LocalPackStatus,
  ProcessingMode,
  RecordExtractionProvenance,
} from "@/types/local-ai";
import {
  parseLocalRunInfo,
  parseRecordExtractionProvenance,
} from "@/types/local-ai";

const PACK_STATUS: LocalPackStatus = {
  platform: "apple_silicon",
  compatible: true,
  enabled: true,
  state: "ready",
  status_reason: null,
  active_revision: "apple-m4-16gb-v1",
  available_revision: "apple-m4-16gb-v1",
  models: [],
  operation: null,
};

const RECORD_PROVENANCE: RecordExtractionProvenance = {
  record_id: "record-1",
  processing_mode: "validated_strict_local",
  schema_version: "clinical-extraction-v1",
  evidence: [],
  unresolved_fields: [],
  rejected_fields: [],
  models: [],
};

test("processing modes remain an exact four-mode contract", () => {
  const modes = [
    "validated_strict_local",
    "custom_local",
    "cloud_assisted",
    "prompt_only",
  ] satisfies ProcessingMode[];

  expect(modes).toHaveLength(4);
});

test("ingestion provenance types keep the summary model out of uploads", () => {
  const invalid = {
    privacy_mode: "validated_strict_local",
    models: [
      {
        // @ts-expect-error Qwen is summary-only, never ingestion provenance.
        role: "summary",
        repository: "mlx-community/Qwen3.5-9B-MLX-4bit",
        revision: "0".repeat(40),
      },
    ],
  } satisfies import("@/types/local-ai").LocalRunInfo;

  expect(invalid.models[0].role).toBe("summary");
});

test("ingestion provenance parsers reject summary roles and more than two models", () => {
  const ocr = {
    role: "ocr",
    repository: "owner/ocr",
    revision: "0".repeat(40),
  };
  const extraction = {
    role: "extraction",
    repository: "owner/extraction",
    revision: "1".repeat(40),
  };

  expect(() =>
    parseLocalRunInfo({
      privacy_mode: "validated_strict_local",
      models: [
        {
          role: "summary",
          repository: "owner/summary",
          revision: "2".repeat(40),
        },
      ],
    })
  ).toThrow("Invalid ingestion model provenance");
  expect(() =>
    parseLocalRunInfo({
      privacy_mode: "validated_strict_local",
      models: [ocr, extraction, ocr],
    })
  ).toThrow("Invalid ingestion model provenance");
  expect(
    parseLocalRunInfo({
      privacy_mode: "validated_strict_local",
      models: [ocr, extraction],
    }).models
  ).toHaveLength(2);

  expect(() =>
    parseRecordExtractionProvenance({
      ...RECORD_PROVENANCE,
      models: [
        {
          role: "summary",
          repository: "owner/summary",
          revision: "2".repeat(40),
          quantization: "4bit",
          runtime: "mlx-vlm 0.5.0",
        },
      ],
    })
  ).toThrow("Invalid ingestion model provenance");

  const lineageModel = {
    ...ocr,
    quantization: "4bit",
    runtime: "mlx-vlm 0.5.0",
  };
  expect(
    parseRecordExtractionProvenance({
      ...RECORD_PROVENANCE,
      models: [lineageModel, lineageModel, lineageModel],
    }).models
  ).toHaveLength(3);
  expect(() =>
    parseRecordExtractionProvenance({
      ...RECORD_PROVENANCE,
      models: Array.from({ length: 513 }, () => lineageModel),
    })
  ).toThrow("Invalid ingestion model provenance");
});

test("summary generation types reject backend-invalid mode combinations", () => {
  const strictLocal: GenerateSummaryRequest = {
    patient_id: "patient-1",
    summary_type: "full",
    output_format: "both",
    processing_mode: "validated_strict_local",
  };
  const customLocal: GenerateSummaryRequest = {
    patient_id: "patient-1",
    summary_type: "full",
    output_format: "natural_language",
    processing_mode: "custom_local",
    custom_system_prompt: "Use the supplied record facts only.",
  };
  const cloudAssisted: GenerateSummaryRequest = {
    patient_id: "patient-1",
    summary_type: "full",
    output_format: "natural_language",
    processing_mode: "cloud_assisted",
    provider: "gemini",
    model: "gemini-3.5-flash",
    custom_user_prompt: "Organize by date.",
  };

  const promptOnly: GenerateSummaryRequest = {
    patient_id: "patient-1",
    summary_type: "full",
    output_format: "natural_language",
    // @ts-expect-error Prompt-only uses /summary/build-prompt, never /generate.
    processing_mode: "prompt_only",
  };
  // @ts-expect-error Strict local cannot override provider or model identity.
  const strictWithProvider: GenerateSummaryRequest = {
    ...strictLocal,
    provider: "gemini",
  };
  // @ts-expect-error Strict local cannot replace server-owned safety prompts.
  const strictWithCustomPrompt: GenerateSummaryRequest = {
    ...strictLocal,
    custom_system_prompt: "Replace the safety prompt.",
  };
  // @ts-expect-error Custom local uses the persisted loopback route.
  const customWithProvider: GenerateSummaryRequest = {
    ...customLocal,
    provider: "ollama",
  };

  expect([
    strictLocal.processing_mode,
    customLocal.processing_mode,
    cloudAssisted.processing_mode,
    promptOnly.processing_mode,
    strictWithProvider.processing_mode,
    strictWithCustomPrompt.processing_mode,
    customWithProvider.processing_mode,
  ]).toHaveLength(7);
});

test("local pack and record evidence helpers use their API routes", async () => {
  const calls: Array<{ method: string; url: string }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    const url = String(input);
    calls.push({ method: init?.method ?? "GET", url });
    const body = url.endsWith("/local-ai/status")
      ? PACK_STATUS
      : RECORD_PROVENANCE;
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    await expect(api.getLocalPackStatus()).resolves.toEqual(PACK_STATUS);
    await expect(api.getRecordEvidence("record-1")).resolves.toEqual(
      RECORD_PROVENANCE
    );
  } finally {
    globalThis.fetch = originalFetch;
  }

  expect(calls.map(({ method, url }) => ({ method, path: new URL(url).pathname })))
    .toEqual([
      { method: "GET", path: "/api/v1/local-ai/status" },
      { method: "GET", path: "/api/v1/records/record-1/evidence" },
    ]);
});

test("pack operation helpers expose typed operation results", async () => {
  const operation = {
    id: "operation-1",
    action: "install",
    state: "running",
    current_role: "ocr",
    bytes_done: 10,
    bytes_total: 20,
    message: null,
    retryable: false,
  } satisfies LocalPackOperation;

  expect(operation.current_role).toBe("ocr");
});

test("pack lifecycle helpers map to the complete operation API", async () => {
  const calls: Array<{ method: string; path: string }> = [];
  const operation: LocalPackOperation = {
    id: "00000000-0000-0000-0000-000000000001",
    action: "verify",
    state: "queued",
    current_role: null,
    bytes_done: 0,
    bytes_total: 20,
    message: null,
    retryable: false,
  };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    const url = new URL(String(input));
    calls.push({ method: init?.method ?? "GET", path: url.pathname });
    if (init?.method === "DELETE") {
      return new Response(null, { status: 204 });
    }
    const body = url.pathname.includes("/operations/")
      ? operation
      : { operation_id: operation.id, state: "queued" };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    await api.installLocalPack();
    await api.updateLocalPack();
    await api.verifyLocalPack();
    await api.rollbackLocalPack();
    await api.getLocalPackOperation(operation.id);
    await api.resumeLocalPackOperation(operation.id);
    await api.retryLocalPackOperation(operation.id);
    await api.removeLocalModel("summary");
    await api.removeLocalPack();
  } finally {
    globalThis.fetch = originalFetch;
  }

  expect(calls).toEqual([
    { method: "POST", path: "/api/v1/local-ai/install" },
    { method: "POST", path: "/api/v1/local-ai/update" },
    { method: "POST", path: "/api/v1/local-ai/verify" },
    { method: "POST", path: "/api/v1/local-ai/rollback" },
    {
      method: "GET",
      path: `/api/v1/local-ai/operations/${operation.id}`,
    },
    {
      method: "POST",
      path: `/api/v1/local-ai/operations/${operation.id}/resume`,
    },
    {
      method: "POST",
      path: `/api/v1/local-ai/operations/${operation.id}/retry`,
    },
    { method: "DELETE", path: "/api/v1/local-ai/models/summary" },
    { method: "DELETE", path: "/api/v1/local-ai" },
  ]);
});

test("generate summary carries its immutable mode and grounded provenance", async () => {
  const request = {
    patient_id: "patient-1",
    summary_type: "full",
    output_format: "both",
    processing_mode: "validated_strict_local",
  } satisfies GenerateSummaryRequest;
  const response = {
    id: "summary-1",
    natural_language: "Grounded summary",
    json_data: {},
    record_count: 1,
    duplicate_warning: null,
    de_identification_report: null,
    model_used: "Qwen/Qwen3.5-9B@0123456789abcdef",
    generated_at: "2026-07-27T12:00:00Z",
    processing_mode: "validated_strict_local",
    model_provenance: {
      processing_mode: "validated_strict_local",
      manifest_sha256: "a".repeat(64),
      pack_revision: "apple-m4-16gb-v1",
      model: {
        role: "summary",
        repository: "Qwen/Qwen3.5-9B",
        revision: "0123456789abcdef0123456789abcdef01234567",
        quantization: "int4",
        runtime: { name: "mlx-vlm", version: "0.5.0" },
      },
    },
    typed_response: {
      sections: [],
      uncertainties: [],
    },
  } satisfies GenerateSummaryApiResponse;
  const originalFetch = globalThis.fetch;
  let posted: unknown;
  globalThis.fetch = (async (_input, init) => {
    posted = JSON.parse(String(init?.body));
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    await expect(api.generateSummary(request)).resolves.toEqual(response);
  } finally {
    globalThis.fetch = originalFetch;
  }

  expect(posted).toEqual(request);
});
