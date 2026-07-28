import type {
  LlmProviderInfo,
  LlmRouting,
  RoutingUpdate,
} from "@/lib/api";
import type { ProcessingMode } from "@/types/local-ai";

export type RoutableProvider = Pick<
  LlmProviderInfo,
  "name" | "is_local" | "configured" | "enabled" | "base_url"
>;

export interface RoutingSaveTicket {
  promise: Promise<void>;
  isLatest: () => boolean;
}

export interface SerializedRoutingSaver {
  enqueue: (body: RoutingUpdate) => RoutingSaveTicket;
  latest: () => Promise<void>;
  reset: () => void;
}

const LOCAL_PROVIDER_NAMES = new Set(["ollama", "lmstudio"]);

export function createSerializedRoutingSaver(
  save: (body: RoutingUpdate) => Promise<void>
): SerializedRoutingSaver {
  let tail: Promise<void> = Promise.resolve();
  let latestVersion = 0;

  return {
    enqueue(body) {
      const version = ++latestVersion;
      const promise = tail.catch(() => undefined).then(() => save(body));
      tail = promise;
      return {
        promise,
        isLatest: () => version === latestVersion,
      };
    },
    latest() {
      return tail;
    },
    reset() {
      latestVersion += 1;
      tail = Promise.resolve();
    },
  };
}

export function isLoopbackUrl(value: string | null): boolean {
  if (!value) return false;
  try {
    const url = new URL(value);
    const ipv4Parts = url.hostname.split(".");
    const isIpv4Loopback =
      ipv4Parts.length === 4 &&
      ipv4Parts[0] === "127" &&
      ipv4Parts.every(
        (part) =>
          /^\d{1,3}$/.test(part) &&
          Number(part) >= 0 &&
          Number(part) <= 255
      );
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      (url.hostname === "localhost" ||
        url.hostname === "[::1]" ||
        isIpv4Loopback)
    );
  } catch {
    return false;
  }
}

export function isCustomLocalProvider(
  provider: RoutableProvider
): boolean {
  return (
    LOCAL_PROVIDER_NAMES.has(provider.name) &&
    provider.is_local &&
    provider.configured &&
    provider.enabled &&
    isLoopbackUrl(provider.base_url)
  );
}

export function isConfiguredCloudProvider(
  provider: RoutableProvider
): boolean {
  return (
    !provider.is_local &&
    provider.configured &&
    provider.enabled
  );
}

export function selectProviderForMode(
  providers: RoutableProvider[],
  routing: LlmRouting,
  mode: "custom_local" | "cloud_assisted"
): RoutableProvider | null {
  const eligible =
    mode === "custom_local"
      ? providers.filter(isCustomLocalProvider)
      : providers.filter(isConfiguredCloudProvider);
  return (
    eligible.find((provider) => provider.name === routing.summary) ??
    eligible.find((provider) => provider.name === routing.default) ??
    eligible[0] ??
    null
  );
}

export function completeProviderRouting(
  routing: LlmRouting,
  provider: string,
  processingMode: Extract<
    ProcessingMode,
    "custom_local" | "cloud_assisted"
  >
): RoutingUpdate {
  return {
    default: provider,
    summary: provider,
    section: provider,
    dedup: provider,
    extraction: provider,
    vision: provider,
    extraction_engine: routing.extraction_engine,
    processing_mode: processingMode,
  };
}

export function completeEffectiveRouting(
  routing: LlmRouting,
  update: Partial<
    Pick<
      LlmRouting,
      "default" | "summary" | "section" | "dedup" | "extraction" | "vision"
    >
  >
): RoutingUpdate {
  return {
    default: update.default ?? routing.default,
    summary: update.summary ?? routing.summary,
    section: update.section ?? routing.section,
    dedup: update.dedup ?? routing.dedup,
    extraction: update.extraction ?? routing.extraction,
    vision: update.vision ?? routing.vision,
    extraction_engine: routing.extraction_engine,
    processing_mode: routing.processing_mode,
  };
}
