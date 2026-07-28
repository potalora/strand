import { expect, test } from "@playwright/test";
import {
  createSerializedRoutingSaver,
  type RoutingSaveTicket,
} from "@/lib/llm-routing";
import type { RoutingUpdate } from "@/lib/api";

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("routing saver commits snapshots in enqueue order and marks only the newest ticket current", async () => {
  const firstGate = deferred();
  const events: string[] = [];
  const saver = createSerializedRoutingSaver(async (body) => {
    const mode = String(body.processing_mode);
    events.push(`start:${mode}`);
    if (mode === "cloud_assisted") await firstGate.promise;
    events.push(`commit:${mode}`);
  });

  const cloud = saver.enqueue({ processing_mode: "cloud_assisted" });
  const custom = saver.enqueue({ processing_mode: "custom_local" });

  await expect.poll(() => events).toEqual(["start:cloud_assisted"]);
  expect(cloud.isLatest()).toBe(false);
  expect(custom.isLatest()).toBe(true);

  firstGate.resolve();
  await custom.promise;
  expect(events).toEqual([
    "start:cloud_assisted",
    "commit:cloud_assisted",
    "start:custom_local",
    "commit:custom_local",
  ]);
  await expect(saver.latest()).resolves.toBeUndefined();
});

test("routing saver continues with a newer snapshot after an older save rejects", async () => {
  const committed: RoutingUpdate[] = [];
  let attempts = 0;
  const saver = createSerializedRoutingSaver(async (body) => {
    attempts += 1;
    if (attempts === 1) throw new Error("transient failure");
    committed.push(body);
  });

  const failed: RoutingSaveTicket = saver.enqueue({
    processing_mode: "cloud_assisted",
  });
  const recovered = saver.enqueue({ processing_mode: "custom_local" });

  await expect(failed.promise).rejects.toThrow("transient failure");
  await expect(recovered.promise).resolves.toBeUndefined();
  expect(committed).toEqual([{ processing_mode: "custom_local" }]);
  expect(recovered.isLatest()).toBe(true);
});

test("routing saver reset clears a failed tail and invalidates the old ticket", async () => {
  const gate = deferred();
  const saver = createSerializedRoutingSaver(async () => {
    await gate.promise;
    throw new Error("stale save failure");
  });

  const stale = saver.enqueue({ processing_mode: "custom_local" });
  saver.reset();

  expect(stale.isLatest()).toBe(false);
  await expect(saver.latest()).resolves.toBeUndefined();

  gate.resolve();
  await expect(stale.promise).rejects.toThrow("stale save failure");
  await expect(saver.latest()).resolves.toBeUndefined();
});
