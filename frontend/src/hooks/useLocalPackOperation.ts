"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type {
  LocalPackOperation,
  LocalPackStatus,
} from "@/types/local-ai";

const POLL_MS = 750;

type PackAction = "install" | "verify" | "update" | "rollback";

const ACTIONS: Record<
  PackAction,
  () => Promise<{ operation_id: string; state: "queued" }>
> = {
  install: () => api.installLocalPack(),
  verify: () => api.verifyLocalPack(),
  update: () => api.updateLocalPack(),
  rollback: () => api.rollbackLocalPack(),
};

export function useLocalPackOperation() {
  const [status, setStatus] = useState<LocalPackStatus | null>(null);
  const [operation, setOperation] = useState<LocalPackOperation | null>(null);
  const [loading, setLoading] = useState(true);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollingInterrupted, setPollingInterrupted] = useState(false);
  const [pollEpoch, setPollEpoch] = useState(0);
  const operationId = useRef<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const next = await api.getLocalPackStatus();
      if (
        !next ||
        !Array.isArray(next.models) ||
        typeof next.state !== "string" ||
        typeof next.compatible !== "boolean"
      ) {
        throw new Error("invalid local pack status");
      }
      setStatus(next);
      window.dispatchEvent(
        new CustomEvent("medtimeline:local-pack-status", { detail: next })
      );
      if (!operationId.current && next.operation) {
        setOperation(next.operation);
        if (["queued", "running"].includes(next.operation.state)) {
          setPollingInterrupted(false);
          operationId.current = next.operation.id;
        }
      }
      setError(null);
    } catch {
      setError("Local model pack status is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (!operationId.current) return;
    let cancelled = false;
    let timer: number | null = null;
    const poll = async () => {
      const id = operationId.current;
      if (!id || cancelled) return;
      try {
        const next = await api.getLocalPackOperation(id);
        if (cancelled) return;
        setOperation(next);
        if (["completed", "failed", "paused"].includes(next.state)) {
          operationId.current = null;
          setPollingInterrupted(false);
          setPendingAction(null);
          await reload();
          return;
        }
      } catch {
        if (!cancelled) {
          operationId.current = null;
          setPollingInterrupted(true);
          setPendingAction(null);
          setError("Local model pack operation could not be refreshed.");
        }
        return;
      }
      if (!cancelled && operationId.current === id) {
        timer = window.setTimeout(() => void poll(), POLL_MS);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [operation?.id, operation?.state, pollEpoch, reload]);

  const start = useCallback(async (action: PackAction) => {
    setPendingAction(action);
    setError(null);
    try {
      const created = await ACTIONS[action]();
      setPollingInterrupted(false);
      operationId.current = created.operation_id;
      setOperation({
        id: created.operation_id,
        action,
        state: "queued",
        current_role: null,
        bytes_done: 0,
        bytes_total: 0,
        message: "Waiting for model pack operation.",
        retryable: false,
      });
    } catch {
      setPendingAction(null);
      setError("The local model pack operation could not be started.");
    }
  }, []);

  const restart = useCallback(
    async (kind: "resume" | "retry") => {
      if (!operation) return;
      setPendingAction(kind);
      setError(null);
      try {
        const next =
          kind === "resume"
            ? await api.resumeLocalPackOperation(operation.id)
            : await api.retryLocalPackOperation(operation.id);
        setPollingInterrupted(false);
        operationId.current = next.id;
        setOperation(next);
      } catch {
        setPendingAction(null);
        setError("The local model pack operation could not be restarted.");
      }
    },
    [operation]
  );

  const retryPolling = useCallback(() => {
    if (
      !operation ||
      !["queued", "running"].includes(operation.state)
    ) {
      return;
    }
    setError(null);
    setPollingInterrupted(false);
    operationId.current = operation.id;
    setPollEpoch((epoch) => epoch + 1);
  }, [operation]);

  const remove = useCallback(async () => {
    setPendingAction("remove");
    setError(null);
    try {
      await api.removeLocalPack();
      setPollingInterrupted(false);
      setOperation(null);
      await reload();
    } catch {
      setError("The local model pack could not be removed.");
    } finally {
      setPendingAction(null);
    }
  }, [reload]);

  return {
    status,
    operation,
    loading,
    busy:
      pendingAction !== null ||
      (!pollingInterrupted &&
        (operation?.state === "queued" ||
          operation?.state === "running")),
    pollingInterrupted,
    error,
    reload,
    start,
    restart,
    retryPolling,
    remove,
  };
}
