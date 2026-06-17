"use client";

import { create } from "zustand";
import { api } from "@/lib/api";
import type { UserResponse } from "@/types/api";

type UserStatus = "idle" | "loading" | "loaded" | "error";

interface UserState {
  user: UserResponse | null;
  status: UserStatus;
  /**
   * Fetch the current user once and cache it. Multiple consumers (the nav and
   * the System pane) call this on mount; only the first triggers a request
   * while one is in flight or already loaded. Pass `{ force: true }` to refetch.
   */
  fetchUser: (opts?: { force?: boolean }) => Promise<void>;
  clearUser: () => void;
}

/**
 * Shared current-user store.
 *
 * The 15-min access token intermittently 401s on `/auth/me` (a token-refresh
 * race). Previously the nav, the settings page, and the System pane EACH
 * fetched `/auth/me` independently and swallowed the 401 → `user` stayed null →
 * the name rendered blank/"Not set" with no retry until the component remounted.
 *
 * This store fixes that by:
 *  - fetching `/auth/me` once and caching it for every consumer;
 *  - retrying a transient failure once before giving up (lib/api.ts already
 *    transparently refreshes the access token + retries the request on a 401;
 *    this is a belt-and-suspenders backstop on top of that);
 *  - never clobbering a cached good user when a later fetch fails.
 *
 * Consumers must distinguish "still loading" (show a skeleton) from "loaded but
 * empty" (show "Not set") via `status` — a transient 401 must NOT blank the name.
 */
export const useUserStore = create<UserState>((set, get) => ({
  user: null,
  status: "idle",
  fetchUser: async (opts) => {
    const { status } = get();
    if (!opts?.force && (status === "loading" || status === "loaded")) return;
    set({ status: "loading" });

    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const user = await api.get<UserResponse>("/auth/me");
        set({ user, status: "loaded" });
        return;
      } catch {
        // Retry once more on a transient failure; on the final miss flag the
        // error but preserve any previously cached user (never blank it).
        if (attempt === 0) continue;
        set((s) => ({ status: "error", user: s.user }));
      }
    }
  },
  clearUser: () => set({ user: null, status: "idle" }),
}));
