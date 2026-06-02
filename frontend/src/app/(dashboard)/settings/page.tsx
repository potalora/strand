"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Lock, Sun, Moon } from "lucide-react";
import { api } from "@/lib/api";
import type { UserResponse } from "@/types/api";
import { useAuthStore } from "@/stores/useAuthStore";
import { usePreferencesStore } from "@/stores/usePreferencesStore";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

function maskEmail(email: string | undefined | null): string {
  if (!email) return "—";
  const [local, domain] = email.split("@");
  if (!domain) return email;
  const head = local.slice(0, 1);
  return `${head}${"•".repeat(Math.max(3, local.length - 1))}@${domain}`;
}

function Field({ l, v }: { l: string; v: string }) {
  return (
    <div className="field">
      <div className="field-l">{l}</div>
      <div className="field-v">{v}</div>
    </div>
  );
}

export default function SettingsPage() {
  const [shown, setShown] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [user, setUser] = useState<UserResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const { theme, setTheme } = useTheme();
  const { clearTokens } = useAuthStore();
  const { skipDeleteConfirm, setSkipDeleteConfirm } = usePreferencesStore();

  useEffect(() => {
    // rAF fires client-side only — mark mounted here so theme reads avoid a
    // hydration mismatch, and trigger the entrance transition in one pass.
    const id = requestAnimationFrame(() => {
      setShown(true);
      setMounted(true);
    });
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    api
      .get<UserResponse>("/auth/me")
      .catch(() => null)
      .then((u) => setUser(u))
      .finally(() => setLoading(false));
  }, []);

  const handleSignOut = async () => {
    // API → POST /auth/logout (revokes the refresh token server-side), then clear local tokens.
    try {
      await api.post("/auth/logout");
    } catch {
      // Best-effort: still clear locally even if the call fails.
    }
    clearTokens();
    window.location.href = "/login";
  };

  if (loading) return <RetroLoadingState text="Loading settings" />;

  const isDark = mounted && theme === "dark";

  return (
    <div className={`screen s24 ${shown ? "on" : ""}`}>
      {/* Header */}
      <div className="page-top">
        <div>
          <p className="kicker">Account &amp; preferences</p>
          <h1 className="h1 display">Settings</h1>
          <p className="h-sub">
            Manage your account and how this organizer behaves. For full data
            controls and record stats, see Admin &gt; System.
          </p>
        </div>
        <SecureChip />
      </div>

      {/* Account */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Account</h3>
        </div>
        <div className="s12">
          <Field l="Name" v={user?.display_name?.trim() || "Not set"} />
          <Field l="Email" v={maskEmail(user?.email)} />
          <Field l="Record owner" v="You" />
        </div>
      </div>

      {/* Preferences */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Preferences</h3>
        </div>

        {/* Appearance */}
        <div
          className="field"
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}
        >
          <div>
            <div className="field-l" style={{ marginBottom: 6 }}>
              Appearance
            </div>
            <div className="field-v" style={{ padding: 0 }}>
              {isDark ? "Dark" : "Light"} theme
            </div>
          </div>
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => setTheme(isDark ? "light" : "dark")}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={15} /> : <Moon size={15} />}
            Switch to {isDark ? "light" : "dark"}
          </button>
        </div>

        {/* Delete confirmation */}
        <div
          className="field"
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}
        >
          <div>
            <div className="field-l" style={{ marginBottom: 6 }}>
              Delete confirmation
            </div>
            <div className="field-v" style={{ padding: 0 }}>
              {skipDeleteConfirm
                ? "Off — records are removed without a prompt"
                : "On — confirm before removing a record"}
            </div>
          </div>
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => setSkipDeleteConfirm(!skipDeleteConfirm)}
          >
            {skipDeleteConfirm ? "Re-enable prompt" : "Disable prompt"}
          </button>
        </div>
      </div>

      {/* Session */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Session</h3>
        </div>
        <p className="h-sub" style={{ margin: "0 0 16px" }}>
          All health data is stored locally and encrypted at rest. It only leaves
          your device when you explicitly request an AI summary or document
          extraction, and only after de-identification.
        </p>
        <button className="btn ghost" onClick={handleSignOut}>
          Sign out
        </button>
      </div>
    </div>
  );
}
