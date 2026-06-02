"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Activity, Eye, EyeOff, Lock } from "lucide-react";
import { api } from "@/lib/api";
import type { UserResponse } from "@/types/api";

const authStyles = `
.auth-wrap {
  min-height: 100svh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 28px 20px;
  background-image:
    radial-gradient(110% 70% at 50% -8%, color-mix(in oklab, var(--primary) 8%, transparent), transparent 62%),
    radial-gradient(80% 55% at 88% 108%, color-mix(in oklab, var(--primary) 5%, transparent), transparent 60%);
}
.auth-card {
  width: 100%;
  max-width: 420px;
  padding: 34px 32px 26px;
}
.auth-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  margin-bottom: 26px;
}
.auth-display {
  font-size: 33px;
  margin: 0;
  color: var(--text);
}
.auth-sub {
  margin-top: 8px;
  margin-bottom: 0;
}
.auth-error {
  margin: 20px 0 0;
  padding: 11px 14px;
  border-radius: var(--radius-sm);
  background: color-mix(in oklab, var(--danger) 9%, var(--card));
  border: 1px solid color-mix(in oklab, var(--danger) 40%, var(--border));
  color: var(--danger);
  font-size: 13.5px;
  line-height: 1.45;
}
.auth-form {
  display: flex;
  flex-direction: column;
  gap: 17px;
  margin-top: 26px;
}
.auth-field {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.auth-field .field-l {
  margin-bottom: 0;
}
.auth-input-wrap {
  position: relative;
  display: flex;
  align-items: center;
}
.auth-input {
  width: 100%;
  font-family: var(--font-body), sans-serif;
  font-size: 14.5px;
  color: var(--text);
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 11px 13px;
  transition: border-color 0.16s, box-shadow 0.16s;
  outline: none;
}
.auth-input.has-toggle {
  padding-right: 42px;
}
.auth-input::placeholder {
  color: var(--text-muted);
}
.auth-input:hover {
  border-color: var(--border-strong);
}
.auth-input:focus {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--ring) 18%, transparent);
}
.auth-toggle {
  position: absolute;
  right: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: 0;
  background: transparent;
  color: var(--text-muted);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: color 0.16s;
}
.auth-toggle:hover {
  color: var(--text-dim);
}
.auth-hint {
  margin: 1px 0 0;
  font-size: 12px;
  line-height: 1.4;
}
.auth-submit {
  width: 100%;
  justify-content: center;
  margin-top: 4px;
  padding: 12px 18px;
}
.auth-switch {
  margin: 20px 0 0;
  text-align: center;
  font-size: 13.5px;
  color: var(--text-muted);
}
.auth-link {
  color: var(--primary);
  font-weight: 600;
  text-decoration: none;
  transition: color 0.16s;
}
.auth-link:hover {
  color: var(--primary-press);
  text-decoration: underline;
}
.auth-trust {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 9px;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
}
.auth-trust-note {
  font-size: 12.5px;
}
`;

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await api.post<UserResponse>("/auth/register", {
        email,
        password,
        display_name: displayName || undefined,
      });
      router.push("/login");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrap">
      <style>{authStyles}</style>
      <div className="card-surface pad auth-card screen on">
        {/* Brand lockup */}
        <div className="auth-brand">
          <span className="brand-mark" aria-hidden="true">
            <Activity size={19} strokeWidth={2.25} />
          </span>
          <span className="brand-name">MedTimeline</span>
        </div>

        {/* Headline */}
        <p className="kicker">Personal Health Record</p>
        <h1 className="display auth-display">Create your record</h1>
        <p className="h-sub auth-sub">
          A private, encrypted home for your health history.
        </p>

        {/* Error */}
        {error && (
          <p className="auth-error" role="alert">
            {error}
          </p>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-field">
            <label htmlFor="displayName" className="field-l">
              Display name
            </label>
            <input
              id="displayName"
              className="auth-input"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              autoComplete="name"
              placeholder="Optional"
            />
          </div>

          <div className="auth-field">
            <label htmlFor="email" className="field-l">
              Email
            </label>
            <input
              id="email"
              className="auth-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              placeholder="you@example.com"
            />
          </div>

          <div className="auth-field">
            <label htmlFor="password" className="field-l">
              Password
            </label>
            <div className="auth-input-wrap">
              <input
                id="password"
                className="auth-input has-toggle"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
                placeholder="At least 8 characters"
              />
              <button
                type="button"
                className="auth-toggle"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                tabIndex={-1}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <p className="auth-hint dim">
              8+ characters with uppercase, lowercase, a number, and a symbol.
            </p>
          </div>

          <button type="submit" className="btn auth-submit" disabled={loading}>
            {loading ? "Creating account…" : "Create account"}
          </button>
        </form>

        <p className="auth-switch">
          Already have an account?{" "}
          <Link href="/login" className="auth-link">
            Sign in
          </Link>
        </p>

        {/* Trust cue */}
        <div className="auth-trust">
          <span className="secure mono">
            <Lock size={12} strokeWidth={2.25} />
            End-to-end encrypted
          </span>
          <span className="muted auth-trust-note">
            Only you can view your records.
          </span>
        </div>
      </div>
    </div>
  );
}
