"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { Sun, Moon, Search, Bell, Activity } from "lucide-react";
import { useAuthStore } from "@/stores/useAuthStore";
import { api } from "@/lib/api";
import type { UserResponse } from "@/types/api";

interface NavItem {
  label: string;
  href: string;
}

// IA modeled on the design: Timeline is the patient-facing narrative; Records,
// Uploads, Duplicates and System live inside Admin. Upload + Summarize are the
// two hero actions, surfaced in the floating dock.
const NAV_ITEMS: NavItem[] = [
  { label: "Overview", href: "/" },
  { label: "Timeline", href: "/timeline" },
  { label: "Summaries", href: "/summaries" },
  { label: "Admin", href: "/admin" },
];

const ADMIN_OWNED = ["/records", "/dedup", "/upload", "/settings"];

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  if (href === "/admin") {
    return pathname.startsWith("/admin") || ADMIN_OWNED.some((p) => pathname.startsWith(p));
  }
  return pathname.startsWith(href);
}

function initialsFrom(user: UserResponse | null): string {
  if (!user) return "··";
  const name = user.display_name?.trim();
  if (name) {
    const parts = name.split(/\s+/);
    return ((parts[0]?.[0] ?? "") + (parts[1]?.[0] ?? "")).toUpperCase() || name.slice(0, 2).toUpperCase();
  }
  return user.email.slice(0, 2).toUpperCase();
}

export function RetroNav() {
  const pathname = usePathname();
  const router = useRouter();
  const { accessToken, clearTokens } = useAuthStore();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [user, setUser] = useState<UserResponse | null>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    api
      .get<UserResponse>("/auth/me")
      .then(setUser)
      .catch(() => {});
  }, []);

  const handleLogout = async () => {
    try {
      await api.post("/auth/logout", undefined, accessToken ?? undefined);
    } catch {
      // Logout even if the server call fails
    }
    clearTokens();
    router.push("/login");
  };

  return (
    <nav
      className="sticky top-0 z-30 border-b"
      style={{ background: "var(--card)", borderColor: "var(--border)", boxShadow: "var(--shadow)" }}
    >
      <div className="topnav">
        <Link href="/" className="top-brand" style={{ textDecoration: "none" }}>
          <span className="brand-mark">
            <Activity size={19} strokeWidth={2} />
          </span>
          <span className="brand-name">MedTimeline</span>
        </Link>

        <div className="top-links">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="top-link"
              aria-current={isActive(pathname, item.href) ? "page" : undefined}
            >
              {item.label}
            </Link>
          ))}
        </div>

        <span className="nav-spring" />

        <Link href="/records" className="icon-btn" aria-label="Search records">
          <Search size={17} />
        </Link>
        {mounted && (
          <button
            className="icon-btn"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle light/dark"
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        )}
        <Link href="/settings" className="icon-btn" aria-label="Settings &amp; notifications">
          <Bell size={17} />
        </Link>
        <button
          onClick={handleLogout}
          className="user-av"
          title="Sign out"
          aria-label="Sign out"
          style={{ cursor: "pointer" }}
        >
          {initialsFrom(user)}
        </button>
      </div>
    </nav>
  );
}
