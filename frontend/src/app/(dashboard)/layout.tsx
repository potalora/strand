"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { RetroNav } from "@/components/retro/RetroNav";
import { FloatingDock } from "@/components/retro/FloatingDock";
import { useAuthStore, useHasHydrated } from "@/stores/useAuthStore";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();
  const hydrated = useHasHydrated();

  useEffect(() => {
    if (hydrated && !isAuthenticated) {
      router.replace("/login");
    }
  }, [isAuthenticated, hydrated, router]);

  if (!hydrated || !isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen">
      <RetroNav />
      <main className="main-narrow">{children}</main>
      <FloatingDock />
    </div>
  );
}
