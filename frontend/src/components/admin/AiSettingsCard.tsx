"use client";

import type { ReactNode } from "react";
import { ValidatedLocalPackCard } from "@/components/admin/ValidatedLocalPackCard";

export function AiSettingsCard({ children }: { children: ReactNode }) {
  return (
    <section aria-label="AI settings">
      <ValidatedLocalPackCard />
      {children}
    </section>
  );
}
