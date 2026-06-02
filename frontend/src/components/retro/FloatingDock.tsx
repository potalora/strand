"use client";

import { usePathname, useRouter } from "next/navigation";
import { Upload, Sparkles } from "lucide-react";
import { useUIStore } from "@/stores/useUIStore";

// Persistent hero actions — the two things every other screen supports.
// API → Upload opens the ingestion flow (POST /upload · /upload/unstructured);
//        Summarize opens the de-identified summary builder (POST /summary/build-prompt → /generate).
export function FloatingDock() {
  const pathname = usePathname();
  const router = useRouter();
  const detailOpen = useUIStore((s) => s.detailOpen);

  const uploadActive = pathname.startsWith("/upload");
  const summarizeActive = pathname.startsWith("/summaries");

  return (
    <div className="fab-dock" data-hidden={detailOpen ? "true" : "false"} aria-hidden={detailOpen}>
      <button
        className={"fab-btn" + (uploadActive ? " on" : "")}
        onClick={() => router.push("/upload")}
      >
        <Upload size={18} /> Upload
      </button>
      <button
        className={"fab-btn ghost" + (summarizeActive ? " on" : "")}
        onClick={() => router.push("/summaries")}
      >
        <Sparkles size={18} /> Summarize
      </button>
    </div>
  );
}
