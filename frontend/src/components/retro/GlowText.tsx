"use client";

import { cn } from "@/lib/utils";

interface GlowTextProps {
  as?: "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "span" | "p";
  glow?: boolean;
  className?: string;
  children: React.ReactNode;
}

// Editorial display type: an italic serif (Source Serif 4) reserved for headings.
const sizeMap: Record<string, string> = {
  h1: "text-[34px] leading-tight",
  h2: "text-[26px] leading-tight",
  h3: "text-[20px] leading-snug",
  h4: "text-[17px] leading-snug",
  h5: "text-[15px]",
  h6: "text-[13px]",
  span: "",
  p: "",
};

const DISPLAY_TAGS = new Set(["h1", "h2", "h3", "h4"]);

export function GlowText({ as: Tag = "h1", className, children }: GlowTextProps) {
  const isDisplay = DISPLAY_TAGS.has(Tag);
  return (
    <Tag
      className={cn(sizeMap[Tag], isDisplay && "display", className)}
      style={{ color: "var(--text)" }}
    >
      {children}
    </Tag>
  );
}
