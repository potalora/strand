"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";

/**
 * Settings was consolidated into the Admin → System pane (Account + Preferences +
 * Session live there now). This route just redirects so existing links and
 * bookmarks to /settings don't 404.
 */
export default function SettingsRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/admin?tab=sys");
  }, [router]);

  return <RetroLoadingState text="Opening system settings" />;
}
