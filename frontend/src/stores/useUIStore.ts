"use client";

import { create } from "zustand";

interface UIState {
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  /** True while a record detail sheet (or other full-height overlay) is open,
   *  so the floating action dock can tuck away to avoid collisions. */
  detailOpen: boolean;
  setDetailOpen: (open: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  detailOpen: false,
  setDetailOpen: (open) => set({ detailOpen: open }),
}));
