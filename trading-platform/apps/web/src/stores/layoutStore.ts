import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface LayoutState {
  sidebarExpanded: boolean;
  toggleSidebar: () => void;
  setSidebarExpanded: (v: boolean) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      sidebarExpanded: false,
      toggleSidebar: () => set((s) => ({ sidebarExpanded: !s.sidebarExpanded })),
      setSidebarExpanded: (sidebarExpanded) => set({ sidebarExpanded }),
    }),
    { name: 'polar-layout' },
  ),
);
