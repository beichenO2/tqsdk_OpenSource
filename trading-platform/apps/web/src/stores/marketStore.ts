import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Market = 'futures' | 'crypto' | 'all';

interface MarketState {
  market: Market;
  setMarket: (m: Market) => void;
}

export const useMarketStore = create<MarketState>()(
  persist(
    (set) => ({
      market: 'all',
      setMarket: (market) => set({ market }),
    }),
    { name: 'polar-market' },
  ),
);
