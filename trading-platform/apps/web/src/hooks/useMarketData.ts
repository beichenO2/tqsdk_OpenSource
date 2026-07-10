import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';

export function useQuotes() {
  return useQuery({
    queryKey: ['quotes'],
    queryFn: () => api.getQuotes(),
    refetchInterval: 3_000,
  });
}

export function usePnlHistory(days = 30) {
  return useQuery({
    queryKey: ['pnl-history', days],
    queryFn: () => api.getPnlHistory(days),
    staleTime: 30_000,
  });
}

export function useRiskAlerts() {
  return useQuery({
    queryKey: ['risk-alerts'],
    queryFn: () => api.getRiskAlerts(),
    refetchInterval: 10_000,
  });
}

export function useKlines(symbol: string | null, duration: number, limit = 200) {
  return useQuery({
    queryKey: ['klines', symbol, duration, limit],
    queryFn: () => api.getKlines(symbol!, duration, limit),
    enabled: !!symbol,
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}
