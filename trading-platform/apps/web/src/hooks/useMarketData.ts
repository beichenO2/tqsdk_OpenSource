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
