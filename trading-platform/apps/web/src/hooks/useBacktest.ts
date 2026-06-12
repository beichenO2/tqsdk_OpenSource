import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';

export function useBacktestResults() {
  return useQuery({
    queryKey: ['backtest-results'],
    queryFn: () => api.getBacktestResults(),
  });
}

export function useCreateBacktest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Parameters<typeof api.createBacktest>[0]) => api.createBacktest(params),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['backtest-results'] });
    },
  });
}

export function useBacktestStrategyOptions() {
  return useQuery({
    queryKey: ['backtest-strategy-options'],
    queryFn: () => api.getBacktestStrategyOptions(),
    staleTime: 60_000,
  });
}

export function useMarketInstruments() {
  return useQuery({
    queryKey: ['market-instruments'],
    queryFn: () => api.getMarketInstruments(),
    staleTime: 60_000,
  });
}
