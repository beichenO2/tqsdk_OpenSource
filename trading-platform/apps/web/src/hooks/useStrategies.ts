import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';

export function useStrategies() {
  return useQuery({
    queryKey: ['strategies'],
    queryFn: () => api.getStrategies(),
    refetchInterval: 10_000,
  });
}

export function useToggleStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, running }: { id: string; running: boolean }) =>
      running ? api.stopStrategy(id) : api.startStrategy(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function usePauseAllStrategies() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.pauseAllStrategies(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}
