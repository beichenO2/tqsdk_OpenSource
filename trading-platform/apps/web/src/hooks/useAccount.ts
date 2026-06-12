import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';

export function useAccount() {
  return useQuery({
    queryKey: ['account'],
    queryFn: () => api.getAccount(),
    refetchInterval: 5_000,
  });
}
