import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useCallback, useMemo } from 'react';
import { api } from '@/services/api';
import { useWebSocket, type WsEvent } from './useWebSocket';

const LIVE_CONFIRM = 'I_UNDERSTAND_LIVE_RISK';

export function useLiveTradingStatus() {
  return useQuery({
    queryKey: ['live-trading-status'],
    queryFn: () => api.getLiveTradingStatus(),
    refetchInterval: 5_000,
  });
}

export function useLiveStrategies() {
  return useQuery({
    queryKey: ['live-strategies'],
    queryFn: () => api.getLiveStrategies(),
    refetchInterval: 5_000,
  });
}

export function useLiveLeaderboard(market?: string) {
  return useQuery({
    queryKey: ['live-leaderboard', market],
    queryFn: () => api.getLiveLeaderboard(market, 50),
    refetchInterval: 10_000,
  });
}

export function useLiveRiskStatus() {
  return useQuery({
    queryKey: ['live-risk-status'],
    queryFn: () => api.getLiveRiskStatus(),
    refetchInterval: 10_000,
  });
}

export function useStartLiveTrading() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Record<string, unknown>) => {
      const mode = String(params.mode || 'paper');
      return api.startLiveTrading(params, mode === 'live' ? LIVE_CONFIRM : undefined);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-trading-status'] });
      qc.invalidateQueries({ queryKey: ['live-strategies'] });
    },
  });
}

export function useStopLiveTrading() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.stopLiveTrading(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-trading-status'] });
      qc.invalidateQueries({ queryKey: ['live-strategies'] });
    },
  });
}

export function useSwitchMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (mode: string) =>
      api.switchMode(mode, mode === 'live' ? LIVE_CONFIRM : undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-trading-status'] });
    },
  });
}

export function useToggleStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (accountId: number) => api.toggleLiveStrategy(accountId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-strategies'] });
    },
  });
}

export function usePlaceLiveOrder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: {
      order: {
        symbol: string;
        exchange?: string;
        direction: string;
        offset?: string;
        price: number | string;
        volume: number;
        strategy_id?: string;
      };
      live?: boolean;
    }) => api.placeLiveOrder(args.order, args.live ? LIVE_CONFIRM : undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-trading-status'] });
      qc.invalidateQueries({ queryKey: ['live-risk-status'] });
    },
  });
}

export function useRiskProbe() {
  return useMutation({
    mutationFn: (params: {
      symbol: string;
      exchange?: string;
      direction?: string;
      offset?: string;
      price?: number | string;
      volume?: number;
    }) => api.riskProbe(params),
  });
}

export function useLiveEvents() {
  const [events, setEvents] = useState<WsEvent[]>([]);

  const wsUrl = useMemo(() => {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/live`;
  }, []);

  const onEvent = useCallback((event: WsEvent) => {
    setEvents((prev) => [...prev.slice(-199), event]);
  }, []);

  const { connected, send } = useWebSocket({
    url: wsUrl,
    channels: ['position_update', 'trade_fill', 'account_update', 'strategy_status', 'signal_generated', 'risk_alert'],
    onEvent,
  });

  return { events, connected, send };
}

export { LIVE_CONFIRM };
