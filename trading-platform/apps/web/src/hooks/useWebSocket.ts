import { useEffect, useRef, useState, useCallback } from 'react';

export interface WsEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

interface UseWebSocketOptions {
  url: string;
  channels?: string[];
  onEvent?: (event: WsEvent) => void;
  reconnectInterval?: number;
  enabled?: boolean;
}

export function useWebSocket({
  url,
  channels = [],
  onEvent,
  reconnectInterval = 3000,
  enabled = true,
}: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (!enabled) return;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        for (const ch of channels) {
          ws.send(JSON.stringify({ action: 'subscribe', channel: ch }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WsEvent;
          setLastEvent(data);
          onEvent?.(data);
        } catch { /* ignore non-JSON */ }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (enabled) {
          reconnectTimer.current = setTimeout(connect, reconnectInterval);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      if (enabled) {
        reconnectTimer.current = setTimeout(connect, reconnectInterval);
      }
    }
  }, [url, channels, onEvent, reconnectInterval, enabled]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, lastEvent, send };
}
