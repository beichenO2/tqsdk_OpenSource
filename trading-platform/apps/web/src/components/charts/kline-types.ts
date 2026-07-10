export const KLINE_PERIODS = [
  { label: '1m', duration: 60 },
  { label: '5m', duration: 300 },
  { label: '15m', duration: 900 },
  { label: '1h', duration: 3600 },
  { label: '1d', duration: 86400 },
] as const;

export type KlinePeriod = (typeof KLINE_PERIODS)[number]['label'];

export interface KLineChartProps {
  /** Contract symbol, e.g. KQ.m@SHFE.rb */
  symbol: string;
  /** Chart canvas height in px (excluding toolbar) */
  height?: number;
  /** Number of bars to fetch from API */
  limit?: number;
  className?: string;
  /** Unix timestamp (seconds) to highlight with a vertical marker */
  highlightTime?: number;
}
