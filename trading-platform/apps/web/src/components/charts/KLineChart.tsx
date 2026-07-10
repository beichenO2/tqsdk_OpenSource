import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  ColorType,
  PriceScaleMode,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { AlertTriangle, BarChart3, Loader2 } from 'lucide-react';
import { useKlines } from '@/hooks/useMarketData';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/cn';
import type { KlineBar } from '@/types';
import { KLINE_PERIODS, type KLineChartProps } from './kline-types';

function readThemeColors() {
  const style = getComputedStyle(document.documentElement);
  const pick = (name: string, fallback: string) =>
    style.getPropertyValue(name).trim() || fallback;
  return {
    text: pick('--muted-foreground', 'oklch(0.62 0.02 245)'),
    grid: pick('--border', 'oklch(0.27 0.02 250)'),
    up: pick('--profit', 'oklch(0.74 0.17 152)'),
    down: pick('--loss', 'oklch(0.65 0.21 20)'),
  };
}

function toCandleData(bar: KlineBar) {
  return {
    time: bar.time as UTCTimestamp,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
  };
}

function toVolumeData(bar: KlineBar, upColor: string, downColor: string) {
  return {
    time: bar.time as UTCTimestamp,
    value: bar.volume,
    color: bar.close >= bar.open ? upColor : downColor,
  };
}

function applySeriesData(
  candle: ISeriesApi<'Candlestick'> | null,
  volume: ISeriesApi<'Histogram'> | null,
  bars: KlineBar[],
  chart: IChartApi | null,
) {
  if (!candle || !volume) return;
  if (bars.length === 0) {
    candle.setData([]);
    volume.setData([]);
    return;
  }
  const colors = readThemeColors();
  candle.setData(bars.map(toCandleData));
  volume.setData(bars.map((b) => toVolumeData(b, colors.up, colors.down)));
  chart?.timeScale().fitContent();
}

export default function KLineChart({
  symbol,
  height = 360,
  limit = 200,
  className,
  highlightTime,
}: KLineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  const [duration, setDuration] = useState<number>(KLINE_PERIODS[1].duration);
  const [logScale, setLogScale] = useState(true);

  const { data: bars = [], isLoading, isError, error, isFetching } = useKlines(symbol, duration, limit);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const colors = readThemeColors();
    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: colors.text,
        fontFamily: 'var(--font-mono)',
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      rightPriceScale: {
        mode: logScale ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
        borderColor: colors.grid,
      },
      timeScale: {
        borderColor: colors.grid,
        timeVisible: true,
        secondsVisible: duration < 3600,
      },
      crosshair: { mode: 0 },
    });

    const candleSeries = chart.addSeries(
      CandlestickSeries,
      {
        upColor: colors.up,
        downColor: colors.down,
        borderVisible: false,
        wickUpColor: colors.up,
        wickDownColor: colors.down,
      },
      0,
    );

    chart.addPane();
    const volumeSeries = chart.addSeries(
      HistogramSeries,
      {
        priceFormat: { type: 'volume' },
      },
      1,
    );

    chartRef.current = chart;
    candleRef.current = candleSeries;
    volumeRef.current = volumeSeries;
    markersRef.current = createSeriesMarkers(candleSeries);

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
      markersRef.current = null;
    };
  }, [height, duration, logScale]);

  useEffect(() => {
    applySeriesData(candleRef.current, volumeRef.current, bars, chartRef.current);
    if (markersRef.current) {
      if (highlightTime) {
        markersRef.current.setMarkers([
          {
            time: highlightTime as UTCTimestamp,
            position: 'aboveBar',
            color: '#3b82f6',
            shape: 'arrowDown',
            text: '决策',
          },
        ]);
      } else {
        markersRef.current.setMarkers([]);
      }
    }
  }, [bars, highlightTime]);

  const periodLabel = KLINE_PERIODS.find((p) => p.duration === duration)?.label ?? `${duration}s`;

  return (
    <div className={cn('rounded-xl border border-border bg-card overflow-hidden', className)}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-border">
        <div className="flex items-center gap-2 min-w-0">
          <BarChart3 className="w-4 h-4 text-brand shrink-0" />
          <span className="text-sm font-medium text-text-primary font-mono truncate">{symbol}</span>
          {isFetching && !isLoading && (
            <Loader2 className="w-3 h-3 text-text-muted animate-spin shrink-0" />
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="flex items-center gap-0.5 rounded-lg border border-border p-0.5">
            {KLINE_PERIODS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setDuration(p.duration)}
                className={cn(
                  'rounded-md px-2 py-0.5 text-xs font-mono transition-colors',
                  duration === p.duration
                    ? 'bg-brand/15 text-brand'
                    : 'text-text-muted hover:text-text-primary',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Button
            variant={logScale ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setLogScale((v) => !v)}
          >
            {logScale ? '对数' : '线性'}
          </Button>
        </div>
      </div>

      <div className="relative" style={{ height }}>
        {isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-card/80">
            <div className="flex items-center gap-2 text-sm text-text-muted">
              <Loader2 className="w-4 h-4 animate-spin" />
              加载 K 线…
            </div>
          </div>
        )}

        {isError && !isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-card/90">
            <div className="flex items-center gap-2 text-sm text-loss px-4 text-center">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error instanceof Error ? error.message : 'K 线加载失败'}</span>
            </div>
          </div>
        )}

        {!isLoading && !isError && bars.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <p className="text-sm text-text-muted">暂无 {periodLabel} K 线数据</p>
          </div>
        )}

        <div ref={containerRef} className="w-full h-full" />
      </div>
    </div>
  );
}
