import { lazy, Suspense } from 'react';
import { BrowserRouter, Navigate, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ToastProvider } from '@/components/ui/Toast';
import { ConfirmProvider } from '@/components/ui/ConfirmDialog';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { Skeleton } from '@/components/ui/Skeleton';
import Layout from '@/components/Layout';
import DataPageReal from '@/pages/DataPage';
import SkillsPageReal from '@/pages/SkillsPage';
import FactorsPageReal from '@/pages/FactorsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

const Dashboard = lazy(() => import('@/pages/Dashboard'));
const Trading = lazy(() => import('@/pages/Trading'));
const Strategies = lazy(() => import('@/pages/Strategies'));
const Backtest = lazy(() => import('@/pages/Backtest'));
const BacktestCompare = lazy(() => import('@/pages/BacktestCompare'));
const Risk = lazy(() => import('@/pages/Risk'));
const SettingsPage = lazy(() => import('@/pages/SettingsPage'));
const PaperTrading = lazy(() => import('@/pages/PaperTrading'));
const StrategyDetail = lazy(() => import('@/pages/StrategyDetail'));
const LiveTrading = lazy(() => import('@/pages/LiveTrading'));
const ResearchRuns = lazy(() => import('@/pages/ResearchRuns'));
const EventsPage = lazy(() => import('@/pages/EventsPage'));
const HealthPage = lazy(() => import('@/pages/HealthPage'));
const OptimizerPageReal = lazy(() => import('@/pages/OptimizerPage'));
const DeployPage = lazy(() => import('@/pages/DeployPage'));
const MLPage = lazy(() => import('@/pages/MLPage'));
const ExplainPage = lazy(() => import('@/pages/ExplainPage'));

function PageFallback() {
  return (
    <div className="p-6 space-y-6">
      <Skeleton className="h-7 w-32" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="bg-surface-secondary border border-border rounded-xl p-4 space-y-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-7 w-24" />
            <Skeleton className="h-3 w-20" />
          </div>
        ))}
      </div>
      <Skeleton className="h-64 w-full rounded-xl" />
    </div>
  );
}

function Wrap({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageFallback />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <ConfirmProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Wrap><Dashboard /></Wrap>} />

                {/* Research workspace */}
                <Route path="research" element={<Wrap><ResearchRuns /></Wrap>} />
                <Route path="research/factors" element={<Wrap><FactorsPageReal /></Wrap>} />
                <Route path="research/strategies" element={<Wrap><Strategies /></Wrap>} />
                <Route path="research/backtest" element={<Wrap><Backtest /></Wrap>} />
                <Route path="research/backtest/compare" element={<Wrap><BacktestCompare /></Wrap>} />
                <Route path="research/optimizer" element={<Wrap><OptimizerPageReal /></Wrap>} />
                <Route path="research/deploy" element={<Wrap><DeployPage /></Wrap>} />
                <Route path="research/ml" element={<Wrap><MLPage /></Wrap>} />

                {/* Trading workspace */}
                <Route path="trading/paper" element={<Wrap><PaperTrading /></Wrap>} />
                <Route path="trading/live" element={<Wrap><LiveTrading /></Wrap>} />
                <Route path="trading/manual" element={<Wrap><Trading /></Wrap>} />

                {/* Monitor workspace */}
                <Route path="monitor/risk" element={<Wrap><Risk /></Wrap>} />
                <Route path="monitor/events" element={<Wrap><EventsPage /></Wrap>} />
                <Route path="monitor/explain" element={<Wrap><ExplainPage /></Wrap>} />
                <Route path="monitor/health" element={<Wrap><HealthPage /></Wrap>} />

                {/* Platform workspace */}
                <Route path="platform/data" element={<Wrap><DataPageReal /></Wrap>} />
                <Route path="platform/skills" element={<Wrap><SkillsPageReal /></Wrap>} />
                <Route path="platform/settings" element={<Wrap><SettingsPage /></Wrap>} />

                {/* Legacy redirects — keep old bookmarks working */}
                <Route path="trading" element={<Navigate to="/trading/manual" replace />} />
                <Route path="strategies" element={<Navigate to="/research/strategies" replace />} />
                <Route path="backtest" element={<Navigate to="/research/backtest" replace />} />
                <Route path="backtest/compare" element={<Navigate to="/research/backtest/compare" replace />} />
                <Route path="risk" element={<Navigate to="/monitor/risk" replace />} />
                <Route path="paper-trading" element={<Navigate to="/trading/paper" replace />} />
                <Route path="live-trading" element={<Navigate to="/trading/live" replace />} />
                <Route path="settings" element={<Navigate to="/platform/settings" replace />} />
                <Route path="strategy/:id" element={<Wrap><StrategyDetail /></Wrap>} />

                {/* Crypto legacy paths → trading workspace */}
                <Route path="crypto" element={<Navigate to="/trading/paper" replace />} />
                <Route path="crypto/*" element={<Navigate to="/trading/paper" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ConfirmProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}
