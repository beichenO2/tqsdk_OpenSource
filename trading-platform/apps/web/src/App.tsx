import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ToastProvider } from '@/components/ui/Toast';
import { ConfirmProvider } from '@/components/ui/ConfirmDialog';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { Skeleton } from '@/components/ui/Skeleton';
import Layout from '@/components/Layout';

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
const CryptoDashboard = lazy(() => import('@/pages/crypto/Dashboard'));
const CryptoTrading = lazy(() => import('@/pages/crypto/Trading'));
const CryptoBacktest = lazy(() => import('@/pages/crypto/Backtest'));
const CryptoPortfolio = lazy(() => import('@/pages/crypto/Portfolio'));
const LiveTrading = lazy(() => import('@/pages/LiveTrading'));

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

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <ConfirmProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<Layout />}>
                <Route
                  index
                  element={
                    <ErrorBoundary>
                      <Suspense fallback={<PageFallback />}>
                        <Dashboard />
                      </Suspense>
                    </ErrorBoundary>
                  }
                />
                <Route path="trading" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><Trading /></Suspense></ErrorBoundary>} />
                <Route path="strategies" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><Strategies /></Suspense></ErrorBoundary>} />
                <Route path="backtest" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><Backtest /></Suspense></ErrorBoundary>} />
                <Route path="backtest/compare" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><BacktestCompare /></Suspense></ErrorBoundary>} />
                <Route path="risk" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><Risk /></Suspense></ErrorBoundary>} />
                <Route path="strategy/:id" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><StrategyDetail /></Suspense></ErrorBoundary>} />
                <Route path="settings" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><SettingsPage /></Suspense></ErrorBoundary>} />
                <Route path="paper-trading" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><PaperTrading /></Suspense></ErrorBoundary>} />
                <Route path="live-trading" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><LiveTrading /></Suspense></ErrorBoundary>} />
                <Route path="crypto" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><CryptoDashboard /></Suspense></ErrorBoundary>} />
                <Route path="crypto/trading" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><CryptoTrading /></Suspense></ErrorBoundary>} />
                <Route path="crypto/backtest" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><CryptoBacktest /></Suspense></ErrorBoundary>} />
                <Route path="crypto/portfolio" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><CryptoPortfolio /></Suspense></ErrorBoundary>} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ConfirmProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}
