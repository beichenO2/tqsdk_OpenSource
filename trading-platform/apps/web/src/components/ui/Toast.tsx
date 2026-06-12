import { useState, useCallback, createContext, useContext, useRef, useEffect, type ReactNode } from 'react';
import { X, CheckCircle2, AlertTriangle, Info, XCircle } from 'lucide-react';
import { cn } from '@/lib/cn';

type ToastVariant = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
}

interface ToastAPI {
  success: (msg: string) => void;
  error: (msg: string) => void;
  warning: (msg: string) => void;
  info: (msg: string) => void;
}

const ToastContext = createContext<ToastAPI>({
  success: () => {},
  error: () => {},
  warning: () => {},
  info: () => {},
});

export function useToast(): ToastAPI {
  return useContext(ToastContext);
}

const icons: Record<ToastVariant, React.ComponentType<{ className?: string }>> = {
  success: CheckCircle2,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

const variantStyles: Record<ToastVariant, string> = {
  success: 'border-profit/30 bg-profit/10 text-profit',
  error: 'border-loss/30 bg-loss/10 text-loss',
  warning: 'border-warning/30 bg-warning/10 text-warning',
  info: 'border-info/30 bg-info/10 text-info',
};

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  const Icon = icons[toast.variant];
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    timerRef.current = setTimeout(() => onDismiss(toast.id), 4000);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [toast.id, onDismiss]);

  return (
    <div
      role="alert"
      className={cn(
        'flex items-center gap-2.5 rounded-lg border px-4 py-3 shadow-lg backdrop-blur-sm',
        'animate-in slide-in-from-right fade-in duration-300',
        variantStyles[toast.variant],
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span className="text-sm font-medium text-text-primary flex-1">{toast.message}</span>
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        className="shrink-0 rounded p-0.5 text-text-muted hover:text-text-primary transition-colors"
        aria-label="关闭"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

let counter = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const add = useCallback((message: string, variant: ToastVariant) => {
    const id = `toast-${++counter}`;
    setToasts((prev) => [...prev.slice(-4), { id, message, variant }]);
  }, []);

  const api: ToastAPI = {
    success: useCallback((msg: string) => add(msg, 'success'), [add]),
    error: useCallback((msg: string) => add(msg, 'error'), [add]),
    warning: useCallback((msg: string) => add(msg, 'warning'), [add]),
    info: useCallback((msg: string) => add(msg, 'info'), [add]),
  };

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 w-80">
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
