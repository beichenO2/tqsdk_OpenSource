import { useState, useCallback, createContext, useContext, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Dialog } from './Dialog';
import { Button } from './Button';

interface ConfirmOptions {
  title: string;
  description: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'destructive' | 'warning' | 'default';
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>;

const ConfirmContext = createContext<ConfirmFn>(() => Promise.resolve(false));

export function useConfirm(): ConfirmFn {
  return useContext(ConfirmContext);
}

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<(ConfirmOptions & { resolve: (v: boolean) => void }) | null>(null);

  const confirm: ConfirmFn = useCallback(
    (opts) =>
      new Promise<boolean>((resolve) => {
        setState({ ...opts, resolve });
      }),
    [],
  );

  const handleClose = (result: boolean) => {
    state?.resolve(result);
    setState(null);
  };

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      {state && (
        <Dialog open onClose={() => handleClose(false)} title={state.title}>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              {state.variant === 'destructive' && (
                <div className="mt-0.5 rounded-full bg-loss/15 p-2">
                  <AlertTriangle className="h-5 w-5 text-loss" />
                </div>
              )}
              {state.variant === 'warning' && (
                <div className="mt-0.5 rounded-full bg-warning/15 p-2">
                  <AlertTriangle className="h-5 w-5 text-warning" />
                </div>
              )}
              <p className="text-sm text-text-secondary leading-relaxed">{state.description}</p>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="secondary" onClick={() => handleClose(false)}>
                {state.cancelText ?? '取消'}
              </Button>
              <Button
                variant={state.variant === 'destructive' ? 'destructive' : 'primary'}
                onClick={() => handleClose(true)}
              >
                {state.confirmText ?? '确认'}
              </Button>
            </div>
          </div>
        </Dialog>
      )}
    </ConfirmContext.Provider>
  );
}
