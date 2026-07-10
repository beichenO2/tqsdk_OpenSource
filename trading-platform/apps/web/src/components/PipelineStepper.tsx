import { cn } from '@/lib/cn';

export interface PipelineStep {
  id: string;
  label: string;
  description?: string;
  status: string;
  done?: boolean;
}

interface PipelineStepperProps {
  steps: PipelineStep[];
  runFailed?: boolean;
  className?: string;
}

function stepColor(status: string, runFailed: boolean): string {
  if (status === 'done') return 'border-profit bg-profit/10 text-profit';
  if (status === 'active') {
    return runFailed
      ? 'border-loss bg-loss/10 text-loss'
      : 'border-brand bg-brand/10 text-brand';
  }
  return 'border-border bg-surface-tertiary text-text-muted';
}

function dotColor(status: string, runFailed: boolean): string {
  if (status === 'done') return 'bg-profit';
  if (status === 'active') return runFailed ? 'bg-loss' : 'bg-brand animate-pulse';
  return 'bg-text-muted/40';
}

export default function PipelineStepper({ steps, runFailed = false, className }: PipelineStepperProps) {
  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-start gap-0">
        {steps.map((step, i) => (
          <div key={step.id} className="flex flex-1 min-w-0 flex-col items-center">
            <div className="flex w-full items-center">
              {i > 0 && (
                <div
                  className={cn(
                    'h-0.5 flex-1',
                    step.status === 'done' || steps[i - 1]?.status === 'done'
                      ? 'bg-profit/50'
                      : 'bg-border',
                  )}
                />
              )}
              <div
                title={step.description}
                className={cn(
                  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-2 text-[10px] font-bold',
                  stepColor(step.status, runFailed),
                )}
              >
                <span className={cn('h-2 w-2 rounded-full', dotColor(step.status, runFailed))} />
              </div>
              {i < steps.length - 1 && (
                <div
                  className={cn(
                    'h-0.5 flex-1',
                    step.status === 'done' ? 'bg-profit/50' : 'bg-border',
                  )}
                />
              )}
            </div>
            <span
              className={cn(
                'mt-1.5 text-center text-[10px] font-mono leading-tight px-0.5',
                step.status === 'active' && !runFailed && 'text-brand font-semibold',
                step.status === 'done' && 'text-profit',
                step.status === 'active' && runFailed && 'text-loss font-semibold',
                step.status === 'pending' && 'text-text-muted',
              )}
            >
              {step.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
