import { useCallback, useRef, type ReactNode, type KeyboardEvent } from 'react';
import { cn } from '@/lib/cn';

interface Tab {
  id: string;
  label: string;
  icon?: ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, active, onChange, className }: TabsProps) {
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent, idx: number) => {
      let next = idx;
      if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = tabs.length - 1;
      else return;

      e.preventDefault();
      tabRefs.current[next]?.focus();
      onChange(tabs[next]!.id);
    },
    [tabs, onChange],
  );

  return (
    <div role="tablist" className={cn('flex gap-1 border-b border-border', className)}>
      {tabs.map((tab, i) => (
        <button
          key={tab.id}
          ref={(el) => { tabRefs.current[i] = el; }}
          role="tab"
          type="button"
          aria-selected={active === tab.id}
          tabIndex={active === tab.id ? 0 : -1}
          onClick={() => onChange(tab.id)}
          onKeyDown={(e) => handleKeyDown(e, i)}
          className={cn(
            'inline-flex items-center gap-1.5 border-b-2 px-3 py-1.5 text-[12.5px] font-mono transition-colors',
            active === tab.id
              ? 'border-brand text-brand'
              : 'border-transparent text-text-muted hover:text-text-secondary',
          )}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  );
}
