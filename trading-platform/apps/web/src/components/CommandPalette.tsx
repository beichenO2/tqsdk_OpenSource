import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { cn } from '@/lib/cn';
import { allCommandItems } from '@/nav/workspaces';

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);

  const items = useMemo(() => {
    const all = allCommandItems();
    const q = query.trim().toLowerCase();
    if (!q) return all;
    return all.filter((item) =>
      item.keywords.some((k) => k.toLowerCase().includes(q)) ||
      item.label.toLowerCase().includes(q) ||
      item.group.toLowerCase().includes(q),
    );
  }, [query]);

  useEffect(() => {
    if (open) {
      setQuery('');
      setActive(0);
      const t = setTimeout(() => inputRef.current?.focus(), 20);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  if (!open) return null;

  const go = (to: string) => {
    navigate(to);
    onClose();
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, Math.max(items.length - 1, 0)));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && items[active]) {
      e.preventDefault();
      go(items[active].to);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center bg-overlay/80 pt-[12vh] px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl rounded-xl border border-border bg-surface-secondary shadow-modal overflow-hidden"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <Search className="h-4 w-4 text-text-muted shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="跳转页面、搜索策略入口…（Esc 关闭）"
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted outline-none"
          />
          <kbd className="hidden sm:inline text-[10px] text-text-muted border border-border rounded px-1.5 py-0.5">
            ⌘K
          </kbd>
        </div>
        <ul className="max-h-[50vh] overflow-auto py-2">
          {items.length === 0 && (
            <li className="px-4 py-6 text-center text-sm text-text-muted">无匹配项</li>
          )}
          {items.map((item, i) => (
            <li key={`${item.group}-${item.to}`}>
              <button
                type="button"
                onClick={() => go(item.to)}
                onMouseEnter={() => setActive(i)}
                className={cn(
                  'w-full flex items-center justify-between px-4 py-2.5 text-left text-sm transition-colors',
                  i === active ? 'bg-brand/10 text-brand' : 'text-text-primary hover:bg-surface-tertiary',
                )}
              >
                <span className="font-medium">{item.label}</span>
                <span className="text-xs text-text-muted">{item.group}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
