import { forwardRef, type InputHTMLAttributes } from 'react';
import { Search, X } from 'lucide-react';
import { cn } from '@/lib/cn';

interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  onClear?: () => void;
}

const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  ({ className, value, onClear, ...props }, ref) => (
    <div className="relative">
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
      <input
        ref={ref}
        type="search"
        value={value}
        className={cn(
          'w-full rounded-lg border border-border bg-surface-tertiary pl-9 pr-8 py-2 text-sm text-text-primary',
          'placeholder:text-text-muted transition-colors',
          'focus:outline-none focus:border-brand focus:ring-1 focus:ring-focus-ring',
          className,
        )}
        {...props}
      />
      {value && onClear && (
        <button
          type="button"
          onClick={onClear}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-text-muted hover:text-text-primary transition-colors"
          aria-label="清空搜索"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  ),
);

SearchInput.displayName = 'SearchInput';
export { SearchInput };
