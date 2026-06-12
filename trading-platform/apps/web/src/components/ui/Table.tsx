import { useCallback, type ReactNode, type KeyboardEvent } from 'react';
import { cn } from '@/lib/cn';
import { ChevronUp, ChevronDown } from 'lucide-react';

export interface Column<T> {
  key: string;
  header: string;
  align?: 'left' | 'right' | 'center';
  sortable?: boolean;
  render: (row: T, index: number) => ReactNode;
  className?: string;
}

interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  selectedKey?: string;
  emptyText?: string;
  sortKey?: string;
  sortDir?: 'asc' | 'desc';
  onSort?: (key: string) => void;
  compact?: boolean;
}

export function Table<T>({
  columns,
  data,
  rowKey,
  onRowClick,
  selectedKey,
  emptyText = '暂无数据',
  sortKey,
  sortDir,
  onSort,
  compact,
}: TableProps<T>) {
  const px = compact ? 'px-3' : 'px-4';
  const py = compact ? 'py-2' : 'py-2.5';

  const handleRowKeyDown = useCallback(
    (e: KeyboardEvent, row: T) => {
      if (onRowClick && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        onRowClick(row);
      }
    },
    [onRowClick],
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-text-muted">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  'font-medium',
                  px,
                  py,
                  col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left',
                  col.sortable && 'cursor-pointer select-none hover:text-text-secondary transition-colors',
                  col.className,
                )}
                onClick={col.sortable && onSort ? () => onSort(col.key) : undefined}
              >
                <span className="inline-flex items-center gap-1">
                  {col.header}
                  {col.sortable && sortKey === col.key && (
                    sortDir === 'asc'
                      ? <ChevronUp className="h-3 w-3" />
                      : <ChevronDown className="h-3 w-3" />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => {
            const key = rowKey(row);
            return (
              <tr
                key={key}
                tabIndex={onRowClick ? 0 : undefined}
                role={onRowClick ? 'button' : undefined}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                onKeyDown={onRowClick ? (e) => handleRowKeyDown(e, row) : undefined}
                className={cn(
                  'border-b border-border/50 transition-colors',
                  onRowClick && 'cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring',
                  selectedKey === key ? 'bg-brand/10' : 'hover:bg-surface-tertiary/50',
                  i % 2 === 1 && 'bg-surface-tertiary/20',
                )}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={cn(
                      px,
                      py,
                      col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left',
                      col.className,
                    )}
                  >
                    {col.render(row, i)}
                  </td>
                ))}
              </tr>
            );
          })}
          {data.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-4 py-12 text-center text-text-muted text-sm">
                {emptyText}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
