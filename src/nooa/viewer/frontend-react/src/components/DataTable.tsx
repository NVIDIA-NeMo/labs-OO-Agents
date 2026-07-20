import { useCallback, useMemo, useState } from 'react';

export interface DataColumn<T> {
  key: string;
  label: string;
  header?: React.ReactNode;
  render?: (row: T) => React.ReactNode;
  className?: string;
  configurable?: boolean;
  sortable?: boolean;
}

export type SortDir = 'asc' | 'desc';

interface DataTableProps<T> {
  data: T[];
  columns: DataColumn<T>[];
  getKey: (row: T, index: number) => string;
  onRowClick?: (row: T) => void;
  total?: number;
  page?: number;
  pageSize?: number;
  onPageChange?: (page: number) => void;
  selectedIndex?: number | null;
  loading?: boolean;
  emptyMessage?: string;
  sortKey?: string | null;
  sortDir?: SortDir;
  onSort?: (key: string | null, dir: SortDir) => void;
  checkedKeys?: Set<string>;
  onCheckChange?: (key: string, checked: boolean) => void;
  onCheckAll?: (checked: boolean) => void;
}

function sortIndicator(dir: SortDir | null, active: boolean) {
  if (!active) return <span className="text-gray-700 ml-1">⇅</span>;
  return <span className="text-gray-300 ml-1">{dir === 'asc' ? '▲' : '▼'}</span>;
}

function cycleSort(
  currentKey: string | null,
  currentDir: SortDir,
  clickedKey: string,
): [string | null, SortDir] {
  if (currentKey !== clickedKey) return [clickedKey, 'asc'];
  if (currentDir === 'asc') return [clickedKey, 'desc'];
  return [null, 'asc'];
}

export function DataTable<T>({
  data,
  columns,
  getKey,
  onRowClick,
  total,
  page = 1,
  pageSize = 50,
  onPageChange,
  selectedIndex,
  loading = false,
  emptyMessage = 'No data found',
  sortKey: controlledSortKey,
  sortDir: controlledSortDir,
  onSort,
  checkedKeys,
  onCheckChange,
  onCheckAll,
}: DataTableProps<T>) {
  const hasCheckboxes = checkedKeys !== undefined && onCheckChange !== undefined;
  const controlled = onSort !== undefined;

  const [localSortKey, setLocalSortKey] = useState<string | null>(null);
  const [localSortDir, setLocalSortDir] = useState<SortDir>('asc');

  const activeSortKey = controlled ? (controlledSortKey ?? null) : localSortKey;
  const activeSortDir = controlled ? (controlledSortDir ?? 'asc') : localSortDir;

  const handleSort = useCallback(
    (key: string) => {
      const [nextKey, nextDir] = cycleSort(activeSortKey, activeSortDir, key);
      if (controlled) {
        onSort!(nextKey, nextDir);
      } else {
        setLocalSortKey(nextKey);
        setLocalSortDir(nextDir);
      }
    },
    [activeSortKey, activeSortDir, controlled, onSort],
  );

  const sortedData = useMemo(() => {
    if (controlled || !activeSortKey) return data;
    const copy = [...data];
    const dir = activeSortDir === 'asc' ? 1 : -1;
    copy.sort((a, b) => {
      const av = String((a as Record<string, unknown>)[activeSortKey] ?? '');
      const bv = String((b as Record<string, unknown>)[activeSortKey] ?? '');
      return av.localeCompare(bv) * dir;
    });
    return copy;
  }, [data, activeSortKey, activeSortDir, controlled]);

  const showPagination = total != null && total > pageSize && onPageChange;
  const totalPages = total != null ? Math.max(1, Math.ceil(total / pageSize)) : 1;

  const handlePrev = useCallback(() => {
    if (page > 1) onPageChange?.(page - 1);
  }, [page, onPageChange]);

  const handleNext = useCallback(() => {
    if (page < totalPages) onPageChange?.(page + 1);
  }, [page, totalPages, onPageChange]);

  return (
    <div>
      <div className="overflow-x-auto overflow-y-auto max-h-[calc(100vh-24rem)]">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-gray-900">
            <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase text-left">
              {hasCheckboxes && (
                <th className="w-8 px-2 py-2">
                  <input
                    type="checkbox"
                    checked={data.length > 0 && data.every((row, i) => checkedKeys!.has(getKey(row, i)))}
                    onChange={(e) => onCheckAll?.(e.target.checked)}
                    className="rounded border-gray-700 bg-gray-800 text-gray-500 focus:ring-0 focus:ring-offset-0 w-3.5 h-3.5 cursor-pointer accent-gray-500"
                  />
                </th>
              )}
              {columns.map((col) => {
                const isSortable = col.sortable !== false;
                const isActive = activeSortKey === col.key;
                return (
                  <th
                    key={col.key}
                    className={`px-3 py-2 font-medium ${col.className ?? ''} ${isSortable ? 'cursor-pointer select-none' : ''}`}
                    onClick={isSortable ? () => handleSort(col.key) : undefined}
                  >
                    <span className="inline-flex items-center gap-1">
                      {col.header ? (
                        <span onClick={(e) => e.stopPropagation()}>
                          {col.header}
                        </span>
                      ) : (
                        col.label
                      )}
                      {isSortable && sortIndicator(isActive ? activeSortDir : null, isActive)}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {loading && data.length === 0 ? (
              <tr>
                <td colSpan={columns.length + (hasCheckboxes ? 1 : 0)} className="text-center py-12 text-gray-500">
                  Loading...
                </td>
              </tr>
            ) : data.length === 0 ? (
              <tr>
                <td colSpan={columns.length + (hasCheckboxes ? 1 : 0)} className="text-center py-12 text-gray-500">
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              sortedData.map((row, idx) => {
                const rowKey = getKey(row, idx);
                return (
                  <tr
                    key={rowKey}
                    data-list-index={idx}
                    onClick={onRowClick ? () => onRowClick(row) : undefined}
                    className={`border-b border-gray-800/50 transition-colors ${
                      onRowClick ? 'cursor-pointer hover:bg-gray-800/50' : ''
                    } ${selectedIndex === idx ? 'bg-gray-800/80 ring-1 ring-gray-600' : ''}`}
                  >
                    {hasCheckboxes && (
                      <td className="w-8 px-2 py-2">
                        <input
                          type="checkbox"
                          checked={checkedKeys!.has(rowKey)}
                          onChange={(e) => {
                            e.stopPropagation();
                            onCheckChange!(rowKey, e.target.checked);
                          }}
                          onClick={(e) => e.stopPropagation()}
                          className="rounded border-gray-700 bg-gray-800 text-gray-500 focus:ring-0 focus:ring-offset-0 w-3.5 h-3.5 cursor-pointer accent-gray-500"
                        />
                      </td>
                    )}
                    {columns.map((col) => (
                      <td key={col.key} className={`px-3 py-2 ${col.className ?? ''}`}>
                        {col.render ? col.render(row) : ''}
                      </td>
                    ))}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {showPagination && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800">
          <span className="text-sm text-gray-400">
            {total} item{total !== 1 ? 's' : ''} &middot; Page {page} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={handlePrev}
              disabled={page <= 1}
              className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <button
              onClick={handleNext}
              disabled={page >= totalPages}
              className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
