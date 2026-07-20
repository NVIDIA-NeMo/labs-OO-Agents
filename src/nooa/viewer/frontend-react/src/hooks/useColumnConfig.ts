import { useState, useMemo, useCallback } from 'react';
import type { DataColumn } from '@/components/DataTable';

interface StoredConfig {
  hidden: string[];
  order: string[];
}

export interface ColumnConfigEntry {
  key: string;
  label: string;
  visible: boolean;
}

export interface ColumnConfig<T> {
  visibleColumns: DataColumn<T>[];
  configEntries: ColumnConfigEntry[];
  hiddenCount: number;
  toggleColumn: (key: string) => void;
  moveColumn: (key: string, direction: 'up' | 'down') => void;
  reorderColumn: (fromKey: string, toKey: string) => void;
  resetConfig: () => void;
}

function readConfig(storageKey: string): StoredConfig | null {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && Array.isArray(parsed.hidden) && Array.isArray(parsed.order)) {
      return parsed as StoredConfig;
    }
  } catch {}
  return null;
}

function writeConfig(storageKey: string, config: StoredConfig) {
  localStorage.setItem(storageKey, JSON.stringify(config));
}

export function useColumnConfig<T>(
  storageKey: string | undefined,
  columns: DataColumn<T>[],
): ColumnConfig<T> {
  const [revision, setRevision] = useState(0);

  const configurableKeys = useMemo(
    () => columns.filter((c) => c.configurable !== false).map((c) => c.key),
    [columns],
  );

  const stored = useMemo(() => {
    if (!storageKey) return null;
    void revision;
    return readConfig(storageKey);
  }, [storageKey, revision]);

  const hiddenSet = useMemo(() => {
    if (!stored) return new Set<string>();
    const validKeys = new Set(configurableKeys);
    return new Set(stored.hidden.filter((k) => validKeys.has(k)));
  }, [stored, configurableKeys]);

  const orderedConfigurableKeys = useMemo(() => {
    if (!stored || stored.order.length === 0) return configurableKeys;
    const keySet = new Set(configurableKeys);
    const ordered: string[] = [];
    for (const k of stored.order) {
      if (keySet.has(k)) {
        ordered.push(k);
        keySet.delete(k);
      }
    }
    for (const k of configurableKeys) {
      if (keySet.has(k)) ordered.push(k);
    }
    return ordered;
  }, [stored, configurableKeys]);

  const visibleColumns = useMemo(() => {
    if (!storageKey) return columns;

    const colByKey = new Map(columns.map((c) => [c.key, c]));
    const result: DataColumn<T>[] = [];
    let configurableIdx = 0;
    const configurableOrdered = orderedConfigurableKeys.filter(
      (k) => !hiddenSet.has(k),
    );

    for (const col of columns) {
      if (col.configurable === false) {
        result.push(col);
      } else {
        while (configurableIdx < configurableOrdered.length) {
          const k = configurableOrdered[configurableIdx];
          configurableIdx++;
          const c = colByKey.get(k);
          if (c) {
            result.push(c);
            break;
          }
        }
      }
    }
    while (configurableIdx < configurableOrdered.length) {
      const k = configurableOrdered[configurableIdx];
      configurableIdx++;
      const c = colByKey.get(k);
      if (c) result.push(c);
    }

    return result;
  }, [storageKey, columns, orderedConfigurableKeys, hiddenSet]);

  const configEntries = useMemo((): ColumnConfigEntry[] => {
    const colByKey = new Map(columns.map((c) => [c.key, c]));
    return orderedConfigurableKeys.map((key) => ({
      key,
      label: colByKey.get(key)?.label || key,
      visible: !hiddenSet.has(key),
    }));
  }, [orderedConfigurableKeys, hiddenSet, columns]);

  const hiddenCount = hiddenSet.size;

  const save = useCallback(
    (updater: (prev: StoredConfig) => StoredConfig) => {
      if (!storageKey) return;
      const prev = readConfig(storageKey) || { hidden: [], order: [] };
      const next = updater(prev);
      writeConfig(storageKey, next);
      setRevision((r) => r + 1);
    },
    [storageKey],
  );

  const toggleColumn = useCallback(
    (key: string) => {
      save((prev) => {
        const hidden = new Set(prev.hidden);
        if (hidden.has(key)) hidden.delete(key);
        else hidden.add(key);
        return { ...prev, hidden: [...hidden] };
      });
    },
    [save],
  );

  const moveColumn = useCallback(
    (key: string, direction: 'up' | 'down') => {
      save((prev) => {
        const currentOrder =
          prev.order.length > 0
            ? prev.order.filter((k) => configurableKeys.includes(k))
            : [...configurableKeys];
        const missing = configurableKeys.filter((k) => !currentOrder.includes(k));
        const order = [...currentOrder, ...missing];

        const idx = order.indexOf(key);
        if (idx < 0) return prev;
        const swapIdx = direction === 'up' ? idx - 1 : idx + 1;
        if (swapIdx < 0 || swapIdx >= order.length) return prev;

        [order[idx], order[swapIdx]] = [order[swapIdx], order[idx]];
        return { ...prev, order };
      });
    },
    [save, configurableKeys],
  );

  const reorderColumn = useCallback(
    (fromKey: string, toKey: string) => {
      if (fromKey === toKey) return;
      save((prev) => {
        const currentOrder =
          prev.order.length > 0
            ? prev.order.filter((k) => configurableKeys.includes(k))
            : [...configurableKeys];
        const missing = configurableKeys.filter((k) => !currentOrder.includes(k));
        const order = [...currentOrder, ...missing];

        const fromIdx = order.indexOf(fromKey);
        if (fromIdx < 0) return prev;
        order.splice(fromIdx, 1);
        const toIdx = order.indexOf(toKey);
        if (toIdx < 0) {
          order.push(fromKey);
        } else {
          order.splice(toIdx, 0, fromKey);
        }
        return { ...prev, order };
      });
    },
    [save, configurableKeys],
  );

  const resetConfig = useCallback(() => {
    if (!storageKey) return;
    localStorage.removeItem(storageKey);
    setRevision((r) => r + 1);
  }, [storageKey]);

  return { visibleColumns, configEntries, hiddenCount, toggleColumn, moveColumn, reorderColumn, resetConfig };
}
