import { useState } from 'react';
import type { ColumnConfig } from '@/hooks/useColumnConfig';
import { ColumnConfigModal } from './ColumnConfigModal';

export function ColumnConfigButton<T>({ config }: { config: ColumnConfig<T> }) {
  const [open, setOpen] = useState(false);
  const { configEntries, hiddenCount, toggleColumn, reorderColumn, resetConfig } = config;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={`text-xs cursor-pointer transition-colors ${
          hiddenCount > 0
            ? 'text-yellow-500 hover:text-yellow-300'
            : 'text-gray-600 hover:text-gray-400'
        }`}
        title={
          hiddenCount > 0
            ? 'Some columns are hidden. Click to configure'
            : 'Configure columns'
        }
      >
        {hiddenCount > 0 ? '⚠ Columns' : '⫶ Columns'}
      </button>
      {open && (
        <ColumnConfigModal
          entries={configEntries}
          onToggle={toggleColumn}
          onReorder={reorderColumn}
          onReset={resetConfig}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
