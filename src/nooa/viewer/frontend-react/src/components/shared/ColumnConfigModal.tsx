import { useState, useEffect, useRef, useCallback } from 'react';
import type { ColumnConfigEntry } from '@/hooks/useColumnConfig';

interface ColumnConfigModalProps {
  entries: ColumnConfigEntry[];
  onToggle: (key: string) => void;
  onReorder: (fromKey: string, toKey: string) => void;
  onReset: () => void;
  onClose: () => void;
}

export function ColumnConfigModal({
  entries,
  onToggle,
  onReorder,
  onReset,
  onClose,
}: ColumnConfigModalProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [dragKey, setDragKey] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const handleDragStart = useCallback((e: React.DragEvent, key: string) => {
    setDragKey(key);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', key);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, key: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDropTarget(key);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent, toKey: string) => {
      e.preventDefault();
      if (dragKey && dragKey !== toKey) {
        onReorder(dragKey, toKey);
      }
      setDragKey(null);
      setDropTarget(null);
    },
    [dragKey, onReorder],
  );

  const handleDragEnd = useCallback(() => {
    setDragKey(null);
    setDropTarget(null);
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onClick={(e) => {
        if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
          onClose();
        }
      }}
    >
      <div
        ref={panelRef}
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl min-w-64 max-w-lg max-h-[90vh] flex flex-col"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <span className="text-sm font-medium text-gray-200">Columns</span>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 text-xs cursor-pointer"
          >
            x
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-1">
          {entries.map((entry) => (
            <div
              key={entry.key}
              draggable
              onDragStart={(e) => handleDragStart(e, entry.key)}
              onDragOver={(e) => handleDragOver(e, entry.key)}
              onDrop={(e) => handleDrop(e, entry.key)}
              onDragEnd={handleDragEnd}
              className={`flex items-center gap-2 px-4 py-1.5 cursor-grab active:cursor-grabbing select-none transition-colors ${
                dragKey === entry.key
                  ? 'opacity-40'
                  : dropTarget === entry.key && dragKey
                    ? 'border-t-2 border-blue-500'
                    : 'hover:bg-gray-800/50'
              }`}
            >
              <span className="text-gray-600 text-xs shrink-0" title="Drag to reorder">
                ⠿
              </span>
              <input
                type="checkbox"
                checked={entry.visible}
                onChange={() => onToggle(entry.key)}
                className="accent-blue-500 cursor-pointer shrink-0"
              />
              <span
                className={`flex-1 text-sm whitespace-nowrap ${entry.visible ? 'text-gray-200' : 'text-gray-500'}`}
              >
                {entry.label}
              </span>
            </div>
          ))}
        </div>

        <div className="px-4 py-3 border-t border-gray-800">
          <button
            onClick={onReset}
            className="text-xs text-gray-500 hover:text-gray-300 cursor-pointer"
          >
            Reset to defaults
          </button>
        </div>
      </div>
    </div>
  );
}
