import { useState, useCallback, useEffect } from 'react';
import type { Annotation, AnnotationCreate } from '@/api/annotations';
import { createAnnotation, updateAnnotation, deleteAnnotation, fetchTags } from '@/api/annotations';

interface AnnotationFormProps {
  sessionId: string;
  spanId: string | null;
  existing: Annotation[];
  onClose: () => void;
  onSaved: () => void;
}

const LABEL_OPTIONS = [
  { value: 'positive', text: 'Good', cls: 'bg-green-900/40 text-green-300 border-green-700' },
  { value: 'negative', text: 'Bad', cls: 'bg-red-900/40 text-red-300 border-red-700' },
] as const;

const DEFAULT_TAGS = [
  'correct',
  'incorrect',
  'hallucination',
  'hypothesis:prompt-unclear',
  'hypothesis:model-limitation',
];

export function AnnotationForm({
  sessionId,
  spanId,
  existing,
  onClose,
  onSaved,
}: AnnotationFormProps) {
  const editing = existing.length > 0 ? existing[0] : null;

  const [label, setLabel] = useState<string | null>(editing?.label ?? null);
  const [score, setScore] = useState<string>(editing?.score != null ? String(editing.score) : '');
  const [comment, setComment] = useState(editing?.comment ?? '');
  const [tags, setTags] = useState<string[]>(editing?.tags ?? []);
  const [tagInput, setTagInput] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTags()
      .then((tagInfos) => {
        const fetched = tagInfos.map((t) => t.tag);
        const merged = [...new Set([...DEFAULT_TAGS, ...fetched])];
        setSuggestions(merged);
      })
      .catch(() => setSuggestions(DEFAULT_TAGS));
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const scoreVal = score !== '' ? parseFloat(score) : null;
      if (editing) {
        await updateAnnotation(editing.id, {
          label,
          score: scoreVal,
          comment: comment || null,
          tags,
        });
      } else {
        const data: AnnotationCreate = {
          session_id: sessionId,
          span_id: spanId,
          name: 'manual',
          label,
          score: scoreVal,
          comment: comment || null,
          tags,
          source: 'human',
        };
        await createAnnotation(data);
      }
      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  }, [editing, sessionId, spanId, label, score, comment, tags, onSaved, onClose]);

  const handleDelete = useCallback(async () => {
    if (!editing) return;
    setSaving(true);
    try {
      await deleteAnnotation(editing.id);
      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete');
    } finally {
      setSaving(false);
    }
  }, [editing, onSaved, onClose]);

  const addTag = useCallback(
    (tag: string) => {
      const t = tag.trim();
      if (t && !tags.includes(t)) setTags((prev) => [...prev, t]);
      setTagInput('');
    },
    [tags],
  );

  const removeTag = useCallback((tag: string) => {
    setTags((prev) => prev.filter((t) => t !== tag));
  }, []);

  const handleTagKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault();
        addTag(tagInput);
      } else if (e.key === 'Backspace' && tagInput === '' && tags.length > 0) {
        setTags((prev) => prev.slice(0, -1));
      }
    },
    [tagInput, tags, addTag],
  );

  const filteredSuggestions = tagInput
    ? suggestions.filter(
        (s) => s.toLowerCase().includes(tagInput.toLowerCase()) && !tags.includes(s),
      )
    : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl w-full max-w-md mx-4"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-gray-200">
            {editing ? 'Edit Annotation' : 'Add Annotation'}
          </h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 text-sm">
            x
          </button>
        </div>

        <div className="px-4 py-3 space-y-4">
          {error && (
            <div className="text-xs text-red-400 bg-red-900/20 rounded px-2 py-1">{error}</div>
          )}

          {/* Label (Good/Bad) */}
          <div>
            <div className="text-xs text-gray-500 mb-1.5">Outcome</div>
            <div className="flex gap-2">
              {LABEL_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setLabel((prev) => (prev === opt.value ? null : opt.value))}
                  className={`px-3 py-1.5 text-xs rounded border transition-colors ${
                    label === opt.value
                      ? opt.cls
                      : 'border-gray-700 text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {opt.text}
                </button>
              ))}
            </div>
          </div>

          {/* Score */}
          <div>
            <div className="text-xs text-gray-500 mb-1.5">Score (0-5)</div>
            <input
              type="number"
              min="0"
              max="5"
              step="0.5"
              value={score}
              onChange={(e) => setScore(e.target.value)}
              placeholder="Optional"
              className="w-24 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-500"
            />
          </div>

          {/* Comment */}
          <div>
            <div className="text-xs text-gray-500 mb-1.5">Comment</div>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={3}
              placeholder="Add notes..."
              className="w-full px-2 py-1.5 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-500 resize-none"
            />
          </div>

          {/* Tags */}
          <div>
            <div className="text-xs text-gray-500 mb-1.5">Tags</div>
            <div className="flex flex-wrap gap-1 mb-1.5">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-0.5 px-1.5 py-0.5 bg-gray-800 text-gray-300 rounded text-[10px]"
                >
                  {tag}
                  <button
                    onClick={() => removeTag(tag)}
                    className="text-gray-500 hover:text-gray-300 ml-0.5"
                  >
                    x
                  </button>
                </span>
              ))}
            </div>
            <div className="relative">
              <input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleTagKeyDown}
                placeholder="Type tag and press Enter"
                className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-500"
              />
              {filteredSuggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg max-h-32 overflow-y-auto z-10">
                  {filteredSuggestions.slice(0, 8).map((s) => (
                    <button
                      key={s}
                      onClick={() => addTag(s)}
                      className="block w-full text-left px-2 py-1 text-xs text-gray-300 hover:bg-gray-700"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Existing annotations list */}
          {existing.length > 1 && (
            <div>
              <div className="text-xs text-gray-500 mb-1">
                {existing.length} annotations on this span
              </div>
              <div className="space-y-1 max-h-24 overflow-y-auto">
                {existing.map((ann) => (
                  <div key={ann.id} className="flex items-center gap-2 text-[10px] text-gray-400">
                    {ann.label && (
                      <span
                        className={ann.label === 'positive' ? 'text-green-400' : 'text-red-400'}
                      >
                        {ann.label}
                      </span>
                    )}
                    {ann.score != null && <span>{ann.score}</span>}
                    {ann.comment && <span className="truncate max-w-[200px]">{ann.comment}</span>}
                    <span className="text-gray-600 ml-auto">
                      {new Date(ann.created_at).toLocaleDateString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800">
          <div>
            {editing && (
              <button
                onClick={handleDelete}
                disabled={saving}
                className="text-xs text-red-400 hover:text-red-300 disabled:opacity-50"
              >
                Delete
              </button>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1 text-xs text-gray-400 hover:text-gray-200 border border-gray-700 rounded"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-500 disabled:opacity-50"
            >
              {saving ? 'Saving...' : editing ? 'Update' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
