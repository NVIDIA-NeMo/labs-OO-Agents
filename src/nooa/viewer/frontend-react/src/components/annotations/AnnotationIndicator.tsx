import { useCallback } from 'react';
import type { Annotation } from '@/api/annotations';

interface AnnotationIndicatorProps {
  annotations: Annotation[];
  onOpenForm: () => void;
  onQuickFeedback: (label: 'positive' | 'negative') => void;
}

export function AnnotationIndicator({
  annotations,
  onOpenForm,
  onQuickFeedback,
}: AnnotationIndicatorProps) {
  const positives = annotations.filter((a) => a.label === 'positive').length;
  const negatives = annotations.filter((a) => a.label === 'negative').length;
  const comments = annotations.filter((a) => a.comment).length;
  const scores = annotations.filter((a) => a.score != null);
  const avgScore =
    scores.length > 0 ? scores.reduce((s, a) => s + (a.score ?? 0), 0) / scores.length : null;

  const hasAny = annotations.length > 0;

  const handlePositive = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onQuickFeedback('positive');
    },
    [onQuickFeedback],
  );

  const handleNegative = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onQuickFeedback('negative');
    },
    [onQuickFeedback],
  );

  const handleOpenForm = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onOpenForm();
    },
    [onOpenForm],
  );

  return (
    <div className="flex items-center gap-1 shrink-0">
      {/* Quick feedback buttons — visible on hover via parent group */}
      <button
        onClick={handlePositive}
        className={`text-xs px-1 rounded transition-colors ${
          positives > 0
            ? 'text-green-400 bg-green-900/30'
            : 'text-gray-600 hover:text-green-400 opacity-0 group-hover/event:opacity-100'
        }`}
        title="Positive feedback"
      >
        +{positives > 0 ? positives : ''}
      </button>
      <button
        onClick={handleNegative}
        className={`text-xs px-1 rounded transition-colors ${
          negatives > 0
            ? 'text-red-400 bg-red-900/30'
            : 'text-gray-600 hover:text-red-400 opacity-0 group-hover/event:opacity-100'
        }`}
        title="Negative feedback"
      >
        -{negatives > 0 ? negatives : ''}
      </button>

      {/* Score badge */}
      {avgScore != null && (
        <span className="text-[10px] px-1 py-0.5 rounded bg-yellow-900/30 text-yellow-400 font-mono">
          {avgScore.toFixed(1)}
        </span>
      )}

      {/* Comment count */}
      {comments > 0 && <span className="text-[10px] text-gray-500">{comments}c</span>}

      {/* Annotation count / open form */}
      <button
        onClick={handleOpenForm}
        className={`text-xs px-1.5 py-0.5 rounded transition-colors ${
          hasAny
            ? 'bg-blue-900/30 text-blue-400 hover:bg-blue-900/50'
            : 'text-gray-600 hover:text-blue-400 opacity-0 group-hover/event:opacity-100'
        }`}
        title={hasAny ? `${annotations.length} annotation(s) — click to edit` : 'Add annotation'}
      >
        {hasAny ? annotations.length : 'a'}
      </button>
    </div>
  );
}
