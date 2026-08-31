import { useEffect, useState } from 'react';
import { onAuthFailure, type ViewerAuthError } from '@/api/http';

/**
 * Shown when the viewer rejects this browser.
 *
 * Without it a 401 is indistinguishable from an empty store: every page renders
 * its "no data" state and the only evidence is a console line. Since the fix is
 * a specific action the user cannot guess, spell it out.
 */
export function AuthBanner() {
  const [err, setErr] = useState<ViewerAuthError | null>(null);

  useEffect(() => onAuthFailure(setErr), []);

  if (!err) return null;

  return (
    <div className="border-b border-amber-700/60 bg-amber-950/60 px-4 py-2.5 text-sm text-amber-100">
      <div className="max-w-[100rem] mx-auto flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-semibold">This browser is not authorized ({err.status}).</span>
        <span className="text-amber-200/90">
          Traces may exist but cannot be loaded. Open the{' '}
          <code className="px-1 rounded bg-amber-900/60 text-amber-100">Share:</code> link printed by{' '}
          <code className="px-1 rounded bg-amber-900/60 text-amber-100">nooa start-dev</code> once —
          it ends in <code className="px-1 rounded bg-amber-900/60 text-amber-100">?token=…</code>{' '}
          and authorizes this browser for future visits.
        </span>
      </div>
    </div>
  );
}
