/**
 * Shared response handling for the viewer API.
 *
 * A 401/403 means this browser has not been authorized, which is a whole-app
 * condition rather than a per-request failure: every panel fails identically.
 * Pages used to swallow it and render an empty list, so an unauthorized viewer
 * looked exactly like a viewer with no data. `assertOk` distinguishes it and
 * publishes it so the shell can say so once, in one place.
 */

export class ViewerAuthError extends Error {
  readonly status: number;

  constructor(status: number, what: string) {
    super(`${what}: not authorized (${status})`);
    this.name = 'ViewerAuthError';
    this.status = status;
  }
}

type Listener = (err: ViewerAuthError | null) => void;

const listeners = new Set<Listener>();
let current: ViewerAuthError | null = null;

function publish(err: ViewerAuthError | null): void {
  current = err;
  for (const cb of listeners) cb(err);
}

/** Subscribe to auth-state changes. Fires immediately with the current state. */
export function onAuthFailure(cb: Listener): () => void {
  listeners.add(cb);
  cb(current);
  return () => {
    listeners.delete(cb);
  };
}

/**
 * Throw on a failed response, tagging auth failures.
 *
 * A successful response clears any prior auth error, so the banner disappears
 * on its own once the token bootstrap has run — no reload needed.
 */
export function assertOk(res: Response, what: string): void {
  if (res.ok) {
    if (current) publish(null);
    return;
  }
  if (res.status === 401 || res.status === 403) {
    const err = new ViewerAuthError(res.status, what);
    publish(err);
    throw err;
  }
  throw new Error(`${what}: ${res.statusText}`);
}
