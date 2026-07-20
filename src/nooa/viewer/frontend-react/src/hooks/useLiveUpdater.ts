import { useEffect, useRef, useCallback } from 'react';

const POLL_INTERVAL_MS = 5000;

export function useLiveUpdater(hasRunning: boolean, onRefresh: () => void) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onRefreshRef = useRef(onRefresh);
  onRefreshRef.current = onRefresh;

  const start = useCallback(() => {
    if (intervalRef.current) return;
    intervalRef.current = setInterval(() => {
      onRefreshRef.current();
    }, POLL_INTERVAL_MS);
  }, []);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!hasRunning) {
      stop();
      return;
    }

    start();

    const onVisibility = () => {
      if (document.hidden) {
        stop();
      } else {
        onRefreshRef.current();
        start();
      }
    };

    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [hasRunning, start, stop]);
}
