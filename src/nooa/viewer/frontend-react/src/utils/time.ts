export function formatRelativeTime(timestamp: string | number): string {
  const date =
    typeof timestamp === "number" || !isNaN(Number(timestamp))
      ? Number(timestamp) > 1e9
        ? new Date(Number(timestamp) * 1000)
        : new Date(Number(timestamp))
      : new Date(timestamp);

  if (isNaN(date.getTime())) return String(timestamp);

  const diffMs = Date.now() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return "just now";
  if (diffMins < 60) return `${diffMins} ${diffMins === 1 ? "minute" : "minutes"} ago`;
  if (diffHours < 24) return `${diffHours} ${diffHours === 1 ? "hour" : "hours"} ago`;
  if (diffDays < 30) return `${diffDays} ${diffDays === 1 ? "day" : "days"} ago`;

  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function formatDurationMs(durationMs: number): string {
  if (!Number.isFinite(durationMs) || durationMs < 0) return "";

  if (durationMs < 1000) {
    if (durationMs === 0) return "0ms";
    if (durationMs < 10) return `${durationMs.toFixed(1)}ms`;
    // Math.round(999.6) === 1000 — roll over to seconds rather than emit "1000ms".
    const roundedMs = Math.round(durationMs);
    if (roundedMs < 1000) return `${roundedMs}ms`;
  }

  if (durationMs < 60_000) {
    const seconds = durationMs / 1000;
    const formatted = seconds < 10 ? seconds.toFixed(2) : seconds.toFixed(1);
    // toFixed rounds, so e.g. 59.99.toFixed(1) === "60.0". Roll over to minutes.
    if (Number(formatted) < 60) return `${formatted}s`;
  }

  // Round to whole seconds first so we never emit "Xm 60s".
  const totalSeconds = Math.round(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}
