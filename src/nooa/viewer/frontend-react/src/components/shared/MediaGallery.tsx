import { createContext, useContext } from "react";

/** One piece of media an agent attached with show(). */
export interface MediaItem {
  kind: string; // image | audio | video | file
  url: string;
  mimeType?: string;
  filename?: string;
}

/** Session of the trace being viewed; media references are per session. */
export const TraceSessionContext = createContext<string>("");

export function useTraceSessionId(): string {
  return useContext(TraceSessionContext);
}

export function mediaRefUrl(sessionId: string, hash: string): string {
  return `/api/trace/media?session_id=${encodeURIComponent(sessionId)}&hash=${encodeURIComponent(hash)}`;
}

/** Read `<prefix>.contents.N.message_content.*` media groups (see otlp_store._media_content_attrs). */
export function extractMessageMedia(
  attrs: Record<string, unknown>,
  prefix: string,
): MediaItem[] {
  const items: MediaItem[] = [];
  for (let k = 0; ; k++) {
    const cp = `${prefix}.contents.${k}.message_content`;
    const kind = attrs[`${cp}.type`];
    if (typeof kind !== "string") break;
    const url = (attrs[`${cp}.image.image.url`] ?? attrs[`${cp}.media.url`]) as
      | string
      | undefined;
    if (kind === "text" || !url) continue;
    items.push({
      kind,
      url,
      mimeType: (attrs[`${cp}.media.mime_type`] as string) || undefined,
      filename: (attrs[`${cp}.media.filename`] as string) || undefined,
    });
  }
  return items;
}

/** Media references recorded on a code-execution span's output.value ("media" list). */
export function mediaFromRefs(refs: unknown, sessionId: string): MediaItem[] {
  if (!Array.isArray(refs)) return [];
  const items: MediaItem[] = [];
  for (const ref of refs) {
    if (!ref || typeof ref !== "object") continue;
    const r = ref as Record<string, unknown>;
    const url =
      typeof r.url === "string"
        ? r.url
        : typeof r.hash === "string" && sessionId
          ? mediaRefUrl(sessionId, r.hash)
          : null;
    if (!url || typeof r.kind !== "string") continue;
    items.push({
      kind: r.kind,
      url,
      mimeType: (r.mime_type as string) || undefined,
      filename: (r.filename as string) || undefined,
    });
  }
  return items;
}

function isSafeUrl(url: string): boolean {
  return (
    url.startsWith("/api/") ||
    url.startsWith("https://") ||
    url.startsWith("http://") ||
    url.startsWith("data:image/") ||
    url.startsWith("data:audio/") ||
    url.startsWith("data:video/") ||
    url.startsWith("data:application/pdf")
  );
}

function MediaTile({ item }: { item: MediaItem }) {
  const label = item.filename || item.mimeType || item.kind;
  if (!isSafeUrl(item.url)) {
    return (
      <span className="text-xs text-gray-500">
        [{item.kind}: unsupported URL]
      </span>
    );
  }
  if (item.kind === "image") {
    return (
      <a
        href={item.url}
        target="_blank"
        rel="noopener noreferrer"
        title="Open full size"
      >
        <img
          src={item.url}
          alt={`shown image (${label})`}
          loading="lazy"
          className="max-h-64 max-w-full rounded border border-gray-700 bg-white/5 object-contain"
        />
      </a>
    );
  }
  if (item.kind === "audio") {
    return (
      <audio controls preload="none" src={item.url} className="max-w-full" />
    );
  }
  if (item.kind === "video") {
    return (
      <video
        controls
        preload="metadata"
        src={item.url}
        className="max-h-64 max-w-full rounded"
      />
    );
  }
  if (item.mimeType === "application/pdf") {
    return (
      <div className="w-full">
        <iframe
          src={item.url}
          title={`PDF ${label}`}
          loading="lazy"
          className="w-full h-96 rounded border border-gray-700 bg-white"
        />
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-400 hover:underline"
        >
          Open PDF ({label})
        </a>
      </div>
    );
  }
  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="text-xs text-blue-400 hover:underline"
    >
      Download {item.kind} ({label})
    </a>
  );
}

/** Thumbnails/players for show() media. Images open full size in a new tab. */
export function MediaGallery({
  items,
  label = "Shown media",
}: {
  items: MediaItem[];
  label?: string;
}) {
  if (items.length === 0) return null;
  return (
    <div
      className="p-3 bg-gray-900 rounded border-l-4 border-teal-600 mb-2"
      data-testid="media-gallery"
    >
      <div className="text-xs text-gray-500 mb-2">
        {label} ({items.length})
      </div>
      <div className="flex flex-wrap gap-2 items-start">
        {items.map((item, i) => (
          <MediaTile key={i} item={item} />
        ))}
      </div>
    </div>
  );
}
