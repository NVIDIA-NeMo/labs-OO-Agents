export interface TruncatedJson {
  limitChars: number;
  previewChars: number;
  preview: string;
}

export function parseTruncatedJson(value: unknown): TruncatedJson | null {
  let parsed: unknown = value;
  if (typeof value === 'string') {
    try {
      parsed = JSON.parse(value);
    } catch {
      return null;
    }
  }

  if (!parsed || typeof parsed !== 'object') return null;
  const envelope = parsed as Record<string, unknown>;
  const metadata = envelope['$nooa'];
  if (!metadata || typeof metadata !== 'object') return null;

  const fields = metadata as Record<string, unknown>;
  if (
    fields.kind !== 'truncated-json' ||
    typeof fields.limit_chars !== 'number' ||
    typeof fields.preview_chars !== 'number' ||
    typeof envelope.preview !== 'string'
  ) {
    return null;
  }

  return {
    limitChars: fields.limit_chars,
    previewChars: fields.preview_chars,
    preview: envelope.preview,
  };
}
