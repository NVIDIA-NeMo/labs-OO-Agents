// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type TraceDirection = "input" | "output";

/** Decode canonical OpenInference I/O, retaining legacy JSON heuristics. */
export function traceValue(
  attrs: Record<string, unknown>,
  direction: TraceDirection,
  ...legacyKeys: string[]
): unknown {
  let value = attrs[`${direction}.value`];
  if (value === undefined || value === null) {
    for (const key of legacyKeys) {
      if (attrs[key] !== undefined && attrs[key] !== null) {
        value = attrs[key];
        break;
      }
    }
  }
  if (typeof value !== "string") return value;

  const mime = attrs[`${direction}.mime_type`];
  if (mime === "application/json" || mime === undefined || mime === null) {
    try {
      return JSON.parse(value);
    } catch {
      // Legacy traces sometimes labelled plain text as JSON or omitted MIME.
    }
  }
  return value;
}

export function previewIncomplete(
  attrs: Record<string, unknown>,
  direction: TraceDirection,
): boolean {
  return attrs[`nooa.${direction}.preview.incomplete`] === true;
}

export function previewPaths(
  attrs: Record<string, unknown>,
  direction: TraceDirection,
): string[] {
  const value = attrs[`nooa.${direction}.preview.paths`];
  if (Array.isArray(value))
    return value.filter((path): path is string => typeof path === "string");
  return [];
}
