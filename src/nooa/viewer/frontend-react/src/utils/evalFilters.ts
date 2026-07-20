import type { TestResult } from "@/api/eval";

export interface EvalFilters {
  keyword: string;
  meta: Record<string, string>;
}

const RESERVED_PARAMS = new Set(["q", "page", "sort", "dir"]);

export function readFiltersFromParams(params: URLSearchParams): EvalFilters {
  const keyword = params.get("q") || "";
  const meta: Record<string, string> = {};
  for (const [key, val] of params.entries()) {
    if (!RESERVED_PARAMS.has(key) && val) {
      meta[key] = val;
    }
  }
  return { keyword, meta };
}

export function buildFilterParams(filters: EvalFilters): string {
  const params = new URLSearchParams();
  if (filters.keyword) params.set("q", filters.keyword);
  for (const [key, val] of Object.entries(filters.meta)) {
    if (val) params.set(key, val);
  }
  const str = params.toString();
  return str ? `?${str}` : "";
}

export function applyEvalFilters(
  tests: TestResult[],
  filters: EvalFilters,
): TestResult[] {
  let result = tests;

  for (const [key, val] of Object.entries(filters.meta)) {
    if (val) {
      result = result.filter((t) => String(t[key] ?? "") === val);
    }
  }

  if (filters.keyword) {
    const kw = filters.keyword.toLowerCase();
    result = result.filter(
      (t) =>
        (t.display_name || "").toLowerCase().includes(kw) ||
        (t.test_id || "").toLowerCase().includes(kw) ||
        (t.test_name || "").toLowerCase().includes(kw),
    );
  }

  return result;
}
