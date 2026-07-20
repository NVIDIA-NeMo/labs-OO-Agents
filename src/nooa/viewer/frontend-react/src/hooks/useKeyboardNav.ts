import { useEffect, useCallback, useRef } from "react";

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    (el as HTMLElement).isContentEditable
  );
}

export interface KeyboardNavCallbacks {
  getItemCount: () => number;
  getSelectedIndex: () => number | null;
  setSelectedIndex: (index: number) => void;
  onExpand?: (index: number) => void;
  onCollapse?: (index: number) => void;
  onActivate?: (index: number) => void;
  onBack?: () => void;
  onSearch?: () => void;
  onAnnotate?: (index: number) => void;
  onPositiveFeedback?: (index: number) => void;
  onNegativeFeedback?: (index: number) => void;
  onToggleRawJson?: (index: number) => void;
  onShowHelp?: () => void;
  getViewState?: (index: number) => string;
}

export function useKeyboardNav(callbacks: KeyboardNavCallbacks) {
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  const scrollToIndex = useCallback((index: number) => {
    const el =
      document.querySelector(`[data-event-index="${index}"]`) ||
      document.querySelector(`[data-list-index="${index}"]`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const cb = cbRef.current;

      if (e.key === "Escape") {
        if (isInputFocused()) {
          (document.activeElement as HTMLElement)?.blur();
          e.preventDefault();
          return;
        }
        return;
      }

      if (isInputFocused()) return;

      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const count = cb.getItemCount();
      const sel = cb.getSelectedIndex();

      switch (e.key) {
        case "?": {
          e.preventDefault();
          cb.onShowHelp?.();
          break;
        }

        case "j":
        case "ArrowDown": {
          e.preventDefault();
          const step = e.shiftKey ? 10 : 1;
          if (count === 0) break;
          const next = sel == null ? 0 : Math.min(count - 1, sel + step);
          cb.setSelectedIndex(next);
          scrollToIndex(next);
          break;
        }

        case "k":
        case "ArrowUp": {
          e.preventDefault();
          const step = e.shiftKey ? 10 : 1;
          if (count === 0) break;
          const prev = sel == null ? 0 : Math.max(0, sel - step);
          cb.setSelectedIndex(prev);
          scrollToIndex(prev);
          break;
        }

        case "ArrowRight":
        case "Enter":
        case "l": {
          if (sel == null) break;
          e.preventDefault();
          const state = cb.getViewState?.(sel);
          if (state === "collapsed" || state === "concise") {
            cb.onExpand?.(sel);
          } else {
            cb.onActivate?.(sel);
          }
          break;
        }

        case "ArrowLeft":
        case "h": {
          if (sel == null) {
            cb.onBack?.();
            break;
          }
          e.preventDefault();
          const state = cb.getViewState?.(sel);
          if (state === "expanded" || state === "concise") {
            cb.onCollapse?.(sel);
          } else {
            cb.onBack?.();
          }
          break;
        }

        case "Backspace": {
          e.preventDefault();
          cb.onBack?.();
          break;
        }

        case "/": {
          e.preventDefault();
          cb.onSearch?.();
          break;
        }

        case "a":
        case "A": {
          e.preventDefault();
          const target = sel ?? 0;
          if (count > 0) {
            cb.setSelectedIndex(target);
            cb.onAnnotate?.(target);
          }
          break;
        }

        case "+":
        case "=": {
          e.preventDefault();
          const target = sel ?? 0;
          if (count > 0) {
            cb.setSelectedIndex(target);
            cb.onPositiveFeedback?.(target);
          }
          break;
        }

        case "-":
        case "_": {
          e.preventDefault();
          const target = sel ?? 0;
          if (count > 0) {
            cb.setSelectedIndex(target);
            cb.onNegativeFeedback?.(target);
          }
          break;
        }

        case "r": {
          if (sel == null) break;
          e.preventDefault();
          cb.onToggleRawJson?.(sel);
          break;
        }
      }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [scrollToIndex]);
}
