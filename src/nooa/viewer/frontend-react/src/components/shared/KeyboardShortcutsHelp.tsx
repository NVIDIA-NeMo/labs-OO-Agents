import { useEffect } from "react";

interface KeyboardShortcutsHelpProps {
  onClose: () => void;
}

const sections = [
  {
    title: "Navigation",
    shortcuts: [
      { keys: ["j", "↓"], action: "Next item" },
      { keys: ["k", "↑"], action: "Previous item" },
      { keys: ["Shift+j", "Shift+↓"], action: "Jump 10 items" },
      { keys: ["Shift+k", "Shift+↑"], action: "Jump 10 items up" },
    ],
  },
  {
    title: "Expand / Collapse",
    shortcuts: [
      { keys: ["Click view buttons"], action: "Set collapsed, concise, or expanded" },
      { keys: ["→", "Enter", "l"], action: "Expand event" },
      { keys: ["←", "h"], action: "Collapse event" },
      { keys: ["r"], action: "Toggle raw JSON" },
      { keys: ["Backspace"], action: "Go back" },
    ],
  },
  {
    title: "Annotations",
    shortcuts: [
      { keys: ["a"], action: "Open annotation form" },
      { keys: ["+"], action: "Positive feedback" },
      { keys: ["-"], action: "Negative feedback" },
    ],
  },
  {
    title: "General",
    shortcuts: [
      { keys: ["/"], action: "Focus search" },
      { keys: ["?"], action: "Show this help" },
      { keys: ["Esc"], action: "Close / blur" },
    ],
  },
];

export function KeyboardShortcutsHelp({ onClose }: KeyboardShortcutsHelpProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "?") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h2 className="text-sm font-medium text-gray-200">
            Keyboard Shortcuts
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 text-sm"
          >
            x
          </button>
        </div>
        <div className="px-4 py-3 space-y-4">
          {sections.map((section) => (
            <div key={section.title}>
              <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-2">
                {section.title}
              </h3>
              <div className="space-y-1.5">
                {section.shortcuts.map((s) => (
                  <div
                    key={s.action}
                    className="flex items-center justify-between"
                  >
                    <span className="text-xs text-gray-400">{s.action}</span>
                    <div className="flex gap-1">
                      {s.keys.map((k) => (
                        <kbd
                          key={k}
                          className="px-1.5 py-0.5 text-[10px] font-mono bg-gray-800 border border-gray-700 rounded text-gray-300"
                        >
                          {k}
                        </kbd>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
