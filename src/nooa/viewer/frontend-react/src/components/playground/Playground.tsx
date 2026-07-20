import { useState, useEffect, useCallback, useRef } from "react";
import {
  fetchPlaygroundModels,
  runInference,
  addCustomModel,
} from "@/api/playground";
import type {
  PlaygroundMessage,
  CustomModel,
  ModelsResponse,
  InferenceResponse,
} from "@/api/playground";
import type { PlaygroundRequest } from "./PlaygroundContext";
import { CodeBox } from "@/components/shared/CodeBox";

const ROLES = ["system", "user", "assistant", "tool"] as const;

const ROLE_BORDER: Record<string, string> = {
  system: "border-gray-600",
  user: "border-sky-700",
  assistant: "border-indigo-700",
  tool: "border-amber-700",
};

interface PlaygroundProps {
  request: PlaygroundRequest;
  onClose: () => void;
}

export function Playground({ request, onClose }: PlaygroundProps) {
  const [messages, setMessages] = useState<PlaygroundMessage[]>(request.messages);
  const [activeTab, setActiveTab] = useState<"structured" | "raw">("structured");
  const [rawJson, setRawJson] = useState("");
  const [rawError, setRawError] = useState<string | null>(null);

  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [temperature, setTemperature] = useState(0.7);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDiff, setShowDiff] = useState(false);

  const [showAddModel, setShowAddModel] = useState(false);

  const editorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchPlaygroundModels()
      .then((m) => {
        setModels(m);
        if (request.model) {
          const traceModel = request.model;
          const all = [...m.builtin.map((b) => b.id), ...m.custom.map((c) => c.model_id)];
          const match =
            all.find((id) => id === traceModel) ||
            all.find((id) => traceModel.endsWith(id)) ||
            all.find((id) => traceModel.includes(id));
          if (match) setSelectedModel(match);
        }
      })
      .catch(() => {});
  }, [request.model]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const syncRawFromMessages = useCallback((msgs: PlaygroundMessage[]) => {
    setRawJson(JSON.stringify(msgs, null, 2));
    setRawError(null);
  }, []);

  const switchTab = useCallback(
    (tab: "structured" | "raw") => {
      if (tab === "raw") {
        syncRawFromMessages(messages);
      } else if (activeTab === "raw") {
        try {
          const parsed = JSON.parse(rawJson);
          if (Array.isArray(parsed)) setMessages(parsed);
        } catch {
          // keep current messages
        }
      }
      setActiveTab(tab);
    },
    [activeTab, messages, rawJson, syncRawFromMessages],
  );

  const updateMessage = useCallback(
    (index: number, field: "role" | "content", value: string) => {
      setMessages((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], [field]: value };
        return next;
      });
    },
    [],
  );

  const removeMessage = useCallback((index: number) => {
    setMessages((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const addMessage = useCallback(() => {
    setMessages((prev) => [...prev, { role: "user", content: "" }]);
  }, []);

  const handleRun = useCallback(async () => {
    if (!selectedModel) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      let msgs = messages;
      if (activeTab === "raw") {
        const parsed = JSON.parse(rawJson);
        if (Array.isArray(parsed)) msgs = parsed;
      }
      const res = await runInference({
        messages: msgs,
        model: selectedModel,
        temperature,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setRunning(false);
    }
  }, [selectedModel, messages, activeTab, rawJson, temperature]);

  const handleAddModel = useCallback(
    async (model: CustomModel) => {
      try {
        await addCustomModel(model);
        const m = await fetchPlaygroundModels();
        setModels(m);
        setSelectedModel(model.model_id);
        setShowAddModel(false);
      } catch (err) {
        alert(err instanceof Error ? err.message : "Failed to add model");
      }
    },
    [],
  );

  const resultContent = result?.response?.content || null;
  const resultToolCalls = result?.response?.tool_calls || [];
  const resultReasoning = result?.response?.reasoning_content || null;

  return (
    <div className="fixed inset-0 z-50 bg-gray-950 flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800 shrink-0">
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-200 text-sm whitespace-nowrap"
        >
          &#9666; Back to Trace
        </button>
        <span className="text-sm text-gray-300 font-medium">Playground</span>
        <div className="ml-auto flex items-center gap-3">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none max-w-[20rem]"
          >
            <option value="">Select model...</option>
            {models && (
              <>
                <optgroup label="Built-in Models">
                  {models.builtin.map((m) => (
                    <option
                      key={m.id}
                      value={m.id}
                      disabled={!!m.api_key_env && !models.available_api_keys.includes(m.api_key_env)}
                    >
                      {m.name || m.id}
                    </option>
                  ))}
                </optgroup>
                {models.custom.length > 0 && (
                  <optgroup label="Custom Models">
                    {models.custom.map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.name || m.model_id}
                      </option>
                    ))}
                  </optgroup>
                )}
              </>
            )}
          </select>
          <button
            onClick={() => setShowAddModel(true)}
            className="text-xs text-gray-500 hover:text-gray-300"
          >
            + Model
          </button>
          <label className="flex items-center gap-1 text-xs text-gray-500">
            Temp
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
              className="w-14 px-1 py-0.5 bg-gray-800 border border-gray-700 rounded text-gray-200 text-xs focus:outline-none"
            />
          </label>
          <button
            onClick={handleRun}
            disabled={running || !selectedModel}
            className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-500 disabled:opacity-50 whitespace-nowrap"
          >
            {running ? "Running..." : "Run Inference"}
          </button>
        </div>
      </div>

      {/* Body: two panels */}
      <div className="flex flex-1 min-h-0">
        {/* Editor panel */}
        <div ref={editorRef} className="flex-1 flex flex-col border-r border-gray-800 min-w-0">
          <div className="flex items-center gap-1 px-3 py-1.5 border-b border-gray-800 shrink-0">
            <button
              onClick={() => switchTab("structured")}
              className={`px-2 py-1 text-xs rounded ${activeTab === "structured" ? "bg-gray-700 text-gray-200" : "text-gray-500 hover:text-gray-300"}`}
            >
              Structured
            </button>
            <button
              onClick={() => switchTab("raw")}
              className={`px-2 py-1 text-xs rounded ${activeTab === "raw" ? "bg-gray-700 text-gray-200" : "text-gray-500 hover:text-gray-300"}`}
            >
              Raw JSON
            </button>
            {activeTab === "structured" && (
              <button
                onClick={addMessage}
                className="ml-auto text-xs text-gray-500 hover:text-gray-300"
              >
                + Add Message
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-3">
            {activeTab === "structured" ? (
              <div className="space-y-2">
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`bg-gray-900 rounded border-l-4 ${ROLE_BORDER[msg.role] || "border-gray-600"}`}
                  >
                    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800">
                      <select
                        value={msg.role}
                        onChange={(e) => updateMessage(i, "role", e.target.value)}
                        className="px-1 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none"
                      >
                        {ROLES.map((r) => (
                          <option key={r} value={r}>{r}</option>
                        ))}
                      </select>
                      {msg.tool_call_id && (
                        <span className="text-[10px] text-gray-500 font-mono">
                          tool_call_id: {msg.tool_call_id.slice(-8)}
                        </span>
                      )}
                      <button
                        onClick={() => removeMessage(i)}
                        className="ml-auto text-xs text-gray-600 hover:text-red-400"
                      >
                        Remove
                      </button>
                    </div>
                    <textarea
                      value={msg.content || ""}
                      onChange={(e) => updateMessage(i, "content", e.target.value)}
                      rows={Math.min(12, Math.max(3, (msg.content || "").split("\n").length + 1))}
                      className="w-full px-3 py-2 bg-transparent text-xs text-gray-200 font-mono resize-y focus:outline-none"
                    />
                    {msg.tool_calls && msg.tool_calls.length > 0 && (
                      <div className="px-3 pb-2">
                        <div className="text-[10px] text-gray-500 mb-1">
                          Tool calls ({msg.tool_calls.length})
                        </div>
                        {msg.tool_calls.map((tc, j) => (
                          <div key={j} className="text-[10px] text-gray-400 font-mono bg-gray-800 rounded px-2 py-1 mb-1">
                            {tc.name}({typeof tc.arguments === "string" ? tc.arguments.slice(0, 80) : JSON.stringify(tc.arguments).slice(0, 80)}...)
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full flex flex-col">
                {rawError && (
                  <div className="text-xs text-red-400 mb-1">{rawError}</div>
                )}
                <textarea
                  value={rawJson}
                  onChange={(e) => {
                    setRawJson(e.target.value);
                    try {
                      JSON.parse(e.target.value);
                      setRawError(null);
                    } catch (err) {
                      setRawError(err instanceof Error ? err.message : "Invalid JSON");
                    }
                  }}
                  className="flex-1 w-full bg-gray-900 text-xs text-gray-200 font-mono p-3 rounded border border-gray-800 resize-none focus:outline-none focus:border-gray-600"
                  spellCheck={false}
                />
              </div>
            )}
          </div>
        </div>

        {/* Results panel */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800 shrink-0">
            <span className="text-xs text-gray-500">Results</span>
            {request.originalOutput && resultContent && (
              <label className="ml-auto flex items-center gap-1 text-xs text-gray-500 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showDiff}
                  onChange={(e) => setShowDiff(e.target.checked)}
                  className="rounded border-gray-700 bg-gray-800 text-gray-500 focus:ring-0 w-3 h-3 accent-gray-500"
                />
                Show Diff
              </label>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-3">
            {error && (
              <div className="p-3 bg-red-900/20 border border-red-800 rounded text-xs text-red-300 mb-3">
                {error}
              </div>
            )}

            {running && (
              <div className="text-sm text-gray-500 py-8 text-center">Running inference...</div>
            )}

            {!running && !result && !error && (
              <div className="text-sm text-gray-600 py-8 text-center">
                Edit messages, select a model, and click Run Inference
              </div>
            )}

            {result && (
              <div className="space-y-3">
                {/* Stats */}
                <div className="flex items-center gap-3 text-xs text-gray-500">
                  <span className="font-mono">{result.model}</span>
                  {result.usage.prompt_tokens != null && (
                    <span>in: {result.usage.prompt_tokens}</span>
                  )}
                  {result.usage.completion_tokens != null && (
                    <span>out: {result.usage.completion_tokens}</span>
                  )}
                </div>

                {showDiff && request.originalOutput && resultContent ? (
                  <DiffView original={request.originalOutput} modified={resultContent} />
                ) : (
                  <>
                    {resultContent && (
                      <CodeBox code={resultContent} language="markdown" showLineNumbers={false} />
                    )}
                  </>
                )}

                {resultToolCalls.length > 0 && (
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Tool Calls</div>
                    {resultToolCalls.map((tc, i) => {
                      const fn = tc.function;
                      let code = fn.arguments;
                      let lang = "json";
                      if (fn.name === "execute_python") {
                        try {
                          const parsed = JSON.parse(fn.arguments);
                          if (parsed.code) {
                            code = parsed.code;
                            lang = "python";
                          }
                        } catch { /* keep as-is */ }
                      }
                      return (
                        <div key={i} className="p-3 bg-gray-900 rounded border-l-4 border-amber-600 mb-2">
                          <div className="text-xs text-gray-500 mb-1">
                            {fn.name} [{(tc.id || "").slice(-8)}]
                          </div>
                          <CodeBox code={code} language={lang} maxHeight="300px" />
                        </div>
                      );
                    })}
                  </div>
                )}

                {resultReasoning && (
                  <details className="border border-purple-700 rounded-md overflow-hidden">
                    <summary className="px-3 py-2 bg-purple-900/20 cursor-pointer text-xs font-semibold text-purple-300">
                      Reasoning
                    </summary>
                    <div className="p-3 bg-[#1a1625] max-h-[300px] overflow-auto">
                      <pre className="m-0 whitespace-pre-wrap break-words text-xs leading-relaxed text-gray-200">
                        {resultReasoning}
                      </pre>
                    </div>
                  </details>
                )}

                {/* Original output for comparison */}
                {request.originalOutput && !showDiff && (
                  <details className="mt-2">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
                      Original Output
                    </summary>
                    <div className="mt-1">
                      <CodeBox code={request.originalOutput} language="markdown" showLineNumbers={false} />
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {showAddModel && (
        <AddModelModal
          onAdd={handleAddModel}
          onClose={() => setShowAddModel(false)}
        />
      )}
    </div>
  );
}

function DiffView({ original, modified }: { original: string; modified: string }) {
  const origLines = original.split("\n");
  const modLines = modified.split("\n");
  const maxLen = Math.max(origLines.length, modLines.length);
  const lines: { type: "added" | "removed" | "unchanged"; text: string }[] = [];

  for (let i = 0; i < maxLen; i++) {
    const o = origLines[i];
    const m = modLines[i];
    if (o === undefined) {
      lines.push({ type: "added", text: m });
    } else if (m === undefined) {
      lines.push({ type: "removed", text: o });
    } else if (o === m) {
      lines.push({ type: "unchanged", text: o });
    } else {
      lines.push({ type: "removed", text: o });
      lines.push({ type: "added", text: m });
    }
  }

  return (
    <div className="bg-gray-900 rounded border border-gray-800 p-3 font-mono text-xs overflow-x-auto">
      {lines.map((line, i) => (
        <div
          key={i}
          className={
            line.type === "added"
              ? "text-green-400 bg-green-900/20"
              : line.type === "removed"
                ? "text-red-400 bg-red-900/20"
                : "text-gray-500"
          }
        >
          <span className="select-none mr-2 opacity-60">
            {line.type === "added" ? "+" : line.type === "removed" ? "-" : " "}
          </span>
          {line.text}
        </div>
      ))}
    </div>
  );
}

function AddModelModal({
  onAdd,
  onClose,
}: {
  onAdd: (model: CustomModel) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [modelId, setModelId] = useState("");
  const [endpoint, setEndpoint] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState("");

  const handleSubmit = () => {
    if (!name || !modelId) return;
    onAdd({
      name,
      model_id: modelId,
      endpoint: endpoint || undefined,
      api_key_env: apiKeyEnv || undefined,
    });
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl w-full max-w-sm mx-4">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-gray-200">Add Custom Model</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 text-sm">x</button>
        </div>
        <div className="px-4 py-3 space-y-3">
          <div>
            <div className="text-xs text-gray-500 mb-1">Display Name</div>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none"
              placeholder="My Model"
            />
          </div>
          <div>
            <div className="text-xs text-gray-500 mb-1">Model ID</div>
            <input
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none"
              placeholder="provider/model-name"
            />
          </div>
          <div>
            <div className="text-xs text-gray-500 mb-1">Endpoint (optional)</div>
            <input
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none"
              placeholder="https://api.example.com/v1"
            />
          </div>
          <div>
            <div className="text-xs text-gray-500 mb-1">API Key Env Var (optional)</div>
            <input
              value={apiKeyEnv}
              onChange={(e) => setApiKeyEnv(e.target.value)}
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none"
              placeholder="OPENAI_API_KEY"
            />
          </div>
        </div>
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-3 py-1 text-xs text-gray-400 hover:text-gray-200 border border-gray-700 rounded"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!name || !modelId}
            className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-500 disabled:opacity-50"
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
}
