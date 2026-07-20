export interface PlaygroundModel {
  id: string;
  name: string;
  provider: string;
  endpoint?: string;
  api_key_env?: string;
}

export interface CustomModel {
  name: string;
  model_id: string;
  endpoint?: string;
  api_key_env?: string;
}

export interface ModelsResponse {
  builtin: PlaygroundModel[];
  custom: CustomModel[];
  available_api_keys: string[];
}

export interface PlaygroundMessage {
  role: string;
  content: string | null;
  tool_calls?: { name: string; arguments: string | Record<string, unknown>; id: string | null }[];
  tool_call_id?: string;
}

export interface InferenceResponse {
  status: string;
  response: {
    role: string;
    content: string | null;
    tool_calls?: { id: string; type: string; function: { name: string; arguments: string } }[];
    reasoning_content?: string | null;
  };
  usage: {
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
  };
  model: string;
}

export async function fetchPlaygroundModels(): Promise<ModelsResponse> {
  const res = await fetch('/api/playground/models');
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function runInference(params: {
  messages: PlaygroundMessage[];
  model: string;
  temperature: number;
  max_tokens?: number;
}): Promise<InferenceResponse> {
  const res = await fetch('/api/playground/inference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: params.messages,
      model: params.model,
      temperature: params.temperature,
      max_tokens: params.max_tokens ?? 4096,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Inference failed: ${res.statusText}`);
  }
  return res.json();
}

export async function addCustomModel(model: CustomModel): Promise<void> {
  const res = await fetch('/api/playground/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(model),
  });
  if (!res.ok) throw new Error(`Failed to add model: ${res.statusText}`);
}

export async function deleteCustomModel(modelId: string): Promise<void> {
  const res = await fetch(`/api/playground/models/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete model: ${res.statusText}`);
}
