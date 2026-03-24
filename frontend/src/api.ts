const API_BASE = '/api';

export interface ChatResponse {
  response: string;
  session_id: string;
  trace_steps: TraceStep[];
  duration_ms: number;
}

export interface TraceStep {
  agent_id: string;
  task_id: string | null;
  step_type: 'observation' | 'thought' | 'action' | 'result' | 'error';
  content: string;
  timestamp: number;
  metadata: Record<string, unknown>;
  tokens_used: number;
}

export interface ToolInfo {
  name: string;
  description: string;
  permission_tier: string;
}

export interface SkillInfo {
  name: string;
  version: string;
  description: string;
}

export async function sendMessage(message: string, sessionId?: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  return res.json();
}

export interface BedrockConfig {
  enabled: boolean;
  region: string;
}

export interface ModelsConfig {
  strong: string;
  fast: string;
  embedding: string;
  fallback_chain: string[];
  bedrock: BedrockConfig;
}

export interface ExecutionConfig {
  timeout_seconds: number;
  max_output_bytes: number;
  restrict_network: boolean;
  max_memory_mb: number;
  max_cpu_seconds: number;
  max_file_descriptors: number;
  blocked_commands: string[];
}

export interface AgentsConfig {
  max_depth: number;
  max_parallel: number;
  loop_detection_window: number;
  max_iterations_per_task: number;
}

export interface AppConfig {
  models: ModelsConfig;
  budget: Record<string, unknown>;
  agents: AgentsConfig;
  memory: Record<string, unknown>;
  execution: ExecutionConfig;
  skills: Record<string, unknown>;
  logging: Record<string, unknown>;
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error(`Config fetch failed: ${res.status}`);
  return res.json();
}

export async function updateConfig(updates: Record<string, unknown>): Promise<AppConfig> {
  const res = await fetch(`${API_BASE}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ updates }),
  });
  if (!res.ok) throw new Error(`Config update failed: ${res.status}`);
  return res.json();
}

export async function fetchTools(): Promise<ToolInfo[]> {
  const res = await fetch(`${API_BASE}/tools`);
  if (!res.ok) throw new Error(`Tools fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchSkills(): Promise<SkillInfo[]> {
  const res = await fetch(`${API_BASE}/skills`);
  if (!res.ok) throw new Error(`Skills fetch failed: ${res.status}`);
  return res.json();
}

export async function uploadSkill(file: File): Promise<{ status: string; skill: SkillInfo }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/skills/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error || `Skill upload failed: ${res.status}`);
  }
  return res.json();
}

export async function uninstallSkill(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/skills/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error || `Skill uninstall failed: ${res.status}`);
  }
}

export async function fetchTrace(limit = 50): Promise<TraceStep[]> {
  const res = await fetch(`${API_BASE}/trace?limit=${limit}`);
  if (!res.ok) throw new Error(`Trace fetch failed: ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// --- API Key management ---

export interface KeyStatus {
  env_var: string;
  has_key: boolean;
}

export interface KeysResponse {
  providers: Record<string, KeyStatus>;
  stored: string[];
}

export async function fetchKeys(): Promise<KeysResponse> {
  const res = await fetch(`${API_BASE}/keys`);
  if (!res.ok) throw new Error(`Keys fetch failed: ${res.status}`);
  return res.json();
}

export async function saveKey(provider: string, apiKey: string): Promise<void> {
  const res = await fetch(`${API_BASE}/keys`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  if (!res.ok) throw new Error(`Key save failed: ${res.status}`);
}

export async function saveBedrockKeys(
  authType: 'api_key' | 'iam',
  credentials: { api_key?: string; access_key_id?: string; secret_access_key?: string }
): Promise<void> {
  const res = await fetch(`${API_BASE}/keys/bedrock`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ auth_type: authType, ...credentials }),
  });
  if (!res.ok) throw new Error(`Bedrock key save failed: ${res.status}`);
}

export async function deleteKey(provider: string): Promise<void> {
  const res = await fetch(`${API_BASE}/keys/${provider}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Key delete failed: ${res.status}`);
}

export interface ModelsListResponse {
  provider: string;
  models: string[];
}

export async function fetchModels(provider: string, tier?: string): Promise<ModelsListResponse> {
  const params = tier ? `?tier=${encodeURIComponent(tier)}` : '';
  const res = await fetch(`${API_BASE}/models/${provider}${params}`);
  if (!res.ok) throw new Error(`Models fetch failed: ${res.status}`);
  return res.json();
}

export type WsMessage =
  | { type: 'trace'; data: TraceStep }
  | { type: 'pong' };

export function connectWebSocket(onMessage: (msg: WsMessage) => void): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data) as WsMessage;
      onMessage(msg);
    } catch {
      // ignore parse errors
    }
  };

  // Heartbeat
  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 30000);

  ws.onclose = () => clearInterval(interval);

  return ws;
}
