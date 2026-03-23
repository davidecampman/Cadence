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

export async function fetchConfig(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error(`Config fetch failed: ${res.status}`);
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
