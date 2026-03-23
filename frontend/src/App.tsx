import { useState, useEffect, useRef, useCallback } from 'react'
import {
  sendMessage,
  fetchConfig,
  updateConfig,
  fetchTools,
  fetchSkills,
  checkHealth,
  connectWebSocket,
  fetchKeys,
  saveKey,
  saveBedrockKeys,
  deleteKey,
  type AppConfig,
  type ChatResponse,
  type TraceStep,
  type ToolInfo,
  type SkillInfo,
  type WsMessage,
  type KeysResponse,
} from './api'
import BackgroundCanvas from './BackgroundCanvas'
import './App.css'

type View = 'chat' | 'tools' | 'skills' | 'config';

interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: number;
  duration_ms?: number;
  trace_steps?: TraceStep[];
}

function App() {
  const [lightMode, setLightMode] = useState(() => {
    const saved = localStorage.getItem('agent-one-theme');
    return saved === 'light';
  });
  const [view, setView] = useState<View>('chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  const [traceOpen, setTraceOpen] = useState(false);
  const [online, setOnline] = useState(false);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configSaving, setConfigSaving] = useState(false);
  const [configSaved, setConfigSaved] = useState(false);
  const [keysInfo, setKeysInfo] = useState<KeysResponse | null>(null);
  const [keyInput, setKeyInput] = useState('');
  const [keySaving, setKeySaving] = useState(false);
  const [keySaved, setKeySaved] = useState<string | null>(null);
  const [bedrockAuthType, setBedrockAuthType] = useState<'api_key' | 'iam'>('iam');
  const [bedrockApiKey, setBedrockApiKey] = useState('');
  const [bedrockAccessKeyId, setBedrockAccessKeyId] = useState('');
  const [bedrockSecretAccessKey, setBedrockSecretAccessKey] = useState('');
  const [configDraft, setConfigDraft] = useState<{
    strong: string;
    fast: string;
    embedding: string;
    fallback_chain: string;
    provider: string;
    apiKey: string;
  } | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const traceEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Sync theme class on body
  useEffect(() => {
    document.body.classList.toggle('light-mode', lightMode);
    localStorage.setItem('agent-one-theme', lightMode ? 'light' : 'dark');
  }, [lightMode]);

  // Check health on mount
  useEffect(() => {
    checkHealth().then(setOnline);
    const interval = setInterval(() => checkHealth().then(setOnline), 10000);
    return () => clearInterval(interval);
  }, []);

  // Load metadata on mount
  useEffect(() => {
    fetchTools().then(setTools).catch(() => {});
    fetchSkills().then(setSkills).catch(() => {});
    fetchConfig().then(setConfig).catch(() => {});
    fetchKeys().then(setKeysInfo).catch(() => {});
  }, []);

  // WebSocket for live trace
  useEffect(() => {
    const ws = connectWebSocket((msg: WsMessage) => {
      if (msg.type === 'trace') {
        setTraceSteps((prev) => [...prev, msg.data]);
      }
    });
    return () => ws.close();
  }, []);

  // Auto-scroll messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-scroll trace
  useEffect(() => {
    traceEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [traceSteps]);

  // Sync config draft when config loads
  useEffect(() => {
    if (config && !configDraft) {
      setConfigDraft({
        strong: config.models.strong,
        fast: config.models.fast,
        embedding: config.models.embedding,
        fallback_chain: config.models.fallback_chain.join(', '),
        provider: '',
        apiKey: '',
      });
    }
  }, [config, configDraft]);

  const handleConfigSave = useCallback(async () => {
    if (!configDraft) return;
    setConfigSaving(true);
    setConfigSaved(false);
    try {
      const fallback = configDraft.fallback_chain
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const newConfig = await updateConfig({
        models: {
          strong: configDraft.strong,
          fast: configDraft.fast,
          embedding: configDraft.embedding,
          fallback_chain: fallback,
        },
      });
      setConfig(newConfig);
      setConfigSaved(true);
      setTimeout(() => setConfigSaved(false), 3000);
    } catch (err) {
      alert(`Failed to save: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setConfigSaving(false);
    }
  }, [configDraft]);

  const handleKeySave = useCallback(async () => {
    if (!configDraft?.provider || !keyInput.trim()) return;
    setKeySaving(true);
    try {
      await saveKey(configDraft.provider, keyInput.trim());
      setKeyInput('');
      setKeySaved(configDraft.provider);
      setTimeout(() => setKeySaved(null), 3000);
      // Refresh stored keys info
      const updated = await fetchKeys();
      setKeysInfo(updated);
    } catch (err) {
      alert(`Failed to save key: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setKeySaving(false);
    }
  }, [configDraft?.provider, keyInput]);

  const handleBedrockKeySave = useCallback(async () => {
    setKeySaving(true);
    try {
      if (bedrockAuthType === 'api_key') {
        if (!bedrockApiKey.trim()) return;
        await saveBedrockKeys('api_key', { api_key: bedrockApiKey.trim() });
      } else {
        if (!bedrockAccessKeyId.trim() || !bedrockSecretAccessKey.trim()) return;
        await saveBedrockKeys('iam', {
          access_key_id: bedrockAccessKeyId.trim(),
          secret_access_key: bedrockSecretAccessKey.trim(),
        });
      }
      setBedrockApiKey('');
      setBedrockAccessKeyId('');
      setBedrockSecretAccessKey('');
      setKeySaved('bedrock');
      setTimeout(() => setKeySaved(null), 3000);
      const updated = await fetchKeys();
      setKeysInfo(updated);
    } catch (err) {
      alert(`Failed to save Bedrock credentials: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setKeySaving(false);
    }
  }, [bedrockAuthType, bedrockApiKey, bedrockAccessKeyId, bedrockSecretAccessKey]);

  const handleKeyDelete = useCallback(async (provider: string) => {
    try {
      await deleteKey(provider);
      const updated = await fetchKeys();
      setKeysInfo(updated);
    } catch (err) {
      alert(`Failed to delete key: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, []);

  const handleSend = useCallback(async (text?: string) => {
    const msg = text || input.trim();
    if (!msg || loading) return;

    setInput('');
    setView('chat');

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: msg,
      timestamp: Date.now() / 1000,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res: ChatResponse = await sendMessage(msg, sessionId);
      setSessionId(res.session_id);

      const agentMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'agent',
        content: res.response,
        timestamp: Date.now() / 1000,
        duration_ms: res.duration_ms,
        trace_steps: res.trace_steps,
      };
      setMessages((prev) => [...prev, agentMsg]);
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'agent',
        content: `Connection error: ${err instanceof Error ? err.message : 'Unknown error'}. Make sure the API server is running.`,
        timestamp: Date.now() / 1000,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const suggestions = [
    'Review my code for security issues',
    'Explain how this project works',
    'Write a Python function to parse CSV',
    'Search for TODO comments in the codebase',
  ];

  return (
    <>
      <BackgroundCanvas lightMode={lightMode} />
      <div className="app-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>Agent One</h1>
          <div className="version">v0.1.0 — Multi-Agent Framework</div>
        </div>

        <div className="sidebar-nav">
          <div className="sidebar-section">
            <h3>Navigation</h3>
            <button
              className={`sidebar-item ${view === 'chat' ? 'active' : ''}`}
              onClick={() => setView('chat')}
            >
              <span className="icon">&#x1F4AC;</span>
              Chat
            </button>
            <button
              className={`sidebar-item ${view === 'tools' ? 'active' : ''}`}
              onClick={() => setView('tools')}
            >
              <span className="icon">&#x1F527;</span>
              Tools
              <span className="badge">{tools.length}</span>
            </button>
            <button
              className={`sidebar-item ${view === 'skills' ? 'active' : ''}`}
              onClick={() => setView('skills')}
            >
              <span className="icon">&#x1F4DA;</span>
              Skills
              <span className="badge">{skills.length}</span>
            </button>
            <button
              className={`sidebar-item ${view === 'config' ? 'active' : ''}`}
              onClick={() => setView('config')}
            >
              <span className="icon">&#x2699;</span>
              Config
            </button>
          </div>

          <div className="sidebar-section">
            <h3>Trace</h3>
            <button
              className={`sidebar-item ${traceOpen ? 'active' : ''}`}
              onClick={() => setTraceOpen(!traceOpen)}
            >
              <span className="icon">&#x1F50D;</span>
              Reasoning Trace
              <span className="badge">{traceSteps.length}</span>
            </button>
          </div>
        </div>

        <div className="sidebar-footer">
          <span className="sidebar-footer-left">
            <span className={`status-dot ${online ? 'online' : 'offline'}`} />
            {online ? 'API Connected' : 'API Disconnected'}
          </span>
          <button
            className="theme-toggle"
            onClick={() => setLightMode((prev) => !prev)}
            title={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}
          >
            {lightMode ? '\u263E' : '\u2600'}
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className="main-content">
        {view === 'chat' && (
          <div className="chat-container">
            <div className="chat-messages">
              {messages.length === 0 && !loading ? (
                <div className="chat-welcome">
                  <div className="logo">&#x1F916;</div>
                  <h2>Agent One</h2>
                  <p>
                    A model-agnostic multi-agent framework with structured planning,
                    tiered memory, and parallel task execution.
                  </p>
                  <div className="chat-suggestions">
                    {suggestions.map((s) => (
                      <button
                        key={s}
                        className="suggestion-chip"
                        onClick={() => handleSend(s)}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((msg) => (
                    <div key={msg.id} className="message">
                      <div className="message-header">
                        <div className={`message-avatar ${msg.role}`}>
                          {msg.role === 'user' ? 'U' : 'A'}
                        </div>
                        <span className="message-sender">
                          {msg.role === 'user' ? 'You' : 'Agent One'}
                        </span>
                        <span className="message-time">{formatTime(msg.timestamp)}</span>
                      </div>
                      <div className={`message-body ${msg.role}`}>{msg.content}</div>
                      {msg.duration_ms && (
                        <div className="message-duration">
                          Completed in {(msg.duration_ms / 1000).toFixed(1)}s
                          {msg.trace_steps && msg.trace_steps.length > 0 && (
                            <>
                              {' '}&middot;{' '}
                              <button
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  color: 'var(--accent)',
                                  cursor: 'pointer',
                                  fontSize: 'inherit',
                                  fontFamily: 'inherit',
                                  padding: 0,
                                }}
                                onClick={() => {
                                  setTraceSteps((prev) => [...prev, ...msg.trace_steps!]);
                                  setTraceOpen(true);
                                }}
                              >
                                {msg.trace_steps.length} trace steps
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  {loading && (
                    <div className="typing-indicator">
                      <div className="message-header">
                        <div className="message-avatar agent">A</div>
                        <span className="message-sender">Agent One</span>
                      </div>
                      <div className="typing-dots">
                        <span />
                        <span />
                        <span />
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <textarea
                  ref={inputRef}
                  className="chat-input"
                  placeholder="Ask Agent One anything..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={loading}
                />
                <button
                  className="chat-send-btn"
                  onClick={() => handleSend()}
                  disabled={!input.trim() || loading}
                  title="Send message"
                >
                  &#x27A4;
                </button>
              </div>
            </div>
          </div>
        )}

        {view === 'tools' && (
          <div className="info-panel">
            <h2>Available Tools</h2>
            {tools.map((tool) => (
              <div key={tool.name} className="info-card">
                <h3>{tool.name}</h3>
                <p>{tool.description}</p>
                <span className="tag">{tool.permission_tier}</span>
              </div>
            ))}
            {tools.length === 0 && (
              <p style={{ color: 'var(--text-muted)' }}>
                {online ? 'No tools loaded.' : 'Connect to the API to see available tools.'}
              </p>
            )}
          </div>
        )}

        {view === 'skills' && (
          <div className="info-panel">
            <h2>Loaded Skills</h2>
            {skills.map((skill) => (
              <div key={skill.name} className="info-card">
                <h3>{skill.name}</h3>
                <p>{skill.description}</p>
                <span className="tag">v{skill.version}</span>
              </div>
            ))}
            {skills.length === 0 && (
              <p style={{ color: 'var(--text-muted)' }}>
                {online ? 'No skills loaded.' : 'Connect to the API to see loaded skills.'}
              </p>
            )}
          </div>
        )}

        {view === 'config' && (
          <div className="info-panel">
            <h2>LLM Providers</h2>
            <p className="config-subtitle">
              Configure which models are used for each task tier. Agent One uses LiteLLM under the hood,
              so any supported provider works (OpenAI, Anthropic, Google, Mistral, etc.).
            </p>

            {configDraft ? (
              <div className="config-form">
                {/* Provider API Key */}
                <div className="config-section">
                  <h3>Provider API Keys</h3>
                  <p className="config-hint">
                    Enter your API keys here. They are encrypted and stored securely on the server.
                    Keys set via environment variables take precedence over stored keys.
                  </p>

                  {/* Stored keys summary */}
                  {keysInfo && keysInfo.stored.length > 0 && (
                    <div className="keys-stored-list">
                      {keysInfo.stored.map((p) => (
                        <div key={p} className="key-stored-item">
                          <span className="key-provider-name">{p}</span>
                          <span className="key-status-badge">Configured</span>
                          <button
                            className="key-delete-btn"
                            onClick={() => handleKeyDelete(p)}
                            title={`Remove ${p} API key`}
                          >
                            &#x2715;
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="config-field">
                    <label>Provider</label>
                    <select
                      value={configDraft.provider}
                      onChange={(e) => { setConfigDraft({ ...configDraft, provider: e.target.value }); setKeyInput(''); }}
                    >
                      <option value="">Select a provider...</option>
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="google">Google (Gemini)</option>
                      <option value="mistral">Mistral</option>
                      <option value="cohere">Cohere</option>
                      <option value="deepseek">DeepSeek</option>
                      <option value="groq">Groq</option>
                      <option value="ollama">Ollama (Local)</option>
                      <option value="bedrock">AWS Bedrock</option>
                    </select>
                  </div>
                  {configDraft.provider && configDraft.provider !== 'ollama' && configDraft.provider !== 'bedrock' && (
                    <div className="config-field" style={{ marginTop: 12 }}>
                      <label>
                        API Key
                        <span className="field-hint">
                          {keysInfo?.providers?.[configDraft.provider]?.env_var
                            ? `Sets ${keysInfo.providers[configDraft.provider].env_var}`
                            : ''}
                          {keysInfo?.providers?.[configDraft.provider]?.has_key
                            ? ' (key already stored)'
                            : ''}
                        </span>
                      </label>
                      <div className="key-input-row">
                        <input
                          type="password"
                          value={keyInput}
                          onChange={(e) => setKeyInput(e.target.value)}
                          placeholder={
                            keysInfo?.providers?.[configDraft.provider]?.has_key
                              ? 'Enter new key to replace existing...'
                              : 'Paste your API key here...'
                          }
                        />
                        <button
                          className="config-save-btn key-save-btn"
                          onClick={handleKeySave}
                          disabled={keySaving || !keyInput.trim()}
                        >
                          {keySaving
                            ? 'Saving...'
                            : keySaved === configDraft.provider
                              ? 'Saved!'
                              : 'Save Key'}
                        </button>
                      </div>
                    </div>
                  )}
                  {configDraft.provider === 'bedrock' && (
                    <div className="config-field" style={{ marginTop: 12 }}>
                      <label>
                        Credential Type
                        <span className="field-hint">
                          {keysInfo?.providers?.bedrock?.has_key ? ' (credentials already stored)' : ''}
                        </span>
                      </label>
                      <select
                        value={bedrockAuthType}
                        onChange={(e) => setBedrockAuthType(e.target.value as 'api_key' | 'iam')}
                        style={{ marginBottom: 12 }}
                      >
                        <option value="iam">IAM Credentials (Access Key + Secret)</option>
                        <option value="api_key">API Key</option>
                      </select>
                      {bedrockAuthType === 'api_key' ? (
                        <div className="key-input-row">
                          <input
                            type="password"
                            value={bedrockApiKey}
                            onChange={(e) => setBedrockApiKey(e.target.value)}
                            placeholder="Paste your Bedrock API key..."
                          />
                          <button
                            className="config-save-btn key-save-btn"
                            onClick={handleBedrockKeySave}
                            disabled={keySaving || !bedrockApiKey.trim()}
                          >
                            {keySaving ? 'Saving...' : keySaved === 'bedrock' ? 'Saved!' : 'Save Key'}
                          </button>
                        </div>
                      ) : (
                        <>
                          <div style={{ marginBottom: 8 }}>
                            <label><span className="field-hint">AWS Access Key ID</span></label>
                            <input
                              type="password"
                              value={bedrockAccessKeyId}
                              onChange={(e) => setBedrockAccessKeyId(e.target.value)}
                              placeholder="AKIA..."
                            />
                          </div>
                          <div style={{ marginBottom: 8 }}>
                            <label><span className="field-hint">AWS Secret Access Key</span></label>
                            <input
                              type="password"
                              value={bedrockSecretAccessKey}
                              onChange={(e) => setBedrockSecretAccessKey(e.target.value)}
                              placeholder="Secret access key..."
                            />
                          </div>
                          <button
                            className="config-save-btn key-save-btn"
                            onClick={handleBedrockKeySave}
                            disabled={keySaving || !bedrockAccessKeyId.trim() || !bedrockSecretAccessKey.trim()}
                          >
                            {keySaving ? 'Saving...' : keySaved === 'bedrock' ? 'Saved!' : 'Save Credentials'}
                          </button>
                        </>
                      )}
                    </div>
                  )}
                  {configDraft.provider && (
                    <div className="config-provider-info">
                      {(configDraft.provider === 'ollama') && (
                        <span className="config-env-var">No API key needed (runs locally)</span>
                      )}
                      <div className="config-model-suggestions">
                        <span className="config-hint">Popular models:</span>
                        {configDraft.provider === 'openai' && (
                          <div className="model-chips">
                            {['gpt-4o', 'gpt-4o-mini', 'o1', 'o3-mini'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'anthropic' && (
                          <div className="model-chips">
                            {['claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'google' && (
                          <div className="model-chips">
                            {['gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-pro'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'mistral' && (
                          <div className="model-chips">
                            {['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'deepseek' && (
                          <div className="model-chips">
                            {['deepseek-chat', 'deepseek-coder'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'groq' && (
                          <div className="model-chips">
                            {['llama-3.3-70b-versatile', 'mixtral-8x7b-32768'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'ollama' && (
                          <div className="model-chips">
                            {['ollama/llama3.2', 'ollama/mistral', 'ollama/codellama'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'cohere' && (
                          <div className="model-chips">
                            {['command-r-plus', 'command-r', 'command-light'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                        {configDraft.provider === 'bedrock' && (
                          <div className="model-chips">
                            {['bedrock/anthropic.claude-sonnet-4-20250514-v1:0', 'bedrock/anthropic.claude-haiku-4-5-20251001-v1:0', 'bedrock/amazon.titan-text-express-v1'].map((m) => (
                              <button key={m} className="model-chip" onClick={() => setConfigDraft({ ...configDraft, strong: m })}>{m}</button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Model Tiers */}
                <div className="config-section">
                  <h3>Model Tiers</h3>
                  <p className="config-hint">
                    Agent One routes tasks to different models based on complexity.
                  </p>
                  <div className="config-field">
                    <label>
                      Strong Model
                      <span className="field-hint">Complex reasoning, code generation</span>
                    </label>
                    <input
                      type="text"
                      value={configDraft.strong}
                      onChange={(e) => setConfigDraft({ ...configDraft, strong: e.target.value })}
                      placeholder="e.g. claude-sonnet-4-20250514"
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Fast Model
                      <span className="field-hint">Planning, memory queries, simple tasks</span>
                    </label>
                    <input
                      type="text"
                      value={configDraft.fast}
                      onChange={(e) => setConfigDraft({ ...configDraft, fast: e.target.value })}
                      placeholder="e.g. claude-haiku-4-5-20251001"
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Embedding Model
                      <span className="field-hint">Memory embeddings and similarity search</span>
                    </label>
                    <input
                      type="text"
                      value={configDraft.embedding}
                      onChange={(e) => setConfigDraft({ ...configDraft, embedding: e.target.value })}
                      placeholder="e.g. text-embedding-3-small"
                    />
                  </div>
                </div>

                {/* Fallback Chain */}
                <div className="config-section">
                  <h3>Fallback Chain</h3>
                  <p className="config-hint">
                    If the primary model fails, these models are tried in order (comma-separated).
                  </p>
                  <div className="config-field">
                    <label>Fallback Models</label>
                    <input
                      type="text"
                      value={configDraft.fallback_chain}
                      onChange={(e) => setConfigDraft({ ...configDraft, fallback_chain: e.target.value })}
                      placeholder="e.g. gpt-4o, claude-sonnet-4-20250514"
                    />
                  </div>
                </div>

                {/* Save Button */}
                <div className="config-actions">
                  <button
                    className="config-save-btn"
                    onClick={handleConfigSave}
                    disabled={configSaving}
                  >
                    {configSaving ? 'Saving...' : configSaved ? 'Saved!' : 'Save Configuration'}
                  </button>
                  <button
                    className="config-reset-btn"
                    onClick={() => {
                      if (config) {
                        setConfigDraft({
                          strong: config.models.strong,
                          fast: config.models.fast,
                          embedding: config.models.embedding,
                          fallback_chain: config.models.fallback_chain.join(', '),
                          provider: configDraft.provider,
                          apiKey: '',
                        });
                      }
                    }}
                  >
                    Reset
                  </button>
                </div>
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>
                {online ? 'Loading configuration...' : 'Connect to the API to see configuration.'}
              </p>
            )}
          </div>
        )}
      </main>

      {/* Trace panel */}
      <aside className={`trace-panel ${traceOpen ? '' : 'hidden'}`}>
        <div className="trace-header">
          <h2>Reasoning Trace</h2>
          <button className="trace-close" onClick={() => setTraceOpen(false)}>
            &#x2715;
          </button>
        </div>
        <div className="trace-steps">
          {traceSteps.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', padding: '12px' }}>
              Trace steps will appear here as the agent works.
            </p>
          ) : (
            traceSteps.map((step, i) => (
              <div key={i} className={`trace-step ${step.step_type}`}>
                <div className="trace-step-type">{step.step_type}</div>
                <div className="trace-step-content">
                  {step.content.length > 200
                    ? step.content.slice(0, 200) + '...'
                    : step.content}
                </div>
                <div className="trace-step-agent">{step.agent_id.slice(0, 12)}</div>
              </div>
            ))
          )}
          <div ref={traceEndRef} />
        </div>
      </aside>
    </div>
    </>
  );
}

export default App
