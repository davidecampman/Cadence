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
  fetchModels,
  uploadSkill,
  uninstallSkill,
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
    const saved = localStorage.getItem('sentinel-theme');
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
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelFilter, setModelFilter] = useState('');
  const [modelTarget, setModelTarget] = useState<'strong' | 'fast' | 'embedding'>('strong');
  const [skillUploading, setSkillUploading] = useState(false);
  const [skillMessage, setSkillMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [disabledTools, setDisabledTools] = useState<Set<string>>(() => {
    const saved = localStorage.getItem('sentinel-disabled-tools');
    return saved ? new Set(JSON.parse(saved)) : new Set();
  });
  const [configTab, setConfigTab] = useState<'providers' | 'budget' | 'agents' | 'memory' | 'execution'>('providers');
  const [configDraft, setConfigDraft] = useState<{
    strong: string;
    fast: string;
    embedding: string;
    fallback_chain: string;
    bedrock_enabled: boolean;
    bedrock_region: string;
    provider: string;
    apiKey: string;
    timeout_seconds: number;
    restrict_network: boolean;
    max_memory_mb: number;
    max_cpu_seconds: number;
    blocked_commands: string;
    max_tokens_per_task: number;
    max_tokens_per_session: number;
    warn_at_percentage: number;
    max_depth: number;
    max_parallel: number;
    loop_detection_window: number;
    max_iterations_per_task: number;
    decay_rate: number;
    max_results: number;
    similarity_threshold: number;
  } | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const traceEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const skillFileRef = useRef<HTMLInputElement>(null);

  // Sync theme class on body
  useEffect(() => {
    document.body.classList.toggle('light-mode', lightMode);
    localStorage.setItem('sentinel-theme', lightMode ? 'light' : 'dark');
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
        bedrock_enabled: config.models.bedrock?.enabled ?? false,
        bedrock_region: config.models.bedrock?.region ?? 'us-east-1',
        provider: '',
        apiKey: '',
        timeout_seconds: config.execution?.timeout_seconds ?? 120,
        restrict_network: config.execution?.restrict_network ?? false,
        max_memory_mb: config.execution?.max_memory_mb ?? 512,
        max_cpu_seconds: config.execution?.max_cpu_seconds ?? 60,
        blocked_commands: config.execution?.blocked_commands?.join(', ') ?? '',
        max_tokens_per_task: config.budget.max_tokens_per_task,
        max_tokens_per_session: config.budget.max_tokens_per_session,
        warn_at_percentage: config.budget.warn_at_percentage,
        max_depth: config.agents.max_depth,
        max_parallel: config.agents.max_parallel,
        loop_detection_window: config.agents.loop_detection_window,
        max_iterations_per_task: config.agents.max_iterations_per_task,
        decay_rate: config.memory.decay_rate,
        max_results: config.memory.max_results,
        similarity_threshold: config.memory.similarity_threshold,
      });
    }
  }, [config, configDraft]);

  // Fetch available models when provider or tier changes
  useEffect(() => {
    const provider = configDraft?.provider;
    if (!provider) {
      setProviderModels([]);
      return;
    }
    setModelsLoading(true);
    setModelFilter('');
    fetchModels(provider, modelTarget)
      .then((res) => setProviderModels(res.models))
      .catch(() => setProviderModels([]))
      .finally(() => setModelsLoading(false));
  }, [configDraft?.provider, modelTarget]);

  const handleConfigSave = useCallback(async () => {
    if (!configDraft) return;
    setConfigSaving(true);
    setConfigSaved(false);
    try {
      const fallback = configDraft.fallback_chain
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const blockedCmds = configDraft.blocked_commands
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const newConfig = await updateConfig({
        models: {
          strong: configDraft.strong,
          fast: configDraft.fast,
          embedding: configDraft.embedding,
          fallback_chain: fallback,
          bedrock: {
            enabled: configDraft.bedrock_enabled,
            region: configDraft.bedrock_region,
          },
        },
        execution: {
          timeout_seconds: configDraft.timeout_seconds,
          restrict_network: configDraft.restrict_network,
          max_memory_mb: configDraft.max_memory_mb,
          max_cpu_seconds: configDraft.max_cpu_seconds,
          blocked_commands: blockedCmds,
        },
        budget: {
          max_tokens_per_task: configDraft.max_tokens_per_task,
          max_tokens_per_session: configDraft.max_tokens_per_session,
          warn_at_percentage: configDraft.warn_at_percentage,
        },
        agents: {
          max_depth: configDraft.max_depth,
          max_parallel: configDraft.max_parallel,
          loop_detection_window: configDraft.loop_detection_window,
          max_iterations_per_task: configDraft.max_iterations_per_task,
        },
        memory: {
          decay_rate: configDraft.decay_rate,
          max_results: configDraft.max_results,
          similarity_threshold: configDraft.similarity_threshold,
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
      // Refresh stored keys info and available models
      const updated = await fetchKeys();
      setKeysInfo(updated);
      fetchModels(configDraft.provider)
        .then((res) => setProviderModels(res.models))
        .catch(() => {});
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

  const handleSkillUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSkillUploading(true);
    setSkillMessage(null);
    try {
      const result = await uploadSkill(file);
      setSkillMessage({ type: 'success', text: `Installed "${result.skill.name}" v${result.skill.version}` });
      const updated = await fetchSkills();
      setSkills(updated);
    } catch (err) {
      setSkillMessage({ type: 'error', text: err instanceof Error ? err.message : 'Upload failed' });
    } finally {
      setSkillUploading(false);
      // Reset input so same file can be re-uploaded
      if (skillFileRef.current) skillFileRef.current.value = '';
    }
  }, []);

  const handleSkillUninstall = useCallback(async (name: string) => {
    if (!confirm(`Uninstall skill "${name}"? This will delete its files from disk.`)) return;
    setSkillMessage(null);
    try {
      await uninstallSkill(name);
      setSkillMessage({ type: 'success', text: `Uninstalled "${name}"` });
      const updated = await fetchSkills();
      setSkills(updated);
    } catch (err) {
      setSkillMessage({ type: 'error', text: err instanceof Error ? err.message : 'Uninstall failed' });
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
          <h1>Sentinel</h1>
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
                  <h2>Sentinel</h2>
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
                          {msg.role === 'user' ? 'You' : 'Sentinel'}
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
                        <span className="message-sender">Sentinel</span>
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
                  placeholder="Ask Sentinel anything..."
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
            <div className="skills-header">
              <h2>Loaded Skills</h2>
              <input
                ref={skillFileRef}
                type="file"
                accept=".zip"
                style={{ display: 'none' }}
                onChange={handleSkillUpload}
              />
              <button
                className="config-save-btn"
                onClick={() => skillFileRef.current?.click()}
                disabled={skillUploading || !online}
                style={{ marginLeft: 'auto', padding: '6px 16px', fontSize: '13px' }}
              >
                {skillUploading ? 'Uploading...' : 'Upload Skill (.zip)'}
              </button>
            </div>
            {skillMessage && (
              <div className={`skill-message ${skillMessage.type}`}>
                {skillMessage.text}
                <button
                  style={{ background: 'none', border: 'none', cursor: 'pointer', marginLeft: 8, color: 'inherit', fontSize: '14px' }}
                  onClick={() => setSkillMessage(null)}
                >
                  &#x2715;
                </button>
              </div>
            )}
            {skills.map((skill) => (
              <div key={skill.name} className="info-card skill-card">
                <div className="skill-card-content">
                  <h3>{skill.name}</h3>
                  <p>{skill.description}</p>
                  <span className="tag">v{skill.version}</span>
                </div>
                <button
                  className="skill-uninstall-btn"
                  onClick={() => handleSkillUninstall(skill.name)}
                  title={`Uninstall ${skill.name}`}
                >
                  &#x1F5D1;
                </button>
              </div>
            ))}
            {skills.length === 0 && (
              <p style={{ color: 'var(--text-muted)' }}>
                {online ? 'No skills loaded. Upload a skill zip to get started.' : 'Connect to the API to see loaded skills.'}
              </p>
            )}
          </div>
        )}

        {view === 'config' && (
          <div className="info-panel">
            <h2>Configuration</h2>

            {/* Tab bar */}
            <div className="config-tabs">
              {([
                ['providers', 'Providers'],
                ['budget', 'Token Budget'],
                ['agents', 'Agents'],
                ['memory', 'Memory'],
                ['execution', 'Execution'],
              ] as const).map(([key, label]) => (
                <button
                  key={key}
                  className={`config-tab${configTab === key ? ' active' : ''}`}
                  onClick={() => setConfigTab(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            {configDraft ? (
              <div className="config-form">

                {/* ═══ PROVIDERS TAB ═══ */}
                {configTab === 'providers' && (
                  <>
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
                      <option value="openrouter">OpenRouter</option>
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
                        <span className="config-hint">
                          Available models{providerModels.length > 0 ? ` (${providerModels.length})` : ''}:
                        </span>
                        {modelsLoading ? (
                          <span className="config-hint" style={{ fontStyle: 'italic' }}>Loading models...</span>
                        ) : providerModels.length > 0 ? (
                          <>
                            <div className="model-target-selector">
                              <span className="config-hint" style={{ marginRight: 8 }}>Assign to:</span>
                              {(['strong', 'fast', 'embedding'] as const).map((tier) => (
                                <button
                                  key={tier}
                                  className={`model-target-btn${modelTarget === tier ? ' active' : ''}`}
                                  onClick={() => setModelTarget(tier)}
                                >
                                  {tier.charAt(0).toUpperCase() + tier.slice(1)}
                                </button>
                              ))}
                            </div>
                            {providerModels.length > 10 && (
                              <input
                                type="text"
                                value={modelFilter}
                                onChange={(e) => setModelFilter(e.target.value)}
                                placeholder="Filter models..."
                                className="model-filter-input"
                                style={{ margin: '6px 0', padding: '4px 8px', fontSize: '0.85em', width: '100%', boxSizing: 'border-box' }}
                              />
                            )}
                            <div className="model-chips">
                              {providerModels
                                .filter((m) => !modelFilter || m.toLowerCase().includes(modelFilter.toLowerCase()))
                                .slice(0, 30)
                                .map((m) => (
                                  <button
                                    key={m}
                                    className={`model-chip${m === configDraft[modelTarget] ? ' selected' : ''}`}
                                    onClick={() => setConfigDraft({ ...configDraft, [modelTarget]: m })}
                                  >
                                    {m}
                                  </button>
                                ))}
                              {providerModels.filter((m) => !modelFilter || m.toLowerCase().includes(modelFilter.toLowerCase())).length > 30 && (
                                <span className="config-hint" style={{ fontSize: '0.8em' }}>
                                  ...and {providerModels.filter((m) => !modelFilter || m.toLowerCase().includes(modelFilter.toLowerCase())).length - 30} more (use filter)
                                </span>
                              )}
                            </div>
                          </>
                        ) : (
                          <span className="config-hint" style={{ fontStyle: 'italic' }}>No models found for this provider</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Model Tiers */}
                <div className="config-section">
                  <h3>Model Tiers</h3>
                  <p className="config-hint">
                    Sentinel routes tasks to different models based on complexity.
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

                {/* AWS Bedrock */}
                <div className="config-section">
                  <h3>AWS Bedrock</h3>
                  <p className="config-hint">
                    When enabled, bare Claude model names (e.g. claude-sonnet-4-20250514) are
                    automatically routed through AWS Bedrock. Save Bedrock credentials above first.
                  </p>
                  <div className="config-field">
                    <label className="bedrock-toggle-label">
                      <input
                        type="checkbox"
                        checked={configDraft.bedrock_enabled}
                        onChange={(e) => setConfigDraft({ ...configDraft, bedrock_enabled: e.target.checked })}
                      />
                      <span>Enable Bedrock routing</span>
                    </label>
                  </div>
                  {configDraft.bedrock_enabled && (
                    <div className="config-field">
                      <label>
                        Region
                        <span className="field-hint">AWS region for Bedrock API calls</span>
                      </label>
                      <input
                        type="text"
                        value={configDraft.bedrock_region}
                        onChange={(e) => setConfigDraft({ ...configDraft, bedrock_region: e.target.value })}
                        placeholder="e.g. us-east-1"
                      />
                    </div>
                  )}
                </div>
                  </>
                )}

                {/* ═══ BUDGET TAB ═══ */}
                {configTab === 'budget' && (
                  <>
                <div className="config-section">
                  <h3>Token Limits</h3>
                  <p className="config-hint">
                    Control how many tokens agents can consume per task and per session.
                    The session budget is enforced across all agent runs, planning, synthesis, and evaluation.
                  </p>
                  <div className="config-field">
                    <label>
                      Max Tokens per Task
                      <span className="field-hint">Circuit breaker for a single agent run</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.max_tokens_per_task}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_tokens_per_task: parseInt(e.target.value) || 0 })}
                      min={1000}
                      step={10000}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Max Tokens per Session
                      <span className="field-hint">Total budget across all requests in a session</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.max_tokens_per_session}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_tokens_per_session: parseInt(e.target.value) || 0 })}
                      min={1000}
                      step={50000}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Warning Threshold (%)
                      <span className="field-hint">Emit a trace warning when this % of session budget is used</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.warn_at_percentage}
                      onChange={(e) => setConfigDraft({ ...configDraft, warn_at_percentage: parseInt(e.target.value) || 0 })}
                      min={0}
                      max={100}
                    />
                  </div>
                </div>
                  </>
                )}

                {/* ═══ AGENTS TAB ═══ */}
                {configTab === 'agents' && (
                  <>
                <div className="config-section">
                  <h3>Orchestration</h3>
                  <p className="config-hint">
                    Tune how the orchestrator decomposes tasks and manages specialist agents.
                  </p>
                  <div className="config-field">
                    <label>
                      Max Parallel Agents
                      <span className="field-hint">How many agents can run concurrently on subtasks</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.max_parallel}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_parallel: parseInt(e.target.value) || 1 })}
                      min={1}
                      max={16}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Max Delegation Depth
                      <span className="field-hint">How deep agents can delegate to sub-agents</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.max_depth}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_depth: parseInt(e.target.value) || 1 })}
                      min={1}
                      max={10}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Max Iterations per Task
                      <span className="field-hint">Circuit breaker: max think-act-observe loops per agent</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.max_iterations_per_task}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_iterations_per_task: parseInt(e.target.value) || 1 })}
                      min={1}
                      max={100}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Loop Detection Window
                      <span className="field-hint">Number of recent outputs to compare for stuck-loop detection</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.loop_detection_window}
                      onChange={(e) => setConfigDraft({ ...configDraft, loop_detection_window: parseInt(e.target.value) || 2 })}
                      min={2}
                      max={20}
                    />
                  </div>
                </div>
                  </>
                )}

                {/* ═══ MEMORY TAB ═══ */}
                {configTab === 'memory' && (
                  <>
                <div className="config-section">
                  <h3>Memory Retrieval</h3>
                  <p className="config-hint">
                    Configure how the tiered memory system stores and retrieves memories.
                    Each agent now has its own private namespace; these settings apply globally.
                  </p>
                  <div className="config-field">
                    <label>
                      Decay Rate
                      <span className="field-hint">How fast memories lose relevance over time (per day, 0 = no decay)</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.decay_rate}
                      onChange={(e) => setConfigDraft({ ...configDraft, decay_rate: parseFloat(e.target.value) || 0 })}
                      min={0}
                      max={1}
                      step={0.01}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Similarity Threshold
                      <span className="field-hint">Minimum relevance score (0-1) for a memory to be returned</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.similarity_threshold}
                      onChange={(e) => setConfigDraft({ ...configDraft, similarity_threshold: parseFloat(e.target.value) || 0 })}
                      min={0}
                      max={1}
                      step={0.05}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Max Results
                      <span className="field-hint">Default maximum memories returned per query</span>
                    </label>
                    <input
                      type="number"
                      value={configDraft.max_results}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_results: parseInt(e.target.value) || 1 })}
                      min={1}
                      max={50}
                    />
                  </div>
                </div>
                  </>
                )}

                {/* ═══ EXECUTION TAB ═══ */}
                {configTab === 'execution' && (
                  <>
                <div className="config-section">
                  <h3>Tool Permissions</h3>
                  <p className="config-hint">
                    Enable or disable individual tools. Disabled tools will not be offered to agents.
                    Changes are saved to your browser.
                  </p>
                  {(() => {
                    const categories: Record<string, ToolInfo[]> = {};
                    tools.forEach((t) => {
                      let cat = 'Other';
                      if (['read_file', 'write_file', 'list_files', 'search_files'].includes(t.name)) cat = 'File Operations';
                      else if (['execute_code', 'shell'].includes(t.name)) cat = 'Code Execution';
                      else if (['memory_save', 'memory_query', 'memory_delete'].includes(t.name)) cat = 'Memory';
                      else if (['web_fetch', 'http_request'].includes(t.name)) cat = 'Web & HTTP';
                      else if (['git_status', 'git_diff', 'git_commit', 'git_log'].includes(t.name)) cat = 'Git';
                      else if (['sql_query'].includes(t.name)) cat = 'Database';
                      else if (['regex_replace', 'diff_patch', 'summarize_text'].includes(t.name)) cat = 'Text Processing';
                      else if (['screenshot', 'image_describe'].includes(t.name)) cat = 'Vision';
                      else if (['env_info', 'install_package', 'check_dependency'].includes(t.name)) cat = 'Environment';
                      else if (['scratch_write', 'scratch_read'].includes(t.name)) cat = 'Scratchpad';
                      else if (['delegate'].includes(t.name)) cat = 'Agent Delegation';
                      (categories[cat] = categories[cat] || []).push(t);
                    });
                    const catOrder = [
                      'File Operations', 'Code Execution', 'Git', 'Web & HTTP',
                      'Database', 'Text Processing', 'Memory', 'Vision',
                      'Environment', 'Scratchpad', 'Agent Delegation', 'Other',
                    ];
                    return catOrder
                      .filter((cat) => categories[cat]?.length)
                      .map((cat) => (
                        <div key={cat} className="tool-category">
                          <div className="tool-category-header">{cat}</div>
                          <div className="tool-toggles">
                            {categories[cat].map((t) => (
                              <label key={t.name} className="tool-toggle-label" title={t.description}>
                                <input
                                  type="checkbox"
                                  checked={!disabledTools.has(t.name)}
                                  onChange={(e) => {
                                    const next = new Set(disabledTools);
                                    if (e.target.checked) {
                                      next.delete(t.name);
                                    } else {
                                      next.add(t.name);
                                    }
                                    setDisabledTools(next);
                                    localStorage.setItem('sentinel-disabled-tools', JSON.stringify([...next]));
                                  }}
                                />
                                <span className="tool-toggle-name">{t.name}</span>
                                <span className={`tool-toggle-tier tier-${t.permission_tier}`}>{t.permission_tier}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      ));
                  })()}
                  {tools.length === 0 && (
                    <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                      Connect to the API to manage tool permissions.
                    </p>
                  )}
                </div>

                <div className="config-section">
                  <h3>Sandbox Limits</h3>
                  <p className="config-hint">
                    Controls for code execution, shell commands, and tool sandboxing.
                  </p>
                  <div className="config-field">
                    <label>
                      Timeout (seconds)
                      <span className="field-hint">Max execution time per tool call</span>
                    </label>
                    <input
                      type="number"
                      min={5}
                      max={600}
                      value={configDraft.timeout_seconds}
                      onChange={(e) => setConfigDraft({ ...configDraft, timeout_seconds: parseInt(e.target.value) || 120 })}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Max Memory (MB)
                      <span className="field-hint">Memory limit per subprocess</span>
                    </label>
                    <input
                      type="number"
                      min={64}
                      max={8192}
                      value={configDraft.max_memory_mb}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_memory_mb: parseInt(e.target.value) || 512 })}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      Max CPU (seconds)
                      <span className="field-hint">CPU time limit per subprocess</span>
                    </label>
                    <input
                      type="number"
                      min={5}
                      max={300}
                      value={configDraft.max_cpu_seconds}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_cpu_seconds: parseInt(e.target.value) || 60 })}
                    />
                  </div>
                  <div className="config-field">
                    <label className="bedrock-toggle-label">
                      <input
                        type="checkbox"
                        checked={configDraft.restrict_network}
                        onChange={(e) => setConfigDraft({ ...configDraft, restrict_network: e.target.checked })}
                      />
                      <span>Restrict network access (Linux only — uses namespace isolation)</span>
                    </label>
                  </div>
                  <div className="config-field">
                    <label>
                      Blocked Commands
                      <span className="field-hint">Comma-separated patterns that are always rejected</span>
                    </label>
                    <input
                      type="text"
                      value={configDraft.blocked_commands}
                      onChange={(e) => setConfigDraft({ ...configDraft, blocked_commands: e.target.value })}
                      placeholder="rm -rf /, mkfs, dd if=, ..."
                    />
                  </div>
                </div>
                  </>
                )}

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
                          bedrock_enabled: config.models.bedrock?.enabled ?? false,
                          bedrock_region: config.models.bedrock?.region ?? 'us-east-1',
                          provider: configDraft.provider,
                          apiKey: '',
                          timeout_seconds: config.execution?.timeout_seconds ?? 120,
                          restrict_network: config.execution?.restrict_network ?? false,
                          max_memory_mb: config.execution?.max_memory_mb ?? 512,
                          max_cpu_seconds: config.execution?.max_cpu_seconds ?? 60,
                          blocked_commands: config.execution?.blocked_commands?.join(', ') ?? '',
                          max_tokens_per_task: config.budget.max_tokens_per_task,
                          max_tokens_per_session: config.budget.max_tokens_per_session,
                          warn_at_percentage: config.budget.warn_at_percentage,
                          max_depth: config.agents.max_depth,
                          max_parallel: config.agents.max_parallel,
                          loop_detection_window: config.agents.loop_detection_window,
                          max_iterations_per_task: config.agents.max_iterations_per_task,
                          decay_rate: config.memory.decay_rate,
                          max_results: config.memory.max_results,
                          similarity_threshold: config.memory.similarity_threshold,
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
