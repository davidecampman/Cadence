import { useState, useEffect, useRef, useCallback } from 'react'
import {
  sendMessage,
  fetchConfig,
  fetchTools,
  fetchSkills,
  checkHealth,
  connectWebSocket,
  type ChatResponse,
  type TraceStep,
  type ToolInfo,
  type SkillInfo,
  type WsMessage,
} from './api'
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
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const traceEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
          <span className={`status-dot ${online ? 'online' : 'offline'}`} />
          {online ? 'API Connected' : 'API Disconnected'}
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
            <h2>Configuration</h2>
            {config ? (
              <pre className="config-json">{JSON.stringify(config, null, 2)}</pre>
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
  );
}

export default App
