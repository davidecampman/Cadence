import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  sendMessageStream,
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
  fileDownloadUrl,
  revealFile,
  fetchChats,
  fetchChat,
  createChat as apiCreateChat,
  updateChat as apiUpdateChat,
  deleteChat as apiDeleteChat,
  addChatMessage,
  initiateOAuth,
  completeOAuth,
  fetchOAuthStatus,
  revokeOAuth,
  fetchDag,
  type AppConfig,
  type TraceStep,
  type ToolInfo,
  type SkillInfo,
  type WsMessage,
  type KeysResponse,
  type ImageAttachment,
  type OAuthStatus,
  type DagGraph,
} from './api'
import BackgroundCanvas from './BackgroundCanvas'
import TaskGraph from './TaskGraph'
import './App.css'

type View = 'chat' | 'tools' | 'skills' | 'dag' | 'config';

interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: number;
  duration_ms?: number;
  trace_steps?: TraceStep[];
  attachments?: ImageAttachment[];
}

interface Chat {
  id: string;
  title: string;
  messages: ChatMessage[];
  sessionId?: string;
  createdAt: number;
}

function loadChatsFromLocalStorage(): Chat[] {
  try {
    const raw = localStorage.getItem('cadence-chats');
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveChatsToLocalStorage(chats: Chat[]) {
  localStorage.setItem('cadence-chats', JSON.stringify(chats));
}

function chatTitle(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === 'user');
  if (!first) return 'New Chat';
  const text = first.content.trim();
  return text.length > 40 ? text.slice(0, 40) + '...' : text;
}

/** Strip LLM citation artifacts (e.g. 【cite】turn0view0【】) that leak into responses. */
function cleanCitations(text: string): string {
  // Remove 【…】 bracketed citation markers from web-search-augmented LLMs
  text = text.replace(/\u3010[^\u3011]*\u3011/g, '');
  // Remove orphaned turn-reference tokens (turn0view0, turn0search3, etc.)
  text = text.replace(/\bturn\d+\w*\d*\b/gi, '');
  // Collapse runs of whitespace left behind
  text = text.replace(/ {2,}/g, ' ');
  return text;
}

/** Custom ReactMarkdown components: download links trigger downloads, others open in new tab. */
const markdownComponents = {
  a: ({ href, children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => {
    if (href && href.includes('/api/files/download')) {
      try {
        const url = new URL(href, window.location.origin);
        const filePath = url.searchParams.get('path') || '';
        const fileName = filePath.split('/').pop() || 'download';
        return <a href={href} download={fileName} {...props}>{children}</a>;
      } catch {
        // Fall through to default link behavior
      }
    }
    return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>;
  },
};

/** Render message content, replacing [[FILE:/path]] markers with download/reveal buttons. */
function renderMessageContent(content: string, role: string = 'user') {
  // Clean citation artifacts from agent responses before rendering
  const cleaned = role === 'agent' ? cleanCitations(content) : content;

  const FILE_RE = /\[\[FILE:(.*?)\]\]/g;
  const parts: (string | { path: string })[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = FILE_RE.exec(cleaned)) !== null) {
    if (m.index > last) parts.push(cleaned.slice(last, m.index));
    parts.push({ path: m[1] });
    last = m.index + m[0].length;
  }
  if (last < cleaned.length) parts.push(cleaned.slice(last));

  // If no file markers found, render based on role
  if (parts.length === 1 && typeof parts[0] === 'string') {
    if (role === 'agent') {
      return <ReactMarkdown components={markdownComponents}>{cleaned}</ReactMarkdown>;
    }
    return <>{cleaned}</>;
  }

  return (
    <>
      {parts.map((part, i) => {
        if (typeof part === 'string') {
          if (role === 'agent') {
            return <ReactMarkdown key={i} components={markdownComponents}>{part}</ReactMarkdown>;
          }
          return <span key={i}>{part}</span>;
        }
        const fileName = part.path.split('/').pop() || part.path;
        return (
          <span key={i} className="file-link-group">
            <a
              href={fileDownloadUrl(part.path)}
              download={fileName}
              className="file-download-btn"
              title={`Download ${part.path}`}
            >
              ⬇ {fileName}
            </a>
            <button
              className="file-reveal-btn"
              title={`Open folder containing ${fileName}`}
              onClick={() => revealFile(part.path)}
            >
              📂
            </button>
          </span>
        );
      })}
    </>
  );
}

function App() {
  const [lightMode, setLightMode] = useState(() => {
    const saved = localStorage.getItem('cadence-theme');
    return saved === 'light';
  });
  const [view, setView] = useState<View>('chat');
  const [chats, setChats] = useState<Chat[]>(loadChatsFromLocalStorage);
  const [activeChatId, setActiveChatId] = useState<string | null>(() => {
    const saved = loadChatsFromLocalStorage();
    return saved.length > 0 ? saved[0].id : null;
  });
  const [_backendReady, setBackendReady] = useState(false);
  const [input, setInput] = useState('');
  // Per-chat loading: tracks which chat IDs are currently processing
  const [loadingChats, setLoadingChats] = useState<Set<string>>(new Set());
  // Per-chat streaming status line shown below the typing indicator
  const [streamingStatus, setStreamingStatus] = useState<Record<string, string>>({});
  // AbortControllers keyed by chatId so each chat's request can be cancelled
  const abortControllersRef = useRef<Record<string, AbortController>>({});
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
  const [oauthStatus, setOauthStatus] = useState<OAuthStatus | null>(null);
  const [oauthLoading, setOauthLoading] = useState(false);
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelFilter, setModelFilter] = useState('');
  const [modelTarget, setModelTarget] = useState<'strong' | 'fast' | 'embedding'>('strong');
  const [skillUploading, setSkillUploading] = useState(false);
  const [skillMessage, setSkillMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [disabledTools, setDisabledTools] = useState<Set<string>>(() => {
    const saved = localStorage.getItem('cadence-disabled-tools');
    return saved ? new Set(JSON.parse(saved)) : new Set();
  });
  const [configTab, setConfigTab] = useState<'providers' | 'budget' | 'agents' | 'memory' | 'execution' | 'conversation'>('providers');
  const [contextTurns, setContextTurns] = useState(0);
  const [maxContextTurns, setMaxContextTurns] = useState(50);
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
    max_history_turns: number;
    compression_enabled: boolean;
    compression_threshold: number;
  } | null>(null);

  const [dagGraph, setDagGraph] = useState<DagGraph>({ nodes: [], edges: [] });

  const [attachments, setAttachments] = useState<ImageAttachment[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const traceEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const skillFileRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Derived: active chat and its messages
  const activeChat = chats.find((c) => c.id === activeChatId) ?? null;
  const messages = activeChat?.messages ?? [];
  // loading is true only for the currently-visible chat
  const loading = activeChatId ? loadingChats.has(activeChatId) : false;

  // Always keep localStorage as a fallback cache
  useEffect(() => {
    saveChatsToLocalStorage(chats);
  }, [chats]);

  // Load chats from backend on mount (if server is available)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const serverChats = await fetchChats();
        if (cancelled) return;
        if (serverChats.length > 0) {
          // Load full messages for each chat
          const fullChats: Chat[] = await Promise.all(
            serverChats.map(async (sc) => {
              const full = await fetchChat(sc.id);
              return {
                id: full.id,
                title: full.title,
                sessionId: full.session_id ?? undefined,
                createdAt: full.created_at * 1000,
                messages: (full.messages ?? []).map((m) => ({
                  id: m.id,
                  role: m.role as 'user' | 'agent',
                  content: m.content,
                  timestamp: m.timestamp,
                  duration_ms: m.duration_ms ?? undefined,
                  trace_steps: (m.trace_steps as TraceStep[] | null) ?? undefined,
                })),
              };
            })
          );
          if (!cancelled) {
            setChats(fullChats);
            setActiveChatId(fullChats[0]?.id ?? null);
          }
        } else {
          // Server is online but has no chats — fresh database.
          // Clear stale localStorage to avoid showing old conversations.
          if (!cancelled) {
            setChats([]);
            setActiveChatId(null);
            localStorage.removeItem('cadence-chats');
          }
        }
        if (!cancelled) setBackendReady(true);
      } catch {
        // Server offline — keep using localStorage chats
        if (!cancelled) setBackendReady(false);
      }
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const createNewChat = useCallback(() => {
    const newChat: Chat = {
      id: crypto.randomUUID(),
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    setView('chat');
    setInput('');
    // Persist to backend
    apiCreateChat(newChat.id, newChat.title, newChat.createdAt / 1000).catch(() => {});
  }, []);

  const deleteChat = useCallback((chatId: string) => {
    setChats((prev) => {
      const next = prev.filter((c) => c.id !== chatId);
      if (activeChatId === chatId) {
        setActiveChatId(next.length > 0 ? next[0].id : null);
      }
      return next;
    });
    // Persist to backend
    apiDeleteChat(chatId).catch(() => {});
  }, [activeChatId]);

  const switchChat = useCallback((chatId: string) => {
    setActiveChatId(chatId);
    setView('chat');
  }, []);

  // Sync theme class on body
  useEffect(() => {
    document.body.classList.toggle('light-mode', lightMode);
    localStorage.setItem('cadence-theme', lightMode ? 'light' : 'dark');
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
    fetchOAuthStatus().then(setOauthStatus).catch(() => {});
  }, []);

  // Handle OAuth callback — when redirected back from OpenAI with ?code=...&state=...
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const state = params.get('state');
    if (code && state && (window.location.pathname === '/auth/callback' || window.location.pathname === '/oauth/callback')) {
      // Clear the URL params so we don't re-process on refresh
      window.history.replaceState({}, '', '/');
      setView('config');
      setOauthLoading(true);
      completeOAuth(code, state)
        .then(() => fetchOAuthStatus())
        .then((status) => {
          setOauthStatus(status);
          setOauthLoading(false);
        })
        .catch((err) => {
          alert(`OAuth authorization failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
          setOauthLoading(false);
        });
    }
  }, []);

  // WebSocket for live trace + DAG updates
  useEffect(() => {
    const ws = connectWebSocket((msg: WsMessage) => {
      if (msg.type === 'trace') {
        setTraceSteps((prev) => [...prev, msg.data]);
      } else if (msg.type === 'dag_update') {
        setDagGraph(msg.data);
      }
    });
    return () => ws.close();
  }, []);

  // Fetch DAG snapshot whenever the user opens the graph view
  useEffect(() => {
    if (view === 'dag') {
      fetchDag().then(setDagGraph).catch(() => {});
    }
  }, [view]);

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
        max_history_turns: config.conversation?.max_history_turns ?? 50,
        compression_enabled: config.conversation?.compression_enabled ?? true,
        compression_threshold: config.conversation?.compression_threshold ?? 30,
      });
      setMaxContextTurns(config.conversation?.max_history_turns ?? 50);
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
        conversation: {
          max_history_turns: configDraft.max_history_turns,
          compression_enabled: configDraft.compression_enabled,
          compression_threshold: configDraft.compression_threshold,
        },
      });
      setConfig(newConfig);
      setMaxContextTurns(configDraft.max_history_turns);
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

  const handleOAuthConnect = useCallback(async () => {
    setOauthLoading(true);
    try {
      const { authorize_url } = await initiateOAuth();
      // Open the authorization URL in a new window/tab
      window.open(authorize_url, '_blank', 'width=600,height=700');
    } catch (err) {
      alert(`Failed to start OAuth flow: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setOauthLoading(false);
    }
  }, []);

  const handleOAuthDisconnect = useCallback(async () => {
    if (!confirm('Disconnect ChatGPT OAuth? You will need to re-authorize to use subscription-based access.')) return;
    try {
      await revokeOAuth();
      setOauthStatus({ authorized: false, account_id: null, scope: null, expires_at: null });
    } catch (err) {
      alert(`Failed to revoke OAuth: ${err instanceof Error ? err.message : 'Unknown error'}`);
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

  // --- File attachment helpers ---
  const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
  const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB

  const fileToAttachment = useCallback((file: File): Promise<ImageAttachment | null> => {
    return new Promise((resolve) => {
      if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
        resolve(null);
        return;
      }
      if (file.size > MAX_FILE_SIZE) {
        resolve(null);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Strip data URL prefix to get pure base64
        const base64 = result.split(',')[1];
        if (base64) {
          resolve({ data: base64, media_type: file.type, name: file.name });
        } else {
          resolve(null);
        }
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  }, []);

  const addFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const results = await Promise.all(fileArray.map(fileToAttachment));
    const valid = results.filter((r): r is ImageAttachment => r !== null);
    if (valid.length > 0) {
      setAttachments((prev) => [...prev, ...valid]);
    }
  }, [fileToAttachment]);

  const removeAttachment = useCallback((index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  }, [addFiles]);

  // Paste handler for images
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    const imageFiles: File[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) imageFiles.push(file);
      }
    }
    if (imageFiles.length > 0) {
      addFiles(imageFiles);
    }
  }, [addFiles]);

  const handleCancel = useCallback(() => {
    if (!activeChatId) return;
    const controller = abortControllersRef.current[activeChatId];
    if (controller) {
      controller.abort();
      delete abortControllersRef.current[activeChatId];
    }
    setLoadingChats((prev) => { const next = new Set(prev); next.delete(activeChatId); return next; });
    setStreamingStatus((prev) => { const next = { ...prev }; delete next[activeChatId]; return next; });
  }, [activeChatId]);

  const handleSend = useCallback(async (text?: string) => {
    const msg = text || input.trim();
    // Block send only for the active chat if it's already loading
    const activeIsLoading = activeChatId ? loadingChats.has(activeChatId) : false;
    if ((!msg && attachments.length === 0) || activeIsLoading) return;

    const currentAttachments = [...attachments];
    setInput('');
    setAttachments([]);
    setView('chat');

    // If no active chat, create one
    let chatId = activeChatId;
    if (!chatId) {
      const newChat: Chat = {
        id: crypto.randomUUID(),
        title: 'New Chat',
        messages: [],
        createdAt: Date.now(),
      };
      chatId = newChat.id;
      setChats((prev) => [newChat, ...prev]);
      setActiveChatId(chatId);
      apiCreateChat(newChat.id, newChat.title, newChat.createdAt / 1000).catch(() => {});
    }

    const displayContent = currentAttachments.length > 0
      ? (msg || '') + `\n[${currentAttachments.length} image${currentAttachments.length > 1 ? 's' : ''} attached]`
      : msg;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: displayContent,
      timestamp: Date.now() / 1000,
      attachments: currentAttachments.length > 0 ? [...currentAttachments] : undefined,
    };

    // Add user message and update title if first message
    const isFirstMessage = (chats.find((c) => c.id === chatId)?.messages.length ?? 0) === 0;
    const newTitle = isFirstMessage ? chatTitle([userMsg]) : undefined;

    setChats((prev) =>
      prev.map((c) => {
        if (c.id !== chatId) return c;
        const updated = { ...c, messages: [...c.messages, userMsg] };
        if (c.messages.length === 0) updated.title = chatTitle([userMsg]);
        return updated;
      })
    );

    // Mark this specific chat as loading
    setLoadingChats((prev) => new Set(prev).add(chatId!));

    // Persist user message to backend
    addChatMessage(chatId, {
      id: userMsg.id,
      role: userMsg.role,
      content: userMsg.content,
      timestamp: userMsg.timestamp,
    }).catch(() => {});
    if (newTitle) {
      apiUpdateChat(chatId, { title: newTitle }).catch(() => {});
    }

    const currentSessionId = chats.find((c) => c.id === chatId)?.sessionId;

    // Create an AbortController so the user can cancel
    const controller = new AbortController();
    abortControllersRef.current[chatId] = controller;

    const clearLoading = () => {
      setLoadingChats((prev) => { const next = new Set(prev); next.delete(chatId!); return next; });
      setStreamingStatus((prev) => { const next = { ...prev }; delete next[chatId!]; return next; });
      delete abortControllersRef.current[chatId!];
    };

    try {
      const res = await sendMessageStream(
        msg || 'Please look at the attached image(s).',
        {
          onThinking: (_thought, _agentId) => {
            // Thinking steps arrive via WebSocket too; use status line for live feedback
            setStreamingStatus((prev) => ({ ...prev, [chatId!]: 'Thinking...' }));
          },
          onStatus: (status, _agentId) => {
            setStreamingStatus((prev) => ({ ...prev, [chatId!]: status }));
          },
        },
        currentSessionId,
        currentAttachments.length > 0 ? currentAttachments : undefined,
        controller.signal,
      );

      setContextTurns(res.context_turns ?? 0);
      if (res.max_context_turns) setMaxContextTurns(res.max_context_turns);

      const agentMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'agent',
        content: res.response,
        timestamp: Date.now() / 1000,
        duration_ms: res.duration_ms,
        trace_steps: res.trace_steps,
      };

      setChats((prev) =>
        prev.map((c) =>
          c.id === chatId
            ? { ...c, messages: [...c.messages, agentMsg], sessionId: res.session_id }
            : c
        )
      );

      // Persist agent message and session_id to backend
      addChatMessage(chatId, {
        id: agentMsg.id,
        role: agentMsg.role,
        content: agentMsg.content,
        timestamp: agentMsg.timestamp,
        duration_ms: agentMsg.duration_ms,
        trace_steps: agentMsg.trace_steps as Record<string, unknown>[] | undefined,
      }).catch(() => {});
      apiUpdateChat(chatId, { session_id: res.session_id }).catch(() => {});
    } catch (err) {
      const isAbort = err instanceof Error && err.name === 'AbortError';
      if (!isAbort) {
        const errorMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'agent',
          content: `Connection error: ${err instanceof Error ? err.message : 'Unknown error'}. Make sure the API server is running.`,
          timestamp: Date.now() / 1000,
        };
        setChats((prev) =>
          prev.map((c) =>
            c.id === chatId ? { ...c, messages: [...c.messages, errorMsg] } : c
          )
        );
      }
    } finally {
      clearLoading();
    }
  }, [input, loadingChats, activeChatId, chats, attachments]);

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
          <h1>Cadence</h1>
          <div className="version">v0.1.0 — Multi-Agent Framework</div>
        </div>

        <div className="sidebar-nav">
          <div className="sidebar-section">
            <div className="sidebar-section-header">
              <h3>Chats</h3>
              <button
                className="new-chat-btn"
                onClick={createNewChat}
                title="New Chat"
              >
                +
              </button>
            </div>
            <div className="chat-list">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`chat-list-item ${view === 'chat' && activeChatId === chat.id ? 'active' : ''}`}
                  onClick={() => switchChat(chat.id)}
                >
                  <span className="chat-list-title">{chat.title}</span>
                  <button
                    className="chat-delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(chat.id);
                    }}
                    title="Delete chat"
                  >
                    &#x2715;
                  </button>
                </div>
              ))}
              {chats.length === 0 && (
                <div className="chat-list-empty">No chats yet</div>
              )}
            </div>
          </div>

          <div className="sidebar-section">
            <h3>Navigation</h3>
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
              className={`sidebar-item ${view === 'dag' ? 'active' : ''}`}
              onClick={() => setView('dag')}
            >
              <span className="icon">&#x25C7;</span>
              Task Graph
              <span className="badge">{dagGraph.nodes.length}</span>
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
          <span className="sidebar-footer-actions">
            <button
              className="theme-toggle"
              onClick={() => setLightMode((prev) => !prev)}
              title={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}
            >
              {lightMode ? '\u25D1' : '\u25D0'}
            </button>
            <button
              className={`config-gear-btn ${view === 'config' ? 'active' : ''}`}
              onClick={() => setView('config')}
              title="Settings"
            >
              &#x2699;&#xFE0E;
            </button>
          </span>
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
                  <h2>Cadence</h2>
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
                          {msg.role === 'user' ? 'You' : 'Cadence'}
                        </span>
                        <span className="message-time">{formatTime(msg.timestamp)}</span>
                      </div>
                      <div className={`message-body ${msg.role}`}>{renderMessageContent(msg.content, msg.role)}</div>
                      {msg.attachments && msg.attachments.length > 0 && (
                        <div className="message-attachments">
                          {msg.attachments.map((att, idx) => {
                            const fileName = att.name || `attachment-${idx + 1}`;
                            const dataUrl = `data:${att.media_type};base64,${att.data}`;
                            return (
                              <a
                                key={idx}
                                href={dataUrl}
                                download={fileName}
                                className="attachment-download-btn"
                                title={`Download ${fileName}`}
                              >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                                  <polyline points="7 10 12 15 17 10" />
                                  <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                                {fileName}
                              </a>
                            );
                          })}
                        </div>
                      )}
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
                        <span className="message-sender">Cadence</span>
                      </div>
                      <div className="typing-dots">
                        <span />
                        <span />
                        <span />
                      </div>
                      {activeChatId && streamingStatus[activeChatId] && (
                        <div className="streaming-status">{streamingStatus[activeChatId]}</div>
                      )}
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            <div
              className={`chat-input-area${isDragging ? ' dragging' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {isDragging && (
                <div className="drop-overlay">
                  <div className="drop-overlay-text">Drop images here</div>
                </div>
              )}
              {contextTurns > 0 && (
                <div className="context-indicator" title={`${contextTurns} of ${maxContextTurns} conversation turns in context`}>
                  <div className="context-bar">
                    <div
                      className={`context-fill${contextTurns / maxContextTurns > 0.8 ? ' warning' : ''}${contextTurns >= maxContextTurns ? ' full' : ''}`}
                      style={{ width: `${Math.min(100, (contextTurns / maxContextTurns) * 100)}%` }}
                    />
                  </div>
                  <span className="context-label">
                    {Math.round((contextTurns / maxContextTurns) * 100)}%
                  </span>
                </div>
              )}
              {attachments.length > 0 && (
                <div className="attachments-preview">
                  {attachments.map((att, idx) => (
                    <div key={idx} className="attachment-thumb">
                      <img
                        src={`data:${att.media_type};base64,${att.data}`}
                        alt={att.name || 'attachment'}
                      />
                      <button
                        className="attachment-remove"
                        onClick={() => removeAttachment(idx)}
                        title="Remove"
                      >
                        &times;
                      </button>
                      {att.name && <span className="attachment-name">{att.name}</span>}
                    </div>
                  ))}
                </div>
              )}
              <div className="chat-input-wrapper">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/gif,image/webp"
                  multiple
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    if (e.target.files) addFiles(e.target.files);
                    e.target.value = '';
                  }}
                />
                <button
                  className="chat-attach-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  title="Attach images"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
                  </svg>
                </button>
                <textarea
                  ref={inputRef}
                  className="chat-input"
                  placeholder={attachments.length > 0 ? "Add a message about your images..." : "Ask Cadence anything... (paste or drop images)"}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  rows={1}
                  disabled={loading}
                />
                {loading ? (
                  <button
                    className="chat-cancel-btn"
                    onClick={handleCancel}
                    title="Stop generation"
                  >
                    &#x25A0;
                  </button>
                ) : (
                  <button
                    className="chat-send-btn"
                    onClick={() => handleSend()}
                    disabled={(!input.trim() && attachments.length === 0)}
                    title="Send message"
                  >
                    &#x27A4;
                  </button>
                )}
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

        {view === 'dag' && (
          <div className="dag-panel">
            <div className="dag-header">
              <h2>Task Graph</h2>
              <div className="dag-legend">
                {(['pending', 'running', 'completed', 'failed', 'skipped'] as const).map((s) => (
                  <span key={s} className={`dag-legend-item dag-status-${s}`}>
                    &#x25CF; {s}
                  </span>
                ))}
              </div>
            </div>
            <div className="dag-canvas-wrap">
              <TaskGraph graph={dagGraph} />
            </div>
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
                ['conversation', 'Conversation'],
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

                {/* ChatGPT OAuth (Codex) */}
                <div className="config-section oauth-section">
                  <h3>ChatGPT OAuth (Codex)</h3>
                  <p className="config-hint">
                    Connect your ChatGPT Plus or Pro subscription to use OpenAI
                    models at your subscription's flat rate &mdash; no per-token
                    API billing.
                  </p>
                  <p className="config-hint" style={{ marginTop: 4 }}>
                    <strong>Fallback chain:</strong> Cadence tries three endpoints
                    in order: (1) ChatGPT Conversation API (same as the ChatGPT
                    desktop/web app), (2) Codex Responses API (separate quota pool),
                    (3) OpenAI API key (per-token, only if configured). This
                    maximizes your subscription usage before any paid API calls.
                  </p>
                  {oauthStatus?.authorized ? (
                    <div className="oauth-status-connected">
                      <div className="oauth-status-row">
                        <span className="oauth-status-indicator connected" />
                        <span className="oauth-status-label">Connected</span>
                        {oauthStatus.account_id && (
                          <span className="oauth-account-id">({oauthStatus.account_id})</span>
                        )}
                      </div>
                      {oauthStatus.scope && (
                        <div className="oauth-detail">
                          <span className="config-hint">Scopes: {oauthStatus.scope}</span>
                        </div>
                      )}
                      {oauthStatus.expires_at && (
                        <div className="oauth-detail">
                          <span className="config-hint">
                            Token expires: {new Date(oauthStatus.expires_at * 1000).toLocaleString()}
                          </span>
                        </div>
                      )}
                      <div className="oauth-detail">
                        <span className="config-hint">
                          Last resort: {keysInfo?.providers?.openai?.has_key
                            ? 'OpenAI API key configured (used only if both conversation and Codex quotas are exhausted)'
                            : 'No API key fallback \u2014 optionally add one above as a safety net'}
                        </span>
                      </div>
                      <div className="oauth-actions">
                        <button
                          className="oauth-refresh-btn"
                          onClick={() => fetchOAuthStatus().then(setOauthStatus).catch(() => {})}
                        >
                          Refresh Status
                        </button>
                        <button
                          className="oauth-disconnect-btn"
                          onClick={handleOAuthDisconnect}
                        >
                          Disconnect
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="oauth-status-disconnected">
                      <div className="oauth-status-row">
                        <span className="oauth-status-indicator disconnected" />
                        <span className="oauth-status-label">Not connected</span>
                      </div>
                      <p className="config-hint" style={{ margin: '8px 0' }}>
                        Click below to authorize Cadence with your OpenAI account.
                        You'll be redirected to OpenAI's login page.
                        Requires a ChatGPT Plus ($20/mo) or Pro ($200/mo)
                        subscription. Cadence will use two separate subscription
                        quota pools (Conversation + Codex) before touching any
                        API key.
                      </p>
                      <button
                        className="oauth-connect-btn"
                        onClick={handleOAuthConnect}
                        disabled={oauthLoading}
                      >
                        {oauthLoading ? 'Starting...' : 'Connect ChatGPT Account'}
                      </button>
                    </div>
                  )}
                </div>

                {/* Model Tiers */}
                <div className="config-section">
                  <h3>Model Tiers</h3>
                  <p className="config-hint">
                    Cadence routes tasks to different models based on complexity.
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
                                    localStorage.setItem('cadence-disabled-tools', JSON.stringify([...next]));
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

                {configTab === 'conversation' && (
                  <>
                <div className="config-section">
                  <h3>Conversation</h3>
                  <p className="config-hint">
                    Controls how much conversation history is sent to the LLM for context.
                    When compression is enabled, older turns are automatically summarized
                    to preserve context while saving tokens.
                  </p>
                  <div className="config-field">
                    <label>
                      Max History Turns
                      <span className="field-hint">Maximum user+assistant pairs to retain</span>
                    </label>
                    <input
                      type="number"
                      min={5}
                      max={200}
                      value={configDraft.max_history_turns}
                      onChange={(e) => setConfigDraft({ ...configDraft, max_history_turns: parseInt(e.target.value) || 50 })}
                    />
                  </div>
                  <div className="config-field">
                    <label>
                      <input
                        type="checkbox"
                        checked={configDraft.compression_enabled}
                        onChange={(e) => setConfigDraft({ ...configDraft, compression_enabled: e.target.checked })}
                      />
                      {' '}Context Compression
                      <span className="field-hint">Summarize older turns instead of dropping them</span>
                    </label>
                  </div>
                  {configDraft.compression_enabled && (
                    <div className="config-field">
                      <label>
                        Compression Threshold
                        <span className="field-hint">Compress when conversation exceeds this many turns</span>
                      </label>
                      <input
                        type="number"
                        min={5}
                        max={100}
                        value={configDraft.compression_threshold}
                        onChange={(e) => setConfigDraft({ ...configDraft, compression_threshold: parseInt(e.target.value) || 30 })}
                      />
                    </div>
                  )}
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
                          max_history_turns: config.conversation?.max_history_turns ?? 50,
                          compression_enabled: config.conversation?.compression_enabled ?? true,
                          compression_threshold: config.conversation?.compression_threshold ?? 30,
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
