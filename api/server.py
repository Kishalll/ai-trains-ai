import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import _orchestrators, get_orchestrator, router, set_roles_dir


def create_app(roles_dir: Path | str = "roles", prewarm_role: str | None = None) -> FastAPI:
    resolved_roles_dir = Path(roles_dir).resolve()
    target_role = prewarm_role or os.environ.get("AI_INSTITUTE_PREWARM_ROLE")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Startup: Pre-warm target role (loads embedding weights & sends keep_alive: -1 to Ollama)
        if target_role:
            try:
                orch = get_orchestrator(target_role)
                orch.warmup()
            except Exception as e:
                print(f"Warning: Could not prewarm role '{target_role}': {e}")
        yield
        # 2. Shutdown: Release all cached orchestrator models from Ollama RAM
        for role_name, orch in list(_orchestrators.items()):
            try:
                orch.unload()
            except Exception:
                pass

    app = FastAPI(
        title="AI-Trains-AI API",
        version="1.0.0",
        description="REST API for role-locked specialized AI assistants",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
            "syntaxHighlight.theme": "monokai",
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    set_roles_dir(resolved_roles_dir)
    app.include_router(router)

    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def playground_ui():
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI-Trains-AI | Assistant Playground</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --user-bubble: #2563eb;
      --assistant-bubble: #1e293b;
      --chip-bg: #334155;
      --success: #22c55e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-primary);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      font-size: 24px;
      background: #334155;
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
    }
    .brand-text h1 {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-main);
    }
    .brand-text p {
      font-size: 12px;
      color: var(--text-muted);
    }
    .nav-controls {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .role-selector-wrap {
      display: flex;
      align-items: center;
      gap: 6px;
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 4px 10px;
    }
    .role-label {
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 600;
      color: var(--text-muted);
      letter-spacing: 0.5px;
    }
    .role-dropdown {
      background: transparent;
      border: none;
      color: var(--primary);
      font-family: inherit;
      font-size: 13px;
      font-weight: 600;
      outline: none;
      cursor: pointer;
    }
    .role-dropdown option {
      background: #1e293b;
      color: #fff;
    }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 500;
      color: var(--success);
      background: rgba(34, 197, 94, 0.1);
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: var(--success);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--success);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.8; }
      50% { transform: scale(1.15); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }
    .btn {
      background: #334155;
      color: var(--text-main);
      text-decoration: none;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      border: 1px solid var(--card-border);
      transition: all 0.2s;
    }
    .btn:hover {
      background: #475569;
      color: #fff;
    }
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
      padding: 16px;
      overflow: hidden;
    }
    #chat-box {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 12px 4px;
    }
    .message {
      display: flex;
      flex-direction: column;
      max-width: 82%;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message.user { align-self: flex-end; }
    .message.assistant { align-self: flex-start; }
    .bubble {
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.6;
      word-break: break-word;
      white-space: pre-wrap;
    }
    .message.user .bubble {
      background: var(--user-bubble);
      color: #fff;
      border-bottom-right-radius: 3px;
    }
    .message.assistant .bubble {
      background: var(--assistant-bubble);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      border-bottom-left-radius: 3px;
    }
    .meta-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 6px;
      font-size: 11px;
      color: var(--text-muted);
      flex-wrap: wrap;
    }
    .meta-tag {
      background: var(--chip-bg);
      padding: 2px 8px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    }
    .meta-latency { color: #38bdf8; }
    .suggestions {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 8px 0;
      flex-shrink: 0;
    }
    .suggestion-chip {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 12px;
      padding: 6px 12px;
      border-radius: 20px;
      white-space: nowrap;
      cursor: pointer;
      transition: all 0.2s;
    }
    .suggestion-chip:hover {
      background: #334155;
      color: var(--text-main);
      border-color: var(--primary);
    }
    .input-bar {
      display: flex;
      gap: 10px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 8px;
      margin-top: 8px;
      flex-shrink: 0;
    }
    .input-bar:focus-within {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    #prompt-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-main);
      font-size: 14px;
      padding: 6px 8px;
      font-family: inherit;
    }
    #prompt-input::placeholder { color: var(--text-muted); }
    .send-btn {
      background: var(--primary);
      color: #0f172a;
      border: none;
      border-radius: 8px;
      padding: 8px 18px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .send-btn:hover { background: var(--primary-hover); }
    .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-icon">🤖</div>
      <div class="brand-text">
        <h1>AI-Trains-AI | Assistant Playground</h1>
        <p>Role-Locked Specialized AI Assistant Framework</p>
      </div>
    </div>
    <div class="nav-controls">
      <div class="role-selector-wrap">
        <span class="role-label">Role:</span>
        <select id="role-select" class="role-dropdown" onchange="handleRoleChange()">
          <option value="">Loading...</option>
        </select>
      </div>
      <div class="status-badge">
        <div class="pulse-dot"></div>
        <span>Online</span>
      </div>
      <a href="/docs" target="_blank" class="btn">⚡ Swagger Docs</a>
      <a href="/redoc" target="_blank" class="btn">📘 ReDoc</a>
    </div>
  </header>

  <main>
    <div id="chat-box"></div>

    <div class="suggestions" id="suggestions-bar">
      <button class="suggestion-chip" onclick="quickSend('Tell me about your role and what you can assist with.')">ℹ️ Role Scope</button>
      <button class="suggestion-chip" onclick="quickSend('What guidelines, policies, or rules do you follow?')">📋 Policies & Guidelines</button>
      <button class="suggestion-chip" onclick="quickSend('What tools, data sources, or capabilities do you have access to?')">⚡ Capabilities & Tools</button>
      <button class="suggestion-chip" onclick="quickSend('Solve this math equation: 2x + 5 = 15')">🛑 Test Scope Refusal</button>
    </div>

    <form class="input-bar" id="chat-form" onsubmit="handleSend(event)">
      <input type="text" id="prompt-input" placeholder="Ask a question or enter a query for this assistant..." autocomplete="off">
      <button type="submit" class="send-btn" id="send-btn">Send</button>
    </form>
  </main>

  <script>
    const chatBox = document.getElementById('chat-box');
    const input = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const roleSelect = document.getElementById('role-select');
    let currentRole = '';
    let sessionId = localStorage.getItem('ai_institute_sess') || 'web-' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('ai_institute_sess', sessionId);

    function appendMessage(role, text, meta) {
      const msgDiv = document.createElement('div');
      msgDiv.className = `message ${role}`;
      
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      msgDiv.appendChild(bubble);

      if (meta && (meta.sources || meta.elapsed)) {
        const metaBar = document.createElement('div');
        metaBar.className = 'meta-bar';
        if (meta.elapsed) {
          const lat = document.createElement('span');
          lat.className = 'meta-latency';
          lat.textContent = `⏱️ ${meta.elapsed}s`;
          metaBar.appendChild(lat);
        }
        if (meta.sources && meta.sources.length) {
          meta.sources.forEach(src => {
            const tag = document.createElement('span');
            tag.className = 'meta-tag';
            tag.textContent = src;
            metaBar.appendChild(tag);
          });
        }
        msgDiv.appendChild(metaBar);
      }

      chatBox.appendChild(msgDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
      return msgDiv;
    }

    async function initRoles() {
      try {
        const res = await fetch('/api/v1/roles');
        const roles = await res.json();
        roleSelect.innerHTML = '';
        if (!roles || roles.length === 0) {
          roleSelect.innerHTML = '<option value="">(No roles configured)</option>';
          appendMessage('assistant', 'Welcome to AI-Trains-AI! No assistant roles are configured yet.\\n\\nTo initialize a role, run:\\n  ai-trains-ai create-role <name> -d "Description"');
          return;
        }
        roles.forEach(r => {
          const opt = document.createElement('option');
          opt.value = r.name;
          opt.textContent = r.name;
          roleSelect.appendChild(opt);
        });
        currentRole = roles[0].name;
        appendMessage('assistant', `Hello! I am the ${currentRole} assistant. How can I assist you today?`);
      } catch (err) {
        roleSelect.innerHTML = '<option value="">(Error loading roles)</option>';
        appendMessage('assistant', 'Unable to retrieve role list from server: ' + err.message);
      }
    }

    function handleRoleChange() {
      currentRole = roleSelect.value;
      if (currentRole) {
        appendMessage('assistant', `Switched to ${currentRole} assistant.`);
      }
    }

    async function handleSend(e) {
      if (e) e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      if (!currentRole) {
        alert('Please configure and select a role first.');
        return;
      }

      appendMessage('user', text);
      input.value = '';
      input.disabled = true;
      sendBtn.disabled = true;

      const loadingMsg = appendMessage('assistant', 'Thinking...');

      try {
        const res = await fetch('/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            role: currentRole,
            message: text,
            session_id: sessionId
          })
        });
        const data = await res.json();
        chatBox.removeChild(loadingMsg);
        appendMessage('assistant', data.response, {
          sources: data.sources,
          elapsed: data.elapsed
        });
      } catch (err) {
        chatBox.removeChild(loadingMsg);
        appendMessage('assistant', 'Error communicating with assistant server: ' + err.message);
      } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
      }
    }

    function quickSend(text) {
      input.value = text;
      handleSend();
    }

    initRoles();
  </script>
</body>
</html>"""

    return app


# Default app instance for ASGI servers
app = create_app(
    roles_dir=os.environ.get("AI_INSTITUTE_ROLES_DIR", "roles"),
    prewarm_role=os.environ.get("AI_INSTITUTE_PREWARM_ROLE"),
)
