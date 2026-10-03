import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import _orchestrators, get_orchestrator, router, set_roles_dir


def create_app(roles_dir: Path | str = "roles", prewarm_role: str | None = None) -> FastAPI:
    resolved_roles_dir = Path(roles_dir).resolve()
    target_role = prewarm_role or os.environ.get("AI_TRAINS_AI_PREWARM_ROLE") or os.environ.get("AI_INSTITUTE_PREWARM_ROLE")

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
        return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>AI-Trains-AI | Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #09090b;
      --surface: #121316;
      --surface-elevated: #18191e;
      --surface-hover: #1e2026;
      --border: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(255, 255, 255, 0.16);
      --border-focus: rgba(255, 255, 255, 0.28);
      --text-primary: #f4f4f5;
      --text-secondary: #a1a1aa;
      --text-tertiary: #71717a;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --online: #10b981;
      --code-bg: #0c0d10;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      font-family: var(--font-sans);
      background: var(--bg);
      color: var(--text-primary);
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    #app {
      display: flex;
      flex-direction: column;
      height: 100%;
      height: 100dvh;
      position: relative;
    }

    /* Header */
    header {
      height: 56px;
      background: rgba(9, 9, 11, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border);
      padding: 0 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      text-decoration: none;
      color: var(--text-primary);
    }
    .brand-icon {
      width: 32px;
      height: 32px;
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-primary);
    }
    .brand-title {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-badge {
      font-size: 11px;
      font-weight: 500;
      color: var(--text-tertiary);
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      padding: 2px 7px;
      border-radius: 12px;
    }

    /* Desktop Controls */
    .desktop-nav {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .role-select-wrap {
      position: relative;
      display: flex;
      align-items: center;
    }
    .role-select {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-primary);
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 6px 30px 6px 12px;
      border-radius: 8px;
      outline: none;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
      transition: border-color 0.15s, background-color 0.15s;
    }
    .role-select:hover {
      border-color: var(--border-hover);
      background: var(--surface-hover);
    }
    .role-select-wrap svg {
      position: absolute;
      right: 10px;
      pointer-events: none;
      color: var(--text-tertiary);
    }
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary);
      padding: 4px 8px;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--online);
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
    }
    .nav-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      text-decoration: none;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }
    .nav-btn:hover {
      background: var(--surface-hover);
      color: var(--text-primary);
      border-color: var(--border-hover);
    }

    /* Mobile Controls */
    .mobile-nav {
      display: none;
      align-items: center;
      gap: 8px;
    }
    .mobile-role-badge {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary);
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 6px;
    }
    .mobile-menu-btn {
      width: 36px;
      height: 36px;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-primary);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }

    /* Mobile Drawer */
    .drawer-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(4px);
      z-index: 40;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease;
    }
    .drawer-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }
    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: min(320px, 85vw);
      background: var(--surface);
      border-left: 1px solid var(--border);
      z-index: 50;
      transform: translateX(100%);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
      padding: 20px;
      gap: 20px;
    }
    .drawer.open {
      transform: translateX(0);
    }
    .drawer-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      padding-bottom: 14px;
    }
    .drawer-title {
      font-size: 15px;
      font-weight: 600;
    }
    .drawer-close {
      background: transparent;
      border: none;
      color: var(--text-tertiary);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 4px;
    }
    .drawer-section {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .drawer-label {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-tertiary);
    }

    /* Main Chat Container */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      max-width: 840px;
      width: 100%;
      margin: 0 auto;
      padding: 0 16px;
      min-height: 0;
      position: relative;
    }
    #chat-box {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding: 20px 0 12px 0;
      scroll-behavior: smooth;
    }
    #chat-box::-webkit-scrollbar {
      width: 5px;
    }
    #chat-box::-webkit-scrollbar-track {
      background: transparent;
    }
    #chat-box::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
    }

    /* Messages */
    .message-row {
      display: flex;
      gap: 12px;
      width: 100%;
      animation: msgFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes msgFadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message-row.user {
      justify-content: flex-end;
    }
    .message-row.assistant {
      justify-content: flex-start;
    }
    .msg-avatar {
      width: 28px;
      height: 28px;
      border-radius: 8px;
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-secondary);
      flex-shrink: 0;
      margin-top: 2px;
    }
    .msg-body {
      display: flex;
      flex-direction: column;
      max-width: min(720px, 86%);
    }
    .message-row.user .msg-body {
      align-items: flex-end;
    }
    .bubble {
      padding: 12px 16px;
      font-size: 14px;
      line-height: 1.6;
      word-break: break-word;
    }
    .message-row.user .bubble {
      background: var(--accent);
      color: #fff;
      border-radius: 18px 18px 4px 18px;
      white-space: pre-wrap;
    }
    .message-row.assistant .bubble {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-primary);
      border-radius: 4px 18px 18px 18px;
    }

    /* Markdown inside assistant */
    .bubble p {
      margin-bottom: 8px;
    }
    .bubble p:last-child {
      margin-bottom: 0;
    }
    .bubble strong {
      font-weight: 600;
      color: #fff;
    }
    .inline-code {
      font-family: var(--font-mono);
      font-size: 12.5px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.06);
      padding: 2px 6px;
      border-radius: 4px;
      color: #38bdf8;
    }
    .code-block {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 10px 0;
      overflow: hidden;
    }
    .code-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.03);
      border-bottom: 1px solid var(--border);
      padding: 6px 12px;
    }
    .code-lang {
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-tertiary);
      text-transform: lowercase;
    }
    .copy-btn {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 11.5px;
      cursor: pointer;
      padding: 2px 6px;
      border-radius: 4px;
      transition: color 0.15s, background-color 0.15s;
    }
    .copy-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-primary);
    }
    .code-block pre {
      padding: 12px 14px;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 12.5px;
      line-height: 1.5;
      color: #e4e4e7;
    }
    .bullet-list {
      margin: 6px 0 8px 20px;
    }
    .bullet-item {
      margin-bottom: 4px;
    }

    /* Meta tags & citations */
    .msg-meta {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-top: 6px;
      flex-wrap: wrap;
    }
    .meta-chip {
      font-size: 11px;
      font-family: var(--font-mono);
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      padding: 2px 7px;
      border-radius: 4px;
      color: var(--text-tertiary);
    }
    .meta-latency {
      color: #38bdf8;
    }
    .meta-source {
      color: #34d399;
    }

    /* Blinking Cursor during Streaming */
    .streaming-cursor {
      display: inline-block;
      width: 7px;
      height: 14px;
      background: #38bdf8;
      vertical-align: middle;
      margin-left: 3px;
      border-radius: 1px;
      animation: cursorBlink 0.8s infinite;
    }
    @keyframes cursorBlink {
      0%, 100% { opacity: 0; }
      50% { opacity: 1; }
    }

    /* Suggestion Chips */
    .suggestions-container {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 8px 0;
      flex-shrink: 0;
      -webkit-overflow-scrolling: touch;
      scrollbar-width: none;
    }
    .suggestions-container::-webkit-scrollbar {
      display: none;
    }
    .suggestion-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      font-family: inherit;
      font-size: 12.5px;
      font-weight: 500;
      padding: 6px 14px;
      border-radius: 20px;
      white-space: nowrap;
      cursor: pointer;
      transition: all 0.15s;
    }
    .suggestion-btn:hover {
      background: var(--surface-hover);
      color: var(--text-primary);
      border-color: var(--border-hover);
    }

    /* Input area */
    .input-wrapper {
      padding: 0 0 max(16px, env(safe-area-inset-bottom)) 0;
      flex-shrink: 0;
    }
    .input-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .input-box:focus-within {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.05);
    }
    #prompt-textarea {
      width: 100%;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-primary);
      font-family: inherit;
      font-size: 14px;
      line-height: 1.5;
      resize: none;
      max-height: 160px;
      min-height: 24px;
    }
    #prompt-textarea::placeholder {
      color: var(--text-tertiary);
    }
    .input-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .input-hint {
      font-size: 11px;
      color: var(--text-tertiary);
    }
    .send-button {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: var(--text-primary);
      color: var(--bg);
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
      flex-shrink: 0;
    }
    .send-button:hover:not(:disabled) {
      opacity: 0.9;
    }
    .send-button:active:not(:disabled) {
      transform: scale(0.96);
    }
    .send-button:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    /* Responsive Adaptation */
    @media (max-width: 768px) {
      header {
        padding: 0 14px;
      }
      .desktop-nav {
        display: none;
      }
      .mobile-nav {
        display: flex;
      }
      main {
        padding: 0 12px;
      }
      .bubble {
        font-size: 13.5px;
      }
      .input-hint {
        display: none;
      }
      .suggestion-btn {
        font-size: 12px;
        padding: 6px 12px;
        min-height: 36px;
      }
    }
  </style>
</head>
<body>
  <div id="app">
    <!-- Header -->
    <header>
      <a href="/" class="brand">
        <div class="brand-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
        </div>
        <div class="brand-title">
          <span>AI-Trains-AI</span>
          <span class="brand-badge" id="header-role-badge">Assistant</span>
        </div>
      </a>

      <!-- Desktop Header Controls -->
      <nav class="desktop-nav">
        <div class="role-select-wrap">
          <select id="desktop-role-select" class="role-select" onchange="handleRoleSwitch(this.value)">
            <option value="">Loading roles...</option>
          </select>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>

        <div class="status-indicator">
          <div class="status-dot"></div>
          <span>Online</span>
        </div>

        <button class="nav-btn" onclick="clearConversation()" title="Start new session">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="1 4 1 10 7 10"></polyline>
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
          </svg>
          <span>Reset</span>
        </button>

        <a href="/docs" target="_blank" class="nav-btn">
          <span>API Docs</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/>
          </svg>
        </a>
      </nav>

      <!-- Mobile Header Controls -->
      <div class="mobile-nav">
        <span class="mobile-role-badge" id="mobile-top-role">Assistant</span>
        <button class="mobile-menu-btn" onclick="toggleMobileDrawer(true)" aria-label="Open menu">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="3" y1="12" x2="21" y2="12"></line>
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <line x1="3" y1="18" x2="21" y2="18"></line>
          </svg>
        </button>
      </div>
    </header>

    <!-- Mobile Slide-over Drawer -->
    <div class="drawer-overlay" id="drawer-overlay" onclick="toggleMobileDrawer(false)"></div>
    <div class="drawer" id="mobile-drawer">
      <div class="drawer-header">
        <span class="drawer-title">Controls & Roles</span>
        <button class="drawer-close" onclick="toggleMobileDrawer(false)">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>

      <div class="drawer-section">
        <label class="drawer-label">Select Assistant Role</label>
        <select id="mobile-role-select" class="role-select" style="width: 100%;" onchange="handleRoleSwitch(this.value); toggleMobileDrawer(false);">
          <option value="">Loading roles...</option>
        </select>
      </div>

      <div class="drawer-section">
        <label class="drawer-label">Session</label>
        <button class="nav-btn" style="width: 100%; justify-content: center;" onclick="clearConversation(); toggleMobileDrawer(false);">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="1 4 1 10 7 10"></polyline>
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
          </svg>
          <span>New Conversation</span>
        </button>
      </div>

      <div class="drawer-section" style="margin-top: auto;">
        <label class="drawer-label">Documentation</label>
        <a href="/docs" target="_blank" class="nav-btn" style="justify-content: center;">
          <span>Swagger API Reference</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/>
          </svg>
        </a>
        <a href="/redoc" target="_blank" class="nav-btn" style="justify-content: center;">
          <span>ReDoc Manual</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/>
          </svg>
        </a>
      </div>
    </div>

    <!-- Main Chat Workspace -->
    <main>
      <div id="chat-box"></div>

      <!-- Suggestion Chips Bar -->
      <div class="suggestions-container" id="suggestions-strip"></div>

      <!-- Input Area -->
      <div class="input-wrapper">
        <form class="input-box" id="chat-form" onsubmit="handleSend(event)">
          <textarea
            id="prompt-textarea"
            rows="1"
            placeholder="Ask a question or enter a query for this assistant..."
            autocomplete="off"
            onkeydown="handleKeyDown(event)"
          ></textarea>
          <div class="input-footer">
            <span class="input-hint">Return to send &middot; Shift+Return for new line</span>
            <button type="submit" class="send-button" id="send-button" disabled title="Send message">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            </button>
          </div>
        </form>
      </div>
    </main>
  </div>

  <script>
    // -------------------------------------------------------------
    // 1. Markdown Formatter Hook (Zero Dependency, Lightweight, Safe)
    // -------------------------------------------------------------
    function renderMarkdown(raw) {
      if (!raw) return '';
      let text = raw.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      
      // Fenced code blocks ```lang ... ```
      text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function(_, lang, code) {
        const cleanCode = code.trim();
        const langLabel = lang ? lang.toLowerCase() : 'code';
        const codeEscaped = encodeURIComponent(cleanCode);
        return `<div class="code-block" data-raw="${codeEscaped}">
          <div class="code-header">
            <span class="code-lang">${langLabel}</span>
            <button type="button" class="copy-btn" onclick="copyCode(this)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              <span>Copy</span>
            </button>
          </div>
          <pre><code>${cleanCode}</code></pre>
        </div>`;
      });

      // Inline code `code`
      text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

      // Bold **text**
      text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

      // Italic *text*
      text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

      // Bullet lists
      text = text.replace(/(?:^|\n)- ([^\n]+)/g, '\n<li class="bullet-item">$1</li>');
      text = text.replace(/((?:<li class="bullet-item">.*<\/li>\s*)+)/g, '<ul class="bullet-list">$1</ul>');

      // Paragraph spacing
      text = text.replace(/\n\n+/g, '<br><br>');
      text = text.replace(/\n/g, '<br>');

      return text;
    }

    function copyCode(btn) {
      const container = btn.closest('.code-block');
      if (!container) return;
      const rawCode = decodeURIComponent(container.getAttribute('data-raw') || '');
      navigator.clipboard.writeText(rawCode).then(() => {
        const textSpan = btn.querySelector('span');
        const originalText = textSpan.textContent;
        textSpan.textContent = 'Copied!';
        setTimeout(() => { textSpan.textContent = originalText; }, 2000);
      });
    }

    // -------------------------------------------------------------
    // 2. Viewport & Mobile Drawer Controller
    // -------------------------------------------------------------
    function toggleMobileDrawer(open) {
      const drawer = document.getElementById('mobile-drawer');
      const overlay = document.getElementById('drawer-overlay');
      if (open) {
        drawer.classList.add('open');
        overlay.classList.add('open');
      } else {
        drawer.classList.remove('open');
        overlay.classList.remove('open');
      }
    }

    // Auto-scroll on mobile virtual keyboard adjustments
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => {
        const chatBox = document.getElementById('chat-box');
        chatBox.scrollTop = chatBox.scrollHeight;
      });
    }

    // -------------------------------------------------------------
    // 3. Session & Role State (Generic Multi-Role)
    // -------------------------------------------------------------
    let currentRole = '';
    let sessionId = localStorage.getItem('ai_trains_ai_sess') || 'web-' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('ai_trains_ai_sess', sessionId);

    const GENERIC_PROMPTS = [
      { label: "Role Scope", query: "Tell me about your role and what you can assist with." },
      { label: "Policies & Rules", query: "What guidelines, policies, or rules do you follow?" },
      { label: "Capabilities", query: "What tools, data sources, or capabilities do you have access to?" },
      { label: "Test Scope Refusal", query: "Solve this math equation: 2x + 5 = 15" }
    ];

    function updateSuggestionChips() {
      const strip = document.getElementById('suggestions-strip');
      strip.innerHTML = '';
      GENERIC_PROMPTS.forEach(item => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'suggestion-btn';
        btn.textContent = item.label;
        btn.onclick = () => {
          const textarea = document.getElementById('prompt-textarea');
          textarea.value = item.query;
          handleSend();
        };
        strip.appendChild(btn);
      });
    }

    async function initRoles() {
      const desktopSelect = document.getElementById('desktop-role-select');
      const mobileSelect = document.getElementById('mobile-role-select');
      const headerRoleBadge = document.getElementById('header-role-badge');
      const mobileTopRole = document.getElementById('mobile-top-role');

      updateSuggestionChips();

      try {
        const res = await fetch('/api/v1/roles');
        const roles = await res.json();
        
        desktopSelect.innerHTML = '';
        mobileSelect.innerHTML = '';

        if (!roles || roles.length === 0) {
          desktopSelect.innerHTML = '<option value="">(No roles configured)</option>';
          mobileSelect.innerHTML = '<option value="">(No roles configured)</option>';
          headerRoleBadge.textContent = 'None';
          mobileTopRole.textContent = 'None';
          appendAssistantGreeting('Welcome to AI-Trains-AI! No assistant roles are configured yet.\n\nTo initialize a role, run:\n  ai-trains-ai create-role <name> -d "Description"');
          return;
        }

        roles.forEach(r => {
          const opt1 = document.createElement('option');
          opt1.value = r.name;
          opt1.textContent = r.name;
          desktopSelect.appendChild(opt1);

          const opt2 = document.createElement('option');
          opt2.value = r.name;
          opt2.textContent = r.name;
          mobileSelect.appendChild(opt2);
        });

        currentRole = roles[0].name;
        desktopSelect.value = currentRole;
        mobileSelect.value = currentRole;
        headerRoleBadge.textContent = currentRole;
        mobileTopRole.textContent = currentRole;

        appendAssistantGreeting(`Hello! I am the ${currentRole} assistant. I am role-locked with strict domain boundaries and direct tool access. How can I assist you today?`);
      } catch (err) {
        desktopSelect.innerHTML = '<option value="">(Error loading roles)</option>';
        mobileSelect.innerHTML = '<option value="">(Error loading roles)</option>';
        appendAssistantGreeting('Unable to retrieve role list from server: ' + err.message);
      }
    }

    function handleRoleSwitch(newRole) {
      if (!newRole || newRole === currentRole) return;
      currentRole = newRole;
      document.getElementById('desktop-role-select').value = newRole;
      document.getElementById('mobile-role-select').value = newRole;
      document.getElementById('header-role-badge').textContent = newRole;
      document.getElementById('mobile-top-role').textContent = newRole;
      clearConversation(false);
      appendAssistantGreeting(`Switched active assistant to ${newRole}. How can I assist you?`);
    }

    function clearConversation(greet = true) {
      const chatBox = document.getElementById('chat-box');
      chatBox.innerHTML = '';
      sessionId = 'web-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('ai_trains_ai_sess', sessionId);
      if (greet && currentRole) {
        appendAssistantGreeting(`Started a fresh conversation session with ${currentRole}.`);
      }
    }

    function appendAssistantGreeting(text) {
      const chatBox = document.getElementById('chat-box');
      const row = document.createElement('div');
      row.className = 'message-row assistant';
      row.innerHTML = `
        <div class="msg-avatar">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
        </div>
        <div class="msg-body">
          <div class="bubble">${renderMarkdown(text)}</div>
        </div>
      `;
      chatBox.appendChild(row);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    // -------------------------------------------------------------
    // 4. Live SSE Token Streaming & Chat Controller
    // -------------------------------------------------------------
    let isStreaming = false;

    async function handleSend(e) {
      if (e) e.preventDefault();
      if (isStreaming) return;

      const textarea = document.getElementById('prompt-textarea');
      const sendBtn = document.getElementById('send-button');
      const text = textarea.value.trim();
      if (!text) return;

      if (!currentRole) {
        alert('Please configure and select an assistant role first.');
        return;
      }

      // Append User message
      const chatBox = document.getElementById('chat-box');
      const userRow = document.createElement('div');
      userRow.className = 'message-row user';
      userRow.innerHTML = `
        <div class="msg-body">
          <div class="bubble">${text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
        </div>
      `;
      chatBox.appendChild(userRow);

      // Reset textarea
      textarea.value = '';
      textarea.style.height = 'auto';
      sendBtn.disabled = true;
      isStreaming = true;

      // Create streaming Assistant row
      const assistantRow = document.createElement('div');
      assistantRow.className = 'message-row assistant';
      assistantRow.innerHTML = `
        <div class="msg-avatar">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
        </div>
        <div class="msg-body">
          <div class="bubble"><span class="stream-content"></span><span class="streaming-cursor"></span></div>
          <div class="msg-meta" style="display: none;"></div>
        </div>
      `;
      chatBox.appendChild(assistantRow);
      chatBox.scrollTop = chatBox.scrollHeight;

      const streamContentSpan = assistantRow.querySelector('.stream-content');
      const cursorSpan = assistantRow.querySelector('.streaming-cursor');
      const metaDiv = assistantRow.querySelector('.msg-meta');
      const bubble = assistantRow.querySelector('.bubble');

      let accumulatedRaw = '';
      let metaPayload = null;

      try {
        const response = await fetch('/api/v1/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            role: currentRole,
            message: text,
            session_id: sessionId
          })
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop(); // Keep uncompleted line in buffer

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              try {
                const data = JSON.parse(trimmed.slice(6));
                if (data.type === 'token') {
                  accumulatedRaw += data.content;
                  streamContentSpan.textContent = accumulatedRaw;
                  chatBox.scrollTop = chatBox.scrollHeight;
                } else if (data.type === 'meta') {
                  metaPayload = data;
                  if (data.session_id) sessionId = data.session_id;
                }
              } catch (parseErr) {
                // Non-JSON chunk ignored
              }
            }
          }
        }

        // Finished streaming: parse full markdown
        cursorSpan.remove();
        bubble.innerHTML = renderMarkdown(accumulatedRaw || metaPayload?.response || '');

        if (metaPayload) {
          metaDiv.style.display = 'flex';
          metaDiv.innerHTML = '';
          if (metaPayload.elapsed) {
            metaDiv.innerHTML += `<span class="meta-chip meta-latency">${metaPayload.elapsed}s</span>`;
          }
          if (metaPayload.sources && metaPayload.sources.length) {
            metaPayload.sources.forEach(src => {
              metaDiv.innerHTML += `<span class="meta-chip meta-source">${src}</span>`;
            });
          }
        }
      } catch (err) {
        cursorSpan.remove();
        bubble.innerHTML = `<span style="color: #f87171;">Error connecting to assistant: ${err.message}</span>`;
      } finally {
        isStreaming = false;
        textarea.focus();
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }

    // -------------------------------------------------------------
    // 5. Input Auto-Grow and Keyboard Handlers
    // -------------------------------------------------------------
    const textarea = document.getElementById('prompt-textarea');
    const sendButton = document.getElementById('send-button');

    textarea.addEventListener('input', function() {
      this.style.height = 'auto';
      const newHeight = Math.min(this.scrollHeight, 160);
      this.style.height = newHeight + 'px';
      sendButton.disabled = this.value.trim().length === 0;
    });

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    }

    // Global shortcut: '/' focuses the input bar when not typing in an input
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== textarea) {
        e.preventDefault();
        textarea.focus();
      }
      if (e.key === 'Escape') {
        toggleMobileDrawer(false);
      }
    });

    // Initialize application
    initRoles();
  </script>
</body>
</html>"""

    return app


# Default app instance for ASGI servers
app = create_app(
    roles_dir=os.environ.get("AI_TRAINS_AI_ROLES_DIR") or os.environ.get("AI_INSTITUTE_ROLES_DIR", "roles"),
    prewarm_role=os.environ.get("AI_TRAINS_AI_PREWARM_ROLE") or os.environ.get("AI_INSTITUTE_PREWARM_ROLE"),
)
