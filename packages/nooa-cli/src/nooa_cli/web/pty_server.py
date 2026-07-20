# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PTY-based web terminal server for NeMo OO Agents.

Serves an xterm.js browser terminal that connects to a real PTY running the
NeMo OO Agents TUI, with a rich content side-channel for plots/HTML.

Endpoints:
  GET  /        — xterm.js UI (HTML embedded)
  WS   /ws/pty  — PTY I/O bridge (base64-framed JSON)
  WS   /ws/rich — rich content push (agent → browser)
  POST /rich    — HTTP endpoint the agent POSTs rich payloads to

Authentication: when ``create_pty_app`` is given an ``auth_token``, every
endpoint requires it.  ``GET /`` accepts the token as a ``?token=`` query
parameter and sets it in a cookie, which the WebSocket handshakes and
``POST /rich`` then validate (a ``?token=`` query parameter also works).
Requests without a valid token get HTTP 403; WebSockets are accepted and
immediately closed with code 4403 before any PTY is spawned.
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import threading
from typing import Any

#: Cookie set by ``GET /?token=...`` and checked by the other endpoints.
_TOKEN_COOKIE = "nooa_term_token"

_log = logging.getLogger("nooa.pty_server")

# ---------------------------------------------------------------------------
# Embedded HTML UI
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NeMo OO Agents — Terminal</title>
<!-- xterm.js -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css"/>
<script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.js"></script>
<!-- Plotly for rich output -->
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<!-- marked.js for markdown -->
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #000000;
    --surface: #0d0d0d;
    --overlay: #222222;
    --text:    #cdd6f4;
    --subtext: #a6adc8;
    --blue:    #89b4fa;
    --green:   #a6e3a1;
    --red:     #f38ba8;
    --yellow:  #f9e2af;
    --mauve:   #cba6f7;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    overflow: hidden;
  }

  #layout {
    display: flex;
    height: 100vh;
    gap: 0;
  }

  /* ---- Terminal pane ---- */
  #term-pane {
    flex: 1 1 0;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: var(--bg);
  }

  #term-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--bg);
    border-bottom: none;
    font-size: 12px;
    color: var(--subtext);
    flex-shrink: 0;
  }

  #term-header .dot { width: 10px; height: 10px; border-radius: 50%; }
  .dot-red    { background: var(--red); }
  .dot-yellow { background: var(--yellow); }
  .dot-green  { background: var(--green); }
  #term-title { flex: 1; text-align: center; }

  #terminal {
    flex: 1 1 0;
    min-height: 0;
    padding: 4px;
  }

  /* status bar at top-right */
  #status-bar {
    position: fixed;
    top: 8px;
    right: 8px;
    font-size: 11px;
    color: var(--subtext);
    pointer-events: none;
    z-index: 100;
  }
</style>
</head>
<body>
<div id="layout">
  <div id="term-pane">
    <div id="term-header">
      <span class="dot dot-red"></span>
      <span class="dot dot-yellow"></span>
      <span class="dot dot-green"></span>
      <span id="term-title">NeMo OO Agents</span>
    </div>
    <div id="terminal"></div>
  </div>
</div>
<div id="status-bar"></div>

<script>
// ---- xterm theme ----
const MOCHA = {
  background:          '#000000',
  foreground:          '#cdd6f4',
  cursor:              '#f5e0dc',
  cursorAccent:        '#000000',
  selectionBackground: '#585b7066',
  black:               '#45475a',
  red:                 '#f38ba8',
  green:               '#a6e3a1',
  yellow:              '#f9e2af',
  blue:                '#89b4fa',
  magenta:             '#f5c2e7',
  cyan:                '#94e2d5',
  white:               '#bac2de',
  brightBlack:         '#585b70',
  brightRed:           '#f38ba8',
  brightGreen:         '#a6e3a1',
  brightYellow:        '#f9e2af',
  brightBlue:          '#89b4fa',
  brightMagenta:       '#f5c2e7',
  brightCyan:          '#94e2d5',
  brightWhite:         '#a6adc8',
};

// ---- xterm init ----
const term = new Terminal({
  theme: MOCHA,
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  fontSize: 14,
  lineHeight: 1.2,
  cursorBlink: true,
  allowProposedApi: true,
  scrollback: 10000,
});
const fitAddon = new FitAddon.FitAddon();
term.loadAddon(fitAddon);
term.open(document.getElementById('terminal'));
fitAddon.fit();

window.addEventListener('resize', () => fitAddon.fit());

// ---- PTY WebSocket ----
const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
// Token is injected by GET / into window.__NEMO_TERM_TOKEN and passed on the WS
// URL, so reconnects don't depend on the host-scoped (port-agnostic) cookie —
// two terminals on different ports of localhost no longer clobber each other.
const tokenQS = window.__NEMO_TERM_TOKEN ? `?token=${window.__NEMO_TERM_TOKEN}` : '';
const ptyWs  = new WebSocket(`${proto}//${location.host}/ws/pty${tokenQS}`);
const richWs = new WebSocket(`${proto}//${location.host}/ws/rich${tokenQS}`);

ptyWs.binaryType = 'arraybuffer';

const statusEl = document.getElementById('status-bar');
function setStatus(msg, color) {
  statusEl.textContent = msg;
  statusEl.style.color = color || 'var(--subtext)';
}

ptyWs.onopen = () => {
  setStatus('connected', 'var(--green)');
  setTimeout(() => { statusEl.textContent = ''; }, 2000);
  // Send initial size
  sendResize();
};

ptyWs.onclose = () => setStatus('disconnected — refresh to reconnect', 'var(--red)');
ptyWs.onerror = () => setStatus('connection error', 'var(--red)');

ptyWs.onmessage = (ev) => {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }
  if (msg.t === 'o') {
    term.write(Uint8Array.from(atob(msg.d), c => c.charCodeAt(0)));
  }
};

// Keyboard input → PTY
term.onData(data => {
  if (ptyWs.readyState !== WebSocket.OPEN) return;
  const b64 = btoa(String.fromCharCode(...new TextEncoder().encode(data)));
  ptyWs.send(JSON.stringify({t: 'i', d: b64}));
});

// Resize → PTY
function sendResize() {
  if (ptyWs.readyState !== WebSocket.OPEN) return;
  ptyWs.send(JSON.stringify({t: 'r', cols: term.cols, rows: term.rows}));
}
term.onResize(({cols, rows}) => {
  ptyWs.send(JSON.stringify({t: 'r', cols, rows}));
  setTimeout(updateRichBlocks, 50);
});

// ---- Rich WebSocket ----
richWs.onmessage = (ev) => {
  let payload;
  try { payload = JSON.parse(ev.data); } catch { return; }
  handleRich(payload);
};

// ---- Scroll-aware inline rich overlay ----
// We manage our own position:absolute overlays anchored to buffer lines via
// persistent xterm markers.  On each scroll/resize we recompute screen
// positions so plots clip smoothly as they scroll off the top of the viewport
// rather than disappearing the moment their anchor marker leaves the viewport
// (which is the behaviour of the Decoration API).

const richBlocks = [];  // {marker, numRows, el, inner}
let plotsContainer = null;

function setupPlotsContainer() {
  const screen = document.querySelector('.xterm-screen');
  if (!screen) return;
  screen.style.position = 'relative';
  plotsContainer = document.createElement('div');
  plotsContainer.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;overflow:hidden;z-index:10;';
  screen.appendChild(plotsContainer);
}

function getCellHeight() {
  const screen = document.querySelector('.xterm-screen');
  if (!screen || !term.rows) return 0;
  return screen.clientHeight / term.rows;
}

function updateRichBlocks() {
  if (!plotsContainer) return;
  const cellH = getCellHeight();
  if (!cellH) return;
  const viewportY = term.buffer.active.viewportY;
  const viewportH = plotsContainer.clientHeight;

  // Remove blocks whose markers were trimmed from the scrollback buffer
  for (let i = richBlocks.length - 1; i >= 0; i--) {
    if (richBlocks[i].marker.line === -1) {
      richBlocks[i].el.remove();
      richBlocks.splice(i, 1);
    }
  }

  for (const block of richBlocks) {
    const screenTop = (block.marker.line - viewportY) * cellH;
    const blockH = block.numRows * cellH;
    const screenBottom = screenTop + blockH;

    if (screenBottom <= 0 || screenTop >= viewportH) {
      block.el.style.display = 'none';
      continue;
    }

    block.el.style.display = '';
    const clipTop = Math.max(0, screenTop);
    const clipBottom = Math.min(screenBottom, viewportH);
    block.el.style.top = clipTop + 'px';
    block.el.style.height = (clipBottom - clipTop) + 'px';
    // Shift inner content up so the visible slice starts at the right place
    block.inner.style.marginTop = Math.min(0, screenTop) + 'px';
    block.inner.style.height = blockH + 'px';
  }
}

function estimateRichRows(payload) {
  switch (payload.kind) {
    case 'plotly':   return 30;
    case 'html':     return 20;
    case 'image':    return 18;
    case 'markdown': return 10;
    case 'json':     return 10;
    default:         return 8;
  }
}

function renderPayloadInto(payload, el) {
  switch (payload.kind) {
    case 'plotly': {
      const div = document.createElement('div');
      div.style.cssText = 'width:100%;height:100%;';
      el.appendChild(div);
      setTimeout(() => {
        try {
          const fig = JSON.parse(payload.figure_json);
          Plotly.newPlot(div, fig.data, Object.assign({}, fig.layout, {
            paper_bgcolor: 'transparent',
            plot_bgcolor:  'transparent',
            font: { color: '#cdd6f4' },
            margin: { t: 40, b: 20, l: 40, r: 10 },
          }), { responsive: true, displayModeBar: false });
        } catch(e) { el.textContent = `Plotly error: ${e}`; }
      }, 0);
      break;
    }
    case 'html': {
      const isFullPage = /^\s*(<html|<!doctype)/i.test(payload.html);
      if (isFullPage) {
        const iframe = document.createElement('iframe');
        iframe.style.cssText = 'width:100%;height:100%;border:none;';
        iframe.sandbox = 'allow-scripts';
        iframe.srcdoc = payload.html;
        el.appendChild(iframe);
      } else {
        const shadow = el.attachShadow({ mode: 'open' });
        const style = document.createElement('style');
        style.textContent = ':host{display:block;padding:8px;color:#cdd6f4;font-size:13px;font-family:monospace}';
        shadow.appendChild(style);
        const div = document.createElement('div');
        div.innerHTML = payload.html;
        shadow.appendChild(div);
      }
      break;
    }
    case 'image': {
      const img = document.createElement('img');
      img.src = payload.src;
      img.alt = payload.alt || '';
      img.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;display:block;margin:4px auto;';
      el.appendChild(img);
      break;
    }
    case 'markdown': {
      const div = document.createElement('div');
      div.className = 'md-render';
      div.style.cssText = 'padding:8px;font-size:13px;line-height:1.5;color:#cdd6f4;';
      div.innerHTML = marked.parse(payload.text || '', { renderer: Object.assign(new marked.Renderer(), { html: () => '' }) });
      el.appendChild(div);
      break;
    }
    case 'json': {
      const pre = document.createElement('pre');
      pre.style.cssText = 'padding:8px;font-size:12px;color:#cdd6f4;overflow:auto;margin:0;';
      try { pre.textContent = JSON.stringify(payload.data, null, 2); }
      catch { pre.textContent = String(payload.data); }
      el.appendChild(pre);
      break;
    }
  }
}

function renderRichInline(payload) {
  if (payload.kind === 'clear') {
    for (const block of richBlocks) {
      block.el.remove();
      try { block.marker.dispose(); } catch {}
    }
    richBlocks.length = 0;
    return;
  }
  if (!plotsContainer) return;
  const numRows = estimateRichRows(payload);

  // Create a persistent marker at the current cursor position.
  // Keep it alive (don't dispose) so marker.line tracks the absolute buffer
  // line even as more content is written below.
  const marker = term.registerMarker(0);
  if (!marker || marker.line === -1) return;

  // Reserve vertical space for live output only.  Replayed plots skip this so
  // they don't push the prompt down with empty lines — the overlay sits on top
  // of whatever buffer content is already there.
  if (!payload._replay) {
    term.write('\r\n'.repeat(numRows + 1));
  }

  // Outer clip element — sized and positioned by updateRichBlocks()
  const el = document.createElement('div');
  el.style.cssText = 'position:absolute;left:0;right:0;overflow:hidden;box-sizing:border-box;pointer-events:auto;';

  // Inner content element — full block height; negative marginTop clips top edge
  const inner = document.createElement('div');
  inner.style.cssText = 'position:relative;background:var(--surface);border:1px solid var(--overlay);border-radius:4px;box-sizing:border-box;overflow:hidden;';

  renderPayloadInto(payload, inner);
  el.appendChild(inner);
  plotsContainer.appendChild(el);

  richBlocks.push({ marker, numRows, el, inner });
  updateRichBlocks();
}

function handleRich(payload) {
  renderRichInline(payload);
}

// Initialise plots overlay (must run after all let/const declarations above)
setupPlotsContainer();
// onRender fires after every render cycle (auto-scroll AND user scroll),
// ensuring plot positions stay in sync whenever the viewport changes.
term.onRender(() => { if (richBlocks.length > 0) updateRichBlocks(); });

// Focus terminal on click
document.getElementById('terminal').addEventListener('click', () => term.focus());
term.focus();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_pty_app(
    tui_argv: list[str],
    env_extra: dict[str, str] | None = None,
    auth_token: str | None = None,
) -> "tuple[Any, Any]":
    """Create the FastAPI PTY terminal application.

    Returns:
        A ``(app, kill_all)`` tuple.  ``kill_all()`` is a synchronous callable
        that terminates every active PTY process — call it from a SIGINT handler
        so connections drain before uvicorn's graceful shutdown timer expires.

    Args:
        tui_argv: Command + arguments to spawn in the PTY (e.g. ``["python", "-m", ...]``).
        env_extra: Extra environment variables to inject into the PTY process.
        auth_token: Per-session token required by every endpoint (via
            ``?token=`` query parameter or the cookie set by ``GET /``).
            ``None`` disables authentication entirely.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.websockets import WebSocket, WebSocketDisconnect  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Web terminal requires FastAPI. Install with: uv add fastapi uvicorn[standard]"
        ) from e

    try:
        import ptyprocess  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Web terminal requires ptyprocess. Install with: uv add ptyprocess"
        ) from e

    from contextlib import asynccontextmanager

    # Track all live PTY processes so the lifespan handler can kill them on shutdown.
    _active_procs: set = set()
    _procs_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        # Shutdown: kill all PTY subprocesses so they don't linger as orphans.
        with _procs_lock:
            procs = list(_active_procs)
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass

    app = FastAPI(
        title="NeMo OO Agents Terminal",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Session-token authentication.  The terminal is a full interactive shell,
    # so every endpoint validates the token (query param or cookie) when one
    # is configured.  auth_token=None disables the checks (--no-auth).
    def _token_valid(provided: str | None) -> bool:
        if auth_token is None:
            return True
        if not provided:
            return False
        # compare_digest(str, str) is ASCII-only and raises on non-ASCII input.
        # Compare UTF-8 bytes instead; malformed surrogate input fails closed.
        try:
            provided_bytes = provided.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return secrets.compare_digest(provided_bytes, auth_token.encode("utf-8"))

    def _http_authorized(request) -> bool:
        return _token_valid(request.query_params.get("token") or request.cookies.get(_TOKEN_COOKIE))

    def _ws_authorized(websocket) -> bool:
        return _token_valid(
            websocket.query_params.get("token") or websocket.cookies.get(_TOKEN_COOKIE)
        )

    # Rich content state shared across all browser connections to this server.
    # _rich_history: ordered list of payloads (up to _RICH_HISTORY_LIMIT).
    #   - populated from live POSTs (including replayed events on resume).
    #   - replayed to each new browser that connects to /ws/rich (handles
    #     page reload without restarting the TUI).
    _RICH_HISTORY_LIMIT = 200
    _rich_history: list[dict] = []
    _rich_subscribers: set[asyncio.Queue] = set()
    _rich_lock = threading.Lock()

    def _add_rich_subscriber(q: asyncio.Queue) -> None:
        with _rich_lock:
            _rich_subscribers.add(q)

    def _remove_rich_subscriber(q: asyncio.Queue) -> None:
        with _rich_lock:
            _rich_subscribers.discard(q)

    async def _broadcast_rich(payload: dict) -> None:
        with _rich_lock:
            subs = list(_rich_subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # Slow consumer; drop rather than block

    # -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if not _http_authorized(request):
            return HTMLResponse(
                "<h1>403 Forbidden</h1>"
                "<p>Missing or invalid session token. Open the URL printed at "
                "server startup (it includes <code>?token=...</code>).</p>",
                status_code=403,
            )
        html = _HTML
        if auth_token is not None:
            # Inject the token so the WS URLs carry it as a query param (see the
            # PTY WebSocket block). token_urlsafe() is JS/HTML-safe; json.dumps
            # yields a proper quoted string literal regardless.
            html = html.replace(
                "// ---- PTY WebSocket ----",
                f"window.__NEMO_TERM_TOKEN = {json.dumps(auth_token)};\n// ---- PTY WebSocket ----",
                1,
            )
        response = HTMLResponse(html)
        if auth_token is not None:
            # Also set a cookie as a fallback for the /rich fetch and any
            # same-origin reconnect that drops the query string.
            response.set_cookie(_TOKEN_COOKIE, auth_token, httponly=True, samesite="strict")
        return response

    # -----------------------------------------------------------------------
    @app.post("/rich")
    async def rich_post(request: Request):
        """Agent POSTs rich payloads here; forwarded to all /ws/rich subscribers."""
        if not _http_authorized(request):
            return JSONResponse({"error": "missing or invalid token"}, status_code=403)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        # Store in history for page-reload replay (skip "clear" — nothing to replay).
        # Strip _replay flag so a page reload treats the content as live.
        if payload.get("kind") != "clear":
            stored = {k: v for k, v in payload.items() if k != "_replay"}
            with _rich_lock:
                _rich_history.append(stored)
                if len(_rich_history) > _RICH_HISTORY_LIMIT:
                    del _rich_history[0]
        else:
            with _rich_lock:
                _rich_history.clear()
        await _broadcast_rich(payload)
        return JSONResponse({"ok": True})

    # -----------------------------------------------------------------------
    @app.websocket("/ws/rich")
    async def rich_ws(websocket: WebSocket):
        if not _ws_authorized(websocket):
            # Accept first so real uvicorn/browser clients observe close code 4403
            # instead of an HTTP-level handshake abort reported as 1006.
            await websocket.accept()
            await websocket.close(code=4403)
            return
        await websocket.accept()
        # Replay stored history before subscribing to live updates so a
        # page reload restores the rich panel without restarting the TUI.
        with _rich_lock:
            history_snapshot = list(_rich_history)
        for payload in history_snapshot:
            try:
                await websocket.send_json(payload)
            except Exception:
                return
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        _add_rich_subscriber(q)
        try:
            while True:
                payload = await q.get()
                await websocket.send_json(payload)
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            _remove_rich_subscriber(q)
            try:
                await websocket.close()
            except BaseException:
                pass

    # -----------------------------------------------------------------------
    @app.websocket("/ws/pty")
    async def pty_ws(websocket: WebSocket):
        """Bridge the browser xterm.js ↔ a real PTY process."""
        import ptyprocess

        if not _ws_authorized(websocket):
            # Accept first so real uvicorn/browser clients observe close code 4403
            # instead of an HTTP-level handshake abort reported as 1006. No PTY is
            # spawned for unauthenticated connections.
            await websocket.accept()
            await websocket.close(code=4403)
            return

        await websocket.accept()

        loop = asyncio.get_running_loop()

        # Queue of bytes chunks from the PTY → browser
        pty_out_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)

        # Merge base env with extras
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        # Spawn PTY
        try:
            proc = ptyprocess.PtyProcess.spawn(
                tui_argv,
                env=env,
                dimensions=(24, 80),
            )
        except Exception as exc:
            _log.error("Failed to spawn PTY process %s: %s", tui_argv, exc)
            try:
                msg = f"\r\n\x1b[31mFailed to start terminal: {exc}\x1b[0m\r\n"
                await websocket.send_json({"t": "o", "d": _b64(msg.encode())})
            except Exception:
                pass
            await websocket.close()
            return

        # Background thread: read PTY output → asyncio queue
        def _read_loop():
            while True:
                try:
                    data = proc.read(4096)
                    if data:
                        loop.call_soon_threadsafe(pty_out_q.put_nowait, data)
                except EOFError:
                    break
                except Exception as exc:
                    _log.debug("PTY read: %s", exc)
                    break
            loop.call_soon_threadsafe(pty_out_q.put_nowait, None)  # sentinel

        threading.Thread(target=_read_loop, daemon=True).start()
        with _procs_lock:
            _active_procs.add(proc)

        # Forward PTY output → WebSocket
        async def _forward_output():
            while True:
                chunk = await pty_out_q.get()
                if chunk is None:
                    break  # PTY EOF
                try:
                    await websocket.send_json({"t": "o", "d": _b64(chunk)})
                except Exception:
                    break

        # Receive WebSocket messages → PTY input / resize
        async def _handle_input():
            try:
                while True:
                    raw = await websocket.receive_text()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    t = msg.get("t")
                    if t == "i":
                        data = base64.b64decode(msg["d"])
                        try:
                            proc.write(data)
                        except Exception:
                            return
                    elif t == "r":
                        cols = int(msg.get("cols", 80))
                        rows = int(msg.get("rows", 24))
                        try:
                            proc.setwinsize(rows, cols)
                        except Exception:
                            pass
            except Exception:
                pass

        # Run both sides; cancel the other when either finishes
        out_task = asyncio.create_task(_forward_output())
        inp_task = asyncio.create_task(_handle_input())
        try:
            _done, pending = await asyncio.wait(
                [out_task, inp_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            with _procs_lock:
                _active_procs.discard(proc)
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                await websocket.close()
            except Exception:
                pass

    def kill_all() -> None:
        """Terminate every active PTY subprocess (call on SIGINT/SIGTERM)."""
        with _procs_lock:
            procs = list(_active_procs)
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass

    return app, kill_all


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
