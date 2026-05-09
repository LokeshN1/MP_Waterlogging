"""
Waterlogging Detection Dashboard — Flask web version.

Replaces the tkinter GUI (requires a compiled _tkinter extension) with a
browser-based dashboard served by Flask.  Works on Linux, macOS, and Windows.

Usage:
    python snippets_detector_gui.py
Then open http://127.0.0.1:5000 in any browser (or it opens automatically).
"""

import base64
import os
import threading
import time
import webbrowser
from io import BytesIO

from flask import Flask, Response, jsonify, render_template_string, send_file

from waterlogging_detection import ModelUnavailableError, predict_waterlogging

# ── Config ────────────────────────────────────────────────────────────────────
SNIPPETS_DIR = "snippets"
ANALYSIS_DIR = "analyis"
REFRESH_INTERVAL_MS = 3000
MAX_HISTORY = 20
MAX_ALERT_LOG = 25
ALERT_CONFIDENCE_DEFAULT = 0.75
ALERT_COOLDOWN_SEC = 12
ANALYSIS_DELAY_SEC = 1.5
PORT = 5000

os.makedirs(SNIPPETS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Shared State ──────────────────────────────────────────────────────────────
state: dict = {
    "last_image_path": "",
    "last_prediction": None,          # (label, confidence) or None
    "waterlogged_count": 0,
    "non_waterlogged_count": 0,
    "alert_enabled": True,
    "alert_sound_enabled": False,      # no winsound on Linux
    "alert_threshold": ALERT_CONFIDENCE_DEFAULT,
    "last_alert_time": 0.0,
    "start_time": time.time(),
    "history": [],                     # list of dicts
    "alert_log": [],                   # list of dicts
    "model_status": "Checking...",
    "mode": "Live snippets feed",
    "verdict": "Waiting",
    "verdict_color": "#4a5568",
    "confidence": 0.0,
    "frame_name": "-",
    "last_update": "-",
    "demo_note": "-",
    "is_alerting": False,
}
state_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_latest_snippet() -> str:
    jpgs = [f for f in os.listdir(SNIPPETS_DIR) if f.lower().endswith(".jpg")]
    if not jpgs:
        return ""
    return os.path.join(SNIPPETS_DIR, sorted(jpgs)[-1])


def _run_prediction(img_path: str) -> None:
    """Run model inference and update shared state (called from background thread)."""
    is_analysis = os.path.basename(os.path.dirname(os.path.abspath(img_path))).lower() == "analyis"
    try:
        label, confidence, _ = predict_waterlogging(img_path)
        frame_base = os.path.basename(img_path)
        timestamp = time.strftime("%H:%M:%S")
        verdict_color = "#c53030" if label == "Waterlogged" else "#2f855a"

        with state_lock:
            state["model_status"] = "Demo mode" if is_analysis else "Online"
            state["last_prediction"] = (label, confidence)
            state["last_image_path"] = img_path
            state["verdict"] = label
            state["verdict_color"] = verdict_color
            state["confidence"] = confidence
            state["frame_name"] = frame_base
            state["last_update"] = timestamp

            if label == "Waterlogged":
                state["waterlogged_count"] += 1
            else:
                state["non_waterlogged_count"] += 1

            # History
            state["history"].insert(
                0, {"time": timestamp, "frame": frame_base, "label": label, "confidence": f"{confidence * 100:.2f}%"}
            )
            if len(state["history"]) > MAX_HISTORY:
                state["history"] = state["history"][:MAX_HISTORY]

            # Alert
            threshold = state["alert_threshold"]
            should_alert = state["alert_enabled"] and label == "Waterlogged" and confidence >= threshold
            state["is_alerting"] = should_alert
            if should_alert:
                now = time.time()
                if now - state["last_alert_time"] >= ALERT_COOLDOWN_SEC:
                    state["last_alert_time"] = now
                    msg = f"Waterlogging detected with {confidence * 100:.2f}% confidence"
                    state["alert_log"].insert(0, {"time": timestamp, "frame": frame_base, "message": msg})
                    if len(state["alert_log"]) > MAX_ALERT_LOG:
                        state["alert_log"] = state["alert_log"][:MAX_ALERT_LOG]
    except ModelUnavailableError as exc:
        with state_lock:
            state["model_status"] = "Unavailable"
            state["verdict"] = f"Model error: {exc}"
            state["verdict_color"] = "#c53030"
            state["confidence"] = 0.0
    except Exception as exc:
        with state_lock:
            state["model_status"] = "Runtime Error"
            state["verdict"] = f"Detection error: {exc}"
            state["verdict_color"] = "#c53030"
            state["confidence"] = 0.0


def _background_monitor() -> None:
    """Poll for new snippets every REFRESH_INTERVAL_MS and run inference."""
    while True:
        latest = _get_latest_snippet()
        with state_lock:
            current = state["last_image_path"]
        if latest and latest != current:
            _run_prediction(latest)
        time.sleep(REFRESH_INTERVAL_MS / 1000)


# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Flood Vision Console — Waterlogging Detection</title>
  <meta name="description" content="Live waterlogging detection dashboard monitoring the snippets folder with AI-powered frame analysis." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0f172a;
      --surface: #1e293b;
      --surface2: #263248;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #4ade80;
      --red: #f87171;
      --orange: #fb923c;
      --radius: 12px;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
    }

    h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }
    h2 { font-size: 1rem; font-weight: 600; color: var(--accent); margin-bottom: 12px; }

    .subtitle { color: var(--muted); font-size: 0.85rem; margin-top: 4px; }

    /* Alert banner */
    #alert-banner {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 0.85rem;
      transition: background 0.3s;
    }
    #alert-banner.normal  { background: #166534; color: #bbf7d0; }
    #alert-banner.alerting { background: #7f1d1d; color: #fecaca; animation: pulse 1s infinite alternate; }
    @keyframes pulse { from { opacity: 1; } to { opacity: 0.7; } }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 24px;
    }

    /* Grid layout */
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
      margin-bottom: 20px;
    }
    @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

    /* Cards */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
    }

    /* Preview */
    #preview-wrap {
      aspect-ratio: 16/9;
      background: #070d1a;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      margin: 12px 0;
      min-height: 240px;
    }
    #preview-wrap img { width: 100%; height: 100%; object-fit: contain; }
    #no-frame-msg {
      color: var(--muted);
      font-size: 0.9rem;
      text-align: center;
      padding: 16px;
    }

    /* Action buttons */
    .btn-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 4px; }
    button {
      padding: 8px 16px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.82rem;
      font-weight: 600;
      border: none;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
    }
    button:hover { opacity: 0.88; transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    .btn-blue   { background: #2563eb; color: #fff; }
    .btn-green  { background: #16a34a; color: #fff; }
    .btn-gray   { background: #475569; color: #fff; }
    .btn-upload { background: #7c3aed; color: #fff; }

    /* Stat rows */
    .stat { display: flex; flex-direction: column; margin-bottom: 14px; }
    .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }
    .stat-value { font-size: 1.1rem; font-weight: 600; }

    /* Verdict badge */
    #verdict-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 1rem;
      letter-spacing: 0.02em;
    }

    /* Confidence bar */
    .bar-wrap { background: var(--surface2); border-radius: 999px; height: 8px; margin: 8px 0 4px; overflow: hidden; }
    #conf-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #38bdf8, #818cf8); transition: width 0.5s ease; width: 0%; }

    /* Sliders / toggles */
    .toggle-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; font-size: 0.85rem; }
    input[type="checkbox"] { accent-color: var(--accent); width: 16px; height: 16px; cursor: pointer; }
    input[type="range"]    { accent-color: var(--accent); flex: 1; cursor: pointer; }
    #threshold-label { font-size: 0.85rem; color: var(--accent); font-weight: 600; min-width: 40px; text-align: right; }

    /* Tables */
    .table-wrap { overflow-x: auto; margin-top: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    thead th {
      text-align: left;
      padding: 8px 12px;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      font-weight: 500;
    }
    tbody td { padding: 7px 12px; border-bottom: 1px solid #1e293b; }
    tbody tr:hover { background: var(--surface2); }
    tbody tr:last-child td { border-bottom: none; }
    .label-water { color: var(--red); font-weight: 600; }
    .label-ok    { color: var(--green); font-weight: 600; }

    /* Uptime chip */
    #uptime-chip {
      display: inline-block;
      background: var(--surface2);
      border-radius: 999px;
      padding: 3px 12px;
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 6px;
    }

    /* File input hidden */
    #file-input { display: none; }

    /* Separator */
    hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

    /* Analysis note */
    #demo-note { font-size: 0.8rem; color: var(--orange); min-height: 18px; margin-top: 4px; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>🌊 Flood Vision Console</h1>
    <p class="subtitle">Live frame monitoring · AI-powered waterlogging detection</p>
  </div>
  <div id="alert-banner" class="normal">✅ ALERT STATUS: NORMAL</div>
</header>

<div class="main-grid">

  <!-- Preview card -->
  <div class="card" id="preview-card">
    <h2>Live Preview</h2>
    <div id="preview-wrap">
      <div id="no-frame-msg">Waiting for frames in the snippets folder…</div>
      <img id="preview-img" src="" alt="Latest frame" style="display:none" />
    </div>
    <p id="demo-note"></p>
    <div class="btn-row">
      <button class="btn-blue"   id="btn-refresh"  onclick="fetchStatus()">⟳ Refresh Now</button>
      <button class="btn-upload" id="btn-upload"   onclick="document.getElementById('file-input').click()">📂 Analyse Image</button>
      <button class="btn-gray"   id="btn-live"     onclick="clearAnalysis()">↩ Back to Live Feed</button>
    </div>
    <input type="file" id="file-input" accept=".jpg,.jpeg,.png" onchange="uploadAnalysis(event)" />
  </div>

  <!-- Side panel -->
  <div class="card">
    <h2>System Status</h2>

    <div class="stat">
      <span class="stat-label">Model</span>
      <span class="stat-value" id="model-status">Checking…</span>
    </div>

    <div class="stat">
      <span class="stat-label">Latest Frame</span>
      <span class="stat-value" style="font-size:0.85rem;word-break:break-all;" id="frame-name">-</span>
    </div>

    <div class="stat">
      <span class="stat-label">Last Update</span>
      <span class="stat-value" style="font-size:0.85rem;" id="last-update">-</span>
    </div>

    <hr />

    <div class="stat">
      <span class="stat-label">Verdict</span>
      <span id="verdict-badge" style="background:#1e3a5f;color:#93c5fd;">Waiting</span>
    </div>

    <div class="bar-wrap"><div id="conf-bar"></div></div>
    <p style="font-size:0.8rem;color:var(--muted);" id="conf-text">Confidence: -</p>

    <hr />

    <div class="stat">
      <span class="stat-label">Session Counts</span>
      <span class="stat-value" style="font-size:0.85rem;" id="counters">Waterlogged: 0 | Non-Waterlogged: 0</span>
    </div>

    <div id="uptime-chip">Uptime: 0s</div>
    <p style="font-size:0.8rem;color:var(--muted);margin-top:8px;" id="mode-text">Mode: Live snippets feed</p>

    <hr />

    <h2 style="margin-bottom:8px;">Alert Controls</h2>

    <div class="toggle-row">
      <input type="checkbox" id="chk-alert" checked onchange="setAlertEnabled(this.checked)" />
      <label for="chk-alert">Enable alerts</label>
    </div>

    <div class="toggle-row" style="margin-top:4px;">
      <label for="range-threshold" style="color:var(--muted);">Trigger confidence</label>
      <input type="range" id="range-threshold" min="50" max="99" value="75" step="1" oninput="setThreshold(this.value)" />
      <span id="threshold-label">75%</span>
    </div>

    <p style="font-size:0.78rem;color:var(--muted);margin-top:4px;" id="cooldown-text">Cooldown: 12s</p>
  </div>

</div>

<!-- Recent Predictions -->
<div class="card" style="margin-bottom:20px;">
  <h2>Recent Predictions</h2>
  <div class="table-wrap">
    <table id="history-table">
      <thead>
        <tr>
          <th>Time</th><th>Frame</th><th>Label</th><th>Confidence</th>
        </tr>
      </thead>
      <tbody id="history-body"></tbody>
    </table>
  </div>
</div>

<!-- Alert Log -->
<div class="card">
  <h2>Alert Log</h2>
  <div class="table-wrap">
    <table id="alert-table">
      <thead>
        <tr>
          <th>Time</th><th>Frame</th><th>Message</th>
        </tr>
      </thead>
      <tbody id="alert-body"></tbody>
    </table>
  </div>
</div>

<script>
  let analysisMode = false;
  let analysisImageB64 = null;
  let startTime = Date.now();

  /* ── Fetch status from API ── */
  async function fetchStatus() {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      applyStatus(d);
    } catch(e) { console.error('Status fetch error', e); }
  }

  function applyStatus(d) {
    // Uptime
    const up = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('uptime-chip').textContent = `Uptime: ${up}s`;

    // Alert banner
    const banner = document.getElementById('alert-banner');
    if (d.is_alerting) {
      banner.className = 'alerting';
      banner.textContent = '⚠️ ALERT STATUS: WATERLOGGING RISK';
    } else {
      banner.className = 'normal';
      banner.textContent = '✅ ALERT STATUS: NORMAL';
    }

    // Stats
    document.getElementById('model-status').textContent = d.model_status;
    document.getElementById('frame-name').textContent = d.frame_name;
    document.getElementById('last-update').textContent = d.last_update;
    document.getElementById('mode-text').textContent = 'Mode: ' + d.mode;
    document.getElementById('counters').textContent =
      `Waterlogged: ${d.waterlogged_count} | Non-Waterlogged: ${d.non_waterlogged_count}`;
    document.getElementById('demo-note').textContent = d.demo_note !== '-' ? '📌 ' + d.demo_note : '';

    // Verdict badge
    const badge = document.getElementById('verdict-badge');
    badge.textContent = d.verdict;
    if (d.verdict === 'Waterlogged') {
      badge.style.background = '#450a0a'; badge.style.color = '#fca5a5';
    } else if (d.verdict === 'Non-Waterlogged') {
      badge.style.background = '#052e16'; badge.style.color = '#86efac';
    } else {
      badge.style.background = '#1e3a5f'; badge.style.color = '#93c5fd';
    }

    // Confidence bar
    const pct = (d.confidence * 100).toFixed(2);
    document.getElementById('conf-bar').style.width = pct + '%';
    document.getElementById('conf-text').textContent = d.confidence > 0 ? `Confidence: ${pct}%` : 'Confidence: -';

    // Preview image (live feed only in non-analysis mode)
    if (!analysisMode && d.has_image) {
      loadPreviewFromApi();
    }

    // History
    const hBody = document.getElementById('history-body');
    hBody.innerHTML = '';
    (d.history || []).forEach(row => {
      const cls = row.label === 'Waterlogged' ? 'label-water' : 'label-ok';
      hBody.insertAdjacentHTML('beforeend',
        `<tr><td>${row.time}</td><td style="font-size:0.78rem;">${row.frame}</td><td class="${cls}">${row.label}</td><td>${row.confidence}</td></tr>`
      );
    });

    // Alert log
    const aBody = document.getElementById('alert-body');
    aBody.innerHTML = '';
    (d.alert_log || []).forEach(row => {
      aBody.insertAdjacentHTML('beforeend',
        `<tr><td>${row.time}</td><td style="font-size:0.78rem;">${row.frame}</td><td style="color:var(--red)">${row.message}</td></tr>`
      );
    });
  }

  async function loadPreviewFromApi() {
    try {
      const r = await fetch('/api/preview?t=' + Date.now());
      if (!r.ok) { showNoFrame(); return; }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const img = document.getElementById('preview-img');
      img.src = url;
      img.style.display = 'block';
      document.getElementById('no-frame-msg').style.display = 'none';
    } catch(e) { showNoFrame(); }
  }

  function showNoFrame() {
    document.getElementById('preview-img').style.display = 'none';
    document.getElementById('no-frame-msg').style.display = 'block';
  }

  /* ── Analysis image upload ── */
  async function uploadAnalysis(event) {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    try {
      const r = await fetch('/api/analyse', { method: 'POST', body: formData });
      const d = await r.json();
      analysisMode = true;
      document.getElementById('mode-text').textContent = 'Mode: Analysis image';
      // Show image in preview
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.getElementById('preview-img');
        img.src = e.target.result;
        img.style.display = 'block';
        document.getElementById('no-frame-msg').style.display = 'none';
      };
      reader.readAsDataURL(file);
      applyStatus(d);
    } catch(e) { alert('Analysis failed: ' + e); }
    event.target.value = '';
  }

  function clearAnalysis() {
    analysisMode = false;
    fetch('/api/clear_analysis', { method: 'POST' });
    document.getElementById('mode-text').textContent = 'Mode: Live snippets feed';
    fetchStatus();
  }

  /* ── Alert controls ── */
  async function setAlertEnabled(val) {
    await fetch('/api/set_alert_enabled', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: val }) });
  }

  async function setThreshold(val) {
    document.getElementById('threshold-label').textContent = val + '%';
    const t = parseInt(val, 10) / 100;
    await fetch('/api/set_threshold', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ threshold: t }) });
    document.getElementById('cooldown-text').textContent = `Cooldown: ${12}s | Threshold: ${val}%`;
  }

  /* ── Poll every REFRESH_INTERVAL_MS ── */
  setInterval(fetchStatus, {{ refresh_ms }});
  fetchStatus();
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, refresh_ms=REFRESH_INTERVAL_MS)


@app.route("/api/status")
def api_status():
    with state_lock:
        s = dict(state)
    uptime = int(time.time() - s["start_time"])
    return jsonify(
        model_status=s["model_status"],
        frame_name=s["frame_name"],
        last_update=s["last_update"],
        verdict=s["verdict"],
        verdict_color=s["verdict_color"],
        confidence=s["confidence"],
        waterlogged_count=s["waterlogged_count"],
        non_waterlogged_count=s["non_waterlogged_count"],
        uptime=uptime,
        mode=s["mode"],
        demo_note=s["demo_note"],
        is_alerting=s["is_alerting"],
        history=s["history"],
        alert_log=s["alert_log"],
        has_image=bool(s["last_image_path"] or _get_latest_snippet()),
    )


@app.route("/api/preview")
def api_preview():
    img_path = _get_latest_snippet()
    if not img_path:
        return Response("No image", status=404)
    return send_file(img_path, mimetype="image/jpeg")


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    from flask import request as freq
    if "image" not in freq.files:
        return jsonify(error="No image"), 400
    f = freq.files["image"]
    dest = os.path.join(ANALYSIS_DIR, f.filename or "upload.jpg")
    f.save(dest)
    _run_prediction(dest)
    with state_lock:
        s = dict(state)
        s["mode"] = "Analysis image"
        state["mode"] = "Analysis image"
        note = "Analysis note: Educational demo prediction active for selected analysis image."
        state["demo_note"] = note
        s["demo_note"] = note
    return api_status().get_json(), 200


@app.route("/api/clear_analysis", methods=["POST"])
def api_clear_analysis():
    with state_lock:
        state["mode"] = "Live snippets feed"
        state["demo_note"] = "-"
        state["last_image_path"] = ""
        state["last_prediction"] = None
    return jsonify(ok=True)


@app.route("/api/set_alert_enabled", methods=["POST"])
def api_set_alert_enabled():
    from flask import request as freq
    data = freq.get_json(force=True)
    with state_lock:
        state["alert_enabled"] = bool(data.get("enabled", True))
    return jsonify(ok=True)


@app.route("/api/set_threshold", methods=["POST"])
def api_set_threshold():
    from flask import request as freq
    data = freq.get_json(force=True)
    with state_lock:
        state["alert_threshold"] = float(data.get("threshold", ALERT_CONFIDENCE_DEFAULT))
    return jsonify(ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Start background monitor thread
    t = threading.Thread(target=_background_monitor, daemon=True)
    t.start()

    url = f"http://127.0.0.1:{PORT}"
    print(f"\n🌊 Flood Vision Console running at  {url}")
    print("   Press Ctrl+C to stop.\n")

    # Open browser after a short delay
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
