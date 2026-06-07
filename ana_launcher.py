#!/usr/bin/env python3
"""
ANA Launcher — Web Server + CLI
Starts a local web UI on localhost (or a cloud port) and serves the ANA agent.
Usage:
  python ana_launcher.py                        # local, port 5000, opens browser
  python ana_launcher.py --port 8080            # custom port
  python ana_launcher.py --no-browser           # skip auto-open (for cloud/server)
  python ana_launcher.py --port 8080 --no-browser
"""

import sys
import os
import json
import threading
import webbrowser
import time
import socket
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mini_100_neuron_agent import ANAAgent
except ImportError as e:
    print(f"[ERROR] Could not import ANA modules: {e}")
    sys.exit(1)

# ── Global agent instance ─────────────────────────────────────────────────────
agent = ANAAgent()

# ── Embedded Web UI HTML ──────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ANA — Autonomous Neuron Agent</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0d0f14; --surface: #151820; --surface2: #1c2030;
      --border: #252a3a; --accent: #6c8eff; --accent2: #a78bfa;
      --text: #e2e8f0; --muted: #64748b;
      --user-bg: #1e2a4a; --ana-bg: #161e2e;
      --success: #34d399; --danger: #f87171;
    }
    body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; height:100vh; display:flex; flex-direction:column; overflow:hidden; }

    header { background:var(--surface); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; gap:14px; flex-shrink:0; }
    .logo { width:38px;height:38px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:16px;color:white;flex-shrink:0; }
    .header-info h1 { font-size:17px;font-weight:700; }
    .header-info p  { font-size:12px;color:var(--muted);margin-top:1px; }
    .status-dot { width:8px;height:8px;background:var(--success);border-radius:50%;margin-left:auto;box-shadow:0 0 8px var(--success);animation:pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.4} }

    .main { display:flex;flex:1;overflow:hidden; }

    .sidebar { width:220px;background:var(--surface);border-right:1px solid var(--border);padding:16px 12px;display:flex;flex-direction:column;gap:6px;flex-shrink:0;overflow-y:auto; }
    .sidebar-title { font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);padding:0 8px;margin-bottom:4px; }
    .cmd-btn { background:transparent;border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer;font-size:12px;text-align:left;transition:all 0.15s;font-family:inherit;width:100%; }
    .cmd-btn:hover { background:var(--surface2);border-color:var(--accent);color:var(--accent); }
    .cmd-btn .cmd-label { font-weight:600;display:block; }
    .cmd-btn .cmd-desc  { color:var(--muted);font-size:11px;margin-top:2px; }
    .sidebar-sep { height:1px;background:var(--border);margin:8px 0; }

    .chat-area { flex:1;display:flex;flex-direction:column;overflow:hidden; }
    #messages { flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth; }
    #messages::-webkit-scrollbar{width:5px}
    #messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

    .msg { display:flex;gap:12px;max-width:780px;animation:fadeIn 0.2s ease; }
    @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1}}
    .msg.user { align-self:flex-end;flex-direction:row-reverse; }
    .msg.ana  { align-self:flex-start; }

    .avatar { width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0; }
    .msg.user .avatar { background:linear-gradient(135deg,#3b5bdb,#6c8eff);color:white; }
    .msg.ana  .avatar { background:linear-gradient(135deg,#5b21b6,#a78bfa);color:white; }

    .bubble { padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.55;max-width:600px; }
    .msg.user .bubble { background:var(--user-bg);border:1px solid #2d3f6b;border-bottom-right-radius:3px; }
    .msg.ana  .bubble { background:var(--ana-bg);border:1px solid var(--border);border-bottom-left-radius:3px; }

    .reasoning { font-size:11px;color:var(--muted);margin-top:6px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:2px solid var(--accent2);font-style:italic; }

    .system-msg { align-self:center;background:var(--surface2);border:1px solid var(--border);padding:6px 14px;border-radius:20px;font-size:12px;color:var(--muted); }

    .dot { width:6px;height:6px;background:var(--muted);border-radius:50%;animation:bounce 1.2s infinite; }
    .dot:nth-child(2){animation-delay:0.2s}.dot:nth-child(3){animation-delay:0.4s}
    @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}

    .input-bar { padding:16px 24px;background:var(--surface);border-top:1px solid var(--border);display:flex;gap:10px;align-items:flex-end;flex-shrink:0; }
    #input { flex:1;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:10px 14px;border-radius:10px;font-size:14px;font-family:inherit;resize:none;min-height:42px;max-height:120px;outline:none;transition:border-color 0.15s;line-height:1.5; }
    #input:focus{border-color:var(--accent)}
    #input::placeholder{color:var(--muted)}
    #send-btn { background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;color:white;width:42px;height:42px;border-radius:10px;cursor:pointer;font-size:18px;display:flex;align-items:center;justify-content:center;transition:opacity 0.15s,transform 0.1s;flex-shrink:0; }
    #send-btn:hover{opacity:0.85}#send-btn:active{transform:scale(0.95)}

    #neuron-panel{display:none;padding:16px 24px;background:var(--surface2);border-top:1px solid var(--border);font-size:12px;}
    #neuron-panel.open{display:block}
    .neuron-title{font-weight:600;margin-bottom:10px;color:var(--accent)}
    .layer-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
    .layer-name{width:60px;color:var(--muted);font-size:11px}
    .bar-track{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden}
    .bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width 0.4s}
    .bar-val{width:50px;text-align:right;color:var(--muted);font-size:11px}
  </style>
</head>
<body>
<header>
  <div class="logo">A</div>
  <div class="header-info">
    <h1>ANA — Autonomous Neuron Agent</h1>
    <p>100-Neuron Brain &nbsp;·&nbsp; Real Reasoning &nbsp;·&nbsp; Local AI</p>
  </div>
  <div class="status-dot" title="Running"></div>
</header>
<div class="main">
  <aside class="sidebar">
    <div class="sidebar-title">Commands</div>
    <button class="cmd-btn" onclick="runCmd('/neurons')"><span class="cmd-label">🧠 Neurons</span><span class="cmd-desc">Show activation heatmap</span></button>
    <button class="cmd-btn" onclick="runCmd('/history')"><span class="cmd-label">📜 History</span><span class="cmd-desc">View conversation log</span></button>
    <button class="cmd-btn" onclick="runCmd('/clear')"><span class="cmd-label">🗑️ Clear</span><span class="cmd-desc">Clear history</span></button>
    <button class="cmd-btn" onclick="runCmd('/save')"><span class="cmd-label">💾 Save State</span><span class="cmd-desc">Save to ana_state.json</span></button>
    <button class="cmd-btn" onclick="runCmd('/load')"><span class="cmd-label">📂 Load State</span><span class="cmd-desc">Load from ana_state.json</span></button>
    <div class="sidebar-sep"></div>
    <div class="sidebar-title">Training</div>
    <button class="cmd-btn" onclick="openTrain('memory')"><span class="cmd-label">📚 Train Memory</span><span class="cmd-desc">/train key: value</span></button>
    <button class="cmd-btn" onclick="openTrain('brain')"><span class="cmd-label">⚡ Train Brain</span><span class="cmd-desc">/learn text → action</span></button>
  </aside>
  <div class="chat-area">
    <div id="messages">
      <div class="system-msg">ANA is running · All processing on-device · Zero external API calls</div>
      <div class="msg ana">
        <div class="avatar">A</div>
        <div><div class="bubble">Hello! I'm ANA — Autonomous Neuron Agent. I have a 100-neuron brain running entirely on this machine. Ask me anything, or use the sidebar to inspect my neural state.</div></div>
      </div>
    </div>
    <div id="typing-wrap" style="padding:0 24px 8px;display:none">
      <div class="msg ana">
        <div class="avatar">A</div>
        <div class="bubble" style="padding:12px 16px">
          <div style="display:flex;gap:5px;align-items:center"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
        </div>
      </div>
    </div>
    <div id="neuron-panel"><div class="neuron-title">🧠 Neuron Activation Heatmap</div><div id="neuron-bars"></div></div>
    <div class="input-bar">
      <textarea id="input" placeholder="Message ANA..." rows="1"></textarea>
      <button id="send-btn" onclick="sendMsg()">↑</button>
    </div>
  </div>
</div>
<script>
  const input=document.getElementById('input'),messages=document.getElementById('messages'),typingWrap=document.getElementById('typing-wrap');
  input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,120)+'px'});
  input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg()}});
  function scrollBottom(){messages.scrollTop=messages.scrollHeight}
  function addMsg(role,text,reasoning){
    const wrap=document.createElement('div');wrap.className=`msg ${role}`;
    const avatar=document.createElement('div');avatar.className='avatar';avatar.textContent=role==='user'?'U':'A';
    const right=document.createElement('div');
    const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=text;
    right.appendChild(bubble);
    if(reasoning&&reasoning.length>0){const r=document.createElement('div');r.className='reasoning';r.textContent='🔗 '+reasoning.join(' → ');right.appendChild(r)}
    if(role==='user'){wrap.appendChild(right);wrap.appendChild(avatar)}else{wrap.appendChild(avatar);wrap.appendChild(right)}
    messages.appendChild(wrap);scrollBottom()
  }
  function addSystem(text){const d=document.createElement('div');d.className='system-msg';d.textContent=text;messages.appendChild(d);scrollBottom()}
  async function sendMsg(){
    const text=input.value.trim();if(!text)return;
    input.value='';input.style.height='auto';
    if(text.startsWith('/')){runCmd(text);return}
    addMsg('user',text,null);typingWrap.style.display='block';scrollBottom();
    try{
      const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
      const data=await res.json();typingWrap.style.display='none';addMsg('ana',data.response,data.reasoning)
    }catch(e){typingWrap.style.display='none';addMsg('ana','Error communicating with ANA.',null)}
  }
  async function runCmd(cmd){
    addMsg('user',cmd,null);typingWrap.style.display='block';scrollBottom();
    try{
      const res=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd})});
      const data=await res.json();typingWrap.style.display='none';
      if(data.type==='neurons'&&data.heatmap)renderNeurons(data.heatmap);else addMsg('ana',data.response,null)
    }catch(e){typingWrap.style.display='none';addMsg('ana','Command error.',null)}
  }
  function openTrain(type){
    if(type==='memory'){const v=prompt('Train memory\\nFormat: key: value');if(v)runCmd('/train '+v)}
    else{const v=prompt('Train brain\\nFormat: text -> action');if(v)runCmd('/learn '+v)}
  }
  function renderNeurons(heatmap){
    const panel=document.getElementById('neuron-panel'),bars=document.getElementById('neuron-bars');
    bars.innerHTML='';let maxVal=0;
    const entries=Object.entries(heatmap);
    entries.forEach(([l,acts])=>{const avg=acts.reduce((s,a)=>s+Math.abs(a),0)/acts.length;if(avg>maxVal)maxVal=avg});
    entries.forEach(([l,acts])=>{
      const avg=acts.reduce((s,a)=>s+Math.abs(a),0)/acts.length;
      const pct=maxVal>0?(avg/maxVal*100).toFixed(1):0;
      const row=document.createElement('div');row.className='layer-row';
      row.innerHTML=`<div class="layer-name">${l}</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><div class="bar-val">${avg.toFixed(4)}</div>`;
      bars.appendChild(row)
    });
    panel.classList.toggle('open');addSystem('Neural heatmap updated ↓');scrollBottom()
  }
</script>
</body>
</html>"""

# ── HTTP Request Handler ──────────────────────────────────────────────────────
class ANAHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/api/chat":
            msg = data.get("message", "").strip()
            if not msg:
                self.send_json({"error": "Empty message"}, 400)
                return
            try:
                result = agent.process_input(msg)
                self.send_json({"response": result.get("response",""), "reasoning": result.get("reasoning",[])})
            except Exception as e:
                self.send_json({"response": f"Error: {e}", "reasoning": []})

        elif self.path == "/api/command":
            cmd = data.get("command", "").strip()
            try:
                self.send_json(handle_command(cmd))
            except Exception as e:
                self.send_json({"type": "text", "response": f"Error: {e}"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

# ── Command dispatcher ────────────────────────────────────────────────────────
def handle_command(cmd):
    if cmd == "/neurons":
        return {"type": "neurons", "heatmap": agent.get_neuron_heatmap(), "response": ""}
    elif cmd == "/history":
        history = agent.get_history()
        if not history: return {"type": "text", "response": "No history yet."}
        lines = [f"{i+1}. {e['input'][:60]}... → {e['action']}" for i,e in enumerate(history)]
        return {"type": "text", "response": "\n".join(lines)}
    elif cmd == "/clear":
        return {"type": "text", "response": agent.clear_history()}
    elif cmd.startswith("/save"):
        parts = cmd.split(); fp = parts[1] if len(parts)>1 else "ana_state.json"
        return {"type": "text", "response": agent.save_state(fp)}
    elif cmd.startswith("/load"):
        parts = cmd.split(); fp = parts[1] if len(parts)>1 else "ana_state.json"
        try: return {"type": "text", "response": agent.load_state(fp)}
        except Exception as e: return {"type": "text", "response": f"Error loading: {e}"}
    elif cmd.startswith("/train "):
        parts = cmd[7:].split(":",1)
        if len(parts)==2: return {"type":"text","response": agent.train_memory(parts[0].strip(),parts[1].strip())}
        return {"type": "text", "response": "Usage: /train key: value"}
    elif cmd.startswith("/learn "):
        parts = cmd[7:].split("->",1)
        if len(parts)==2: return {"type":"text","response": agent.train_brain(parts[0].strip(),parts[1].strip())}
        return {"type": "text", "response": "Usage: /learn text -> action"}
    elif cmd == "/exit":
        return {"type": "text", "response": "Use Ctrl+C in terminal to exit."}
    else:
        return {"type": "text", "response": f"Unknown command: {cmd}"}

# ── Find free port ────────────────────────────────────────────────────────────
def find_free_port(start=5000):
    for port in range(start, start+20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try: s.bind(("0.0.0.0", port)); return port
            except OSError: continue
    return start

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ANA Web Launcher")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Skip auto-opening browser")
    args = parser.parse_args()

    # Railway / Render inject PORT env var
    env_port = int(os.environ.get("PORT", 0))
    port = args.port or env_port or find_free_port(5000)
    host = "0.0.0.0"  # bind all interfaces so cloud can reach it

    url = f"http://localhost:{port}"
    server = HTTPServer((host, port), ANAHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    print()
    print("=" * 60)
    print("  ANA — Autonomous Neuron Agent")
    print("  100-Neuron Brain | Real Reasoning | Web UI")
    print("=" * 60)
    print()
    print(f"  🌐  Web UI:  {url}")
    print()
    if not args.no_browser:
        print("  Opening browser automatically...")
        time.sleep(0.8)
        try: webbrowser.open(url)
        except Exception: pass
    else:
        print("  Running in server mode (no browser)")
    print()
    print("  Press Ctrl+C to stop ANA.")
    print("=" * 60)
    print()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down ANA. Goodbye!")
        server.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()
