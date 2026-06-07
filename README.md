<div align="center">

<img src="https://media.base44.com/images/public/6a23b82bd84dd64b846ca29c/e6ead34e5_generated_image.png" width="100%" alt="ANA Chat Interface" />

# ANA — Autonomous Neuron Agent

**A self-contained local AI agent with a 100-neuron neural network brain.**  
Zero cloud APIs. Zero external dependencies. Runs entirely on your machine.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Deploy on Railway](https://img.shields.io/badge/Deploy-Railway-blueviolet?style=flat-square&logo=railway)](https://railway.app/new/template)
[![No Dependencies](https://img.shields.io/badge/Dependencies-Zero-orange?style=flat-square)](requirements.txt)

</div>

---

## What is ANA?

ANA (**A**utonomous **N**euron **A**gent) is a fully self-contained AI agent built from scratch in pure Python. No PyTorch. No TensorFlow. No OpenAI API. Just raw neural network mathematics and a sleek web interface you can run anywhere.

**Architecture:** `Input(32) → L1(25, GELU) → L2(25, ReLU) → L3(25, Tanh) → L4(15, Sigmoid) → Out(10, Softmax)`  
**Total neurons:** 100 | **Total parameters:** ~3,200

---

## Screenshots

### 💬 Chat Interface

<img src="https://media.base44.com/images/public/6a23b82bd84dd64b846ca29c/e6ead34e5_generated_image.png" width="100%" alt="ANA Chat Interface — dark theme with sidebar commands and conversation" />

> Real-time chat with ANA. The agent reasons through every input using its neural network and returns both a response and a reasoning chain.

---


> Click **Neurons** in the sidebar to see a live visualization of each layer's average activation. Watch how different inputs light up different parts of the network.

---





> When you run ANA, the terminal shows the local URL and the browser opens automatically.

---

## Features

| Feature | Details |
|---|---|
| 🧠 **100-Neuron Brain** | 5-layer network with GELU, ReLU, Tanh, Sigmoid, Softmax activations |
| 🌐 **Web UI** | Built-in dark-theme chat interface — no separate frontend needed |
| 📚 **4-Tier Memory** | Sensory → Working → Episodic → Long-term memory consolidation |
| 🎯 **Goal Planner** | Multi-step goal decomposition and action sequencing |
| ⚡ **Online Learning** | Train ANA in real-time via the web UI or CLI commands |
| 🔍 **Self-Attention** | Transformer-style attention for context understanding |
| 🏥 **Health Monitor** | Built-in metrics, circuit breakers, rate limiting |
| 💾 **Persistence** | Save and load full agent state (weights + memory + history) |
| 🐳 **Docker Ready** | One command to run anywhere — local, Railway, Render, etc. |
| 0️⃣ **Zero Dependencies** | Pure Python stdlib. No pip installs needed to run. |

---


Open **http://localhost:8080**

### Option 3 — Windows EXE (no Python needed)

1. Go to [Releases](https://github.com/AadhhyaSharma/ana-agent/releases) or [Actions](https://github.com/AadhhyaSharma/ana-agent/actions)
2. Download `ANA_windows.zip`
3. Unzip → run `ANA.exe`
4. Browser opens automatically ✅

---



## CLI Commands

Use these in the chat input or via the sidebar buttons:

| Command | Description |
|---|---|
| `/neurons` | Show neuron activation heatmap |
| `/history` | View conversation history |
| `/clear` | Clear conversation history |
| `/save [file]` | Save agent state (default: `ana_state.json`) |
| `/load [file]` | Load agent state |
| `/train key: value` | Train long-term memory |
| `/learn text -> action` | Train brain with input-action pair |

---

## Training Examples

```
/train Python: A high-level programming language known for simplicity
/train Machine Learning: A subset of AI that learns from data
/learn what is AI -> explain_concept
/learn write code -> generate_code
```

---

## Architecture Deep Dive

```
Input Layer (32 dims)
      │
      ▼
Layer 1: 25 neurons, GELU activation, Layer Norm
      │
      ▼
Layer 2: 25 neurons, ReLU activation
      │
      ▼
Layer 3: 25 neurons, Tanh activation
      │
      ▼
Layer 4: 15 neurons, Sigmoid activation
      │
      ▼
Output: 10 neurons, Softmax → action probabilities
```

**Learning:** Adam optimizer with gradient clipping and adaptive plasticity  
**Memory:** LRU cache + semantic similarity + consolidation scheduler  
**IPC:** Built-in inter-process communication for multi-agent setups

---

## Files

```
ana-agent/
├── ana_launcher.py          ← Web server + UI (start here)
├── mini_100_neuron_agent.py ← Core agent + REPL
├── core_infra.py            ← Config, logging, events, health
├── neural_engine.py         ← Training, optimizers, checkpointing
├── Dockerfile               ← Docker container
├── railway.json             ← Railway deployment config
├── render.yaml              ← Render deployment config
└── requirements.txt         ← (empty — zero dependencies)
```

---

## Command Line Options

```bash
python ana_launcher.py                    # Default: port 5000, opens browser
python ana_launcher.py --port 8080        # Custom port
python ana_launcher.py --no-browser       # Server mode (no auto-open)
python ana_launcher.py --port 8080 --no-browser  # Cloud/Docker mode
```

The `PORT` environment variable is also respected (set automatically by Railway/Render).

---

## Built By

Made by [@AadhhyaSharma](https://github.com/AadhhyaSharma) · 15-year-old AI researcher from Jaipur, India 🇮🇳

---

<div align="center">
<sub>ANA runs entirely on-device. No data leaves your machine. No API keys needed.</sub>
</div>
