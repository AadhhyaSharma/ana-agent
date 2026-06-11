<div align="center">

# 🍫 COCOA Bot — ANA (Autonomous Neural Agent)

**A tiny, from-scratch AI agent built to study how neural networks think.**  
100 neurons. Pure Python. No PyTorch. No APIs. No black boxes — just math you can watch.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-orange?style=flat-square)](requirements.txt)
[![Windows EXE](https://img.shields.io/badge/Download-Windows%20EXE-brightgreen?style=flat-square)](../../releases/latest)

</div>

---

## What is COCOA / ANA?

COCOA (my nickname for it) is an **Autonomous Neural Agent** — a very small AI agent I built from scratch in pure Python, primarily to study **mechanistic interpretability**: the art of understanding *what* neurons actually do and *why* a neural net behaves the way it does.

It has exactly **100 neurons** across 5 layers, all hand-coded without any ML libraries. You can chat with it, watch its neurons activate in real time, train it on new knowledge, and poke around its internals.

> ⚠️ **Expectation check:** This is NOT ChatGPT. COCOA hasn't been trained on any real data. It uses pattern matching, a keyword memory system, and its small neural network to generate responses. The point isn't to get smart answers — it's to *see the machine think*.

I built this in about two weeks as a personal research project to understand interpretability from the ground up. Hope you find it as interesting as I did.

---

## Why I built this

When you use a big language model, the reasoning is completely hidden. I wanted to build something small enough that I could watch every neuron fire, trace exactly why it chose a response, and actually understand the mechanics.

COCOA is that experiment. Every layer, every weight, every activation function is written in plain Python — no abstractions hiding what's happening. Click **Neurons** in the sidebar and you can literally watch which layers light up when you type a message.

---

## Network Architecture

```
Input (32 dims — text encoded as a fixed-length vector)
      │
      ▼
Layer 1: 25 neurons · GELU activation · Layer Normalization
      │
      ▼
Layer 2: 25 neurons · ReLU activation
      │
      ▼
Layer 3: 25 neurons · Tanh activation
      │
      ▼
Layer 4: 15 neurons · Sigmoid activation
      │
      ▼
Output:  10 neurons · Softmax → action probabilities
```

**Total neurons:** 100  
**Total trainable parameters:** ~3,200  
**Optimizer:** Adam with gradient clipping  
**Memory system:** 4-tier (Sensory → Working → Episodic → Long-term)  
**Attention:** Single-head self-attention for context

Everything — matrix math, backprop, Adam, layer norm, attention — is implemented from scratch using only Python's standard library.

---

## Features

| | Feature | What it does |
|---|---|---|
| 🧠 | **Neuron Heatmap** | Live visualization of each layer's activations as you chat |
| 💬 | **Web Chat UI** | Dark-theme interface served locally — no install needed |
| 📚 | **4-Tier Memory** | Sensory → Working → Episodic → Long-term consolidation |
| ⚡ | **Online Learning** | Train COCOA new facts or behaviors in real time |
| 🎯 | **Goal Planner** | Decomposes inputs into multi-step action sequences |
| 💾 | **Save / Load State** | Persist the full agent (weights + memory + history) to JSON |
| 0️⃣ | **Zero Dependencies** | Runs on pure Python stdlib — `pip install` nothing |
| 🪟 | **Windows EXE** | Pre-built `.exe` — no Python required |

---

## Running COCOA

### Option 1 — Python (any OS)

```bash
python ana_launcher.py
```

Opens `http://localhost:5000` in your browser automatically.

```bash
python ana_launcher.py --port 8080          # custom port
python ana_launcher.py --no-browser         # server / headless mode
```

### Option 2 — Windows EXE (no Python needed)

1. Go to [**Releases**](../../releases/latest)
2. Download `ANA_windows.zip`
3. Unzip → double-click `ANA.exe`
4. Browser opens automatically ✅

### Option 3 — Docker

```bash
docker build -t cocoa-bot .
docker run -p 5000:5000 cocoa-bot
```

---

## Chat Commands

Type these directly in the chat input, or use the sidebar buttons:

| Command | What it does |
|---|---|
| `/neurons` | Show neuron activation heatmap |
| `/history` | View the full conversation log |
| `/clear` | Clear conversation history |
| `/save` | Save agent state to `ana_state.json` |
| `/load` | Load agent state from `ana_state.json` |
| `/train key: value` | Teach COCOA a new long-term memory fact |
| `/learn text → action` | Train the neural network on an input-action pair |

### Training examples

```
/train Python: a high-level language known for clean, readable syntax
/train mechanistic interpretability: studying what individual neurons do inside a neural network
/learn what is AI -> explain_concept
/learn write code -> generate_code
```

---

## Project Files

```
cocoa-bot/
├── ana_launcher.py           ← Web server + UI  (start here)
├── mini_100_neuron_agent.py  ← The 100-neuron brain + REPL
├── neural_engine.py          ← Optimizers, loss functions, training loop
├── core_infra.py             ← Config, logging, events, health monitor
├── Dockerfile                ← Docker setup
├── railway.json              ← Railway deployment config
├── render.yaml               ← Render deployment config
└── requirements.txt          ← Empty — zero external dependencies
```

---

## Mechanistic Interpretability

This is the whole point of the project. Some things you can actually observe:

- **Neuron heatmap** — see which layers activate most for different input types
- **Layer 1 (GELU)** tends to respond to broad semantic features of the input
- **Layer 3 (Tanh)** applies the strongest suppression, acting as a gating layer
- **Layer 4 (Sigmoid)** produces the probability signals feeding the output
- **Output softmax** picks one of 10 action classes — you can watch the probabilities shift

Since weights are saved to plain JSON, you can open `ana_state.json` and inspect every weight directly.

---

## Built by

Made by [@AadhhyaSharma](https://github.com/AadhhyaSharma) — 15-year-old AI researcher from Jaipur, India 🇮🇳  
Built in ~2 weeks as a mechanistic interpretability study project.

---

<div align="center">
<sub>COCOA runs entirely on your machine. No data leaves your device. No API keys. No cloud.</sub>
</div>
