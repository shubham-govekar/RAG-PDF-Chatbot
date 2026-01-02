# PDF RAG Chatbot — Local Hybrid Retrieval (Privacy-first)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

A lightweight, privacy-first RAG (Retrieval-Augmented Generation) chatbot that runs completely locally. It combines vector search with keyword matching for robust retrieval, then re-ranks and validates the best chunks before sending them to an LLM (via Ollama) to generate answers.

---

## Table of Contents
- [Overview](#overview)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project addresses two common problems with RAG systems: privacy (no cloud/API keys needed) and reliability (hybrid retrieval ensures better recall). The system performs:

- A hybrid search (vector + keyword)
- Result merging and re-ranking with FlashRank
- A score-based quality check: low-confidence retrievals are rejected
- LLM generation via Ollama (configurable model; default: `qwen2.5:1.5b`) when confidence is high

---

## How it works

```mermaid
graph LR
  A[User Query] --> B(Hybrid Search)
  B --> C{Re-ranker Check}
  C -- Score is Low --> D[Respond: I do not know]
  C -- Score is High --> E[LLM generates answer (via Ollama)]
```

Key points:
- Vector search captures semantic similarity
- Keyword matching ensures exact-term recall
- FlashRank re-ranks candidate chunks quickly on CPU
- If the top chunk score is below threshold, the system safely declines to answer

---

## Tech stack
- App: Python + Streamlit (local UI)
- LLM: Ollama running a local model (default: `qwen2.5:1.5b`)
- Embeddings / Vector DB: ChromaDB (in-memory)
- Re-ranker: FlashRank (CPU)

---

## Requirements
- Python 3.11+  
- Ollama installed and running locally (https://ollama.com)  
- Recommended: a machine with a modern CPU; GPU not required for core components

---

## Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install and run Ollama, then pull the model:

```bash
# Install Ollama via https://ollama.com
# Pull the model referenced in `config.py` (default: `qwen2.5:1.5b`)
ollama pull qwen2.5:1.5b
ollama run # keep Ollama running in background
```

---

## Quick start
Run the Streamlit app locally:

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501, upload a PDF from the sidebar, and start chatting.

Notes:
- ChromaDB runs in-memory by default — the DB resets when the app is closed.
- If the re-ranker deems results low-confidence, the app will reply that it doesn't know rather than hallucinate.

---

## Project structure
- `app.py` — Streamlit frontend and app orchestration
- `config.py` — configuration and application settings
- `src/advanced_chunking.py` — PDF chunking and preprocessing
- `src/embeddings.py` — embedding computation and helpers
- `src/hybrid_retrieval.py` — hybrid search implementation (BM25 + Chroma)
- `src/retrieval.py` — retrieval utilities
- `src/generation.py` — prompt management and Ollama integration
- `src/ui.py` — UI helpers for Streamlit
- `src/utils.py` — utility functions
- `models/` — local models and tokenizers (e.g., FlashRank / MiniLM)

---

## Contributing
Contributions are welcome! Please open an issue to discuss larger changes and submit PRs for bug fixes or feature additions. Keep contributions small and well-documented.

---

## License
This project is released under the MIT License. See the `LICENSE` file for details.

---
