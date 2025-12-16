# Advanced PDF RAG Chatbot with Hybrid Retrieval

A local, high-precision Retrieval-Augmented Generation (RAG) system engineered to solve common limitations in standard RAG pipelines. This project implements Hybrid Search (Dense Vector + Sparse Keyword), Cross-Encoder Re-ranking, and strict Relevance Gating to minimize hallucinations.

## Overview

Standard RAG implementations often suffer from retrieval inaccuracies (missing specific keywords) and hallucinated responses when the retrieved context is irrelevant. This project addresses these issues through a multi-stage pipeline:

1.  **Hybrid Retrieval:** Combines semantic understanding (Vector Search via ChromaDB) with precise keyword matching (BM25) to capture both conceptual and literal queries.
2.  **Cross-Encoder Re-ranking:** Uses a high-fidelity model (FlashRank/TinyBERT) to re-score retrieval candidates, ensuring the Large Language Model (LLM) receives only the most pertinent information.
3.  **Relevance Gating:** Implements a hard rejection threshold based on Cross-Encoder scores. If the system detects low relevance between the query and the documents, it refuses to answer rather than fabricating a response.
4.  **Local Execution:** Fully privacy-focused architecture running locally using Ollama and ephemeral vector stores.

## Technical Architecture

The system follows a modular pipeline architecture:

```mermaid
graph LR
    A[User Query] --> B{Intent Classifier}
    B -- Search --> C[Hybrid Retriever]
    C --> D[Vector Search] & E[BM25 Keyword Search]
    D & E --> F[Merged Results]
    F --> G[Cross-Encoder Re-ranker]
    G --> H{Relevance Gate}
    H -- Score < Threshold --> I[Reject Query]
    H -- Score >= Threshold --> J[Llama 3.2 Context]

## Key Features

* **Hybrid Search Strategy:** Utilizes a weighted ensemble of Cosine Similarity (Dense) and BM25 (Sparse) to handle various query types effectively.
* **FlashRank Re-ranking:** Post-process retrieval using a cross-encoder to fix the "Lost in the Middle" phenomenon common in long-context windows.
* **Parent-Child Chunking:** Retrieves smaller, granular chunks for scoring accuracy while delivering larger "parent" windows to the LLM for better context coherence.
* **Ephemeral Vector Store:** Initializes a stateless ChromaDB instance in memory for each session, ensuring data privacy and clean state management.
* **Self-Correction:** The system evaluates the confidence of retrieved chunks and halts generation if the confidence score falls below a set threshold.

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** Streamlit
* **LLM Inference:** Ollama (Local)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace / SentenceTransformers
* **Re-ranking:** FlashRank
* **Sparse Search:** RankBM25

## Installation

### 1. Prerequisites
Ensure you have Python installed. You also need Ollama running locally for the LLM inference.

1.  Download Ollama from ollama.com.
2.  Pull the target model:
    ollama pull llama3.2

### 2. Setup Environment

Clone the repository and install the required dependencies.

git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
pip install -r requirements.txt

## Usage

1.  Start the local development server:
    streamlit run app.py

2.  Navigate to http://localhost:8501 in your browser.

3.  Upload a PDF document via the sidebar. The system will automatically ingest, chunk, and index the file.

4.  Input your query. You can view the internal reasoning trace (Intent -> Retrieval -> Re-ranking -> Score) by expanding the "View Detailed Logic" section in the interface.

## Project Structure

* app.py: Main application entry point and UI logic.
* src/: Core logic modules.
    * hybrid_retrieval.py: Implementation of the Dense+Sparse retrieval strategy and re-ranking.
    * generation.py: LLM prompting and stream handling.
    * embeddings.py: Vector embedding generation service.
    * advanced_chunking.py: Parent-Child document splitting logic.
* config.py: Configuration settings for thresholds, models, and pathing.

## License

MIT License