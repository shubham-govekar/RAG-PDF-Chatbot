"""
Configuration for RAG Chatbot - Phase 3
"""

# ============================================================
# MODEL CONFIGURATION
# ============================================================
OLLAMA_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
RERANKER_CACHE_DIR = "./models"

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================
# --- UPDATED VARIABLE NAMES FOR PHASE 3 ---
TOP_K_RETRIEVAL = 20    # Number of chunks to fetch initially (Vector/BM25)
TOP_K_RERANK = 5        # Number of chunks to send to LLM after re-ranking
# ------------------------------------------

# Legacy names (Keep these just in case older files need them)
INITIAL_RETRIEVAL_COUNT = 20
FINAL_RESULTS_COUNT = 5
SIMILARITY_THRESHOLD = 0.40

# Hybrid search settings
USE_HYBRID_SEARCH = True  # Toggle between hybrid and vector-only
HYBRID_ALPHA = 0.5  # BM25 weight (0.5 = equal weight for BM25 and vector)

# ============================================================
# CHUNKING CONFIGURATION
# ============================================================
# Phase 1: Fixed-size chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200

# NEW: Phase 2 - Semantic chunking
USE_SEMANTIC_CHUNKING = False
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
MAX_CHUNK_SIZE = 800
MIN_CHUNK_SIZE = 200

# ============================================================
# GENERATION CONFIGURATION
# ============================================================
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 40
MAX_HISTORY_MESSAGES = 5

# ============================================================
# UI CONFIGURATION
# ============================================================
PAGE_TITLE = "PDF RAG Chatbot"
PAGE_ICON = "📚"

EXAMPLE_QUESTIONS = [
    "What is this document about?",
    "Summarize the main findings",
    "What are the key conclusions?",
    "List the main topics discussed"
]

# ============================================================
# PERFORMANCE TUNING
# ============================================================
EMBEDDING_BATCH_SIZE = 16
ENABLE_RERANKING = True
ENABLE_QUERY_EXPANSION = True

# ============================================================
# CHROMADB CONFIGURATION
# ============================================================
COLLECTION_NAME = "pdf_rag_collection"
DISTANCE_METRIC = "cosine"

# ============================================================
# EVALUATION
# ============================================================
ENABLE_EVALUATION_MODE = False
EVALUATION_OUTPUT_PATH = "./evaluation_results.json"