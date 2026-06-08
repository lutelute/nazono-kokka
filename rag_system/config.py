"""Centralized configuration for the RAG judicial system.

All configurable parameters for the RAG pipeline are defined here.
Environment variables can override default values where noted.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEGAL_FRAMEWORK_DIR = PROJECT_ROOT / "legal_framework"
PRECEDENTS_DIR = PROJECT_ROOT / "precedents"
PRECEDENTS_CRIMINAL_DIR = PRECEDENTS_DIR / "criminal"
PRECEDENTS_CIVIL_DIR = PRECEDENTS_DIR / "civil"
PRECEDENTS_CONSTITUTIONAL_DIR = PRECEDENTS_DIR / "constitutional"
PRECEDENTS_METADATA_PATH = PRECEDENTS_DIR / "metadata.json"

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    str(PROJECT_ROOT / "chroma_db"),
)

# ---------------------------------------------------------------------------
# Ollama / LLM Settings
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

LLM_MODEL_NAME = "schroneko/llama-3.1-swallow-8b-instruct-v0.1"
LLM_FALLBACK_MODEL = "llama3.1:8b"

try:
    LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
except (ValueError, TypeError):
    LLM_TEMPERATURE = 0.1

try:
    LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "4096"))
except (ValueError, TypeError):
    LLM_NUM_CTX = 4096

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Document Chunking
# ---------------------------------------------------------------------------

try:
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
except (ValueError, TypeError):
    CHUNK_SIZE = 1000

try:
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
except (ValueError, TypeError):
    CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

try:
    RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))
except (ValueError, TypeError):
    RETRIEVAL_K = 5

# Number of candidates to fetch *before* reranking / fusion narrows them
# down to the final ``RETRIEVAL_K``.  A wider candidate pool gives the
# reranker more material to work with at the cost of latency.
try:
    RETRIEVAL_FETCH_K = int(os.environ.get("RETRIEVAL_FETCH_K", "20"))
except (ValueError, TypeError):
    RETRIEVAL_FETCH_K = 20

# ---------------------------------------------------------------------------
# Hybrid Search (sparse BM25 + dense vector fusion)
# ---------------------------------------------------------------------------

# Reciprocal Rank Fusion constant.  Larger values flatten the contribution
# of high ranks; 60 is the value from the original RRF paper.
try:
    RRF_K = int(os.environ.get("RRF_K", "60"))
except (ValueError, TypeError):
    RRF_K = 60

# Relative weight of the dense (vector) ranking vs. the sparse (BM25)
# ranking during fusion.  0.0 = BM25 only, 1.0 = dense only.
try:
    HYBRID_DENSE_WEIGHT = float(os.environ.get("HYBRID_DENSE_WEIGHT", "0.5"))
except (ValueError, TypeError):
    HYBRID_DENSE_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Reranking (cross-encoder)
# ---------------------------------------------------------------------------

# Multilingual cross-encoder that scores (query, document) pairs jointly.
# Supports Japanese.  Lazily loaded and gracefully skipped if unavailable.
RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL_NAME",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

# ---------------------------------------------------------------------------
# ChromaDB Collection
# ---------------------------------------------------------------------------

CHROMA_COLLECTION_NAME = "nazono_kokka_legal"
