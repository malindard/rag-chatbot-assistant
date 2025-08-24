"""
Configuration settings for RAG Chatbot Assistant
- Embeddings: FastEmbed (BAAI/bge-small-en-v1.5)
- Vector store: FAISS
- Chat LLM: GROQ (Llama 3.1 8b)
"""
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # fast n lightweight sentence transformer
LLM_BACKEND = os.getenv("LLM_BACKEND", "GROQ").upper()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_MAX_RETRIES_PER_MODEL = int(os.getenv("LLM_MAX_RETRIES_PER_MODEL", "2"))
LLM_BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", "1"))

# Generation parameters
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))        # lower = more focused, higher = more creative 
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
MIN_COSINE_SIMILARITY = float(os.getenv("MIN_COSINE_SIMILARITY", "0.15"))   # max cosine similarity floor
MAX_DISTINCT_CITATIONS = int(os.getenv("MAX_DISTINCT_CITATIONS", "3"))      # max dedupe citations

CACHE_DIR = str(MODELS_DIR)

# Validate token
if GROQ_API_KEY:
    print(f"✅ Token loaded: {GROQ_API_KEY[:10]}...")
else:
    print("⚠️ No API key found. Set GROQ_API_KEY environment variable or create .env file")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))    # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # overlap between chunks
MAX_CHUNKS_FOR_CONTEXT = int(os.getenv("MAX_CHUNKS_FOR_CONTEXT", "3"))  # max relevant chunks to include in prompt
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2500"))   # max context length for LLM

# Vector store settings
VECTOR_STORE_PATH = str(MODELS_DIR / "faiss_index")

# Streamlit settings
PAGE_TITLE = "Chatbot Assistant"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Hybrid retrieval toggles
USE_HYBRID_RETRIEVAL = True  # flip to enable/disable

# BM25
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "0.1"))

# Dense
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "6"))

# RRF
RRF_K = int(os.getenv("RRF_K", "60"))
FUSED_TOP_K = int(os.getenv("FUSED_TOP_K", str(MAX_CHUNKS_FOR_CONTEXT)))
