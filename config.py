"""
Configuration settings for RAG Chatbot Assistant
- Embeddings: FastEmbed (BAAI/bge-small-en-v1.5)
- Vector store: FAISS
- Chat LLM: OpenRouter (DeepSeek by default)
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
LLM_BACKEND = os.getenv("LLM_BACKEND", "OPENROUTER").upper()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free")

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Generation parameters
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))        # lower = more focused, higher = more creative 
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))    # max tokens in response

CACHE_DIR = str(MODELS_DIR)

# Validate token
if OPENROUTER_API_KEY:
    print(f"✅ Token loaded: {OPENROUTER_API_KEY[:10]}...")
else:
    print("⚠️ No API key found. Set OPENROUTER_API_KEY environment variable or create .env file")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))    # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # overlap between chunks
MAX_CHUNKS_FOR_CONTEXT = int(os.getenv("MAX_CHUNKS_FOR_CONTEXT", "6"))  # max relevant chunks to include in prompt
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))   # max context length for LLM

# Vector store settings
VECTOR_STORE_PATH = str(MODELS_DIR / "faiss_index")

# Streamlit settings
PAGE_TITLE = "Chatbot Assistant"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Sample HR documents
HR_DOCUMENTS = [
    "employee_leave_policy.md",
    "remote_work_guidelines.md",
    "performance_review.md",
    "code_of_conduct.md",
]