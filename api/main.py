import hashlib
import threading
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore, build_vector_store
from src.rag_system import RAGEngine, DEFAULT_REFUSAL
from src.llm_handler import ChatLLM

# global singletons
ENGINE_LOCK = threading.Lock()
ENGINE: Optional[RAGEngine] = None
VSTORE: Optional[VectorStore] = None

def index_exists() -> bool:
    base = Path(config.VECTOR_STORE_PATH)
    return base.with_suffix(".index").exists() and base.with_suffix(".docs").exists()

def ensure_engine(load_only: bool = True) -> Optional[RAGEngine]:
    global ENGINE, VSTORE
    if ENGINE:
        return ENGINE
    if load_only and not index_exists():
        return None
    # (re)load vector store
    VSTORE = VectorStore()
    ok = VSTORE.load_index(config.VECTOR_STORE_PATH)
    if not ok:
        return None
    ENGINE = RAGEngine(VSTORE, ChatLLM())
    return ENGINE

# FastAPI app
app = FastAPI(title="Docs RAG API", version="1.0.0")

# relax cors for dev/portfolio, tighten for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models (lightweight)
class QueryRequest(BaseModel):
    question: str
    show_citations: bool = True
    refusal_message: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str

# startup
@app.on_event("startup")
def _startup():
    # try to load the index if present
    ensure_engine(load_only=True)

# endpoints
@app.get("/health")
def health():
    return {"status": "ok", "index_present": index_exists()}

@app.get("/stats")
def stats():
    e = ensure_engine(load_only=True)
    if not e or not VSTORE:
        return {"index_present": False}
    vs_stats = VSTORE.get_stats()
    return {
        "index_present": True,
        "vectors": vs_stats.get("total_vectors", 0),
        "embedding_dim": vs_stats.get("embedding_dimension", None),
        "model_name": vs_stats.get("model_name", None),
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    e = ensure_engine(load_only=True)
    if not e:
        raise HTTPException(status_code=409, detail="Index not built. Upload documents and /rebuild first.")
    answer = e.answer(req.question, refusal_message=req.refusal_message or DEFAULT_REFUSAL)
    if not req.show_citations:
        # strip citations with the same regex your UI uses
        import re as _re
        _cite_re = _re.compile(r"\[source:\s*[^\]]+\]")
        answer = _cite_re.sub("", answer).strip()
    return QueryResponse(answer=answer)

@app.get("/query/stream")
def query_stream(question: str, refusal_message: Optional[str] = None, show_citations: bool = True):
    e = ensure_engine(load_only=True)
    if not e:
        raise HTTPException(status_code=409, detail="Index not built. Upload documents and /rebuild first.")

    def _generator():
        buf = []
        import re as _re
        _cite_re = _re.compile(r"\[source:\s*[^\]]+\]")
        for token in e.answer_stream(question, refusal_message=refusal_message or DEFAULT_REFUSAL):
            if show_citations:
                yield token
            else:
                buf.append(token)
                if len(buf) % 5 == 0:  # flush in small chunks
                    chunk = "".join(buf)
                    chunk = _cite_re.sub("", chunk)
                    yield chunk
                    buf.clear()
        if not show_citations and buf:
            chunk = "".join(buf)
            chunk = _cite_re.sub("", chunk)
            yield chunk

    return StreamingResponse(_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        raw = await f.read()
        h = hashlib.sha256(raw).hexdigest()[:12]
        safe = f.filename.replace("\\", "_").replace("/", "_").strip()
        out = data_dir / f"{h}_{safe}"
        if out.exists():
            skipped.append(f.filename)
        else:
            out.write_bytes(raw)
            saved.append(f.filename)
    return {"saved": saved, "skipped": skipped, "message": "Upload complete. Call /rebuild to index."}

@app.post("/rebuild")
def rebuild():
    """
    Rebuild index from files in config.DATA_DIR
    Serialized with a lock to avoid concurrent rebuilds
    """
    with ENGINE_LOCK:
        processor = DocumentProcessor()
        docs = processor.process_directory(str(config.DATA_DIR))
        if not docs:
            raise HTTPException(status_code=400, detail="No supported documents found in data directory.")
        vs = build_vector_store(docs, force_rebuild=True)
        if not vs:
            raise HTTPException(status_code=500, detail="Failed to build vector store.")
        # swap global engine/vector store atomically
        global VSTORE, ENGINE
        VSTORE = vs
        ENGINE = RAGEngine(VSTORE, ChatLLM())
    return {"status": "ok", "vectors": VSTORE.get_stats().get("total_vectors", 0)}

@app.delete("/reset")
def reset():
    """Delete data/ and models/ to start fresh"""
    import shutil
    shutil.rmtree(Path(config.DATA_DIR), ignore_errors=True)
    shutil.rmtree(Path(config.MODELS_DIR), ignore_errors=True)
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    global VSTORE, ENGINE
    VSTORE = None
    ENGINE = None
    return {"status": "cleared"}
