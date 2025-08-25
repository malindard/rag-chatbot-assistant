# app/streamlit_app.py
import os
import re
import hashlib
from pathlib import Path
import streamlit as st

import config
from rag_chatbot.indexing.document_processor import DocumentProcessor
from rag_chatbot.stores.vector_store import VectorStore, build_vector_store
from rag_chatbot.llm.llm_handler import ChatLLM
from rag_chatbot.pipeline.rag_system import RAGEngine

st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON, layout=config.LAYOUT)

# ---------------------------
# Helpers
# ---------------------------

CITE_RE = re.compile(r"\[source:\s*[^\]]+\]")

def strip_citations(text: str) -> str:
    cleaned = CITE_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def _has_any_docs() -> bool:
    data_dir = Path(config.DATA_DIR)
    if not data_dir.exists():
        return False
    allowed = {".md", ".markdown", ".txt", ".pdf"}
    return any(p.suffix.lower() in allowed for p in data_dir.glob("*"))

def _has_index() -> bool:
    base = Path(config.VECTOR_STORE_PATH)
    return base.with_suffix(".index").exists() and base.with_suffix(".docs").exists()

def _safe_name(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").strip()

def _dedup_save(upload, folder: Path) -> str:
    raw = upload.read()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    safe = _safe_name(upload.name)
    target = folder / f"{digest}_{safe}"
    if not target.exists():
        target.write_bytes(raw)
        return f"Saved: {upload.name}"
    return f"Duplicate skipped: {upload.name}"

# ---------------------------
# Cached resources
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_vector_store():
    """Load an existing FAISS index if present. Do NOT auto-build."""
    vs = VectorStore()
    if vs.load_index(config.VECTOR_STORE_PATH):
        return vs
    return None

def _clear_vs_cache():
    try:
        get_vector_store.clear()
    except Exception:
        pass

def ensure_engine():
    """Return a cached RAGEngine if index is present and loadable, else None."""
    if "engine" in st.session_state and st.session_state.engine:
        return st.session_state.engine

    vs = get_vector_store()
    if not vs:
        return None

    llm = ChatLLM()
    engine = RAGEngine(vs, llm)
    st.session_state.engine = engine
    return engine

def rebuild_index():
    """Process documents and rebuild FAISS index."""
    processor = DocumentProcessor()
    docs = processor.process_directory(str(config.DATA_DIR))
    vs = build_vector_store(docs, force_rebuild=True)
    if vs:
        _clear_vs_cache()  # refresh the cached loader
        st.session_state.engine = RAGEngine(vs, ChatLLM())
        return True
    return False

def reset_workspace():
    """Clear data and models; force user to upload and rebuild."""
    try:
        import shutil
        shutil.rmtree(Path(config.DATA_DIR), ignore_errors=True)
        shutil.rmtree(Path(config.MODELS_DIR), ignore_errors=True)
    except Exception:
        pass
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    _clear_vs_cache()
    st.session_state.engine = None
    st.session_state.messages = []

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    st.title("Controls")

    if st.button("Reset (clear data & index)", use_container_width=True):
        reset_workspace()
        st.success("Workspace reset. Upload documents to begin.")
        st.stop()

    st.subheader("📤 Upload documents")
    uploads = st.file_uploader(
        "Add .md / .txt / .pdf and then build the index",
        type=["md", "markdown", "txt", "pdf"],
        accept_multiple_files=True
    )
    if uploads:
        data_dir = Path(config.DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        msgs = []
        for f in uploads:
            msgs.append(_dedup_save(f, data_dir))
        for m in msgs:
            st.write("• " + m)
        st.warning("Uploaded. Click **Rebuild index** below.")

    if st.button("Rebuild index", type="primary", use_container_width=True):
        with st.spinner("Indexing…"):
            ok = rebuild_index()
            if ok:
                st.success("Index built. You can start chatting.")
            else:
                st.error("Index build failed. Check logs.")

    st.subheader("Display")
    show_citations = st.checkbox(
        "Show sources (citations)",
        value=True,
        help="Toggle inline [source: ...] markers"
    )

    st.subheader("Config (read-only)")
    st.code(
        f"Data dir:   {config.DATA_DIR}\n"
        f"Models dir: {config.MODELS_DIR}\n"
        f"LLM model:  {getattr(config, 'GROQ_MODEL', 'llama-3.1-8b-instant')}\n"
        f"Hybrid:     {getattr(config, 'USE_HYBRID_RETRIEVAL', False)}",
        language="bash"
    )

# ---------------------------
# Main layout
# ---------------------------

st.title("Ask Your Docs")
st.caption("👋 Welcome! Upload your documents on the left. I’ll build an index and let you query them instantly.")

# Onboarding gate: require docs & index before chatting
if not _has_any_docs():
    st.info("No documents found. Upload .md/.txt/.pdf in the sidebar, then click **Rebuild index**.")
    st.stop()

if not _has_index():
    st.warning("Documents exist, but the index has not been built yet. Click **Rebuild index** in the sidebar.")
    st.stop()

engine = ensure_engine()
if engine is None:
    st.error("Index could not be loaded. Try clicking **Rebuild index** again.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# System hello (only first time)
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! I answer **only** from your uploaded documents. "
            "Ask something like: _“What's the main topic of this document?_"
        )

# Render past messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_q = st.chat_input("Type your question…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if show_citations:
                    # Stream raw tokens (with citations)
                    final_text = st.write_stream(engine.answer_stream(user_q))
                else:
                    # Live stream but render a citation-free view
                    placeholder = st.empty()
                    buf = []
                    for token in engine.answer_stream(user_q):
                        buf.append(token)
                        cleaned = strip_citations("".join(buf))
                        placeholder.markdown(cleaned)
                    final_text = strip_citations("".join(buf))
            except Exception as e:
                st.error("The answer engine had an issue.")
                st.code(f"{e}")
                final_text = ("⚠️ The answer engine had a temporary issue processing your request. "
                              "Please try again later.")

        # Post-check / UX notes
        if show_citations and "[source:" not in final_text:
            st.info("ℹ️ No verified source detected. Try a more specific question or rebuild your index.")
        if not show_citations:
            st.caption("Sources hidden (toggle in the sidebar to show).")

        st.session_state.messages.append({"role": "assistant", "content": final_text})

# (Optional) simple debug expander for retrieval signals
with st.expander("Debug: paths & status"):
    st.write(f"Has docs: {_has_any_docs()} | Has index: {_has_index()}")
    st.write(f"DATA_DIR: {config.DATA_DIR}")
    st.write(f"MODELS_DIR: {config.MODELS_DIR}")
    st.write(f"VECTOR_STORE_PATH: {config.VECTOR_STORE_PATH}")
