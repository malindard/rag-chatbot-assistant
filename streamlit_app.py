import re
import streamlit as st
import hashlib
from pathlib import Path
import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore, build_vector_store
from src.llm_handler import ChatLLM
from src.rag_system import RAGEngine

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store():
    vs = VectorStore()
    if not vs.load_index(config.VECTOR_STORE_PATH):
        st.info("No existing vector store found. Building a new one from data/ ...")
        proc = DocumentProcessor()
        docs = proc.process_directory(str(config.DATA_DIR))
        if not docs:
            st.warning("No documents found in data/. Please add .md or .markdown files and click 'Rebuild index'.")
            return None
        vs_built = build_vector_store(docs, force_rebuild=True)
        return vs_built
    return vs

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatLLM()

def ensure_engine():
    vs = get_vector_store()
    if vs is None:
        return None
    llm = get_llm()
    return RAGEngine(vs, llm)

def rebuild_index(show_toast=True):
    try:
        proc = DocumentProcessor()
        docs = proc.process_directory(str(config.DATA_DIR))
        if not docs:
            st.error("No documents found in data/. Add .md files first.")
            return False
        # Build fresh store
        _ = build_vector_store(docs, force_rebuild=True)
        # Clear cached vector store
        get_vector_store.clear()
        if show_toast:
            st.toast("Index rebuilt ✅", icon="✅")
        return True
    except Exception as e:
        st.error(f"Failed to rebuild index: {e}")
        st.exception(e)
        return False

def show_stats(vs: VectorStore):
    stats = vs.get_stats()
    st.metric("Chunks", stats["total_documents"])
    st.metric("Vectors", stats["total_vectors"])
    st.caption(f"Embedding dim: {stats['embedding_dimension']} | Model: {stats['model_name']}")

CITE_RE = re.compile(r"\[source:\s*[^\]]+\]")
def strip_citations(text: str) -> str:
    # remove citation markers and tidy whitespsce
    cleaned = CITE_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON, layout=config.LAYOUT)
st.title("HR Policy RAG Assistant")
st.caption("Grounded answers + citations from your Markdown policy docs.")

with st.sidebar:
    st.subheader("Controls")
    st.subheader("📤 Upload HR Documents")
    uploaded_files = st.file_uploader(
        "Upload HR policies (.pdf, .md, .txt)", 
        type=["pdf", "md", "txt"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        data_dir = Path(config.DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            # Deduplicate using hash
            raw = f.read()
            sha = hashlib.sha256(raw).hexdigest()[:12]
            safe_name = f"{sha}_{f.name}"
            out_path = data_dir / safe_name

            if not out_path.exists():
                with open(out_path, "wb") as fout:
                    fout.write(raw)
                st.success(f"Saved: {f.name}")
            else:
                st.info(f"Skipped duplicate: {f.name}")

        st.warning("ℹ️ Uploaded docs saved. Click 'Rebuild index' below to include them.")
        
    if st.button("🔁 Rebuild index", use_container_width=True):
        rebuild_index()

    st.subheader("Display")
    show_citations = st.checkbox("Show sources (citations)", value=True, help="Toggle inline [source: ...] markers")

    st.divider()
    st.subheader("Config (read-only)")
    st.write(f"**Data dir:** `{config.DATA_DIR}`")
    st.write(f"**Vector store:** `{config.VECTOR_STORE_PATH}`")
    st.write(f"**LLM model:** `{getattr(config, 'GROQ_MODEL', 'N/A')}`")
    st.write(f"**Max chunks:** {config.MAX_CHUNKS_FOR_CONTEXT}")
    st.write(f"**Max context chars:** {config.MAX_CONTEXT_LENGTH}")
    st.write(f"**Temperature:** {config.TEMPERATURE}")
    st.write(f"**Max tokens:** {config.MAX_NEW_TOKENS}")
    st.write(f"**Min cosine:** {getattr(config, 'MIN_COSINE_SIMILARITY', 0.25)}")

    st.divider()
    vs_cached = get_vector_store()
    if vs_cached:
        st.subheader("Index stats")
        show_stats(vs_cached)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask an HR policy question (e.g., *How many sick days do I get?*)."}
    ]

# Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_q = st.chat_input("Type your HR question...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        engine = ensure_engine()
        if engine is None:
            st.error("Vector store is not ready. Add documents to data/ and click 'Rebuild index'.")
        else:
            with st.spinner("Thinking..."):
                try:
                    if show_citations:
                        # with streaming text
                        final_text = st.write_stream(engine.answer_stream(user_q))
                    else:
                        # stream but render a live, citation free
                        placeholder = st.empty()
                        buf = []
                        for token in engine.answer_stream(user_q):
                            buf.append(token)
                            cleaned = strip_citations("".join(buf))
                            placeholder.markdown(cleaned)
                        final_text = strip_citations("".join(buf))
                except Exception as e:
                    # Show detailed error inline so you can debug free-route issues
                    st.error("The answer engine had an issue.")
                    st.code(f"{e}")
                    final_text = ("⚠️ The answer engine had a temporary issue processing your request. "
                                  "Please try again or switch model in your .env.")
            # post-check: if no citations present, append a gentle notice
            if show_citations and "[source:" not in final_text:
                st.info("ℹ️ No verified source detected. If this seems incorrect, try a more specific question or contact HR.")
            if not show_citations:
                st.caption("Sources hidden (toggle in the sidebar to show).")
st.divider()
st.caption(
    "Answers are derived only from your local policy documents. "
    "If context is missing or weak, the assistant will refuse and suggest contacting HR."
)