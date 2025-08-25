# 🤖 Ask Your Docs – RAG-Powered Document Q&A

Welcome to **Ask Your Docs**, a document-based assistant where users can upload files and instantly query them.  
It uses **hybrid retrieval (FAISS + BM25 + RRF)** and a **Groq LLM backend** to return concise, citation-backed answers — no hallucinations.

🎯 [Live Demo](https://rag-chatbot-assistant.streamlit.app/)  

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-green)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **Document Upload** – Accepts PDF, Markdown, and TXT  
- 🔎 **Hybrid Retrieval** – Combines dense embeddings (FAISS) + sparse search (BM25) with Reciprocal Rank Fusion  
- 📑 **Verified Answers** – Uses only uploaded docs with inline citations  
- 💬 **Chat UI** – Streamlit-based interface for interactive Q&A  
- ⚡ **LLM Backend** – Powered by Groq (Llama-3.1-8B/70B Instant) for low-latency inference  

---

## 🧰 Tech Stack

| Tool               | Role |
|--------------------|------|
| **FastEmbed**      | Dense embeddings (`BAAI/bge-small-en-v1.5`) |
| **FAISS**          | Vector similarity search |
| **BM25**           | Sparse keyword retrieval |
| **RRF Fusion**     | Combines dense & sparse hits |
| **Groq LLM**       | Answer generation (via Groq API) |
| **Streamlit**      | Frontend UI |
| **PyPDF2 / Markdown** | Document parsing & chunking |

---

## 📁 Project Structure

```
├── streamlit_app.py           # Streamlit chat interface
├── api/
│   └── main.py                # FastAPI entrypoint (REST API)
├── rag_chatbot/
│   ├── indexing/              # Document parsing & chunking
│   │   └── document_processor.py
│   ├── retrievers/            # Retrieval logic
│   │   ├── bm25_retriever.py
│   │   └── hybrid_retriever.py
│   ├── stores/                # Vector storage
│   │   └── vector_store.py
│   ├── llm/                   # LLM integration
│   │   └── llm_handler.py
│   └── pipeline/              # Orchestration (RAG engine)
│       └── rag_system.py
├── config.py                  # Configs & hyperparams
├── requirements.txt
└── README.md
```

---

## 🚀 Project Highlights

- Designed and implemented a **hybrid RAG pipeline** with Reciprocal Rank Fusion (RRF) to combine FAISS (dense) and BM25 (sparse) retrievals.  
- Built **Streamlit frontend** for interactive Q&A and **FastAPI backend** for REST-based integration.  
- Added **document upload capability** allowing users to query their own PDFs, Markdown, or TXT files.  
- Integrated **Groq LLM (Llama-3.1-8B Instant)** for low-latency, streaming responses.  
- Implemented **citation toggle** to improve UX (answers can be shown with or without sources).  

---

## 💻 Run It Locally

```bash
git clone https://github.com/malindard/rag-chatbot-assistant.git
cd rag-chatbot-assistant
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or run the API:

```bash
uvicorn api.main:app --reload
```

---

## ⚠️ Notes

- Answers are **strictly from uploaded documents** — no fabrication.  
- Citations appear like:  
  > “Employees are entitled to 12 days of annual leave.” [source: policy.pdf §Leave Policy]  
- If missing:  
  > “Not specified in the provided documents.”

---

## 📄 License

MIT License — free to use, fork, and extend.  

🙌 Contributions are welcome!  
Feel free to **open an issue** for bugs, suggestions, or questions, and send a **pull request** if you'd like to improve the project.  