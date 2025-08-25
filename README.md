# 🤖 Ask Your Docs – RAG-Powered Document Q&A

Welcome to **Ask Your Docs**, a document-based chatbot that lets you upload files and instantly query them.  
It uses **hybrid retrieval (FAISS + BM25 + RRF)** and a **Groq LLM backend** to give concise answers with verified citations — no hallucinations.

🎯 [Live Demo](https://rag-chatbot-assistant.streamlit.app/)  

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-green)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ What It Does

Think of this as your personal **AI knowledge assistant**. With your uploaded docs, it:

1. **Processes** PDF / Markdown / TXT files into clean text chunks  
2. **Indexes** them with dense embeddings (FAISS + FastEmbed)  
3. **Fuses** dense & sparse results using Reciprocal Rank Fusion  
4. **Answers** your questions *only* from the docs — with citations  
5. **Runs** in a Streamlit chat UI for quick interaction  

---

## 🧰 Tech Stack

| Tool             | Role                                   |
|------------------|----------------------------------------|
| FastEmbed        | Text embeddings (BAAI/bge-small-en-v1.5) |
| FAISS            | Vector store for similarity search |
| BM25 (rank-bm25) | Sparse keyword retrieval |
| RRF Fusion       | Combines dense & sparse hits |
| Groq (Llama 3.1) | Chat LLM backend |
| Streamlit        | Frontend UI |
| PyPDF2 / LangChain | Document parsing & chunking |

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

Run the app:

```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
├── streamlit_app.py        # Streamlit chat interface
├── src/
│   ├── document_processor.py  # PDF/MD/TXT processing & chunking
│   ├── vector_store.py        # FAISS + embeddings
│   ├── bm25_retriever.py      # Sparse keyword search
│   ├── hybrid_retriever.py    # Reciprocal Rank Fusion
│   ├── rag_system.py          # RAG engine (retrieval + LLM)
│   └── llm_handler.py         # Groq LLM client
├── config.py                # Centralized configs
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

- Answers are **strictly from uploaded documents** — no guessing.  
- Citations appear like:  
  > “Employees are entitled to 12 days of annual leave.” [source: policy.pdf §Leave Policy]  
- If a detail is missing, it will say:  
  > “Not specified in the provided documents.”

---

## 📄 License

This project is open source under the **MIT License** — fork it, build on top, or showcase it!
