import re
from typing import Optional
from textwrap import dedent
from langchain.docstore.document import Document
from src.vector_store import VectorStore
from src.llm_handler import ChatLLM
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import rrf_fuse
import config

CITE_RE = re.compile(r"\[source:\s*[^\]]+\]")

SYSTEM_PROMPT = dedent("""
                       You are a document-based assistant. You are working on a chatbot that requires users to upload their documents.
                       Therefore you must follow these rules strictly:
                       - Use ONLY the information found in the CONTEXT to answer.
                        - If the answer is missing or unclear in the CONTEXT, say so explicitly and suggest what additional document(s) would help
                        (e.g., “This isn’t in the provided documents — try uploading the transcript/contract/policy for …”).
                        - Do NOT guess or invent details. If a detail is not present, write “Not specified in the provided documents.”
                        - Include 1–3 citations for concrete claims using this format:
                        [source: <filename>{ optional: " §" + <Section> }]
                        (omit the section if none is available).
                        - Keep answers concise and factual. Do not provide legal or medical advice.
                        - If multiple documents disagree, state the discrepancy and cite each source.
                        - Prefer lists or short paragraphs. Quote exact phrases when helpful.

""").strip()

ANSWER_TEMPLATE = """
USER QUESTION:
{question}

CONTEXT (from policy documents):
{context}

INSTRUCTIONS:
- Answer ONLY based on the context above.
- If the answer is not supported by the context, say so and suggest what to upload next.
- Include 1–3 citations like [source: filename §Section]
"""

# default fallback when retrieval returns nothing
DEFAULT_REFUSAL = ("I couldn’t find this in the documents you provided. "
                   "Try uploading the relevant file(s) or asking a more specific question.")

class RAGEngine:
    def __init__(self, vector_store: VectorStore, llm: ChatLLM):
        self.vs = vector_store
        self.llm = llm
        self.bm25 = BM25Retriever(self.vs.documents) if config.USE_HYBRID_RETRIEVAL else None
    
    def _retrieve(self, question: str):
        # dense
        dense = self.vs.topk_with_citations(question, k=config.DENSE_TOP_K)
        dense = [h for h in dense if h.get("score",0.0) >= config.MIN_COSINE_SIMILARITY]

        if not config.USE_HYBRID_RETRIEVAL or self.bm25 is None:
            return dense  # vector-only mode

        # sparse
        sparse = self.bm25.topk(question, k=config.BM25_TOP_K)
        sparse = [h for h in sparse if h["score"] >= config.BM25_MIN_SCORE]

        # fuse
        fused = rrf_fuse(
            dense_hits=[{**h, "rank": i+1} for i,h in enumerate(dense)],
            sparse_hits=[{**h, "rank": h["rank"]} for h in sparse]
        )
        if not fused:
            return dense or sparse
        return fused

    def answer(self, question: str, refusal_message: Optional[str] = None) -> str:
        # 1. retrieve
        hits = self._retrieve(question)
        if not hits:
            return refusal_message or DEFAULT_REFUSAL

        # 2. build context
        context = self.vs.build_context(hits, max_chars=config.MAX_CONTEXT_LENGTH)

        # 3. send to LLM (now guarded)
        user_prompt = ANSWER_TEMPLATE.format(question=question, context=context)
        try:
            text = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            return (f"⚠️ The answer engine had an issue: {e}\n"
                    "Please try again, or ask a slightly shorter question.")

        # 4. guardrails
        if "[source:" not in text:
            # no verified citation found, append a gentle nudge (don’t hallucinate)
            text = text.strip() + "\n\n" + (DEFAULT_REFUSAL)

        found = []
        def _keep_first_n(m):
            if len(found) < config.MAX_DISTINCT_CITATIONS:
                found.append(m.group(0))
                return m.group(0)
            return ""
        text = CITE_RE.sub(_keep_first_n, text)
        return text.strip()
    
    def answer_stream(self, question: str, refusal_message: Optional[str] = None):
        """Stream tokens as they generate"""
        hits = self._retrieve(question)
        if not hits:
            # yield a one-shot refusal so streamlit displays something
            yield (refusal_message or DEFAULT_REFUSAL)
            return

        context = self.vs.build_context(hits, max_chars=config.MAX_CONTEXT_LENGTH)
        user_prompt = ANSWER_TEMPLATE.format(question=question, context=context)

        # delegate token streaming to the LLM client
        for token in self.llm.generate_stream(SYSTEM_PROMPT, user_prompt):
            yield token