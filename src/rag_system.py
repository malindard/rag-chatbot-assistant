import re
from textwrap import dedent
from langchain.docstore.document import Document
from src.vector_store import VectorStore
from src.llm_handler import ChatLLM
import config

CITE_RE = re.compile(r"\[source:\s*[^\]]+\]")

SYSTEM_PROMPT = dedent("""
You are an HR policy assistant.
Follow these rules strictly:
1) Use ONLY the provided CONTEXT. Do not use outside knowledge.
2) Include 1–3 citations in the exact form: [source: filename §Section].
3) If the answer is not clearly supported, say you don't know and suggest contacting HR.
4) Do not provide legal or medical advice.
""").strip()

ANSWER_TEMPLATE = """
USER QUESTION:
{question}

CONTEXT (from policy documents):
{context}

INSTRUCTIONS:
- Answer ONLY based on the context above.
- If not supported, say you don't know and suggest contacting HR.
- Always add 1–3 citations like [source: filename §Section].
"""

class RAGEngine:
    def __init__(self, vector_store: VectorStore, llm: ChatLLM):
        self.vs = vector_store
        self.llm = llm

    def answer(self, question: str) -> str:
        # 1. retrieve
        hits = self.vs.topk_with_citations(question, k=config.MAX_CHUNKS_FOR_CONTEXT)
        hits = [h for h in hits if h.get("score", 0.0) >= config.MIN_COSINE_SIMILARITY] # filter by cosine floor
        if not hits:
            return "❌ I couldn't find relevant information in the policy documents. Please check with HR."

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
            return "❌ I cannot provide a verified answer from the documents. Please consult HR directly."

        found = []
        def _keep_first_n(m):
            if len(found) < config.MAX_DISTINCT_CITATIONS:
                found.append(m.group(0))
                return m.group(0)
            return ""
        text = CITE_RE.sub(_keep_first_n, text)
        return text.strip()