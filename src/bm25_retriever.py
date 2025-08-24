import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> List[str]:
    # simple, fast tokenizer, add another later
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

class BM25Retriever:
    def __init__(self, docs: List[Any]):
        self.docs = docs
        self.corpus_tokens = [ _tokenize(d.page_content) for d in docs ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def topk(self, query: str, k: int = 20) -> List[Dict]:
        q = _tokenize(query)
        scores = self.bm25.get_scores(q)
        # collect top-k indices by score
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for rank, i in enumerate(idxs, start=1):
            d = self.docs[i]
            source_full = d.metadata.get("source", "unknown")
            source_name = Path(source_full).name            # use filename
            section_path = d.metadata.get("section_path", "")
            # prefer 'chunk_index', fallback to 'chunk_in_section'
            chunk_idx = d.metadata.get("chunk_index",
                          d.metadata.get("chunk_in_section", i))
            results.append({
                "doc": d,
                "score": float(scores[i]),
                "rank": rank,
                "source": source_name,  # match dense
                "section_path": section_path,
                "chunk_index": chunk_idx,
                "uid": (source_name, section_path, chunk_idx),  # mattch dense
            })
        return results
