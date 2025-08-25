from typing import List, Dict
import config

def rrf_fuse(dense_hits: List[Dict], sparse_hits: List[Dict]) -> List[Dict]:
    # build dicts keyed by uid
    fused = {}
    def add_list(hits, kind):
        for h in hits:
            uid = h["uid"]
            rank = h.get("rank", 9999)
            # RRF contribution
            contrib = 1.0 / (config.RRF_K + rank)
            if uid not in fused:
                fused[uid] = {
                    "doc": h["doc"],
                    "source": h["source"],
                    "section_path": h.get("section_path",""),
                    "chunk_index": h.get("chunk_index", 0),
                    "citation": h.get("citation",""),
                    "dense_score": None,
                    "sparse_score": None,
                    "rrf_score": 0.0,
                }
            fused[uid]["rrf_score"] += contrib
            if kind == "dense":
                fused[uid]["dense_score"] = h.get("score")
            else:
                fused[uid]["sparse_score"] = h.get("score")

    add_list(dense_hits, "dense")
    add_list(sparse_hits, "sparse")

    # sort by fused score, keep top K
    items = list(fused.values())
    items.sort(key=lambda x: x["rrf_score"], reverse=True)
    return items[:config.FUSED_TOP_K]
