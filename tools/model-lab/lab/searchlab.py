"""Search for the dashboard: the same three methods as memoria's
src/memoria/search.py (FTS5 keyword, brute-force cosine, RRF fusion with
the same k), but returning the scores behind each rank so the UI can
show the math.

One deliberate difference, shown in the UI: a typed query is turned into
an FTS5 OR-expression of its words (minus stopwords). memoria's
keyword_search passes the text straight to MATCH, which means an
implicit AND of every word, so a natural-language query like "dog
playing on the beach" only matches items containing all five words.
"""

import re

import numpy as np

from memoria.search import RRF_K

STOPWORDS = set("""a an and are as at be by for from has have i in is it its me my of on or our that the
their them this to was were with we you your photo photos picture pictures video videos image images
show find some""".split())


def fts_expression(query: str, mode: str = "or") -> str:
    words = [w for w in re.findall(r"[\w']+", query.lower()) if w not in STOPWORDS]
    if not words:
        return ""
    quoted = [f'"{w}"' for w in dict.fromkeys(words)]
    return (" OR " if mode == "or" else " AND ").join(quoted)


def keyword(conn, query: str, table="search_fts", limit=50, mode="or"):
    expr = fts_expression(query, mode)
    if not expr:
        return expr, []
    rows = conn.execute(f"SELECT asset_id, bm25({table}) AS score FROM {table} WHERE {table} MATCH ? "
                        f"ORDER BY score LIMIT ?", (expr, limit)).fetchall()
    # bm25() is "lower is better" (negative); flip the sign for display.
    return expr, [{"id": r[0], "score": -r[1]} for r in rows]


class VectorIndex:
    """All SigLIP2 vectors in one matrix, loaded once (ADR-0022)."""

    def __init__(self, conn, model: str = "siglip2"):
        # Production vectors live in `embeddings`; comparison models in
        # `alt_embeddings`. Each index only ever holds one model's vectors.
        if model == "siglip2":
            rows = conn.execute("SELECT asset_id, vector, model_version FROM embeddings").fetchall()
        else:
            rows = conn.execute("SELECT asset_id, vector, model_version FROM alt_embeddings WHERE model=?",
                                (model,)).fetchall()
        self.model = model
        self.ids = [r[0] for r in rows]
        self.pos = {i: n for n, i in enumerate(self.ids)}
        self.model_version = rows[0][2] if rows else None
        m = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1))
        self.matrix = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)

    def search(self, qvec, limit=50, min_similarity=None, exclude=None):
        if not self.ids:
            return []
        q = qvec / (np.linalg.norm(qvec) + 1e-10)
        sims = self.matrix @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            if exclude and self.ids[i] == exclude:
                continue
            out.append({"id": self.ids[i], "score": float(sims[i]),
                        "below_threshold": bool(min_similarity is not None and sims[i] < min_similarity)})
            if len(out) >= limit:
                break
        return out

    def vector(self, asset_id):
        n = self.pos.get(asset_id)
        return None if n is None else self.matrix[n]


def hybrid(keyword_hits, vector_hits, k=RRF_K, min_similarity=None, limit=50):
    """RRF, as in memoria.search.reciprocal_rank_fusion, keeping each
    method's contribution so the UI can show 1/(k+rank) per method."""
    vec = [h for h in vector_hits if not (min_similarity is not None and h["score"] < min_similarity)]
    fused: dict[str, dict] = {}
    for method, hits in (("keyword", keyword_hits), ("vector", vec)):
        for rank, h in enumerate(hits, start=1):
            e = fused.setdefault(h["id"], {"id": h["id"], "score": 0.0, "parts": {}})
            part = 1.0 / (k + rank)
            e["parts"][method] = {"rank": rank, "contribution": part, "raw": h["score"]}
            e["score"] += part
    return sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:limit]


def rank_of(hits, asset_id):
    return next((n for n, h in enumerate(hits, start=1) if h["id"] == asset_id), None)
