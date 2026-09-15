"""Hybrid search (ADR-0010): FTS5 keyword hits fused with brute-force
vector similarity (ADR-0022) via Reciprocal Rank Fusion - not vector
search alone. Metadata filters narrow the candidate set with plain SQL
first, then both keyword and vector search are scoped to that narrowed
set, per ADR-0022's "filter first, then rank" pattern.

No AI model involved anywhere in this file - everything here is testable
against synthetic rows and hand-built vectors, which is exactly how it's
tested (tests/test_search.py), well ahead of the Mac Studio.
"""

import datetime

import numpy as np

# From the original RRF paper (Cormack, Clarke, Buettcher, SIGIR 2009)
# and its common default since. Not tuned against this project's own data
# yet - no eval harness exists to tune it against (ADR-0006).
RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """rankings: a list of ranked id-lists (best first), from independent
    retrieval methods (e.g. keyword hits, vector hits). An id missing from
    a given ranking simply contributes no term from it - RRF doesn't need
    every method to have an opinion on every candidate.

    Returns a single fused ranking (best first): for each id, sum
    1/(k + rank) across every ranking it appears in (rank is 1-based).
    Ties are broken by first-seen order (Python's sort is stable and
    dicts preserve insertion order) - deterministic, not meaningful.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


def vector_search(
    conn,
    query_vector: np.ndarray,
    candidate_ids: list[str] | None = None,
    limit: int = 50,
    min_similarity: float | None = None,
) -> list[str]:
    """Brute-force cosine similarity (ADR-0022) against the embeddings
    table. `candidate_ids=None` searches everything; an empty list is a
    real "nothing matched the filter" case and short-circuits to []
    rather than silently searching everything.

    Unlike keyword search, cosine similarity always ranks *something* -
    even a genuinely irrelevant result looks "closest" relative to
    everything else in the store. `min_similarity` (ADR-0026) is the
    escape hatch: results below it are excluded rather than force-filled
    into the top-k, so a query with no real match in the library can
    return empty instead of confidently handing back noise. No default
    value is set here - there's no real embedding data yet to calibrate
    what "not a real match" actually looks like (see ADR-0026)."""
    if candidate_ids is not None and not candidate_ids:
        return []

    if candidate_ids is not None:
        placeholders = ",".join("?" * len(candidate_ids))
        rows = conn.execute(
            f"SELECT asset_id, vector FROM embeddings WHERE asset_id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
    else:
        rows = conn.execute("SELECT asset_id, vector FROM embeddings").fetchall()

    if not rows:
        return []

    ids = [r[0] for r in rows]
    vectors = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    query = query_vector.astype(np.float32)

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []
    vector_norms = np.linalg.norm(vectors, axis=1)
    similarities = (vectors @ query) / (vector_norms * query_norm + 1e-10)

    order = np.argsort(-similarities)[:limit]
    if min_similarity is not None:
        order = [i for i in order if similarities[i] >= min_similarity]
    return [ids[i] for i in order]


def keyword_search(
    conn, query_text: str, candidate_ids: list[str] | None = None, limit: int = 50
) -> list[str]:
    """FTS5 keyword search (ADR-0010) against search_fts, ranked by
    FTS5's own bm25 relevance. The candidate filter is pushed into the
    SQL query itself, not applied after fetching - filtering after a
    LIMIT would silently under-return whenever most of the top FTS hits
    fall outside the filtered set."""
    if candidate_ids is not None and not candidate_ids:
        return []

    if candidate_ids is not None:
        placeholders = ",".join("?" * len(candidate_ids))
        rows = conn.execute(
            f"SELECT asset_id FROM search_fts WHERE search_fts MATCH ? "
            f"AND asset_id IN ({placeholders}) ORDER BY rank LIMIT ?",
            [query_text, *candidate_ids, limit],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT asset_id FROM search_fts WHERE search_fts MATCH ? ORDER BY rank LIMIT ?",
            (query_text, limit),
        ).fetchall()
    return [r[0] for r in rows]


def semantic_search(
    conn,
    query_text: str,
    query_vector: np.ndarray,
    filters: dict | None = None,
    limit: int = 20,
    fanout: int = 50,
    min_similarity: float | None = None,
) -> list[str]:
    """The hybrid search behind the agent's semantic_search tool
    (ADR-0005). filters: optional {"kind": "image"|"video", "year": int}.
    fanout: how many candidates keyword/vector search each retrieve
    before fusion - larger than `limit` so RRF has real signal from both
    methods to work with, not just whatever each one's own top-`limit`
    happened to contain. min_similarity: passed through to vector_search
    (ADR-0026) - a real value isn't set by default here yet either, for
    the same reason (nothing to calibrate it against until real
    embeddings exist).
    """
    candidate_ids = _filtered_asset_ids(conn, filters) if filters else None

    keyword_ranked = keyword_search(conn, query_text, candidate_ids, limit=fanout)
    vector_ranked = vector_search(
        conn, query_vector, candidate_ids, limit=fanout, min_similarity=min_similarity
    )

    fused = reciprocal_rank_fusion([keyword_ranked, vector_ranked])
    return fused[:limit]


def _filtered_asset_ids(conn, filters: dict) -> list[str]:
    clauses, params = [], []
    if "kind" in filters:
        clauses.append("kind = ?")
        params.append(filters["kind"])
    if "year" in filters:
        year = filters["year"]
        clauses.append("date_created >= ? AND date_created < ?")
        params.append(_year_start(year))
        params.append(_year_start(year + 1))
    where = " AND ".join(clauses) if clauses else "1=1"
    rows = conn.execute(f"SELECT uuid FROM assets WHERE {where}", params).fetchall()
    return [r[0] for r in rows]


def _year_start(year: int) -> float:
    return datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
