"""Model comparison: a small, automatic retrieval eval plus caption stats.

Each example query was written for one specific asset, so "where does
that asset rank?" gives a score for every search setup: Recall@k (was
it in the top k?) and MRR (mean of 1/rank, so rank 1 = 1.0, rank 2 =
0.5, not found = 0).

Caveats, shown in the UI: the caption queries were written from the 30B
caption, so they favour keyword search over the 30B index; they are
silver-quality at best. ADR-0006's hand-checked ground truth is the real
eval. The vector comparison (so400m vs giant) is fair: neither model saw
the queries being written.
"""

import json
import time

import numpy as np

from . import searchlab


def _metrics(ranks):
    n = len(ranks)
    if not n:
        return None
    return {"n": n,
            "r1": sum(1 for r in ranks if r and r <= 1) / n,
            "r5": sum(1 for r in ranks if r and r <= 5) / n,
            "r10": sum(1 for r in ranks if r and r <= 10) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "not_found": sum(1 for r in ranks if not r) / n}


def retrieval_eval(conn, embed_queries, vec_index, fts_tables: dict, vector_models: list[str]):
    qs = conn.execute("SELECT asset_id, query, model_version FROM example_queries").fetchall()
    groups = {"caption": [q for q in qs if q[2] != "photos-people"],
              "name": [q for q in qs if q[2] == "photos-people"]}
    total = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    out = {}
    for gname, items in groups.items():
        if not items:
            continue
        ranks: dict[str, list] = {}
        kw_hits = {k: [] for k in fts_tables}
        vec_hits = {}
        for vm in vector_models:
            idx = vec_index(vm)
            qv = embed_queries([q[1] for q in items], vm)
            vec_hits[vm] = [idx.search(v, limit=total) for v in qv]
        for i, (asset_id, query, _) in enumerate(items):
            for k, table in fts_tables.items():
                _, hits = searchlab.keyword(conn, query, table=table, limit=total)
                kw_hits[k].append(hits)
                ranks.setdefault(f"keyword · {k}", []).append(searchlab.rank_of(hits, asset_id))
            for vm in vector_models:
                ranks.setdefault(f"vector · {vm}", []).append(searchlab.rank_of(vec_hits[vm][i], asset_id))
            for k in fts_tables:
                for vm in vector_models:
                    fused = searchlab.hybrid(kw_hits[k][i], vec_hits[vm][i], limit=total)
                    ranks.setdefault(f"hybrid · {k} + {vm}", []).append(searchlab.rank_of(fused, asset_id))
        out[gname] = {m: _metrics(r) for m, r in ranks.items()}
    return out


def caption_stats(conn):
    # "model@strategy" rows are frame-selection experiments on videos only;
    # they're compared on the asset pages, not here.
    models = [r[0] for r in conn.execute("SELECT DISTINCT model FROM vlm_results WHERE model NOT LIKE '%@%' ORDER BY model")]
    stats = {}
    tags = {}
    for m in models:
        rows = conn.execute("SELECT v.asset_id, v.caption, v.tags, v.ocr_text, v.output_tokens, v.tokens_per_sec, "
                            "v.seconds, a.kind FROM vlm_results v JOIN assets a ON a.uuid=v.asset_id "
                            "WHERE v.model=? AND v.error IS NULL", (m,)).fetchall()
        tags[m] = {r[0]: set(json.loads(r[2] or "[]")) for r in rows}
        img = [r[6] for r in rows if r[7] == "image"]
        vid = [r[6] for r in rows if r[7] == "video"]
        stats[m] = {"n": len(rows),
                    "caption_chars": float(np.mean([len(r[1] or "") for r in rows])),
                    "tags": float(np.mean([len(tags[m][r[0]]) for r in rows])),
                    "ocr_rate": float(np.mean([1 if (r[3] or "").strip() else 0 for r in rows])),
                    "output_tokens": float(np.mean([r[4] or 0 for r in rows])),
                    "tps": float(np.mean([r[5] for r in rows if r[5]])),
                    "sec_image": float(np.mean(img)) if img else None,
                    "sec_video": float(np.mean(vid)) if vid else None}
    base = "qwen3-vl-30b"
    for m in models:
        if m == base or base not in tags:
            continue
        common = [a for a in tags[m] if a in tags[base]]
        jac = [len(tags[m][a] & tags[base][a]) / max(1, len(tags[m][a] | tags[base][a])) for a in common]
        stats[m]["tag_jaccard_vs_30b"] = float(np.mean(jac)) if jac else None
    return stats


def compute(conn, embed_queries, vec_index, fts_tables, vector_models):
    t0 = time.time()
    return {"retrieval": retrieval_eval(conn, embed_queries, vec_index, fts_tables, vector_models),
            "captions": caption_stats(conn), "vector_models": vector_models,
            "fts_indexes": list(fts_tables), "seconds": time.time() - t0, "computed_at": time.time()}
