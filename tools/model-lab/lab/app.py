"""Model lab dashboard: a local-only web app over data/model-lab/lab.sqlite.

    tools/model-lab/start.sh      # http://127.0.0.1:8765

Binds to 127.0.0.1 only. Every asset (JS, CSS) is served from this
directory - no CDNs, fonts or telemetry - so it works fully offline.
Query text is embedded by SigLIP2 in this process (on the Mac's GPU).
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import searchlab
from .store import DATA, MEDIA, ROOT, connect

STATIC = Path(__file__).parent / "static"
PHOTOS_DB = ROOT / "data" / "Photos.sqlite"
MEDIA_FILES = {"thumb.jpg", "full.jpg", "model.jpg", "video.mp4", "f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"}

app = FastAPI(title="memoria model lab", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_local = threading.local()
_siglip: dict = {}   # key -> {"model", "error", "load_seconds"}
_vec: dict = {}
FTS_TABLES = {"30b": "search_fts", "8b": "search_fts_8b", "32b": "search_fts_32b",
              "gemma4": "search_fts_gemma4", "qwen3.8": "search_fts_qwen38"}
VECTOR_MODELS = ("siglip2", "siglip2-giant")


def db() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = connect()
    return _local.conn


def vec_index(model: str = "siglip2") -> searchlab.VectorIndex:
    if model not in VECTOR_MODELS:
        raise HTTPException(400, f"unknown vector model {model}")
    if model not in _vec:
        _vec[model] = searchlab.VectorIndex(db(), model)
    return _vec[model]


def available_vector_models() -> list[str]:
    have = {r[0] for r in db().execute("SELECT DISTINCT model FROM alt_embeddings")}
    return ["siglip2"] + [m for m in VECTOR_MODELS[1:] if m in have]


_load_lock = threading.Lock()


def _load_siglip(key: str):
    with _load_lock:
        if key in _siglip:
            return
        _siglip[key] = {"model": None, "error": None, "load_seconds": None}
        try:
            from .siglip import Siglip
            s = Siglip(key)
            _siglip[key].update(model=s, load_seconds=s.load_seconds)
        except Exception as e:  # noqa: BLE001
            _siglip[key]["error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load_siglip, args=("siglip2",), daemon=True).start()


def embed_query(text: str, key: str = "siglip2") -> np.ndarray:
    return embed_queries([text], key)[0]


def embed_queries(texts: list[str], key: str = "siglip2") -> np.ndarray:
    """The query text goes through the SAME model's text encoder as the
    image vectors it's compared with - text and image vectors from
    different models live in different spaces."""
    if key not in _siglip:
        _load_siglip(key)
    for _ in range(1200):
        if _siglip[key]["model"] or _siglip[key]["error"]:
            break
        time.sleep(0.1)
    if not _siglip[key]["model"]:
        raise HTTPException(503, f"{key} text encoder not available: {_siglip[key]['error']}")
    return _siglip[key]["model"].embed_texts(texts)


# ---------- pages ----------

for page in ("index", "asset", "search", "map", "stats", "compare", "ask", "mlx"):
    def _page(page=page):
        return FileResponse(STATIC / f"{page}.html")
    app.add_api_route("/" if page == "index" else f"/{page}", _page, include_in_schema=False)


FRAME_DIRS = {"fixed4": "frames", "change": "frames_change", "dense": "frames_dense"}


@app.get("/media/{uuid}/frames/{strategy}/{name}")
def frame(uuid: str, strategy: str, name: str):
    if strategy not in FRAME_DIRS or not all(c in "0123456789ABCDEFabcdef-" for c in uuid) \
            or not (name.startswith("f") and name.endswith(".jpg") and name[1:-4].isdigit()):
        raise HTTPException(404)
    p = MEDIA / uuid / FRAME_DIRS[strategy] / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/media/{uuid}/{name}")
def media(uuid: str, name: str):
    if name not in MEDIA_FILES or not all(c in "0123456789ABCDEFabcdef-" for c in uuid):
        raise HTTPException(404)
    p = MEDIA / uuid / (f"frames/{name}" if name.startswith("f") and name[1].isdigit() else name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ---------- data ----------

def _asset_rows(where="1=1", params=()):
    c = db()
    return c.execute(f"""
        SELECT a.uuid, a.kind, a.date_created, a.status, a.note, a.original_filename, l.strata, l.dup_group, l.duration,
               l.width, l.height, l.has_audio, l.uti, l.subtype,
               q.aesthetic, json_extract(tq.detail, '$.quality') AS quality,
               (SELECT label FROM zero_shot z WHERE z.asset_id=a.uuid ORDER BY softmax DESC LIMIT 1) AS label,
               (SELECT softmax FROM zero_shot z WHERE z.asset_id=a.uuid ORDER BY softmax DESC LIMIT 1) AS label_p,
               tm.has_speech, l.tz_offset, l.tz_name, l.latitude, l.longitude, l.face_count,
               (SELECT json_group_array(person_name) FROM asset_people p WHERE p.asset_id=a.uuid) AS people
        FROM assets a JOIN lab_assets l ON l.uuid=a.uuid
        LEFT JOIN quality_scores q ON q.asset_id=a.uuid
        LEFT JOIN timings tq ON tq.asset_id=a.uuid AND tq.model='q-align'
        LEFT JOIN transcript_meta tm ON tm.asset_id=a.uuid
        WHERE {where} ORDER BY a.date_created""", params).fetchall()


def _brief(r):
    return {"uuid": r["uuid"], "kind": r["kind"], "date": r["date_created"], "filename": r["original_filename"],
            "year": time.gmtime(r["date_created"]).tm_year if r["date_created"] else None,
            "status": r["status"], "strata": (r["strata"] or "").split(","), "label": r["label"],
            "label_p": r["label_p"], "aesthetic": r["aesthetic"], "quality": r["quality"],
            "duration": r["duration"], "has_speech": r["has_speech"], "dup_group": r["dup_group"],
            "people": sorted(json.loads(r["people"] or "[]")), "faces": r["face_count"],
            "local": _local_time(r["date_created"], r["tz_offset"]), "tz_name": r["tz_name"],
            "gps": [r["latitude"], r["longitude"]] if r["latitude"] is not None else None}


def _local_time(ts, offset):
    """Capture time in the photo's own time zone (Photos stores the offset);
    None when Photos doesn't know the zone."""
    if ts is None or offset is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts + offset)) + f" (UTC{offset / 3600:+g})"


@app.get("/api/assets")
def assets():
    return [_brief(r) for r in _asset_rows()]


def _briefs(ids):
    if not ids:
        return {}
    rows = _asset_rows(f"a.uuid IN ({','.join('?' * len(ids))})", ids)
    return {r["uuid"]: _brief(r) for r in rows}


@app.get("/api/asset/{uuid}")
def asset(uuid: str):
    c = db()
    rows = _asset_rows("a.uuid=?", (uuid,))
    if not rows:
        raise HTTPException(404)
    r = rows[0]
    out = _brief(r) | {"width": r["width"], "height": r["height"], "note": r["note"], "uti": r["uti"],
                       "has_audio": r["has_audio"]}
    out["vlm"] = {v["model"]: {**dict(v), "tags": json.loads(v["tags"] or "[]")}
                  for v in c.execute("SELECT * FROM vlm_results WHERE asset_id=?", (uuid,))}
    out["zero_shot"] = [dict(z) for z in c.execute("SELECT * FROM zero_shot WHERE asset_id=? ORDER BY softmax DESC", (uuid,))]
    out["segments"] = [dict(s) for s in c.execute("SELECT * FROM transcript_segments WHERE asset_id=? ORDER BY idx", (uuid,))]
    tm = c.execute("SELECT * FROM transcript_meta WHERE asset_id=?", (uuid,)).fetchone()
    out["transcript_meta"] = dict(tm) if tm else None
    out["timings"] = {t["model"]: {"seconds": t["seconds"], "detail": json.loads(t["detail"]) if t["detail"] else None}
                      for t in c.execute("SELECT * FROM timings WHERE asset_id=?", (uuid,))}
    qs = c.execute("SELECT * FROM quality_scores WHERE asset_id=?", (uuid,)).fetchone()
    out["quality_row"] = dict(qs) if qs else None
    emb = c.execute("SELECT model_version FROM embeddings WHERE asset_id=?", (uuid,)).fetchone()
    out["embedding_model"] = emb[0] if emb else None
    out["queries"] = [q["query"] for q in c.execute("SELECT query FROM example_queries WHERE asset_id=? ORDER BY idx", (uuid,))]
    frames = {}
    if c.execute("SELECT name FROM sqlite_master WHERE name='video_frames'").fetchone():
        for fr in c.execute("SELECT strategy, idx, t, px FROM video_frames WHERE asset_id=? ORDER BY strategy, idx", (uuid,)):
            files = sorted(p.name for p in (MEDIA / uuid / FRAME_DIRS[fr["strategy"]]).glob("f*.jpg"))
            if fr["idx"] < len(files):
                frames.setdefault(fr["strategy"], []).append({"t": fr["t"], "px": fr["px"], "file": files[fr["idx"]]})
    out["frames"] = frames
    return out


def _run_search(q, min_sim=None, index="30b", mode="or", limit=50, vmodel="siglip2"):
    table = FTS_TABLES.get(index, "search_fts")
    expr, kw = searchlab.keyword(db(), q, table=table, limit=limit, mode=mode)
    vh = vec_index(vmodel).search(embed_query(q, vmodel), limit=limit, min_similarity=min_sim)
    hy = searchlab.hybrid(kw, vh, min_similarity=min_sim, limit=limit)
    return expr, kw, vh, hy


@app.get("/api/search")
def search(q: str, min_sim: float | None = None, index: str = "30b", mode: str = "or", limit: int = 24,
           vmodel: str = "siglip2"):
    t0 = time.time()
    expr, kw, vh, hy = _run_search(q, min_sim, index, mode, limit=50, vmodel=vmodel)
    ms = (time.time() - t0) * 1000
    ids = list({h["id"] for h in kw[:limit] + vh[:limit] + hy[:limit]})
    return {"query": q, "fts_expression": expr, "rrf_k": searchlab.RRF_K, "ms": ms, "vmodel": vmodel,
            "keyword": kw[:limit], "vector": vh[:limit], "hybrid": hy[:limit], "assets": _briefs(ids),
            "keyword_total": len(kw),
            "vector_above_threshold": sum(1 for h in vh if not h["below_threshold"]) if min_sim is not None else None}


@app.get("/api/findability/{uuid}")
def findability(uuid: str, min_sim: float | None = None):
    qs = [q["query"] for q in db().execute("SELECT query FROM example_queries WHERE asset_id=? ORDER BY idx", (uuid,))]
    total = db().execute("SELECT count(*) FROM embeddings").fetchone()[0]
    out = []
    for q in qs:
        expr, kw, vh, hy = _run_search(q, min_sim, limit=total)
        hk = next((h for h in hy if h["id"] == uuid), None)
        extra = {}
        for k, table in _active_fts().items():
            if k != "30b":
                _, h = searchlab.keyword(db(), q, table=table, limit=total)
                extra[f"keyword {k}"] = searchlab.rank_of(h, uuid)
        for vm in available_vector_models()[1:]:
            h = vec_index(vm).search(embed_query(q, vm), limit=total)
            extra[f"vector {vm}"] = searchlab.rank_of(h, uuid)
        out.append({"query": q, "fts_expression": expr, "extra": extra,
                    "keyword_rank": searchlab.rank_of(kw, uuid), "keyword_hits": len(kw),
                    "vector_rank": searchlab.rank_of(vh, uuid),
                    "vector_sim": next((h["score"] for h in vh if h["id"] == uuid), None),
                    "hybrid_rank": searchlab.rank_of(hy, uuid), "hybrid": hk,
                    "top3": [h["id"] for h in hy[:3]]})
    return {"rrf_k": searchlab.RRF_K, "total": total, "queries": out}


def _active_fts():
    return {k: t for k, t in FTS_TABLES.items() if db().execute(f"SELECT count(*) FROM {t}").fetchone()[0]}


@app.get("/api/compare")
def compare(refresh: bool = False):
    from . import compare as cmp
    c = db()
    sig = [c.execute(q).fetchone()[0] for q in (
        "SELECT count(*) FROM example_queries", "SELECT count(*) FROM embeddings",
        "SELECT count(*) FROM alt_embeddings", "SELECT count(*) FROM vlm_results WHERE error IS NULL",
        "SELECT count(*) FROM search_fts_32b", "SELECT count(*) FROM search_fts_gemma4",
        "SELECT count(*) FROM search_fts_qwen38")]
    cache = DATA / "compare.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        if data.get("sig") == sig:
            return data
    data = cmp.compute(c, embed_queries, vec_index, _active_fts(), available_vector_models()) | {"sig": sig}
    cache.write_text(json.dumps(data))
    return data


class AskRequest(BaseModel):
    question: str
    facts: str = ""
    model: str = "8b"
    k: int = 20


@app.post("/api/ask")
def ask(req: AskRequest):
    """POST so questions and facts (e.g. birthdays) never land in URLs or
    server logs."""
    from . import ask as ask_mod
    if req.model not in ask_mod.MODELS:
        raise HTTPException(400, "model must be 8b or 30b")
    out = ask_mod.answer(db(), req.question, embed_query, vec_index, req.facts, req.model, k=max(5, min(60, req.k)))
    return out | {"assets": _briefs([i["id"] for i in out["items"]])}


MLX_DIR = ROOT / "tools" / "model-lab" / "mlx_handson"
MLX_EXERCISES = [("01_load_and_inspect", "01_inspect"), ("02_quantize", "02_quantize"),
                 ("03_weights_up_close", "03_weights"), ("04_change_the_weights", "04_change")]


@app.get("/api/mlx")
def mlx_results():
    out = []
    for script, result in MLX_EXERCISES:
        src = (MLX_DIR / f"{script}.py").read_text()
        res = DATA / "mlx" / "results" / f"{result}.json"
        data = json.loads(res.read_text()) if res.exists() else None
        if data:
            for key in ("images",):
                if key in data:
                    data[key] = [{k: v for k, v in im.items() if k != "path"} for im in data[key]]
        out.append({"script": script, "command": f".venv/bin/python tools/model-lab/mlx_handson/{script}.py",
                    "doc": src.split('"""')[1].strip(), "source": src, "result": data})
    return out


@app.get("/api/similar/{uuid}")
def similar(uuid: str, limit: int = 12, vmodel: str = "siglip2"):
    idx = vec_index(vmodel)
    v = idx.vector(uuid)
    if v is None:
        return {"results": [], "assets": {}}
    hits = idx.search(v, limit=limit, exclude=uuid)
    return {"results": hits, "assets": _briefs([h["id"] for h in hits])}


@app.get("/api/map")
def embedding_map(vmodel: str = "siglip2"):
    idx = vec_index(vmodel)
    cache = DATA / ("projection.json" if vmodel == "siglip2" else f"projection-{vmodel}.json")
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("n") == len(idx.ids):
            return data | {"assets": _briefs(idx.ids)}
    from sklearn.decomposition import PCA
    m = idx.matrix
    pca = PCA(n_components=2, random_state=0).fit(m)
    p = pca.transform(m)
    data = {"n": len(idx.ids), "ids": idx.ids, "pca": p.tolist(),
            "pca_variance": pca.explained_variance_ratio_.tolist()}
    try:
        import umap
        u = umap.UMAP(n_neighbors=12, min_dist=0.15, metric="cosine", random_state=0).fit_transform(m)
        data["umap"] = u.tolist()
    except Exception as e:  # noqa: BLE001
        data["umap_error"] = str(e)
    cache.write_text(json.dumps(data))
    return data | {"assets": _briefs(idx.ids)}


def _library_totals():
    cache = DATA / "library_stats.json"
    if cache.exists():
        return json.loads(cache.read_text())
    conn = sqlite3.connect(f"file:{PHOTOS_DB}?mode=ro", uri=True)
    total, images, videos, vid_seconds = conn.execute(
        "SELECT count(*), sum(ZKIND=0), sum(ZKIND=1), sum(CASE WHEN ZKIND=1 THEN ZDURATION ELSE 0 END) "
        "FROM ZASSET WHERE ZTRASHEDSTATE=0").fetchone()
    data = {"total_all": conn.execute("SELECT count(*) FROM ZASSET").fetchone()[0], "total": total,
            "images": images, "videos": videos, "video_seconds": vid_seconds}
    cache.write_text(json.dumps(data))
    return data


PRODUCTION_MODELS = ["qwen3-vl-30b", "siglip2", "whisper", "q-align"]


@app.get("/api/stats")
def stats():
    c = db()
    runs = [dict(r) for r in c.execute("SELECT * FROM model_runs ORDER BY started_at")]
    per = {}
    for r in c.execute("""SELECT t.model, a.kind, count(*) n, avg(t.seconds) mean, min(t.seconds) mn, max(t.seconds) mx,
                                 group_concat(t.seconds) all_s
                          FROM timings t JOIN assets a ON a.uuid=t.asset_id GROUP BY t.model, a.kind"""):
        xs = sorted(float(x) for x in r["all_s"].split(","))
        per.setdefault(r["model"], {})[r["kind"]] = {
            "n": r["n"], "mean": r["mean"], "min": r["mn"], "max": r["mx"],
            "p50": xs[len(xs) // 2], "p90": xs[min(len(xs) - 1, int(len(xs) * 0.9))]}
    tps = {r["model"]: {"mean_tps": r["tps"], "prompt_tokens": r["pt"], "output_tokens": r["ot"]}
           for r in c.execute("SELECT model, avg(tokens_per_sec) tps, avg(prompt_tokens) pt, avg(output_tokens) ot "
                              "FROM vlm_results WHERE error IS NULL GROUP BY model")}
    whisper_rtf = c.execute("SELECT sum(t.seconds) / NULLIF(sum(json_extract(t.detail,'$.audio_seconds')),0) "
                            "FROM timings t WHERE t.model='whisper'").fetchone()[0]
    lib = _library_totals()
    proj = {}
    for model, kinds in per.items():
        if model == "whisper":
            secs = (whisper_rtf or 0) * lib["video_seconds"]
            basis = f"{whisper_rtf:.3f} s of compute per second of audio x {lib['video_seconds'] / 3600:.0f} h of video"
        else:
            img = kinds.get("image", {}).get("mean", 0)
            vid = kinds.get("video", {}).get("mean", img)
            secs = img * lib["images"] + vid * lib["videos"]
            basis = f"{img:.2f} s x {lib['images']:,} photos + {vid:.2f} s x {lib['videos']:,} videos"
        proj[model] = {"seconds": secs, "basis": basis}
    counts = {"assets": c.execute("SELECT count(*) FROM assets").fetchone()[0],
              "failed": c.execute("SELECT count(*) FROM assets WHERE status='failed'").fetchone()[0],
              "failures": [dict(r) for r in c.execute("SELECT note, count(*) n FROM assets WHERE status='failed' GROUP BY note")]}
    return {"runs": runs, "per_asset": per, "tokens": tps, "whisper_rtf": whisper_rtf, "library": lib,
            "projection": proj, "counts": counts, "siglip_query_load_seconds": _siglip.get("siglip2", {}).get("load_seconds"),
            "production": PRODUCTION_MODELS}


@app.get("/api/status")
def status():
    return {"vector_models": available_vector_models(),
            "fts_indexes": [k for k, t in FTS_TABLES.items()
                            if db().execute(f"SELECT count(*) FROM {t}").fetchone()[0]],
            "loaded": {k: bool(v["model"]) for k, v in _siglip.items()}}


@app.exception_handler(sqlite3.OperationalError)
def _sqlite_error(_, exc):
    return JSONResponse({"detail": f"query error: {exc}"}, status_code=400)
