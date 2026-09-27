"""The lab's SQLite store: memoria's own schema (src/memoria/db.py), used
as-is, plus a few lab-only tables for side-by-side comparison, timings
and memory - things the production store doesn't need.

Where the two overlap, the lab writes the production tables the way the
real pipeline will (captions/tags/embeddings/transcripts/quality_scores/
search_fts, each with model_version per ADR-0019), so the dashboard's
search goes through the same shapes memoria's search.py expects.

- captions / search_fts hold the 30B (production-candidate) output; the
  8B output lives in vlm_results and a second FTS table, search_fts_8b,
  so keyword search can be compared across the two.
- embeddings holds packed float32 SigLIP2 vectors (ADR-0022,
  brute-force); siglip_vec mirrors them into a sqlite-vec vec0 table so
  both paths can be tried.
"""

import os
import sqlite3
import sys
from pathlib import Path

import sqlite_vec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from memoria import db as memoria_db  # noqa: E402

DATA = Path(os.environ.get("MODEL_LAB_DATA", ROOT / "data" / "model-lab"))
DB_PATH = DATA / "lab.sqlite"
MEDIA = DATA / "media"
CACHE = ROOT / "data" / "cache"
MANIFEST = DATA / "manifest.json"
EMBED_DIM = 1152  # SigLIP2 so400m

LAB_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS lab_assets (
    uuid TEXT PRIMARY KEY REFERENCES assets(uuid),
    relpath TEXT, uti TEXT, subtype INTEGER, duration REAL,
    original_size INTEGER, strata TEXT, dup_group TEXT,
    width INTEGER, height INTEGER,
    has_audio INTEGER, frame_count INTEGER
);

-- One row per (asset, VLM): the full prompt and raw response are kept so
-- the dashboard can show exactly what the model saw and said.
CREATE TABLE IF NOT EXISTS vlm_results (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    model TEXT NOT NULL,
    model_version TEXT NOT NULL,
    caption TEXT, tags TEXT, ocr_text TEXT,     -- tags: JSON array
    prompt TEXT, raw_response TEXT,
    images_sent INTEGER,
    prompt_tokens INTEGER, output_tokens INTEGER,
    seconds REAL, prompt_seconds REAL, output_seconds REAL,
    tokens_per_sec REAL,
    error TEXT,
    computed_at REAL NOT NULL,
    PRIMARY KEY (asset_id, model)
);

CREATE TABLE IF NOT EXISTS zero_shot (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    label TEXT NOT NULL,
    prompt TEXT NOT NULL,
    logit REAL, prob REAL, softmax REAL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (asset_id, label)
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    idx INTEGER NOT NULL,
    start REAL, end REAL, text TEXT,
    no_speech_prob REAL, avg_logprob REAL,
    PRIMARY KEY (asset_id, idx)
);

CREATE TABLE IF NOT EXISTS transcript_meta (
    asset_id TEXT PRIMARY KEY REFERENCES assets(uuid),
    language TEXT, has_speech INTEGER, audio_seconds REAL,
    model_version TEXT NOT NULL
);

-- Per-asset, per-model wall time (the VLMs also have finer detail in
-- vlm_results).
CREATE TABLE IF NOT EXISTS timings (
    asset_id TEXT NOT NULL,
    model TEXT NOT NULL,
    seconds REAL NOT NULL,
    detail TEXT,
    PRIMARY KEY (asset_id, model)
);

-- One row per model run over the test set.
CREATE TABLE IF NOT EXISTS model_runs (
    model TEXT PRIMARY KEY,
    model_version TEXT,
    role TEXT,
    runtime TEXT,
    license TEXT,
    disk_bytes INTEGER,
    load_seconds REAL,
    peak_memory_bytes INTEGER,
    memory_note TEXT,
    assets_done INTEGER, assets_failed INTEGER,
    total_seconds REAL,
    started_at REAL, finished_at REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS example_queries (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    idx INTEGER NOT NULL,
    query TEXT NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (asset_id, idx)
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_fts_8b USING fts5(asset_id UNINDEXED, text);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts_32b USING fts5(asset_id UNINDEXED, text);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts_gemma4 USING fts5(asset_id UNINDEXED, text);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts_qwen38 USING fts5(asset_id UNINDEXED, text);

-- Embeddings from alternative (comparison) models. The production
-- embedding lives in memoria's `embeddings` table; vectors from
-- different models can't be compared with each other (ADR-0019), so
-- each model's vectors are only ever searched against their own kind.
CREATE TABLE IF NOT EXISTS alt_embeddings (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL,
    PRIMARY KEY (asset_id, model)
);

CREATE VIRTUAL TABLE IF NOT EXISTS siglip_vec USING vec0(
    asset_id TEXT PRIMARY KEY,
    embedding float[{EMBED_DIM}] distance_metric=cosine
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(memoria_db.SCHEMA)
    conn.executescript(LAB_SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def record_run(conn, model: str, **fields) -> None:
    cols = ["model", *fields]
    conn.execute(
        f"INSERT INTO model_runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
        f"ON CONFLICT(model) DO UPDATE SET {','.join(f'{c}=excluded.{c}' for c in fields)}",
        [model, *fields.values()],
    )
    conn.commit()


def media_dir(uuid: str) -> Path:
    return MEDIA / uuid
