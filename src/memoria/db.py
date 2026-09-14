"""Schema for memoria's own store (ADR-0004: sqlite-vec, one file for
metadata + vectors + tags + quality + FTS5, no second storage system).

Every asset is keyed by Photos' own ZUUID (not a filesystem path - unlike
the prior dedup project, this store is Photos-database-centric, since it
needs to join AI-computed data against Photos' own metadata: dates,
named people, existing keywords).

`status`/`note`/`decided_at`-style suggestion columns and the
duplicate_groups/group_members join-table shape are carried over
deliberately from PhotoDedupReport's proven dedup.sqlite pattern
(ADR-0008, ADR-0014) - not reinvented.

Every table that holds AI-computed output carries `model_version`
(ADR-0019): a model upgrade means a deliberate full re-index, never a
silent per-asset mix of old and new model output, so which model
produced a value has to be recorded alongside the value itself.
"""

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    uuid TEXT PRIMARY KEY,                     -- Photos' ZASSET.ZUUID
    kind TEXT NOT NULL,                        -- 'image' | 'video'
    original_filename TEXT,
    date_created REAL,                         -- unix epoch seconds
    photos_modified_at REAL,                   -- Photos' own edit timestamp (ADR-0018 change detection)
    trashed INTEGER NOT NULL DEFAULT 0,        -- mirrors ZASSET.ZTRASHEDSTATE
    indexed_at REAL,                           -- last time THIS project fully indexed it; NULL = never
    removed_at REAL,                           -- set once detected gone from Photos (ADR-0018); NULL = still present
    status TEXT NOT NULL DEFAULT 'pending',
    note TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    tag TEXT NOT NULL,
    source TEXT NOT NULL,                      -- e.g. 'qwen3-vl'
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL,
    PRIMARY KEY (asset_id, tag, source)
);

CREATE TABLE IF NOT EXISTS captions (
    asset_id TEXT PRIMARY KEY REFERENCES assets(uuid),
    caption TEXT NOT NULL,
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    asset_id TEXT PRIMARY KEY REFERENCES assets(uuid),
    vector BLOB NOT NULL,                      -- packed float32 bytes (numpy .tobytes()); similarity
                                                -- computed brute-force in application code, not a sqlite-vec
                                                -- vec0 index - see ADR-0022 for why, and the escape hatch
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_scores (
    asset_id TEXT PRIMARY KEY REFERENCES assets(uuid),
    sharpness REAL,
    exposure REAL,
    aesthetic REAL,                            -- Q-Align (ADR-0021: non-commercial license, personal use only)
    activity REAL,                             -- video only (ADR-0009's boring/activity signal)
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS expression_scores (
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    face_index INTEGER NOT NULL,               -- which detected face within the asset, 0-based
    smile REAL,
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL,
    PRIMARY KEY (asset_id, face_index)
);

CREATE TABLE IF NOT EXISTS duplicate_groups (
    id INTEGER PRIMARY KEY,
    match_type TEXT NOT NULL,                  -- 'exact' | 'pixel' | 'perceptual'
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_group_members (
    group_id INTEGER NOT NULL REFERENCES duplicate_groups(id),
    asset_id TEXT NOT NULL REFERENCES assets(uuid),
    suggested_keep INTEGER NOT NULL DEFAULT 0, -- ADR-0014: a suggestion, never acted on automatically
    PRIMARY KEY (group_id, asset_id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    asset_id TEXT PRIMARY KEY REFERENCES assets(uuid),
    text TEXT NOT NULL,
    model_version TEXT NOT NULL,
    computed_at REAL NOT NULL
);

-- ADR-0010: hybrid keyword+vector search. Kept as a separate FTS5 table
-- (not a column) so it can be rebuilt independently of the tables above.
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
    asset_id UNINDEXED,
    text
);
"""


def connect(path: str) -> sqlite3.Connection:
    """Open (creating if needed) memoria's own store and ensure the schema
    exists. Not read-only - this is *our* file, unlike Photos.sqlite."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def connect_photos_readonly(photos_sqlite_path: str) -> sqlite3.Connection:
    """Open Apple's Photos.sqlite read-only (ADR-0001: never write to it).
    Same `file:...?mode=ro` URI pattern already proven safe in
    PhotoDedupReport - a plain path string here would open for read-write
    and risks the live library, so this is the one connection helper in
    this project that must never be "simplified" to a plain sqlite3.connect."""
    return sqlite3.connect(f"file:{photos_sqlite_path}?mode=ro", uri=True)
