"""Tests for the schema itself - not exhaustive coverage of every column,
but the properties worth guaranteeing before any real code builds on this:
the schema creates cleanly, is idempotent to re-open, foreign keys are
actually enforced (not just declared), and the read-only Photos connection
helper genuinely refuses writes."""

import sqlite3

import pytest

from memoria.db import connect, connect_photos_readonly


def test_schema_creates_all_expected_tables(tmp_path):
    conn = connect(str(tmp_path / "test.sqlite"))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    expected = {
        "assets", "tags", "captions", "embeddings", "quality_scores",
        "expression_scores", "duplicate_groups", "duplicate_group_members",
        "transcripts", "search_fts", "asset_people", "pets",
        "pet_reference_photos", "pet_matches",
    }
    assert expected <= tables


def test_connect_is_idempotent(tmp_path):
    path = str(tmp_path / "test.sqlite")
    connect(path).close()
    conn = connect(path)  # must not raise on re-running CREATE TABLE IF NOT EXISTS
    conn.execute("SELECT 1 FROM assets")


def test_foreign_keys_are_enforced(tmp_path):
    conn = connect(str(tmp_path / "test.sqlite"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO tags (asset_id, tag, source, model_version, computed_at) "
            "VALUES ('does-not-exist', 'dog', 'qwen3-vl', 'v1', 0.0)"
        )
        conn.commit()


def test_fts_table_is_queryable(tmp_path):
    conn = connect(str(tmp_path / "test.sqlite"))
    conn.execute("INSERT INTO assets (uuid, kind) VALUES ('a1', 'image')")
    conn.execute("INSERT INTO search_fts (asset_id, text) VALUES ('a1', 'birthday cake candles')")
    conn.commit()
    rows = conn.execute("SELECT asset_id FROM search_fts WHERE search_fts MATCH 'cake'").fetchall()
    assert rows == [("a1",)]


def test_expression_scores_keyed_by_stable_face_key_not_an_index(tmp_path):
    # ADR-0025: face_key must be Photos' own identifier, not an invented
    # per-run index - this just confirms the column exists under that name
    # and two different faces on the same asset can coexist.
    conn = connect(str(tmp_path / "test.sqlite"))
    conn.execute("INSERT INTO assets (uuid, kind) VALUES ('a1', 'image')")
    conn.execute(
        "INSERT INTO expression_scores (asset_id, face_key, smile, model_version, computed_at) "
        "VALUES ('a1', 'photos-face-abc', 0.9, 'v1', 0.0)"
    )
    conn.execute(
        "INSERT INTO expression_scores (asset_id, face_key, smile, model_version, computed_at) "
        "VALUES ('a1', 'photos-face-def', 0.2, 'v1', 0.0)"
    )
    conn.commit()
    rows = conn.execute("SELECT face_key FROM expression_scores ORDER BY face_key").fetchall()
    assert rows == [("photos-face-abc",), ("photos-face-def",)]


def test_asset_people_and_pet_tables_enforce_foreign_keys(tmp_path):
    conn = connect(str(tmp_path / "test.sqlite"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO asset_people (asset_id, person_key, person_name, cached_at) "
            "VALUES ('does-not-exist', 'p1', 'Sarah', 0.0)"
        )
        conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO pet_matches (asset_id, pet_id, similarity, model_version, computed_at) "
            "VALUES ('does-not-exist', 1, 0.8, 'v1', 0.0)"
        )
        conn.commit()


def test_photos_connection_is_read_only(tmp_path):
    # A real (empty) sqlite file to open read-only against - standing in
    # for Photos.sqlite without needing the actual 750GB library.
    photos_path = tmp_path / "Photos.sqlite"
    setup_conn = sqlite3.connect(str(photos_path))
    setup_conn.execute("CREATE TABLE ZASSET (Z_PK INTEGER)")
    setup_conn.commit()
    setup_conn.close()

    conn = connect_photos_readonly(str(photos_path))
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO ZASSET (Z_PK) VALUES (1)")
