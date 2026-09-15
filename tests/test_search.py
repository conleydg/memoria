"""Tests for the hybrid search layer (ADR-0010/0022) against synthetic
data - no AI model involved. Fake embeddings, fake tags, fake FTS text,
built by hand so the fusion/filtering/ranking logic can be verified
before any real model exists to produce real ones.

Run with: pytest
"""

import numpy as np
import pytest

from memoria.db import connect
from memoria.search import (
    keyword_search,
    reciprocal_rank_fusion,
    semantic_search,
    vector_search,
)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion - pure function, no DB at all
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_single_ranking_passes_through_in_order(self):
        assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_item_ranked_well_in_both_beats_item_in_only_one(self):
        # "a" is #1 in both rankings; "b" only appears in the first.
        fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        assert fused[0] == "a"

    def test_disjoint_rankings_include_everything(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        assert set(fused) == {"a", "b", "c", "d"}

    def test_empty_rankings_produce_empty_result(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_matches_hand_computed_rrf_score(self):
        # k=10 for round numbers: score(a) = 1/(10+1) + 1/(10+2) = 0.1758...
        #                          score(b) = 1/(10+2) + 1/(10+1) = same as a
        #                          score(c) = 1/(10+1) only = 0.0909...
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10)
        assert set(fused[:2]) == {"a", "b"}  # tied for first, order between them is arbitrary
        assert "c" not in fused  # never appeared anywhere


# ---------------------------------------------------------------------------
# vector_search - brute-force cosine similarity against hand-built vectors
# ---------------------------------------------------------------------------

def _insert_asset(conn, uuid, kind="image", date_created=0.0):
    conn.execute(
        "INSERT INTO assets (uuid, kind, date_created) VALUES (?, ?, ?)",
        (uuid, kind, date_created),
    )


def _insert_embedding(conn, asset_id, vector: np.ndarray):
    conn.execute(
        "INSERT INTO embeddings (asset_id, vector, model_version, computed_at) VALUES (?, ?, ?, ?)",
        (asset_id, vector.astype(np.float32).tobytes(), "test-v1", 0.0),
    )


class TestVectorSearch:
    def test_closest_vector_ranks_first(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        for uid in ("close", "far", "opposite"):
            _insert_asset(conn, uid)
        _insert_embedding(conn, "close", np.array([1.0, 0.1, 0.0]))
        _insert_embedding(conn, "far", np.array([0.0, 1.0, 1.0]))
        _insert_embedding(conn, "opposite", np.array([-1.0, -0.1, 0.0]))
        conn.commit()

        results = vector_search(conn, np.array([1.0, 0.0, 0.0]))
        assert results[0] == "close"
        assert results[-1] == "opposite"

    def test_empty_candidate_list_short_circuits_to_empty(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        _insert_asset(conn, "a1")
        _insert_embedding(conn, "a1", np.array([1.0, 0.0]))
        conn.commit()
        assert vector_search(conn, np.array([1.0, 0.0]), candidate_ids=[]) == []

    def test_candidate_ids_restrict_the_search(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        for uid in ("a1", "a2"):
            _insert_asset(conn, uid)
        _insert_embedding(conn, "a1", np.array([1.0, 0.0]))
        _insert_embedding(conn, "a2", np.array([1.0, 0.0]))  # equally close, but excluded below
        conn.commit()

        results = vector_search(conn, np.array([1.0, 0.0]), candidate_ids=["a1"])
        assert results == ["a1"]

    def test_no_embeddings_returns_empty(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        assert vector_search(conn, np.array([1.0, 0.0])) == []


# ---------------------------------------------------------------------------
# keyword_search - FTS5, including the filter-then-limit correctness case
# ---------------------------------------------------------------------------

def _insert_fts(conn, asset_id, text):
    conn.execute("INSERT INTO search_fts (asset_id, text) VALUES (?, ?)", (asset_id, text))


class TestKeywordSearch:
    def test_finds_matching_text(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        for uid in ("a1", "a2"):
            _insert_asset(conn, uid)
        _insert_fts(conn, "a1", "birthday cake candles")
        _insert_fts(conn, "a2", "beach sunset")
        conn.commit()

        assert keyword_search(conn, "cake") == ["a1"]

    def test_filter_is_applied_before_limit_not_after(self, tmp_path):
        # 3 assets match the keyword; only 1 passes the candidate filter.
        # A naive "fetch top-N then filter" implementation would return []
        # here if N were smaller than the number of non-matching hits
        # ranked above the one real match - this pins the fix.
        conn = connect(str(tmp_path / "t.sqlite"))
        for uid in ("a1", "a2", "a3"):
            _insert_asset(conn, uid)
        _insert_fts(conn, "a1", "dog park")
        _insert_fts(conn, "a2", "dog park")
        _insert_fts(conn, "a3", "dog park")
        conn.commit()

        results = keyword_search(conn, "dog", candidate_ids=["a3"], limit=1)
        assert results == ["a3"]

    def test_empty_candidate_list_short_circuits_to_empty(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        _insert_asset(conn, "a1")
        _insert_fts(conn, "a1", "dog park")
        conn.commit()
        assert keyword_search(conn, "dog", candidate_ids=[]) == []


# ---------------------------------------------------------------------------
# semantic_search - end to end: filter, then fuse keyword + vector hits
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_fuses_keyword_and_vector_signal(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        _insert_asset(conn, "best")   # matches both keyword and vector well
        _insert_asset(conn, "vector_only")
        _insert_asset(conn, "keyword_only")

        _insert_fts(conn, "best", "birthday cake")
        _insert_fts(conn, "keyword_only", "birthday cake")
        _insert_embedding(conn, "best", np.array([1.0, 0.0]))
        _insert_embedding(conn, "vector_only", np.array([1.0, 0.0]))
        conn.commit()

        results = semantic_search(conn, "cake", np.array([1.0, 0.0]))
        assert results[0] == "best"
        assert set(results) == {"best", "vector_only", "keyword_only"}

    def test_year_filter_excludes_out_of_range_assets(self, tmp_path):
        import datetime

        conn = connect(str(tmp_path / "t.sqlite"))
        in_2020 = datetime.datetime(2020, 6, 1, tzinfo=datetime.timezone.utc).timestamp()
        in_2021 = datetime.datetime(2021, 6, 1, tzinfo=datetime.timezone.utc).timestamp()
        _insert_asset(conn, "y2020", date_created=in_2020)
        _insert_asset(conn, "y2021", date_created=in_2021)
        _insert_fts(conn, "y2020", "christmas tree")
        _insert_fts(conn, "y2021", "christmas tree")
        _insert_embedding(conn, "y2020", np.array([1.0, 0.0]))
        _insert_embedding(conn, "y2021", np.array([1.0, 0.0]))
        conn.commit()

        results = semantic_search(conn, "christmas", np.array([1.0, 0.0]), filters={"year": 2020})
        assert results == ["y2020"]

    def test_kind_filter_excludes_other_kind(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        _insert_asset(conn, "img1", kind="image")
        _insert_asset(conn, "vid1", kind="video")
        _insert_fts(conn, "img1", "sunset")
        _insert_fts(conn, "vid1", "sunset")
        _insert_embedding(conn, "img1", np.array([1.0, 0.0]))
        _insert_embedding(conn, "vid1", np.array([1.0, 0.0]))
        conn.commit()

        results = semantic_search(conn, "sunset", np.array([1.0, 0.0]), filters={"kind": "image"})
        assert results == ["img1"]

    def test_no_matches_returns_empty_not_an_error(self, tmp_path):
        conn = connect(str(tmp_path / "t.sqlite"))
        _insert_asset(conn, "a1")
        conn.commit()
        assert semantic_search(conn, "nonexistent", np.array([1.0, 0.0])) == []
