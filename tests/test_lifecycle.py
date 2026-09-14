"""Tests for lifecycle.classify (ADR-0018). The goal is the same kind of
exhaustive edge-case coverage as PhotoDedupReport's test_dedup_logic.py:
this decides what gets re-indexed or dropped, so every boundary case
(missing dates, trashed-but-still-present, an asset that vanishes
entirely) needs an explicit answer, not an assumption.

Run with: pytest
"""

from memoria.lifecycle import LifecycleDiff, classify


def photos_asset(uuid, trashed=False, modified_at=None):
    return {"uuid": uuid, "trashed": trashed, "modified_at": modified_at}


def indexed_asset(uuid, modified_at=None):
    return {"uuid": uuid, "modified_at": modified_at}


class TestNewAssets:
    def test_asset_not_in_index_is_new(self):
        diff = classify([photos_asset("a1")], [])
        assert diff.new == {"a1"}
        assert diff.changed == frozenset()
        assert diff.removed == frozenset()

    def test_no_new_assets_when_everything_already_indexed(self):
        diff = classify([photos_asset("a1")], [indexed_asset("a1")])
        assert diff.new == frozenset()


class TestChangedAssets:
    def test_newer_modified_at_is_changed(self):
        diff = classify(
            [photos_asset("a1", modified_at=200.0)],
            [indexed_asset("a1", modified_at=100.0)],
        )
        assert diff.changed == {"a1"}

    def test_same_modified_at_is_not_changed(self):
        diff = classify(
            [photos_asset("a1", modified_at=100.0)],
            [indexed_asset("a1", modified_at=100.0)],
        )
        assert diff.changed == frozenset()

    def test_older_modified_at_is_not_changed(self):
        # Clock skew / a stale re-read shouldn't trigger a re-index.
        diff = classify(
            [photos_asset("a1", modified_at=50.0)],
            [indexed_asset("a1", modified_at=100.0)],
        )
        assert diff.changed == frozenset()

    def test_missing_modified_at_on_either_side_is_never_changed(self):
        assert classify(
            [photos_asset("a1", modified_at=None)],
            [indexed_asset("a1", modified_at=100.0)],
        ).changed == frozenset()
        assert classify(
            [photos_asset("a1", modified_at=200.0)],
            [indexed_asset("a1", modified_at=None)],
        ).changed == frozenset()


class TestRemovedAssets:
    def test_missing_from_photos_entirely_is_removed(self):
        diff = classify([], [indexed_asset("a1")])
        assert diff.removed == {"a1"}

    def test_trashed_in_photos_is_removed(self):
        diff = classify(
            [photos_asset("a1", trashed=True)],
            [indexed_asset("a1")],
        )
        assert diff.removed == {"a1"}

    def test_present_and_not_trashed_is_not_removed(self):
        diff = classify(
            [photos_asset("a1", trashed=False)],
            [indexed_asset("a1")],
        )
        assert diff.removed == frozenset()

    def test_trashed_asset_is_removed_not_also_changed(self):
        # A trashed asset with a bumped modified_at should be classified
        # once, as removed - not double-counted into both sets.
        diff = classify(
            [photos_asset("a1", trashed=True, modified_at=999.0)],
            [indexed_asset("a1", modified_at=1.0)],
        )
        assert diff.removed == {"a1"}
        assert diff.changed == frozenset()


class TestCombinedScenarios:
    def test_disjoint_new_changed_removed_in_one_pass(self):
        diff = classify(
            photos_assets=[
                photos_asset("new1"),
                photos_asset("changed1", modified_at=200.0),
                photos_asset("untouched1", modified_at=100.0),
            ],
            indexed_assets=[
                indexed_asset("changed1", modified_at=100.0),
                indexed_asset("untouched1", modified_at=100.0),
                indexed_asset("gone1"),
            ],
        )
        assert diff.new == {"new1"}
        assert diff.changed == {"changed1"}
        assert diff.removed == {"gone1"}

    def test_empty_diff_reports_is_empty(self):
        assert classify([], []).is_empty()
        assert not classify([photos_asset("a1")], []).is_empty()


def test_lifecycle_diff_defaults_to_all_empty():
    assert LifecycleDiff().is_empty()
