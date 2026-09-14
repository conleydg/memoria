"""Asset lifecycle classification (ADR-0018): the watcher (ADR-0013) needs
to detect not just new assets but edited and deleted/trashed ones too.

Pure functions, no sqlite/file I/O - same reasoning as PhotoDedupReport's
dedup_logic.py: the decision logic is small, has real edge cases worth
testing exhaustively, and shouldn't need a real Photos.sqlite or a real
memoria store to verify it's correct.

An asset dict has at minimum: uuid, trashed (bool), modified_at (float
epoch seconds or None). `modified_at` is Photos' own last-modified
signal for that asset - exactly what changes when a photo is edited.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LifecycleDiff:
    new: frozenset[str] = field(default_factory=frozenset)
    changed: frozenset[str] = field(default_factory=frozenset)
    removed: frozenset[str] = field(default_factory=frozenset)

    def is_empty(self) -> bool:
        return not (self.new or self.changed or self.removed)


def classify(photos_assets: list[dict], indexed_assets: list[dict]) -> LifecycleDiff:
    """photos_assets: current live state read from Photos.sqlite.
    indexed_assets: what memoria's own store currently has on record for
    assets it has seen before (not yet marked removed).

    - new: in Photos, never seen in our store before.
    - changed: in both, but Photos' modified_at is newer than what we
      recorded at last index, or trashed state flipped false -> true
      (a photo trashed needs the same "removed" treatment as changed,
      handled here as changed so the caller re-evaluates trashed state
      via the removed_at check below rather than this function guessing
      the caller's exact removal semantics).
    - removed: we have it on record as present, but it's either missing
      entirely from the current Photos read or its trashed flag is now
      true.
    """
    photos_by_uuid = {a["uuid"]: a for a in photos_assets}
    indexed_by_uuid = {a["uuid"]: a for a in indexed_assets}

    new = frozenset(photos_by_uuid.keys() - indexed_by_uuid.keys())

    removed = frozenset(
        uuid
        for uuid, indexed in indexed_by_uuid.items()
        if uuid not in photos_by_uuid or photos_by_uuid[uuid]["trashed"]
    )

    changed = frozenset(
        uuid
        for uuid, current in photos_by_uuid.items()
        if uuid in indexed_by_uuid
        and uuid not in removed
        and _is_newer(current.get("modified_at"), indexed_by_uuid[uuid].get("modified_at"))
    )

    return LifecycleDiff(new=new, changed=changed, removed=removed)


def _is_newer(current_modified_at, indexed_modified_at) -> bool:
    """A missing/None modified_at on either side is never treated as
    'changed' - no signal to act on beats a false-positive re-index."""
    if current_modified_at is None or indexed_modified_at is None:
        return False
    return current_modified_at > indexed_modified_at
