# ADR-0025: `expression_scores` keyed by Photos' own face identity, not an invented index

**Status:** Accepted (2026-09-14)

## Context

The original schema keyed `expression_scores` by `face_index` - a 0-based position into however many faces the expression model detected on a given run. That's fragile: if a re-index (ADR-0019's model-upgrade re-index, or any re-run) detects the same faces in a different order, or detects a different number, the index no longer lines up with the same physical face, silently corrupting the association between a score and the face it was actually measured on.

Photos' own `ZDETECTEDFACE` already assigns a stable identifier to each detected face, independent of processing order - the same identifier this project already reads for `asset_people` (ADR-0023).

## Decision

Replace `face_index` with `face_key`, referencing Photos' own stable per-face identifier. `expression_scores` is keyed by `(asset_id, face_key)`.

## Consequences

- A re-run of the expression scorer can never silently misattribute a score to the wrong face, regardless of detection order.
- Consistent with `asset_people` (ADR-0023) - both cache data anchored to Photos' own stable identifiers, not values this project invents itself.
- Requires actually reading the face identifier out of `ZDETECTEDFACE` during indexing, not just running the expression model on cropped face regions in whatever order they come back - a small real requirement on the indexing code, not just a schema rename.

## Alternatives considered

- **Keep `face_index`, document the ordering risk:** rejected - a documented footgun is still a footgun, and the fix (use the identifier Photos already provides) costs nothing extra to implement correctly the first time.
