# ADR-0008: Reuse perceptual-hash duplicate grouping from the prior dedup project

**Status:** Accepted (2026-09-14)

## Context

"Find the best version of near-duplicate photos taken in succession" is functionally the same problem already solved by a prior project (PhotoDedupReport) against the same library: exact/pixel/perceptual duplicate grouping via sha256/pixel_hash/phash, with quality-based best-pick within each group.

## Decision

Fold perceptual-hash duplicate grouping into the indexing pipeline as a fifth per-asset processing step (classical CV — no ML model, same category as keyframe extraction), writing group membership and a suggested rank into `sqlite-vec`. Reuse the matching technique already proven in `dedup.sqlite` rather than rebuilding it independently.

## Consequences

- No need to re-derive or re-validate a duplicate-matching approach that's already working in production against this exact library.
- "Show me the best photo from each burst" becomes a plain filtered query against precomputed group data, not a new capability to design from scratch.
- The output is a *suggestion* only — see ADR-0014. The user still picks and deletes manually, exactly as in the prior project's existing workflow.

## Alternatives considered

- **Building a new duplicate-detection approach from scratch:** rejected — no reason to re-solve an already-solved problem against the same library.
