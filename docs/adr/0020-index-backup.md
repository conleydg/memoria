# ADR-0020: Back up the index file via the existing external-drive process

**Status:** Accepted (2026-09-14)

## Context

`sqlite-vec` (ADR-0004) will end up holding weeks or months of AI-computed tags, embeddings, quality scores, duplicate groups, and transcripts — real computed value, not just a cache of something trivially reproducible (re-running the full pipeline, ADR-0003, is expensive). No backup or recovery plan for this file existed anywhere in the design. A related project against the same library (PhotoDedupReport) already backs up its own equivalent file and other derived data to an external drive.

## Decision

Back up `sqlite-vec` on the same schedule and to the same external drive already used for the prior dedup project's backups, rather than designing a new backup mechanism. A single-file SQLite database is straightforward to back up by simple periodic copy — no special tooling needed beyond what's already in place.

## Consequences

- A disk failure on the Mac Studio costs, at most, the time since the last backup — not a full reprocessing of the whole library from scratch.
- Reuses existing infrastructure and habits (the external drive, the backup cadence) rather than introducing a new backup system to maintain.
- Backing up while the batch indexing job (ADR-0013) is actively writing needs a moment-of-copy that doesn't corrupt an in-progress write — SQLite's own consistency guarantees during a filesystem-level copy should be checked once this is actually built, not assumed.

## Alternatives considered

- **No backup, treat the index as fully reproducible:** rejected — reprocessing the whole library through the full model pipeline is expensive (ADR-0003) and reproducibility isn't perfect anyway once model-upgrade re-indexing (ADR-0019) means "what model produced this" is meaningful history worth keeping.
