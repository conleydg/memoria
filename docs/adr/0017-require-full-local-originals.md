# ADR-0017: Require Photos originals fully downloaded locally, not iCloud-optimized

**Status:** Accepted (2026-09-14)

## Context

Apple Photos' "Optimize Mac Storage" setting keeps only lower-resolution proxies on disk for assets not recently accessed, with full originals fetched from iCloud on demand. The indexing pipeline (Qwen3-VL, SigLIP2, Whisper, etc., ADR-0003/0009) needs real pixel/audio data to produce good tags, embeddings, and transcripts — a low-res proxy would silently degrade every downstream result, with no obvious signal that anything was wrong. This was never previously stated as an explicit precondition anywhere in the design.

## Decision

This project requires Photos library originals to be fully downloaded and present on local disk — "Optimize Mac Storage" must be off for the library being indexed. This is consistent with the project's existing internet-independence stance (ADR-0001's privacy/local requirements): needing iCloud to fetch an original on demand would make indexing depend on connectivity anyway.

## Consequences

- Requires enough local disk space to hold the full library (~750GB as of this writing) — a real, named constraint on the hardware, not an incidental detail.
- The indexing pipeline can assume real pixel/audio data is always available locally; it does not need to detect or handle a "proxy-only" asset as a distinct case.
- If this constraint is ever violated (storage optimization gets re-enabled, accidentally or otherwise), results degrade silently rather than failing loudly — worth a startup check that flags any asset whose local file looks like a low-res proxy, once the indexing pipeline is actually built.

## Alternatives considered

- **Force-download originals on demand during indexing:** rejected for now — adds a network dependency and iCloud-fetch failure handling to a project whose whole premise is offline independence, for a problem better avoided by just requiring the setting be off in the first place.
