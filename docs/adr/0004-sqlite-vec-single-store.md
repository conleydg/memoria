# ADR-0004: sqlite-vec as the single metadata + vector store

**Status:** Accepted (2026-09-14)

## Context

Candidates researched: sqlite-vec, LanceDB, Chroma, usearch/FAISS. The store needs to hold ~150-300K image/video-keyframe embeddings plus relational metadata (dates, people, quality scores, duplicate groups, transcripts) with hybrid filtering (vector similarity *and* metadata predicates like `year = 2020`), fully offline, on a single machine. A prior project against the same library (PhotoDedupReport) already uses a single SQLite file (`dedup.sqlite`) for exact/perceptual duplicate groups, quality scores, and file metadata together.

## Decision

Use **sqlite-vec**: one SQLite file for metadata, vectors, tags, quality/duplicate/transcript data, and an FTS5 full-text index, queried with plain SQL throughout.

## Consequences

- No second storage system to run, back up, or reason about — matches the existing `dedup.sqlite` pattern, so operational knowledge (backup, inspection, querying) carries over directly.
- Hybrid vector+metadata queries and hybrid keyword+vector search (FTS5 + vectors, combined via Reciprocal Rank Fusion) are both plain SQL joins, not a second query language.
- LanceDB's native multimodal schema support and automatic versioning are given up — acceptable at this scale, but the strongest reason to revisit this decision if the project's needs change materially.
- Immich, a mature and popular self-hosted alternative, independently converged on the same "one hybrid vector+metadata store" pattern (Postgres+VectorChord) rather than a separate vector-only DB — external validation of this general approach, if not the exact technology.

## Alternatives considered

- **LanceDB:** runner-up — better native multimodal/versioning support, but a second storage system to operate for a benefit not yet needed.
- **Chroma:** simple API, but no disk-based indexing story (wants to fit in memory) — not disqualifying at 64GB but no advantage over sqlite-vec either.
- **usearch/FAISS:** bare similarity-search libraries, not databases — would require hand-rolling metadata storage and filtering; only worth it if sqlite-vec proves too slow, which is unlikely at this scale.
