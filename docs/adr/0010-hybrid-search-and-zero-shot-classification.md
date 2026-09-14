# ADR-0010: Hybrid keyword+vector search; zero-shot screenshot/document classification

**Status:** Accepted (2026-09-14)

## Context

Researching comparable open-source projects surfaced two near-free improvements given the stack already chosen. **Tagle**, a small local-first photo search project, uses Reciprocal Rank Fusion (RRF) to combine keyword and semantic search results rather than relying on vector search alone. **SnapSort** classifies screenshots/receipts/documents via zero-shot CLIP-style classification against text label prompts, with no separate trained model. Most real Photos libraries are heavily diluted with screenshots and documents that dilute every other search.

## Decision

- `semantic_search` becomes **hybrid**: FTS5 keyword hits and SigLIP2 vector hits, combined via Reciprocal Rank Fusion, rather than vector-only.
- SigLIP2's existing embedding also produces a zero-shot screenshot/document flag, by comparing the same embedding against candidate label prompts ("screenshot," "receipt," "photo") — no additional model.

## Consequences

- `sqlite-vec` needs an FTS5 full-text index over tags/captions/transcripts alongside the vector index (already reflected in ADR-0004).
- Both additions ride on infrastructure already being built (SigLIP2, the store) rather than adding new pipeline stages — genuinely low-cost given the stack already committed to.
- Keyword+vector fusion should improve exact-term recall (names, specific words) that pure semantic search sometimes misses.

## Alternatives considered

- **Vector-only search:** the original default; kept as the fallback if RRF fusion doesn't clearly improve results once measured against the retrieval eval (ADR-0006).
- **A trained screenshot classifier:** rejected in favor of zero-shot — no training data collection needed, and the underlying embedding already exists.
