# ADR-0019: Model upgrades trigger a deliberate full re-index, never a silent mixed state

**Status:** Accepted (2026-09-14)

## Context

The local model landscape moves fast (ADR-0003 itself notes several relevant models emerged within one research pass). At some point a meaningfully better tagging or embedding model will exist. If newly-indexed assets silently start using a new model while older assets keep embeddings/tags from an old one, search and ranking quality becomes inconsistent in ways that are hard to diagnose — two visually similar photos could rank very differently purely because they were indexed by different model versions. Immich's own handling of exactly this problem (re-embedding a live library on a CLIP model swap, noted during OSS research) was flagged as worth studying but never turned into a decision here.

## Decision

Record which model (name + version + quantization) produced each embedding/tag/score, per asset, in `sqlite-vec`. A model upgrade is a deliberate, explicit full re-index of the affected pipeline stage — never an automatic, silent switch applied only to new assets going forward. Mixed-model states are allowed only transiently, during a re-index in progress, never as a steady-state.

## Consequences

- Requires a `model_version` (or similar) column per relevant score/embedding/tag field, not just the value itself — a schema requirement to design in from the start rather than retrofit.
- A model upgrade is a real, potentially expensive operation (re-running the biggest model over the whole library, ADR-0003), not a quick swap — this is a deliberate cost accepted in exchange for consistent, comparable results across the whole library.
- Search/ranking code never needs to reason about "which model version produced this value" at query time, since the store never holds a silently-inconsistent mix as steady state.

## Alternatives considered

- **Let new assets use the newest model, leave old assets as-is indefinitely:** rejected — produces exactly the inconsistent, hard-to-diagnose ranking behavior this decision exists to avoid.
- **Automatically re-index everything the moment a new model is detected:** rejected — an expensive operation shouldn't trigger itself silently; re-indexing on model upgrade is a deliberate action the user takes knowingly.
