# ADR-0026: Vector search can return "no confident match," not just a forced top-k

**Status:** Accepted (2026-09-14) — mechanism built now, real threshold value deferred

## Context

Cosine similarity always produces a ranking relative to whatever exists in the store, even when nothing is actually a good match - unlike FTS5 keyword search, which naturally returns nothing when no photo's tags/caption contain the query terms. Left as originally written, `vector_search` (ADR-0022) would confidently return a "closest" result even for a query the library has zero real matches for (e.g. an animal never photographed), and `semantic_search` would hand that back with no signal that it isn't a real match. That's a direct path to the agent (ADR-0005) presenting an irrelevant photo as if it answered the question, instead of honestly reporting no results - a trust-eroding failure mode for a personal tool.

## Decision

`vector_search` takes an optional `min_similarity` threshold; results below it are excluded rather than force-included to fill out a top-k. The actual threshold *value* is not set yet - there's no real embedding data to calibrate it against before the Mac Studio and a real model exist. The retrieval eval (ADR-0006) will include deliberate zero-result queries (something genuinely absent from the library) specifically to calibrate and verify this threshold once real embeddings exist.

## Consequences

- The mechanism is real and tested now, against synthetic vectors at an arbitrary test threshold - not blocked on hardware.
- Ships with `min_similarity=None` (no filtering) as the default until a real value is calibrated, so existing behavior/tests are unaffected until that calibration happens.
- The eval ground truth set needs at least a few queries with a *known, true* zero-result answer, not just positive-match queries - a methodological addition to ADR-0006's ground truth collection.
- The agent-behavior eval track (added to ADR-0006 in an earlier gap-review pass) should explicitly include this case: given a known zero-result query, does the agent honestly report "no photos found" rather than presenting a low-confidence result as if it were a match.

## Alternatives considered

- **Always return a top-k regardless of match quality:** rejected - this is the status quo being fixed, and the actual failure mode motivating this ADR.
- **Guess a threshold value now:** rejected - a wrong guess is worse than an honest "not yet calibrated," since it would be presented with the same confidence as a validated one.
