# ADR-0007: Precompute stable traits, judge subjective ones live and pairwise

**Status:** Accepted (2026-09-14)

## Context

Beyond objective quality (sharpness, exposure), there's real demand for subjective ranking: "best smiles," "how happy people look," "funniest photo." These aren't all the same kind of problem. Facial expression/smile intensity is a stable, well-trodden classical CV task. Humor is not — it's contextual and comparative, and testing showed that classical expression models don't reliably catch "silly" faces (tongue out, exaggerated grimace) since those aren't real emotion categories the models are trained on.

## Decision

Split by whether a trait is stable and precomputable, or genuinely contextual:

- **Precompute:** smile intensity / happiness via a cheap classical face-expression model (Expression scorer, ADR-0003's sibling in the indexing pipeline), run once per detected face, stored in `sqlite-vec`.
- **Judge live, pairwise, at query time:** traits with no stable universal score — humor, "silly," "best moment," mood. The agent narrows to a candidate set first (`semantic_search` + filters), then asks the VLM to compare candidates directly, pairwise, over just that narrowed set. Never an absolute score for these traits.

## Consequences

- Avoids trying to precompute a "funniness score" for every one of ~150K assets, which would be both expensive and conceptually unsound (humor isn't a stable per-photo property the way sharpness is).
- The live-judge path only runs over a small narrowed candidate set per query, keeping its cost bounded.
- This is the general pattern for any future "find me the best/funniest/X-est" feature request, not a one-off special case.

## Alternatives considered

- **Precompute a universal score for every subjective trait, same as quality:** rejected — humor/mood aren't stable per-photo properties, and precomputing every conceivable subjective axis for the whole library doesn't scale to trait after trait.
