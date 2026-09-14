# ADR-0016: Pet identification via few-shot SigLIP2 matching, not a dedicated model

**Status:** Accepted (2026-09-14)

## Context

Unlike people (ADR-0015), Apple's Photos has no equivalent native pet-identification feature, and no mature, production-grade open-source pet-identification model exists — this space is still research-grade (e.g. DogFaceNet, the PetFace dataset), with nothing packaged and reliable enough to adopt directly.

## Decision

Reuse the SigLIP2 embeddings already computed for search (ADR-0003, ADR-0004). Identify a pet via few-shot nearest-neighbor matching: the user labels a handful of clear reference photos per pet (e.g. 3-5 photos of "Rex"), and other photos whose embedding lands close to those references are suggested as a match — the same underlying mechanism as semantic search, applied to a small labeled reference set instead of a text query, and no new model to train or serve.

## Consequences

- No new model, no training pipeline — the marginal cost of this feature is genuinely small given SigLIP2 is already in the stack.
- Reliability will be visibly lower than person recognition: SigLIP2 wasn't trained to distinguish individual animals of the same breed, so it should work well for visually distinct pets (different species, very different coloring) and struggle more on, say, two similar-looking dogs of the same breed.
- Because of that reliability gap, pet-match suggestions get the same treatment as every other uncertain output in this system (ADR-0014) — a suggestion to confirm, never presented as a confident identification.
- Requires a small new user-facing step (labeling reference photos) that no other feature in this project currently needs — the only place the user actively trains something, rather than the system inferring everything passively.

## Alternatives considered

- **Wait for a mature open-source pet-ID model:** rejected as the default path — nothing production-ready exists yet, and few-shot matching against an embedding already being computed is available now at near-zero marginal cost.
- **Train a dedicated pet classifier:** rejected — requires a labeled dataset and training pipeline disproportionate to the value of this one feature, for a personal-scale library where a handful of reference photos per pet is far cheaper.
