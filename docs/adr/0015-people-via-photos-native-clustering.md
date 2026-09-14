# ADR-0015: Read Apple's own face clustering for people, don't rebuild it

**Status:** Accepted (2026-09-14)

## Context

Apple Photos already runs on-device face detection *and* clustering for every asset in the library — confirmed directly against `Photos.sqlite`: the `ZPERSON`/`ZDETECTEDFACE` tables are heavily populated, with detection/clustering already covering nearly the whole library even though only a handful of clusters typically get named by the user. `ZPERSON.ZFULLNAME` holds the name once a cluster is named (empty string if unnamed). This is the same category of "already solved, just read it" opportunity as Photos' own object/scene search (ADR-0006's silver eval set) and its duplicate detection (ADR-0008).

## Decision

Read `ZPERSON`/`ZDETECTEDFACE` directly, read-only, alongside the rest of the `Photos Library` metadata (ADR-0001). Where a cluster is already named, that name flows straight into the index — "photos of Sarah" works immediately. Where a cluster is unnamed, the recommended path is to name it in Photos.app itself, using Apple's native People UI — since this project never writes back to Photos (ADR-0001) but reading is exactly what it already does everywhere else, and the indexer picks up the name on its next refresh with zero new UI. A lightweight in-app "who is this" naming queue is a fallback only if, in practice, too many unnamed-but-relevant clusters accumulate that the user doesn't want to name inside Photos.

## Consequences

- No face detection or clustering model needs to be built, trained, or run — a real scope reduction on an otherwise ambitious project.
- Person-based search quality is bounded by how much naming has already happened (or will happen) in Photos.app itself, not by anything this project controls.
- Keeps person-handling consistent with the project's read-only relationship to Photos (ADR-0001) and its general "reuse what Apple/prior work already solved well" posture (ADR-0008).

## Alternatives considered

- **Build independent face detection/clustering (e.g. via the VLM or a dedicated model):** rejected — Apple's own on-device pipeline already does this well for the whole library; re-solving it would be pure duplicated effort for no quality gain.
- **Build an in-app naming UI as the primary path:** rejected as the default — Photos.app's own People UI already exists and requires no new development; only worth building if the read-only-from-Photos path proves insufficient in practice.
