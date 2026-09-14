# ADR-0018: Detect edited and deleted/trashed assets, not just new ones

**Status:** Accepted (2026-09-14)

## Context

The operational model (ADR-0013) describes a watcher that finds assets newer than the last-indexed watermark — but says nothing about assets that already exist in the index and then change: a photo gets edited (crop, filter, a burst's featured pick changes) or gets trashed/deleted in Photos (`ZTRASHEDSTATE`). Left unhandled, the index would silently go stale in both directions — an edited photo keeps its old tags/embedding forever, a deleted one stays searchable and rankable indefinitely.

## Decision

Extend the watcher (ADR-0013) beyond "new since watermark" to also check, on the same cadence:

- **Edits:** compare each asset's modification signal in `Photos.sqlite` against what was indexed last time; a changed asset gets queued for re-indexing through the full pipeline, same as a new asset.
- **Deletions/trash:** an asset with `ZTRASHEDSTATE = 1` (or missing from `ZASSET` entirely, once permanently purged) gets its entry removed from `sqlite-vec` — not from Photos, only from this project's own index — including cleanup of any duplicate-group membership that referenced it.

## Consequences

- The batch indexing job (ADR-0013) needs three queues, not one: new, changed, and removed — a small addition to what was already planned, not a new subsystem.
- Duplicate groups (ADR-0008) need to handle a member disappearing mid-group without breaking the "never suggest deleting every copy" safety property already proven in the prior dedup project.
- Still consistent with ADR-0001: this only ever reads Photos' state to decide what to do to *this project's own* store; Photos itself is never touched.

## Alternatives considered

- **Full re-scan of the whole library on every run instead of incremental change detection:** rejected — defeats the purpose of the incremental, skip-already-processed pipeline design (ADR-0013's reference to the Tagle pattern), and doesn't scale as the library grows.
