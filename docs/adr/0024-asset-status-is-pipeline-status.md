# ADR-0024: `assets.status` is pipeline status, not a review decision

**Status:** Accepted (2026-09-14)

## Context

`status`/`note` was carried over from PhotoDedupReport's proven pattern without a clear meaning at the whole-asset level. In `dedup.sqlite`, `status` on a file means a human review decision (keep / delete-candidate). Nothing at the `assets` level in memoria maps to a human decision like that - `duplicate_group_members.suggested_keep` already covers the one place a real keep/delete-style suggestion exists. Left as originally written, `assets.status` was a copied pattern with no actual meaning.

Separately, an earlier gap review flagged that the batch indexing job (ADR-0013) has no defined error/retry handling for a single asset failing mid-batch (a corrupt file, an unsupported format, a model crash on one weird image).

## Decision

`assets.status` means **pipeline processing status**: `pending` (not yet indexed) → `indexing` (batch job currently working on it) → `indexed` (done) → `failed` (the pipeline hit an error on this asset). `note` holds the failure reason when `status = 'failed'`. This gives the previously-unaddressed per-asset failure case a real place to land, rather than leaving a batch job with no way to record "this one asset didn't work" short of crashing the whole run.

## Consequences

- Directly closes part of the "no failure handling" gap named earlier - a failed asset is now a queryable, visible state (`SELECT * FROM assets WHERE status = 'failed'`), not a silent gap or a crashed batch job.
- The batch job's retry/skip policy for `failed` assets isn't decided here - just that there's now a place to record the outcome.
- No longer conflated with a human review decision - `duplicate_group_members.suggested_keep` remains the one place a keep/delete-style suggestion actually lives.

## Alternatives considered

- **Drop `status`/`note` from `assets` entirely:** rejected - a real gap (no per-asset failure tracking) exists and this column pair is a natural place to close it, rather than removing it and reopening that gap.
