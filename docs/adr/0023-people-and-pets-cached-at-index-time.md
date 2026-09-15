# ADR-0023: People and pets cached at index time, not joined live against Photos

**Status:** Accepted (2026-09-14) — with a known, tracked limitation

## Context

ADR-0015 (people) and ADR-0016 (pets) establish *how* those signals are produced, but not where the result lives. Two options: join live against `Photos.sqlite` at query time, or cache the relevant bit into memoria's own store during indexing.

## Decision

Cache at index time. A new `asset_people` table records which named people (from Photos' own `ZPERSON`/`ZDETECTEDFACE`) appear in each asset, keyed by Photos' own person identifier (not just the name string, so a later re-sync can reconcile a renamed person against the same identity). New `pets`/`pet_reference_photos`/`pet_matches` tables hold the few-shot pet-matching data from ADR-0016. Query time never opens `Photos.sqlite` at all — only the batch indexing job does.

This keeps the two-loop operational split (ADR-0013) clean: only indexing touches Photos, the on-demand query/agent loop is fully self-contained against memoria's one file (ADR-0004), and the eval harness can run against pure synthetic data without a real `Photos.sqlite` present.

## Known limitation (tracked, not solved here)

The asset-lifecycle watcher (ADR-0018) detects changes via each asset's own `modified_at` timestamp. **Renaming a person cluster in Photos.app is a change to `ZPERSON`, not necessarily something that bumps the modification timestamp on every photo that person appears in** — unverified either way against a real `Photos.sqlite`, and not solved here. Practical consequence: a person renamed in Photos may not get picked up by the incremental watcher. For v1, this is accepted as a known gap — person-name staleness is corrected only by an explicit full re-index, not the incremental one. Worth verifying and fixing properly if it turns out to matter in practice.

## Consequences

- A small, deliberate duplication of Photos' own data (names + associations only, not face imagery) - acceptable given the alternative breaks the one-file principle at query time.
- The rename-staleness gap above is real and explicitly not fixed by this decision - tracked here rather than silently assumed away.
- Pet reference photos and matches are now first-class, queryable alongside everything else (`pet_matches.similarity`, suggestion-only per ADR-0014), not a bolt-on.

## Alternatives considered

- **Live join against Photos.sqlite at query time:** rejected - couples the lightweight on-demand agent (ADR-0013) to a hard runtime dependency on Photos.sqlite being present and unlocked, and complicates testing against synthetic data.
