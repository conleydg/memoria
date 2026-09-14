# ADR-0001: Standalone app with its own index, never writing back to Apple Photos

**Status:** Accepted (2026-09-14)

## Context

Apple Photos' `Photos.sqlite` has a `ZKEYWORD` table that would let a tool write custom tags directly into Photos, making them natively searchable there. Photos' own native search is already confirmed to work well for object/scene terms and compound event+year queries.

## Decision

Build a standalone application with its own database, reading `Photos.sqlite` and the underlying files read-only. Never write into `ZKEYWORD` or any other live Photos table.

## Consequences

- Full control over ranking, subjective judgment, and search UX that Photos' own search doesn't offer — the actual point of this project.
- No risk to the live Photos library from a bug in this project's code.
- Tags/rankings aren't visible inside Photos.app itself — this project owns its own dashboard/UI instead of extending Photos'.
- Matches the pattern already established by a prior project (PhotoDedupReport) against the same library: read-only access, own SQLite file, decisions surfaced for manual action rather than automated.

## Alternatives considered

- **Write tags into `ZKEYWORD`:** rejected — writes into Apple's live database carry real risk for a marginal convenience gain, and constrains ranking/UX to whatever Photos' own search supports.
