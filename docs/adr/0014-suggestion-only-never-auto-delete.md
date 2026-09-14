# ADR-0014: Every output is a suggestion — nothing here deletes or modifies the library automatically

**Status:** Accepted (2026-09-14)

## Context

This principle runs through nearly every feature: best-in-burst duplicate picks (ADR-0008), "might be boring" video flags, suggested Christmas favorites. It was established over a prior project against the same library (PhotoDedupReport), where the consistent workflow was to surface candidate lists for manual review and action, never to automate the irreversible step — even in cases where a more automated approach was initially agreed to.

## Decision

Every score, group, or ranking this system produces is a suggestion surfaced for review. The system never writes back to Apple Photos (ADR-0001) and never deletes a file. Keep/delete/act decisions stay manual, always — this is a hard architectural boundary, not a default that individual features can opt out of.

## Consequences

- No feature in this system needs a "confirm delete" flow or an undo mechanism — the system is structurally incapable of the irreversible action in the first place.
- Every ranking/grouping/flagging feature's job ends at "surface a good suggestion," which simplifies each feature's own design — it never needs to also handle the consequences of being wrong.
- The cost is convenience: even a highly confident duplicate-group suggestion still requires manual review, every time.

## Alternatives considered

- **Automate low-risk actions (e.g. auto-delete exact pixel-identical duplicates):** rejected — even "obviously safe" automated deletion was explicitly rejected in the prior project after being initially agreed to; manual action is a hard boundary here, not a risk-based judgment call made per feature.
