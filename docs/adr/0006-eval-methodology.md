# ADR-0006: Separate retrieval and ranking evals, pairwise-only for subjective judgment

**Status:** Proposed (2026-09-14) — methodology designed, not yet built; sequencing relative to the indexing pipeline was deliberately left open mid-design in favor of finishing the architecture first.

## Context

"Did search find the right photos" and "is this the *best* photo in the group" are different problems requiring different evaluation approaches, and there's no existing open-source eval framework that fits image tagging/retrieval directly (most assume text-chunk RAG). Building a large hand-labeled dataset from scratch is impractical for a solo project.

## Decision

Two separate eval tracks:

- **Retrieval eval:** seed cheap "silver" ground truth from Photos' own native search (already confirmed to work well on object/scene terms) across ~30-50 queries; hand-correct a small ~10-15 query subset into a trusted "gold" set. Track Recall@k and NDCG@10 against both.
- **Ranking eval** ("best photo in this group"): hand-pick the best photo in ~50-100 sampled groups (fast — humans compare quickly even if they search slowly). Score the quality model by agreement rate / Kendall's tau, optionally cross-checked with a local VLM as a *pairwise* judge only — never an absolute aesthetic score, since pairwise LLM judgments hold up against human judgment far better than absolute scores do.

A small custom harness will be built rather than adopting a heavy existing framework.

**Two further tracks, added after an initial gap review found the original two insufficient:**

- **Subjective-judgment eval:** the VLM pairwise judge (ADR-0007) has no measurement plan of its own. Sample a set of pairwise comparisons the judge makes (e.g. "which is funnier"), get a human pairwise call on the same pairs, and track agreement rate — the same pairwise-only discipline used everywhere else in this project, applied to evaluating the judge itself.
- **Agent-behavior eval:** distinct from whether retrieved results are good — does the agent (ADR-0005) call tools in a sensible order, avoid inventing photo IDs that don't exist, and recognize when it doesn't have enough information to answer rather than guessing? Track this via a small set of scripted multi-step queries with known-correct tool-call sequences, checked against what the agent actually calls.

Expression scorer accuracy (ADR-0007) on this specific library is treated as a spot-check against the ranking eval's hand-picked groups, not a separate track — reuse the same ~50-100 sampled groups rather than building a third dataset.

## Consequences

- Ground truth costs a few hours of manual review, not weeks of labeling.
- The "pairwise judge only, never absolute score" discipline established here is reused project-wide, including the runtime VLM pairwise judge (ADR-0007).
- The silver set (from Photos' own search) provides an ongoing regression signal even without further manual labeling; the gold set is the trusted number to actually report.

## Alternatives considered

- **Adopting an existing RAG eval framework:** rejected — the ones surveyed assume text-chunk retrieval, not images, and would need as much adaptation work as building a small purpose-built harness.
