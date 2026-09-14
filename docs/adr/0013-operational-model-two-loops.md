# ADR-0013: Two operational loops — a cheap watcher and an expensive batch job — not one continuous service

**Status:** Accepted (2026-09-14)

## Context

The system needs to stay current indefinitely as new photos and videos get added to the library — a different problem from "run the pipeline once." Keeping the indexing models (Qwen3-VL et al., ADR-0003) resident and running continuously would waste RAM and power for a tool where new content arrives sporadically, and the query side is used on-demand, not continuously.

## Decision

Two separate loops:

1. **Watcher** — a cheap, AI-free check of `Photos.sqlite` for assets newer than the last-indexed watermark, run frequently (e.g. every 15-60 minutes) via `launchd`.
2. **Batch indexing job** — triggered once enough new assets accumulate (or on a schedule, e.g. nightly): loads the indexing models, processes the batch, releases the memory. Checks "already processed" per-asset before doing real work, not just relying on the watcher's timestamp.

The **query/agent side loads its model on demand**, when a question is actually asked, rather than staying resident all day.

## Consequences

- No large model sits loaded in memory/power 24/7 for infrequent use — a personal tool used a few times a week doesn't justify that cost.
- New photos are reflected within one watcher cycle, not instantly — an acceptable latency for a personal archive tool.
- Reuses an operational pattern (cheap cron-style watcher, heavier action only on state change) already running successfully elsewhere for unrelated infrastructure, rather than inventing a new one.
- Known, unaddressed edge case: a nightly batch job could overlap with an evening query, briefly wanting two large models' worth of memory at once. Not expected to be a real problem given the memory headroom (ADR-0003), but not yet solved with an explicit lock/coordination mechanism.

## Alternatives considered

- **One continuously-running service with all models resident:** rejected — wastes memory and power for a tool with sporadic real usage, for no benefit over the two-loop design.
