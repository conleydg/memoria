# ADR-0022: Brute-force vector search first, sqlite-vec's `vec0` deferred

**Status:** Accepted (2026-09-14)

## Context

`sqlite-vec` (ADR-0004) gives fast approximate nearest-neighbor search only through its `vec0` virtual table type — a plain column of vector bytes (what the `embeddings` table currently stores) is storage, not search. Adopting `vec0` now, before any real embedding exists, before the eval harness exists, and before real query latency has ever been measured, would mean designing around an unmeasured need.

It's also not free to adopt right now. `vec0` is a loadable SQLite extension, not built-in — checked directly against this machine's Python: `sqlite3.Connection` has no `enable_load_extension` method at all, meaning this specific Python build was compiled without loadable-extension support (a common default for Python.org's official macOS installers, for security reasons - loading an extension executes native code from a file). `vec0` genuinely cannot load here as-is. (FTS5, by contrast, is compiled directly into this SQLite - confirmed via `PRAGMA compile_options` - which is why it already works with zero setup.)

At this project's actual scale (~150-300K vectors), brute-force cosine similarity - a vectorized matrix multiply of the query vector against every stored vector - is well-documented to be fast enough that approximate indexes often aren't worth their overhead below roughly 1M vectors.

## Decision

Store embeddings as packed float32 bytes (`numpy.tobytes()`) in a plain table, and compute similarity with a brute-force scan in application code at query time - no vector index, no loadable extension, no new dependency. Adopt `vec0` later only if the eval harness (ADR-0006) measures real query latency as a genuine problem, not preemptively.

Separately: when setting up the Mac Studio's environment, deliberately choose a Python build with loadable-extension support (e.g. a Homebrew-built Python), regardless of whether `vec0` ends up needed - removes this whole class of surprise for free, since there's no cost to choosing it now.

## Consequences

- Zero new dependencies for the current build; the schema and code that already exist and are tested don't need to change.
- Storing packed float32 bytes now means a future migration to `vec0` (if it ever happens) is a one-time script loading existing bytes into a virtual table, not a re-encoding of stored data.
- The migration, if it happens, is contained behind the agent's `semantic_search` tool (ADR-0005) - nothing upstream of that tool needs to know whether search is brute-force or `vec0`-backed.
- Hybrid queries (vector + metadata filter, e.g. year/kind/person) are arguably simpler under brute-force: filter with plain SQL first, then rank only the narrowed candidate set - not a `vec0`-specific join pattern to learn.
- A real ceiling exists if the library grows far beyond current scale or per-query latency budgets tighten significantly - not a concern today, worth re-measuring if either changes materially.

## Alternatives considered

- **Adopt `vec0` now:** rejected - no present workload to validate index configuration against, a genuine environment blocker on this machine's current Python, and the same "avoid premature specialized infrastructure" reasoning ADR-0011 already applied to the graph-database question.
