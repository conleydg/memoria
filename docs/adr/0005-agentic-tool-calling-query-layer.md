# ADR-0005: A tool-calling agent loop, not a fixed query language

**Status:** Accepted (2026-09-14)

## Context

The motivating query — "show me the best Christmas pictures from every year" — isn't answerable by a single search call. It requires searching per year, ranking each year's results by quality, and composing a final answer. A fixed query language (or a single embedding lookup) can't express that kind of multi-step, conditional plan.

## Decision

Build a local tool-calling model (Qwen3 family — see ADR-0003) that plans multi-step queries against a small set of tools: `semantic_search(query, filters)`, `get_photo_metadata(id)`, `rank_by_quality(ids)`, with a further fallback to a live VLM pairwise judge (ADR-0007) for traits with no precomputed score. The agent decomposes a query, calls tools, reads results, and can retry a bad step before composing a final answer.

## Consequences

- Handles genuinely multi-step queries (per-year search-then-rank) that a single search call cannot.
- Also a deliberate scope choice for the project's learning/portfolio goals — a real agentic system, not just a search box, was an explicit goal alongside the core search feature.
- No heavyweight orchestration framework (e.g. LangGraph) is required at this scope — a small set of tools plus a thin planning loop is sufficient, keeping the system's moving parts easy to reason about.

## Alternatives considered

- **A single embedding search with no agent layer:** rejected — cannot express "one best result per year," which is the actual motivating use case, only "all results matching X."
