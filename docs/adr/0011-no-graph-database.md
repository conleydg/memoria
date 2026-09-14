# ADR-0011: No dedicated graph database for cross-connector relationships

**Status:** Accepted (2026-09-14)

## Context

Once other connectors exist (ADR-0012), queries like "photos of people mentioned in this email thread" become relationship-traversal problems, which graph databases (Neo4j, etc.) are built for. But graph databases earn their complexity at billions of edges with heavy multi-hop production traversal — a personal archive of people/places/events across a few connectors is thousands of entities, not billions.

## Decision

Model relationships as a plain edges table (`entity_a, relation, entity_b`) inside the same SQLite store, queried with joins or recursive CTEs for multi-hop lookups, rather than introducing a separate graph database.

## Consequences

- Preserves the "one file, no second storage system" principle (ADR-0004) even as connectors are added later.
- No new infrastructure to operate for a relationship-modeling need that doesn't exist yet — this project has no other connectors built (ADR-0012).
- If the relational approach genuinely proves insufficient later, the natural escape hatch is an *embedded* graph engine (e.g. Kuzu — file-based, no server), not a server-based graph database, to stay consistent with this project's local-first, single-process operating philosophy.

## Alternatives considered

- **Adopting a graph database now, in anticipation of future connectors:** rejected — premature; no second connector exists yet to generate a real relationship-modeling need, and the eventual need (if it materializes) is well within what a SQLite edges table handles at this scale.
