# ADR-0012: Design for connectors, build only the photos/video connector

**Status:** Accepted (2026-09-14)

## Context

The agent/tool-calling loop and the store pattern are inherently not photo-specific — the same shape of problem applies to email or text messages. Generalizing this project into a broader local "personal AI with connectors" system is appealing: it's a stronger portfolio story, and text/email retrieval is arguably *lower* technical risk than the vision pipeline already scoped (local text RAG is more mature than local VLM tagging). But each connector is a genuinely separate subsystem (different extraction, different chunking/retrieval patterns), and going wide now risks a project that never ships — especially right after deliberately choosing to hold off building until the target hardware arrives.

There's also a privacy dimension unique to email/texts: indexing them means indexing *other people's* words (senders, conversation partners), not just the user's own memories, which is a materially different consideration than photos even though the "fully local, no cloud" requirement covers the "does it leave the machine" concern either way.

## Decision

Keep the agent and store pattern connector-agnostic in naming and shape (generic tool names like `semantic_search`, not `search_photos`; a schema that could carry a connector/source-type distinction later) — but build and ship **only** the photos/video connector for now. Email and text-message connectors are named future candidates, not current work, and would warrant their own explicit privacy conversation before being built, not just an extension of the existing no-cloud rule.

## Consequences

- No rework needed later if a second connector is eventually built — the abstraction is designed in from the start, at near-zero present cost (it's just naming discipline).
- Scope stays bounded to what's already been carefully researched and designed (this repo, as of September 2026) rather than expanding into unscoped territory.
- The "other people's data" privacy question for email/texts is explicitly deferred, not accidentally skipped — it will need its own decision before any such connector is built.

## Alternatives considered

- **Build multiple connectors from the start:** rejected — scope risk to an already-ambitious single-connector project (which already includes an eval harness, model-serving comparisons, and an agentic layer as deliberate portfolio scope).
- **Design only for photos, no forward-looking abstraction:** rejected — the cost of generic naming now is near zero, while a later rename/refactor across the agent and schema would not be.
