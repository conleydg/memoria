# memoria

Local, private, AI-powered semantic search over your own photo and video library. No cloud AI services, no account, nothing leaves the machine.

## Why

Apple Photos' native on-device search is already good — object/scene terms and compound queries like "christmas tree 2020" both work well, and it's fully private. But it has a real gap: results come back as an unordered pile of matches, with no sense of *which one is best*, and no way to reason about subjective qualities — the best smiles, the funniest shot, whether anything actually happens in a given video.

The motivating query for this project is simple to say and genuinely hard to answer well:

> Show me the best Christmas pictures from every year.

That's not a search problem alone — it's search *plus* ranking *plus*, sometimes, a judgment call a keyword index can't make. This project is also an explicit learning vehicle: a way to build real depth in local AI engineering (model serving, evaluation, agentic systems), not just call a hosted API and call it done.

## Constraints, non-negotiable

- **Fully local.** No photo or video data is ever sent to a third-party AI service, under any privacy terms. No dependency on internet connectivity to use your own library.
- **Suggestions, not actions.** Every score, group, or ranking this system produces is a suggestion surfaced for review. Nothing here writes back to Apple Photos or deletes a file — that stays manual, always.
- **Originals must be fully local.** "Optimize Mac Storage" must be off for the library being indexed — see [ADR-0017](docs/adr/0017-require-full-local-originals.md).

See [docs/adr/](docs/adr/) for the reasoning behind every non-obvious decision below, and [docs/how-ai-was-used.md](docs/how-ai-was-used.md) for an honest account of the AI-assisted process that produced this repo — including where it hasn't been tested yet.

## How it works

Full diagram and component breakdown: [docs/architecture.md](docs/architecture.md).

Short version: an offline indexing pipeline runs several small, specialized local models over each photo and video — tagging, embedding, facial expression scoring, quality scoring, duplicate detection, audio transcription — and writes everything into one SQLite file (`sqlite-vec`). A local tool-calling agent answers natural-language queries at runtime by querying that same store, with a fallback to live pairwise AI judgment for traits too subjective or contextual to precompute (humor, "best moment").

## A connector, not the whole system

This is designed to be the first of what's meant to become several local **connectors** into a broader personal-AI system — email and text messages are named future candidates. The agent and store are kept generically shaped on purpose. **Only the photos/video connector is being built right now** — see [ADR-0012](docs/adr/0012-connector-architecture-photos-first.md).

## Development

The schema and asset-lifecycle logic (`src/memoria/`) don't depend on the target hardware or any AI model, so they're built and tested now, ahead of the rest — see [`docs/adr/0017`](docs/adr/0017-require-full-local-originals.md) through [`0020`](docs/adr/0020-index-backup.md) for why these were the first pieces worth writing.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## Status

Design phase as of September 2026. The target hardware (a Mac Studio, 64GB unified memory) hasn't arrived yet, so this repo currently holds research and architecture decisions, not code — deliberately: the design process is part of what this project is meant to demonstrate.

- [x] Research: local model landscape, eval methodology, open-source prior art
- [x] Architecture design
- [ ] Eval harness
- [ ] Indexing pipeline
- [ ] Agent / query layer
- [ ] UI

## License

MIT — see [LICENSE](LICENSE).
