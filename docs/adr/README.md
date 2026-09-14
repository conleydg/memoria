# Architecture Decision Records

Each file records one decision: the context that motivated it, what was decided, the consequences (including the honest costs), and the alternatives considered.

| # | Title | Status |
|---|---|---|
| [0001](0001-standalone-app-no-writeback.md) | Standalone app with its own index, never writing back to Apple Photos | Accepted |
| [0002](0002-images-and-video-v1-scope.md) | Images and video both in scope from v1 | Accepted |
| [0003](0003-local-model-stack.md) | Qwen3-VL + SigLIP2, served via MLX directly | Accepted |
| [0004](0004-sqlite-vec-single-store.md) | sqlite-vec as the single metadata + vector store | Accepted |
| [0005](0005-agentic-tool-calling-query-layer.md) | A tool-calling agent loop, not a fixed query language | Accepted |
| [0006](0006-eval-methodology.md) | Separate retrieval and ranking evals, pairwise-only for subjective judgment | Proposed |
| [0007](0007-subjective-traits-precompute-vs-live-judge.md) | Precompute stable traits, judge subjective ones live and pairwise | Accepted |
| [0008](0008-duplicate-grouping-reuse.md) | Reuse perceptual-hash duplicate grouping from the prior dedup project | Accepted |
| [0009](0009-video-transcription.md) | Local transcription (Whisper) as a per-video indexing step | Accepted |
| [0010](0010-hybrid-search-and-zero-shot-classification.md) | Hybrid keyword+vector search; zero-shot screenshot/document classification | Accepted |
| [0011](0011-no-graph-database.md) | No dedicated graph database for cross-connector relationships | Accepted |
| [0012](0012-connector-architecture-photos-first.md) | Design for connectors, build only the photos/video connector | Accepted |
| [0013](0013-operational-model-two-loops.md) | Two operational loops — a cheap watcher and an expensive batch job | Accepted |
| [0014](0014-suggestion-only-never-auto-delete.md) | Every output is a suggestion — nothing deletes or modifies the library | Accepted |
| [0015](0015-people-via-photos-native-clustering.md) | Read Apple's own face clustering for people, don't rebuild it | Accepted |
| [0016](0016-pet-identification-few-shot-siglip2.md) | Pet identification via few-shot SigLIP2 matching | Accepted |
| [0017](0017-require-full-local-originals.md) | Require Photos originals fully downloaded locally | Accepted |
| [0018](0018-asset-lifecycle-edits-and-deletions.md) | Detect edited and deleted/trashed assets, not just new ones | Accepted |
| [0019](0019-model-upgrade-policy.md) | Model upgrades trigger a deliberate full re-index | Accepted |
| [0020](0020-index-backup.md) | Back up the index file via the existing external-drive process | Accepted |
| [0021](0021-model-licensing-check.md) | Model licensing check — Q-Align is non-commercial only | Accepted |
| [0022](0022-brute-force-vector-search-first.md) | Brute-force vector search first, sqlite-vec's `vec0` deferred | Accepted |
