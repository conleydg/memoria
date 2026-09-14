# ADR-0003: Qwen3-VL + SigLIP2, served via MLX directly

**Status:** Accepted (2026-09-14)

## Context

The target hardware is a Mac Studio with 64GB unified memory (Apple Silicon, no CUDA). Candidates researched for the vision-language (tagging/captioning) role included Qwen3-VL/Qwen3.6-VL, LLaVA-NeXT/OneVision, Florence-2, and Moondream; for embeddings, SigLIP2, CLIP/OpenCLIP, and MetaCLIP2/Meta Perception Encoder. For serving, MLX, Ollama, and llama.cpp were compared. Notably, Ollama switched its own Apple Silicon backend to MLX in ~March 2026 (v0.19) — the old "MLX vs Ollama" framing is largely obsolete, since Ollama is now a convenience wrapper around MLX rather than a competing engine.

## Decision

- **Tagging/captioning:** Qwen3-VL, 30B-A3B mixture-of-experts variant, 4-bit MLX quantization (~18GB resident). LLaVA's MLX support was only partial as of mid-2026; Florence-2 is weaker at free-text captioning.
- **Embeddings:** SigLIP2 — now generally outperforms CLIP at equivalent size for fine-grained retrieval, with multilingual/dynamic-resolution support.
- **Serving:** build directly against MLX rather than through Ollama's API — MLX shows 21-87% higher throughput than llama.cpp across model sizes in 2026 benchmarks, and building against it directly gives more control than going through Ollama's abstraction, now that the "which engine" question is moot.
- **Reuse for query-time reasoning:** the same Qwen3-VL instance also serves as the agent loop's planner and the VLM pairwise judge — a vision-capable model reasons over plain text fine, so a second dedicated text-only model isn't needed.

## Consequences

- ~18GB is the single largest memory commitment in the system; everything else (SigLIP2, Whisper, Q-Align, classical CV models) is much smaller, leaving comfortable headroom in 64GB.
- Reusing one model for three roles (tagging, agent planning, pairwise judging) simplifies the runtime memory story at the cost of that model being a bottleneck if any one role needs a different model later (e.g. a smaller/faster planner).
- Model landscape moves fast (Qwen3.6-VL, MetaCLIP2 emerged within the same research pass) — this decision should be re-validated against current options once real building starts, not assumed permanent.

## Alternatives considered

- **Ollama as the primary serving layer:** rejected as unnecessary now that its backend *is* MLX — no upside over building against MLX directly, only an extra abstraction layer.
- **CLIP over SigLIP2:** kept as a fallback if SigLIP2 tooling friction proves higher than expected in practice.
