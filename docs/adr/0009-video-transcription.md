# ADR-0009: Local transcription (Whisper) as a per-video, not per-frame, indexing step

**Status:** Accepted (2026-09-14)

## Context

Video is 70.9% of the library by size (ADR-0002), and searching only what's *visible* misses a real capability — finding a video by what was *said* in it (a story, a toast, a name mentioned) — which Photos' native search has no equivalent for. Transcription operates on the whole audio track, not per-frame, so it doesn't fit the same processing shape as the other four indexing models (which each run per image/keyframe).

## Decision

Run Whisper (MLX Whisper or whisper.cpp, both fully local) once per video's audio track, writing a timestamped transcript directly into `sqlite-vec`, bypassing the per-frame processor bus entirely.

## Consequences

- Enables a genuinely new search capability ("find the video where dad tells the story about...") rather than a nicer version of something Photos already does.
- Feeds a second use: transcript presence/density becomes an input to the Quality scorer's video "activity" signal (ADR-0007-adjacent — distinguishing a quiet, meaningful moment from an actually static/boring clip).
- Adds a second model family (separate from the Qwen3/SigLIP2 lineage) to the serving story, though a small one (~1.5GB) relative to Qwen3-VL.

## Alternatives considered

- **Skipping transcription for v1:** rejected — video is the majority of the library and the "search what was said" capability was judged a genuine differentiator worth the added pipeline branch.
