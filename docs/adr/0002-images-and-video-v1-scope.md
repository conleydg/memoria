# ADR-0002: Images and video both in scope from v1

**Status:** Accepted (2026-09-14)

## Context

The target library is ~150K assets / ~750GB, of which video is 70.9% (534.5GB) and images 27.8% (209.8GB) by size. Treating video as a "later" add-on would mean the majority of the library's storage — and a lot of its emotionally significant content — is out of scope for the first real version.

## Decision

Build for images and video together from the start, including video-specific pipeline steps (keyframe extraction, transcription) rather than shipping an images-only v1 and bolting video on afterward.

## Consequences

- The indexing pipeline needs a video-specific branch (keyframe extraction, audio transcription) from day one — more upfront design work than an images-only pipeline.
- Avoids a retrofit later that would otherwise touch most of the architecture (the store schema, the agent's tools, the eval harness) a second time.
- The bulk of the library (by size) is actually searchable/useful from the first working version.

## Alternatives considered

- **Images-only v1, video later:** rejected — video is the majority of the library by size, and deferring it risks a costly retrofit rather than a clean incremental extension.
