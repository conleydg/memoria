# How AI was used to build this

This project was designed in close collaboration with Claude (Anthropic's AI coding assistant) — visible directly in the commit history (`Co-Authored-By: Claude Sonnet 5`) rather than hidden. This document is an honest account of that process, including its real limits, not a pitch.

It's worth separating two different things "AI expertise" can mean here, because this document only covers one of them:

1. **How an AI coding assistant was used to build this project** — the process this document describes.
2. **The local-AI engineering embedded in the product itself** — model choices, evaluation methodology, serving tradeoffs. That's covered by [`architecture.md`](architecture.md) and the [ADRs](adr/README.md), not here.

## The actual pattern

Mostly conversational co-design, not "generate a project": for each significant decision, the loop was *discuss → get a recommendation with real tradeoffs → push back or confirm → capture it as an ADR before moving on*. Nothing got built until it had been argued through.

**Research was delegated, not done inline.** Two passes — the local model/serving landscape, and comparable open-source projects — ran as background agent tasks so the raw search process didn't clutter the working conversation. Each returned a synthesized, decision-useful summary (concrete options, tradeoffs, a recommendation) rather than a dump of search results, and the conversation kept moving on other things while they ran.

**Every non-obvious decision became an ADR**, not just a line in chat history. This is the part most AI-assisted projects skip — most produce code with no record of *why* a choice was made, which means the reasoning is unrecoverable the moment the chat is closed. 21 ADRs exist because the discipline was "if it's worth deciding, it's worth writing down why," applied consistently rather than after the fact.

## Verification, not blind trust

A few concrete moments where this mattered, not just a general claim:

- Connecting to GitHub over SSH triggered a host-key mismatch warning — the kind of warning that's usually benign (a key rotation) but occasionally means something real. Rather than overriding it, the offered fingerprint was checked against GitHub's officially published values before trusting it.
- Model licenses were actually checked, not assumed. This caught something real: Q-Align, one of the models this project planned to use, turned out to be non-commercial-only licensed — fine for personal use, but a genuine constraint that would have gone unnoticed without checking.
- A self-review pass, done deliberately after the initial design felt complete, surfaced real gaps: an unaddressed privacy consequence of video transcription, no plan for edited/deleted assets, no backup story for the index, missing eval coverage for two of the newer mechanisms. All five became ADRs.

## The honest limits

**Everything so far is research, writing, and judgment calls — the areas AI is currently strongest at.** No code has been written yet. That's a different and harder test: does generated code get reviewed line by line, does a hallucinated API get caught, does the eval harness (once built) actually stop a bad model change from shipping. This document can't claim success on a test that hasn't happened yet.

**The gap review was self-review.** The same session that designed the architecture also audited it for gaps — a real structural limitation, since an AI (or anyone) reviewing their own work can't see what they structurally didn't think of in the first place. It found real issues, which is worth something, but it isn't the same as an independent second opinion, human or a fresh AI session with no investment in the existing design. That's worth doing at some point, not assuming this pass covered it.

**This document was written by the same process it describes.** Flagging that plainly rather than pretending otherwise.

## What's still to be tested

The real test of "was this a good way to use AI" starts once code exists: whether the eval harness (ADR-0006) actually catches regressions, whether code gets reviewed rather than trusted because it runs, whether the agent's own tool-calling behavior gets caught when it's wrong. This document should be revisited and expanded once that phase is underway — not treated as a finished answer.
