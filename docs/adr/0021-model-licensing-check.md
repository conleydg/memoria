# ADR-0021: Model licensing check — Q-Align is non-commercial only

**Status:** Accepted (2026-09-14)

## Context

None of the model choices in ADR-0003/0009 had their licenses actually verified against this project's use (personal, local, source published in a public GitHub repo). Checked directly:

| Model | License | Fit |
|---|---|---|
| Qwen3-VL | Apache 2.0 | Permissive, no issue. |
| SigLIP2 | Apache 2.0 (confirmed via the model's Hugging Face page) | Permissive, no issue. |
| Whisper | MIT | Permissive, no issue. |
| **Q-Align / OneAlign** | **S-Lab License 1.0** (confirmed via the repository's LICENSE file) | **Non-commercial only** — the license text requires contacting the contributors for any commercial use. |

## Decision

Proceed with Q-Align for aesthetic/quality scoring (ADR-0003), since this project is personal, non-commercial, local-only use — squarely within what the S-Lab License 1.0 permits. Record the restriction explicitly so it's never assumed away later: this specific component would need to be swapped for a permissively-licensed alternative (e.g. NIMA, already named as a fallback in ADR-0003) before this project, or any component built on it, could be used commercially.

## Consequences

- No action needed for current use.
- A real constraint if this project's direction ever changes (e.g. turning any part of it into a product or service) — Q-Align specifically, not the rest of the stack, would need to be replaced.
- Publishing this repo's source code publicly (MIT-licensed, per the repo's own LICENSE) is unaffected — the restriction is on using the Q-Align *model weights/inference*, not on this project's own original code.

## Alternatives considered

- **Swap Q-Align for NIMA now, to avoid any non-commercial dependency:** rejected for now — current use is personal and non-commercial, well within what the license permits; premature to give up Q-Align's better quality (ADR-0003) for a restriction that doesn't currently apply.
