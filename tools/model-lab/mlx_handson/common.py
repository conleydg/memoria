"""Shared helpers for the hands-on MLX exercises.

These scripts load Qwen3-VL 8B *directly* with Apple's MLX (via mlx-vlm),
without Ollama in between, so you can see and change what's inside:
parameters, data types, quantization, individual weight matrices.

Everything runs locally. Results are written to data/model-lab/mlx/ and
shown on the dashboard's "MLX hands-on" page.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[3]  # ~/memoria
sys.path.insert(0, str(ROOT / "tools" / "model-lab"))
DATA = ROOT / "data" / "model-lab"
OUT = DATA / "mlx"
MODELS = OUT / "models"
RESULTS = OUT / "results"
BF16_REPO = "mlx-community/Qwen3-VL-8B-Instruct-bf16"
BF16_REVISION = "76cfb0c70d92"
PROMPT = "Describe this photo in one or two plain sentences."


def bf16_path() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(BF16_REPO, revision=BF16_REVISION, local_files_only=True)


def model_path(variant: str) -> str:
    """'bf16' -> the downloaded original; 'q8', 'q4', ... -> your own
    quantized copies made by 02_quantize.py."""
    return bf16_path() if variant == "bf16" else str(MODELS / variant)


def load(variant: str = "bf16"):
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.time()
    path = model_path(variant)
    model, processor = vlm_load(path)
    mx.eval(model.parameters())  # force weights into memory now, so load time is honest
    return model, processor, load_config(path), time.time() - t0


def sample_images(n: int = 6) -> list[dict]:
    """A fixed, varied handful of test photos (one per sample stratum)."""
    conn = sqlite3.connect(DATA / "lab.sqlite")
    out, seen = [], set()
    for stratum in ("people", "pets", "screenshot", "low-light", "event", "year-spread", "near-duplicate"):
        row = conn.execute(
            "SELECT a.uuid FROM assets a JOIN lab_assets l ON l.uuid=a.uuid WHERE a.kind='image' "
            "AND a.status='indexed' AND ','||l.strata||',' LIKE ? ORDER BY a.uuid LIMIT 1", (f"%,{stratum},%",)).fetchone()
        if row and row[0] not in seen:
            seen.add(row[0])
            out.append({"uuid": row[0], "stratum": stratum, "path": str(DATA / "media" / row[0] / "model.jpg")})
        if len(out) >= n:
            break
    return out


def caption(model, processor, config, image_path: str, max_tokens: int = 100) -> dict:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
    t0 = time.time()
    r = generate(model, processor, prompt, image=[image_path], max_tokens=max_tokens, temperature=0.0, verbose=False)
    return {"text": r.text.strip(), "seconds": time.time() - t0, "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.generation_tokens, "prompt_tps": r.prompt_tps, "gen_tps": r.generation_tps,
            "peak_memory_gb": r.peak_memory}


def param_summary(model) -> dict:
    """Parameter counts and bytes, grouped by top-level component. For
    quantized layers MLX stores packed integers plus per-group scales and
    biases, so 'stored bytes' is what actually sits in memory."""
    groups: dict[str, dict] = {}
    for name, arr in tree_flatten(model.parameters()):
        top = name.split(".")[0]
        g = groups.setdefault(top, {"tensors": 0, "stored_values": 0, "bytes": 0, "dtypes": {}})
        g["tensors"] += 1
        g["stored_values"] += arr.size
        g["bytes"] += arr.nbytes
        g["dtypes"][str(arr.dtype)] = g["dtypes"].get(str(arr.dtype), 0) + arr.size
    total = {"bytes": sum(g["bytes"] for g in groups.values()),
             "stored_values": sum(g["stored_values"] for g in groups.values())}
    return {"groups": groups, "total": total}


def siglip_similarity(texts_a: list[str], texts_b: list[str]) -> list[float]:
    """How similar two captions *mean*, scored by SigLIP2's text encoder
    (cosine similarity of the two text vectors). 1.0 = same meaning."""
    from lab.siglip import Siglip
    s = siglip_similarity.model = getattr(siglip_similarity, "model", None) or Siglip()
    a, b = s.embed_texts(texts_a), s.embed_texts(texts_b)
    return [float(x) for x in (a * b).sum(axis=1)]


def save(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(data | {"saved_at": time.time()}, indent=1, default=str))
    print(f"results written to {p.relative_to(ROOT)}")
    return p


def folder_bytes(path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
