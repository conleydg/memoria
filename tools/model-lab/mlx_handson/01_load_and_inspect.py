"""Exercise 1: load a model yourself and look inside.

    .venv/bin/python tools/model-lab/mlx_handson/01_load_and_inspect.py

What Ollama hides: a "model" is just a folder of files. config.json
describes the architecture (how many layers, how wide), and
*.safetensors files hold the learned numbers (the parameters). Loading
means reading those numbers into (unified) memory. This script loads
Qwen3-VL 8B at full bf16 precision and reports:

- how the parameters split between the vision encoder (turns pixels into
  "image tokens"), the language model (reads and writes text), and the
  small projector that connects them
- data type and memory: bf16 = 2 bytes per parameter
- the module tree (the actual layers)
- speed on a few of your photos, next to the same model run through
  Ollama at 4-bit

Try: change PROMPT in common.py, or pass --images 3 for a quicker run.
"""

import argparse
import json
import time

import mlx.core as mx

from common import bf16_path, caption, folder_bytes, load, param_summary, sample_images, save


def module_tree(module, depth=0, max_depth=3, name="model"):
    lines = []
    children = module.children()
    kids = {k: v for k, v in children.items() if hasattr(v, "children") or isinstance(v, list)}
    label = type(module).__name__
    lines.append(f"{'  ' * depth}{name}: {label}")
    if depth >= max_depth:
        return lines
    for k, v in kids.items():
        if isinstance(v, list):
            if v and hasattr(v[0], "children"):
                lines.append(f"{'  ' * (depth + 1)}{k}: {len(v)} x {type(v[0]).__name__}")
                lines += module_tree(v[0], depth + 2, max_depth, f"{k}[0]")[1:] if depth + 2 <= max_depth else []
        else:
            lines += module_tree(v, depth + 1, max_depth, k)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=6)
    args = ap.parse_args()

    path = bf16_path()
    cfg = json.load(open(f"{path}/config.json"))
    print("loading bf16 weights ...")
    model, processor, config, load_s = load("bf16")
    active = mx.get_active_memory()
    summary = param_summary(model)
    text_cfg = cfg.get("text_config", cfg)
    vis_cfg = cfg.get("vision_config", {})
    arch = {
        "model_type": cfg.get("model_type"),
        "text": {k: text_cfg.get(k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                             "num_attention_heads", "num_key_value_heads", "vocab_size",
                                             "max_position_embeddings")},
        "vision": {k: vis_cfg.get(k) for k in ("depth", "hidden_size", "num_heads", "patch_size",
                                               "spatial_merge_size", "out_hidden_size")},
        "torch_dtype": cfg.get("torch_dtype"),
    }
    print(f"loaded in {load_s:.1f}s, {active / 1e9:.1f} GB active memory")
    for k, g in summary["groups"].items():
        print(f"  {k:16s} {g['stored_values'] / 1e9:6.2f} B values  {g['bytes'] / 1e9:5.1f} GB  {g['dtypes']}")

    images = sample_images(args.images)
    caption(model, processor, config, images[0]["path"], max_tokens=5)  # warm-up (compiles GPU kernels)
    runs = []
    for im in images:
        r = caption(model, processor, config, im["path"])
        runs.append({"uuid": im["uuid"], "stratum": im["stratum"], **r})
        print(f"  {im['stratum']:14s} {r['seconds']:.1f}s  {r['gen_tps']:.0f} tok/s  ({len(r['text'])} chars)")

    save("01_inspect", {
        "variant": "bf16", "path": path, "disk_bytes": folder_bytes(path), "load_seconds": load_s,
        "active_memory_bytes": active, "peak_memory_bytes": mx.get_peak_memory(),
        "architecture": arch, "params": summary, "tree": module_tree(model),
        "runs": runs, "mean_gen_tps": sum(r["gen_tps"] for r in runs) / len(runs),
        "mean_seconds": sum(r["seconds"] for r in runs) / len(runs), "time": time.time(),
    })


if __name__ == "__main__":
    main()
