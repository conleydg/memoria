"""Exercise 2: quantize the model yourself, and see what it costs.

    .venv/bin/python tools/model-lab/mlx_handson/02_quantize.py
    .venv/bin/python tools/model-lab/mlx_handson/02_quantize.py --bits 8 4 3 2 --group-size 64

Quantization stores each weight in fewer bits. MLX's "affine" scheme cuts
every row of a weight matrix into groups of `group_size` numbers. For each
group it stores a scale and an offset (bias), and each weight becomes a
small integer: w ~ scale * q + bias, with q in 0 .. 2^bits - 1.
So 4-bit with group size 64 costs 4 bits per weight plus 2 x 16 bits per
64 weights, about 4.5 bits per weight instead of 16.

This script makes 8/4/3/2-bit copies from the bf16 original
(mlx_vlm.convert, the same tool the "mlx-community" uploads were made
with), then loads each one and captions the same photos. It measures
size, memory and speed, and how close each caption's *meaning* stays to
the bf16 caption (SigLIP2 text similarity, 1.0 = same meaning).

Expect: 8-bit ~ indistinguishable, 4-bit very close (that's why everyone
uses it), 3-bit starting to drift, 2-bit falling apart.

Disk: the copies take ~10 / 6 / 4.5 / 3.5 GB under data/model-lab/mlx/models/.
Delete that folder when you're done.
"""

import argparse
import gc
import shutil
import time

import mlx.core as mx

from common import MODELS, bf16_path, caption, folder_bytes, load, sample_images, save, siglip_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--reconvert", action="store_true")
    args = ap.parse_args()
    from mlx_vlm import convert

    MODELS.mkdir(parents=True, exist_ok=True)
    conversions = {}
    for b in args.bits:
        out = MODELS / f"q{b}"
        if out.exists() and not args.reconvert:
            continue
        shutil.rmtree(out, ignore_errors=True)
        print(f"quantizing to {b}-bit (group size {args.group_size}) ...")
        t0 = time.time()
        convert(bf16_path(), mlx_path=str(out), quantize=True, q_bits=b, q_group_size=args.group_size)
        conversions[f"q{b}"] = time.time() - t0
        gc.collect()
        mx.clear_cache()

    images = sample_images(args.images)
    variants = ["bf16"] + [f"q{b}" for b in args.bits]
    results = {}
    for v in variants:
        print(f"== {v}")
        model, processor, config, load_s = load(v)
        active = mx.get_active_memory()
        caption(model, processor, config, images[0]["path"], max_tokens=5)  # warm-up
        runs = [caption(model, processor, config, im["path"]) for im in images]
        results[v] = {"disk_bytes": folder_bytes(bf16_path() if v == "bf16" else MODELS / v),
                      "load_seconds": load_s, "active_memory_bytes": active,
                      "peak_memory_bytes": mx.get_peak_memory(),
                      "convert_seconds": conversions.get(v),
                      "mean_gen_tps": sum(r["gen_tps"] for r in runs) / len(runs),
                      "mean_seconds": sum(r["seconds"] for r in runs) / len(runs),
                      "captions": [r["text"] for r in runs], "runs": runs}
        print(f"   {results[v]['disk_bytes'] / 1e9:.1f} GB on disk, {active / 1e9:.1f} GB in memory, "
              f"{results[v]['mean_gen_tps']:.0f} tok/s")
        del model, processor
        gc.collect()
        mx.clear_cache()

    base = results["bf16"]["captions"]
    for v in variants:
        sims = siglip_similarity(results[v]["captions"], base)
        results[v]["similarity_to_bf16"] = sims
        results[v]["mean_similarity"] = sum(sims) / len(sims)
        print(f"{v}: meaning similarity to bf16 = {results[v]['mean_similarity']:.3f}")
    save("02_quantize", {"group_size": args.group_size, "images": images, "variants": variants, "results": results})


if __name__ == "__main__":
    main()
