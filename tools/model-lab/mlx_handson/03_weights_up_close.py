"""Exercise 3: look at the actual numbers.

    .venv/bin/python tools/model-lab/mlx_handson/03_weights_up_close.py
    .venv/bin/python tools/model-lab/mlx_handson/03_weights_up_close.py --layer 30 --matrix self_attn.q_proj

"Parameters" sound abstract; they're just big grids of numbers. This
script picks one weight matrix inside the language model (by default the
MLP "down projection" of decoder layer 18 of 36) and shows:

- its shape, and the distribution of its values (a histogram): almost
  all near zero, with a few large outliers. Those outliers are exactly
  what makes low-bit quantization hard.
- one group of 64 consecutive weights, before and after 4-bit
  quantization, with the scale and offset MLX stores for that group
- the rounding error at 8, 6, 4, 3 and 2 bits for the whole matrix
- where the model's 8 billion parameters actually live (attention vs MLP
  vs embeddings vs vision)
"""

import argparse
from collections import defaultdict

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from common import load, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--matrix", default="mlp.down_proj")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    model, _, cfg, _ = load("bf16")
    params = dict(tree_flatten(model.parameters()))
    key = next(k for k in params if f"layers.{args.layer}.{args.matrix}.weight" in k and k.startswith("language_model"))
    w = params[key]
    wf = np.array(w.astype(mx.float32))
    print(key, w.shape, w.dtype)

    hist, edges = np.histogram(wf, bins=80, range=(float(np.percentile(wf, 0.05)), float(np.percentile(wf, 99.95))))
    std = float(wf.std())
    stats = {"shape": list(w.shape), "dtype": str(w.dtype), "values": int(wf.size), "mean": float(wf.mean()),
             "std": std, "min": float(wf.min()), "max": float(wf.max()),
             "share_beyond_4std": float((np.abs(wf) > 4 * std).mean()),
             "largest_abs": float(np.abs(wf).max()), "largest_over_std": float(np.abs(wf).max() / std)}

    # One group of `group_size` weights, quantized to 4 bits and back.
    row = int(np.argmax(np.abs(wf).max(axis=1)))  # the row containing the biggest outlier: the interesting case
    col = int(np.argmax(np.abs(wf[row]))) // args.group_size * args.group_size
    wq, scales, biases = mx.quantize(w, group_size=args.group_size, bits=4)
    deq = np.array(mx.dequantize(wq, scales, biases, group_size=args.group_size, bits=4).astype(mx.float32))
    g = col // args.group_size
    group = {"row": row, "col": col, "original": wf[row, col:col + args.group_size].tolist(),
             "reconstructed": deq[row, col:col + args.group_size].tolist(),
             "scale": float(np.array(scales.astype(mx.float32))[row, g]),
             "bias": float(np.array(biases.astype(mx.float32))[row, g]),
             "levels": 16}

    errors = []
    for bits in (8, 6, 4, 3, 2):
        q, s, b = mx.quantize(w, group_size=args.group_size, bits=bits)
        d = np.array(mx.dequantize(q, s, b, group_size=args.group_size, bits=bits).astype(mx.float32))
        err = d - wf
        errors.append({"bits": bits, "bits_per_weight": bits + 32 / args.group_size,
                       "mean_abs_error": float(np.abs(err).mean()),
                       "relative_error": float(np.linalg.norm(err) / np.linalg.norm(wf)),
                       "max_abs_error": float(np.abs(err).max())})
        print(f"{bits}-bit: relative error {errors[-1]['relative_error']:.3%}")

    # Where do the parameters live?
    buckets = defaultdict(int)
    for name, arr in params.items():
        if name.startswith("vision_tower"):
            b = "vision encoder"
        elif "embed_tokens" in name or "lm_head" in name:
            b = "token embeddings / output head"
        elif ".mlp." in name:
            b = "language MLP (feed-forward)"
        elif "self_attn" in name:
            b = "language attention"
        else:
            b = "norms, projector and other"
        buckets[b] += arr.size
    biggest = sorted(((n, list(a.shape), a.size) for n, a in params.items()), key=lambda x: -x[2])[:10]
    save("03_weights", {"key": key, "stats": stats, "hist": {"counts": hist.tolist(), "edges": edges.tolist()},
                        "group": group, "errors": errors, "group_size": args.group_size,
                        "where": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
                        "total_params": int(sum(a.size for a in params.values())),
                        "biggest": [{"name": n, "shape": s, "size": z} for n, s, z in biggest]})


if __name__ == "__main__":
    main()
