"""Exercise 4: change the weights and watch what happens.

    .venv/bin/python tools/model-lab/mlx_handson/04_change_the_weights.py
    .venv/bin/python tools/model-lab/mlx_handson/04_change_the_weights.py --noise 0.01 0.05 --ablate 0 20 35

A trained model is its numbers. Nudge them and the behaviour changes.
Starting from the bf16 model, this captions the same photos after each
edit, then restores the original weights before the next one:

1. Noise: add random noise to every language-model weight matrix, scaled
   to each matrix's own spread (0.5% ... 20% of its standard deviation).
   Small noise does nothing (models are robust), large noise produces
   nonsense. Where's the cliff?
2. Knock out one layer: zero the feed-forward (MLP) output of a single
   decoder layer, early, middle or late. The residual connection lets
   information skip over it, so one missing layer often barely matters,
   but not always: the first layers are usually load-bearing.
3. Blind the model: zero the small projector ("merger") that hands the
   vision encoder's output to the language model. The language model is
   intact but now receives no picture, so it describes... something. This
   shows the image only reaches the language model through that one
   bridge.

Each caption is scored against the unedited caption with SigLIP2 text
similarity (1.0 = same meaning).
"""

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten

from common import caption, load, sample_images, save, siglip_similarity


def snapshot(module):
    return dict(tree_flatten(module.parameters()))


def restore(module, saved):
    module.load_weights(list(saved.items()), strict=False)
    mx.eval(module.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.005, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--ablate", type=int, nargs="+", default=None, help="decoder layer indices to knock out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, processor, config, _ = load("bf16")
    n_layers = len(model.language_model.model.layers)
    ablate = args.ablate or [0, n_layers // 2, n_layers - 1]
    images = sample_images(args.images)
    caption(model, processor, config, images[0]["path"], max_tokens=5)  # warm-up
    original = snapshot(model)

    def run(label, detail):
        caps = [caption(model, processor, config, im["path"])["text"] for im in images]
        print(f"  {label:28s} done")
        return {"label": label, "detail": detail, "captions": caps}

    experiments = [run("unedited", "the original bf16 weights")]

    # 1. Noise, scaled per matrix.
    mx.random.seed(args.seed)
    lm_weights = [(k, v) for k, v in original.items()
                  if k.startswith("language_model") and k.endswith(".weight") and v.ndim == 2]
    for level in args.noise:
        # One matrix at a time, in the weights' own bf16, evaluated right
        # away: building float32 copies of all 8B weights at once needs
        # ~4x the model's memory and runs out on a 64 GB Mac.
        for k, w in lm_weights:
            std = mx.std(w.astype(mx.float32)).astype(w.dtype)
            nw = w + level * std * mx.random.normal(w.shape, dtype=w.dtype)
            mx.eval(nw)
            model.load_weights([(k, nw)], strict=False)
        experiments.append(run(f"noise {level:.1%}", f"every language-model matrix += {level:.1%} x its std x random"))
        restore(model, original)
        mx.clear_cache()

    # 2. Knock out single layers' MLP.
    for i in ablate:
        k = f"language_model.model.layers.{i}.mlp.down_proj.weight"
        model.load_weights([(k, mx.zeros_like(original[k]))], strict=False)
        experiments.append(run(f"layer {i} MLP removed", f"decoder layer {i} of {n_layers}: feed-forward output zeroed"))
        restore(model, {k: original[k]})

    # 3. Blind the model: zero the vision->language projector.
    merger = {k: v for k, v in original.items() if k.startswith("vision_tower") and "merger" in k}
    if merger:
        model.load_weights([(k, mx.zeros_like(v)) for k, v in merger.items()], strict=False)
        experiments.append(run("vision projector zeroed", f"{len(merger)} tensors of the merger set to 0: the picture never reaches the language model"))
        restore(model, merger)

    base = experiments[0]["captions"]
    for e in experiments:
        e["similarity"] = siglip_similarity(e["captions"], base)
        e["mean_similarity"] = sum(e["similarity"]) / len(e["similarity"])
        print(f"{e['label']:28s} similarity to unedited {e['mean_similarity']:.3f}")
    save("04_change", {"images": images, "n_layers": n_layers, "experiments": experiments})


if __name__ == "__main__":
    main()
