"""SigLIP2 image embeddings + zero-shot labels (ADR-0010), via
transformers on the Mac's GPU (MPS).

    python -m lab.siglip

A photo gets one 1152-d vector from model.jpg; a video gets the mean of
its 4 keyframe vectors (re-normalized). Vectors go in memoria's
`embeddings` table (packed float32, ADR-0022) and in the sqlite-vec
`siglip_vec` table.

Zero-shot: the same image vector is compared against the text vectors of
a few label prompts. SigLIP scores each label independently with a
sigmoid (so the probabilities don't have to sum to 1); a softmax over
the same logits is stored too, for picking one label.
"""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .memwatch import ProcessPeak
from .store import connect, media_dir, record_run

# "siglip2" is the production embedding (memoria's `embeddings` table);
# "siglip2-giant" is a larger comparison model (lab `alt_embeddings`).
MODELS = {
    "siglip2": ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"),
    "siglip2-giant": ("google/siglip2-giant-opt-patch16-384", "a713301b217d38485fb2204c808367d10bc3cc40"),
}
MODEL_ID, REVISION = MODELS["siglip2"]
MODEL_VERSION = f"{MODEL_ID}@{REVISION[:7]} (fp16, MPS)"


def model_version(key: str) -> str:
    mid, rev = MODELS[key]
    return f"{mid}@{rev[:7]} (fp16, MPS)"
LABELS = {
    "screenshot": "a screenshot of a phone or computer screen",
    "receipt": "a photo of a receipt",
    "document": "a photo of a document or a page of text",
    "photo": "a photograph of a real-world scene",
}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Siglip:
    def __init__(self, key: str = "siglip2"):
        t0 = time.time()
        mid, rev = MODELS[key]
        self.key, self.model_version = key, model_version(key)
        self.model = AutoModel.from_pretrained(mid, revision=rev, dtype=torch.float16).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(mid, revision=rev)
        self.load_seconds = time.time() - t0

    @staticmethod
    def _tensor(out):
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        feats = self._tensor(self.model.get_image_features(**inputs)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # SigLIP2 was trained on lowercase text padded to 64 tokens.
        inputs = self.processor(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(DEVICE)
        feats = self._tensor(self.model.get_text_features(**inputs)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def logit_params(self) -> tuple[float, float]:
        return float(self.model.logit_scale.exp().detach()), float(self.model.logit_bias.detach())


def image_vector(s: Siglip, uuid: str, kind: str) -> np.ndarray:
    d = media_dir(uuid)
    paths = sorted((d / "frames").glob("f*.jpg")) if kind == "video" else [d / "model.jpg"]
    vecs = s.embed_images([Image.open(p).convert("RGB") for p in paths])
    v = vecs.mean(axis=0)
    return (v / np.linalg.norm(v)).astype(np.float32)


def main(key: str = "siglip2"):
    """key "siglip2": production embedding + zero-shot labels.
    key "siglip2-giant": comparison embedding only, into alt_embeddings."""
    primary = key == "siglip2"
    conn = connect()
    rows = conn.execute("SELECT uuid, kind FROM assets WHERE status != 'failed' ORDER BY date_created").fetchall()
    started = time.time()
    with ProcessPeak() as peak:
        s = Siglip(key)
        version = s.model_version
        scale, bias = s.logit_params()
        label_vecs = s.embed_texts(list(LABELS.values()))
        done = failed = 0
        for r in rows:
            t0 = time.time()
            try:
                v = image_vector(s, r["uuid"], r["kind"])
                if DEVICE == "mps":
                    torch.mps.synchronize()
                secs = time.time() - t0
                now = time.time()
                if primary:
                    conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                                 (r["uuid"], v.tobytes(), version, now))
                    conn.execute("DELETE FROM siglip_vec WHERE asset_id=?", (r["uuid"],))
                    conn.execute("INSERT INTO siglip_vec (asset_id, embedding) VALUES (?,?)", (r["uuid"], v.tobytes()))
                    logits = scale * (label_vecs @ v) + bias
                    probs = 1 / (1 + np.exp(-logits))
                    soft = np.exp(logits - logits.max())
                    soft /= soft.sum()
                    conn.execute("DELETE FROM zero_shot WHERE asset_id=?", (r["uuid"],))
                    conn.executemany("INSERT INTO zero_shot VALUES (?,?,?,?,?,?,?)",
                                     [(r["uuid"], lab, p, float(lg), float(pr), float(sm), version)
                                      for (lab, p), lg, pr, sm in zip(LABELS.items(), logits, probs, soft)])
                else:
                    conn.execute("INSERT OR REPLACE INTO alt_embeddings VALUES (?,?,?,?,?)",
                                 (r["uuid"], key, v.tobytes(), version, now))
                conn.execute("INSERT OR REPLACE INTO timings VALUES (?,?,?,?)", (r["uuid"], key, secs, None))
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("siglip failed:", r["uuid"], type(e).__name__, str(e)[:200])
            conn.commit()
        mps_peak = torch.mps.driver_allocated_memory() if DEVICE == "mps" else 0
    total = time.time() - started
    record_run(conn, key, model_version=version,
               role="image embeddings + zero-shot labels" if primary else "image embeddings (comparison model)",
               runtime=f"transformers {__import__('transformers').__version__} / torch {torch.__version__} ({DEVICE})",
               license="Apache-2.0", disk_bytes=_hf_size(MODELS[key][0]), load_seconds=s.load_seconds,
               peak_memory_bytes=max(peak.peak, mps_peak),
               memory_note=f"process RSS peak {peak.peak / 1e9:.1f} GB; MPS driver allocated {mps_peak / 1e9:.1f} GB",
               assets_done=done, assets_failed=failed, total_seconds=total, started_at=started,
               finished_at=time.time(), notes=(f"labels: {LABELS}; " if primary else "") +
               f"dim={len(v)}; logit_scale={scale:.2f}, logit_bias={bias:.2f}")
    print(f"{key}: done {done}, failed {failed}, {total:.0f}s, dim {len(v)}")


def _hf_size(repo: str) -> int:
    from huggingface_hub import scan_cache_dir
    return next((r.size_on_disk for r in scan_cache_dir().repos if r.repo_id == repo), 0)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "siglip2")
