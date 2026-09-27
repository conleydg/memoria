"""Q-Align (OneAlign) quality + aesthetic scores, 1-5 scale.

OneAlign ships as mPLUG-Owl2 remote code that only works with an older
transformers (4.36.x), so this runs in its own venv
(tools/model-lab/.venv-qalign, see README) and touches the lab store
with plain sqlite3 only - no imports from the rest of the lab package.

    tools/model-lab/.venv-qalign/bin/python tools/model-lab/lab/qalign.py

License: the HF card says MIT, but the Q-Align repository's LICENSE is
S-Lab License 1.0 (non-commercial), which is what ADR-0021 records. We
follow the stricter reading: personal, non-commercial use only.
"""

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("MODEL_LAB_DATA", ROOT / "data" / "model-lab"))
MODEL_ID = "q-future/one-align"
REVISION = "dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_VERSION = f"{MODEL_ID}@{REVISION[:7]} (fp16, {DEVICE})"


def main():
    conn = sqlite3.connect(DATA / "lab.sqlite", timeout=30)
    rows = conn.execute("SELECT uuid, kind FROM assets WHERE status != 'failed' ORDER BY date_created").fetchall()
    if len(sys.argv) > 1:
        rows = rows[: int(sys.argv[1])]
    proc = psutil.Process()
    peak = [0]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            stop.wait(0.25)

    threading.Thread(target=watch, daemon=True).start()
    started = time.time()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True,
                                                 attn_implementation="eager", torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    load_seconds = time.time() - t0
    done = failed = 0
    for uuid, kind in rows:
        d = DATA / "media" / uuid
        t0 = time.time()
        try:
            with torch.no_grad():
                if kind == "video":
                    frames = [[Image.open(p).convert("RGB") for p in sorted((d / "frames").glob("f*.jpg"))]]
                    q = model.score(frames, task_="quality", input_="video")
                    a = model.score(frames, task_="aesthetics", input_="video")
                else:
                    img = [Image.open(d / "model.jpg").convert("RGB")]
                    q = model.score(img, task_="quality", input_="image")
                    a = model.score(img, task_="aesthetics", input_="image")
            if DEVICE == "mps":
                torch.mps.synchronize()
            secs = time.time() - t0
            q, a = float(q.flatten()[0]), float(a.flatten()[0])
            # quality_scores.sharpness/exposure are for classical CV
            # signals (not computed here); Q-Align's technical-quality score
            # goes in the lab's timings detail and `exposure` stays NULL.
            conn.execute(
                "INSERT OR REPLACE INTO quality_scores (asset_id, sharpness, exposure, aesthetic, activity, "
                "model_version, computed_at) VALUES (?,?,?,?,?,?,?)",
                (uuid, None, None, a, None, MODEL_VERSION, time.time()))
            conn.execute("INSERT OR REPLACE INTO timings VALUES (?,?,?,?)",
                         (uuid, "q-align", secs, json.dumps({"quality": q, "aesthetic": a})))
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("q-align failed:", uuid, type(e).__name__, str(e)[:200], flush=True)
        conn.commit()
        if (done + failed) % 20 == 0:
            print(f"q-align {done + failed}/{len(rows)}, {time.time() - started:.0f}s", flush=True)
    stop.set()
    mps = torch.mps.driver_allocated_memory() if DEVICE == "mps" else 0
    total = time.time() - started
    import transformers
    fields = dict(model_version=MODEL_VERSION, role="quality + aesthetic score (1-5)",
                  runtime=f"transformers {transformers.__version__} / torch {torch.__version__} ({DEVICE})",
                  license="S-Lab 1.0 per repo (HF card: MIT) - non-commercial", disk_bytes=_hf_size(),
                  load_seconds=load_seconds, peak_memory_bytes=max(peak[0], mps),
                  memory_note=f"process RSS peak {peak[0] / 1e9:.1f} GB; MPS driver allocated {mps / 1e9:.1f} GB",
                  assets_done=done, assets_failed=failed, total_seconds=total, started_at=started,
                  finished_at=time.time(), notes="two passes per asset (quality, aesthetics)")
    cols = ["model", *fields]
    conn.execute(f"INSERT OR REPLACE INTO model_runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                 ["q-align", *fields.values()])
    conn.commit()
    print(f"q-align: done {done}, failed {failed}, {total:.0f}s")


def _hf_size():
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    blobs = base / f"models--{MODEL_ID.replace('/', '--')}" / "blobs"
    return sum(p.stat().st_size for p in blobs.glob("*")) if blobs.exists() else 0


if __name__ == "__main__":
    main()
