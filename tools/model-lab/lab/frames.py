"""Which frames should a vision model see of a video? Three strategies,
compared on the test videos with the production caption model (30B-A3B):

- fixed4: the default. 4 frames at 10/35/60/85% of the clip, 1024 px,
  whatever its length.
- change: frames where the *content* changes. A candidate frame every
  second is embedded with SigLIP2, then frames are picked greedily so
  each new one is as different as possible from those already chosen
  (farthest-point sampling on cosine distance), until the next-best
  frame is too similar (distance < 0.08) or 12 are chosen; at least 4.
  768 px. (Classic cut detection - ffmpeg's scene score - was tried
  first and found almost nothing: home videos are one continuous take,
  so there are no cuts to detect. Embeddings see content changes
  within a take: someone walking in, the camera turning.)
- dense:  one frame every ~5 s, clamped to 8-16 frames, at 512 px.
  The idea was that smaller frames cost fewer tokens. In practice
  Ollama's Qwen3-VL gives every image at least ~1,000 tokens (it scales
  small images up), so a 512 px frame costs about as much as a 1024 px
  one: 8-16 frames means 2-4x the prompt tokens of the default 4.
  These runs use a 32K context (the default 8K overflows past ~7 frames).

    python -m lab.frames                  # extract frames + caption with the 30B
    python -m lab.frames --caption-only   # just (re)caption items without a result

Frames go in media/<uuid>/frames_<strategy>/, their timestamps in the
video_frames table, and captions in vlm_results as "qwen3-vl-30b@<strategy>".
Strategy "scene" (cut detection) is kept as a function for reference.
"""

import base64
import json
import re
import subprocess
import time

from .store import CACHE, connect, media_dir
from .vlm import LABELS, OPTIONS, describe, model_version, unload

FFMPEG = "/opt/homebrew/bin/ffmpeg"
MODEL = "qwen3-vl:30b-a3b-instruct-q4_K_M"
BASE_LABEL = LABELS[MODEL]
FIXED = (0.10, 0.35, 0.60, 0.85)

PROMPT = """You are indexing a personal video library so it can be searched later.
These are {n} frames taken from one {dur:.0f}-second video, in order, at these times (seconds): {times}.
Describe the video as a whole.

Return JSON with:
- "caption": 1-3 plain sentences describing what happens in the video: the main subjects, what they are doing, the setting, and anything notable (season, event, weather, time of day). Do not guess names.
- "tags": 8-20 short lowercase search tags (objects, animals, activities, scene, event, style).
- "ocr_text": any clearly legible text in the frames, verbatim, or "" if there is none."""


def ensure_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS video_frames (
        asset_id TEXT NOT NULL, strategy TEXT NOT NULL, idx INTEGER NOT NULL,
        t REAL NOT NULL, px INTEGER NOT NULL,
        PRIMARY KEY (asset_id, strategy, idx))""")


def scene_times(src, threshold=0.3) -> list[float]:
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(src), "-an", "-vf",
                        f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]


def evenly(duration, n) -> list[float]:
    return [duration * (i + 0.5) / n for i in range(n)]


def pick_scene(duration, cuts, lo=4, hi=12) -> list[float]:
    # A frame just after each cut (the new shot), plus the opening shot.
    ts = sorted({0.5 if duration > 1 else 0.0, *[min(c + 0.3, duration - 0.05) for c in cuts]})
    if len(ts) > hi:  # keep an even spread of the cuts
        ts = [ts[round(i * (len(ts) - 1) / (hi - 1))] for i in range(hi)]
    if len(ts) < lo:  # static clip: top up with evenly spaced frames, keep an even spread
        pool = sorted(set(ts) | set(evenly(duration, lo)))
        ts = [pool[round(i * (len(pool) - 1) / (lo - 1))] for i in range(lo)]
    return ts


def change_times(src, duration, siglip, lo=4, hi=12, min_dist=0.08) -> list[float]:
    """Farthest-point sampling over SigLIP2 embeddings of 1 fps candidates."""
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image
    step = max(1.0, duration / 120)  # at most ~120 candidates
    times = [float(t) for t in np.arange(0, duration, step)]  # the fps filter emits frame i at ~i*step
    if len(times) <= lo:
        return evenly(duration, lo)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([FFMPEG, "-v", "error", "-i", str(src), "-vf", f"fps=1/{step},scale=384:-2",
                        "-q:v", "4", f"{tmp}/c%04d.jpg"], check=True)
        files = sorted(Path(tmp).glob("c*.jpg"))
        vecs = np.concatenate([siglip.embed_images([Image.open(p).convert("RGB") for p in files[i:i + 32]])
                               for i in range(0, len(files), 32)])
    times = times[:len(vecs)]
    chosen = [0]
    mind = 1 - vecs @ vecs[0]  # cosine distance to the chosen set
    while len(chosen) < hi:
        nxt = int(np.argmax(mind))
        if mind[nxt] < min_dist and len(chosen) >= lo:
            break
        chosen.append(nxt)
        mind = np.minimum(mind, 1 - vecs @ vecs[nxt])
    return sorted(float(times[i]) for i in chosen)


def extract(src, times, outdir, px):
    outdir.mkdir(exist_ok=True)
    for old in outdir.glob("f*.jpg"):
        old.unlink()
    for i, t in enumerate(times):
        # Seeking right at the end of a stream can land past the last
        # decodable frame; step back a little and retry.
        for back in (0, 0.5, 1.5, 3.0):
            r = subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{max(0.0, t - back):.3f}", "-i", str(src),
                                "-frames:v", "1", "-vf", f"scale='min({px},iw)':-2", "-q:v", "3",
                                str(outdir / f"f{i:02d}.jpg")], capture_output=True)
            if r.returncode == 0 and (outdir / f"f{i:02d}.jpg").exists():
                break
        else:
            raise RuntimeError(f"could not extract a frame near {t:.1f}s")


def strategies(duration, src, siglip):
    return {
        "fixed4": ([duration * p for p in FIXED], 1024),
        "change": (change_times(src, duration, siglip), 768),
        "dense": (evenly(duration, max(8, min(16, round(duration / 5)))), 512),
    }


FRAME_OPTIONS = {**OPTIONS, "num_ctx": 32768}


def main():
    import sys
    caption_only = "--caption-only" in sys.argv
    conn = connect()
    ensure_table(conn)
    videos = conn.execute("SELECT a.uuid, l.relpath, l.duration FROM assets a JOIN lab_assets l ON l.uuid=a.uuid "
                          "WHERE a.kind='video' AND a.status != 'failed' ORDER BY a.date_created").fetchall()
    # 1. Extract frames for every strategy.
    if caption_only:
        return caption_all(conn, videos)
    from .siglip import Siglip
    siglip = Siglip()
    t0 = time.time()
    for v in videos:
        src, d = CACHE / v["relpath"], v["duration"] or 1
        for name, (times, px) in strategies(d, src, siglip).items():
            times = [min(t, max(0.0, d - 0.2)) for t in times]
            outdir = media_dir(v["uuid"]) / ("frames" if name == "fixed4" else f"frames_{name}")
            if name != "fixed4":  # fixed4 frames already exist from lab.prepare
                extract(src, times, outdir, px)
            conn.execute("DELETE FROM video_frames WHERE asset_id=? AND strategy=?", (v["uuid"], name))
            conn.executemany("INSERT INTO video_frames VALUES (?,?,?,?,?)",
                             [(v["uuid"], name, i, t, px) for i, t in enumerate(times)])
        conn.commit()
    print(f"frames extracted for {len(videos)} videos in {time.time() - t0:.0f}s")
    del siglip
    counts = {s: conn.execute("SELECT avg(n) FROM (SELECT count(*) n FROM video_frames WHERE strategy=? "
                              "GROUP BY asset_id)", (s,)).fetchone()[0] for s in ("fixed4", "change", "dense")}
    print("average frames per video:", {k: round(v, 1) for k, v in counts.items()})

    caption_all(conn, videos)


def caption_all(conn, videos):
    # 2. Caption each non-default strategy with the 30B.
    version = model_version(MODEL)
    unload(MODEL)
    for name in ("change", "dense"):
        label = f"{BASE_LABEL}@{name}"
        done_ids = {r[0] for r in conn.execute("SELECT asset_id FROM vlm_results WHERE model=? AND error IS NULL", (label,))}
        done = failed = 0
        started = time.time()
        for v in videos:
            if v["uuid"] in done_ids:
                continue
            rows = conn.execute("SELECT t FROM video_frames WHERE asset_id=? AND strategy=? ORDER BY idx",
                                (v["uuid"], name)).fetchall()
            files = sorted((media_dir(v["uuid"]) / f"frames_{name}").glob("f*.jpg"))
            images = [base64.b64encode(p.read_bytes()).decode() for p in files]
            prompt = PROMPT.format(n=len(images), dur=v["duration"] or 0,
                                   times=", ".join(f"{r[0]:.1f}" for r in rows))
            try:
                o = describe(MODEL, v["uuid"], "video", images=images, prompt=prompt, options=FRAME_OPTIONS)
                secs = o["wall"] - o["load_seconds"]
                conn.execute("INSERT OR REPLACE INTO vlm_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (v["uuid"], label, version, o["caption"], json.dumps(o["tags"]), o["ocr_text"],
                              o["prompt"], o["raw"], o["images_sent"], o["prompt_tokens"], o["output_tokens"],
                              secs, o["prompt_seconds"], o["output_seconds"], o["tps"], None, time.time()))
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                conn.execute("INSERT OR REPLACE INTO vlm_results (asset_id, model, model_version, error, computed_at) "
                             "VALUES (?,?,?,?,?)", (v["uuid"], label, version, f"{type(e).__name__}: {str(e)[:200]}",
                                                    time.time()))
            conn.commit()
        print(f"{label}: done {done}, failed {failed}, {time.time() - started:.0f}s", flush=True)
    unload(MODEL)
    for name, label in (("fixed4", BASE_LABEL), ("change", f"{BASE_LABEL}@change"), ("dense", f"{BASE_LABEL}@dense")):
        r = conn.execute("SELECT count(*), avg(images_sent), avg(prompt_tokens), avg(seconds), avg(length(caption)) "
                         "FROM vlm_results v JOIN assets a ON a.uuid=v.asset_id WHERE a.kind='video' AND v.model=? "
                         "AND v.error IS NULL", (label,)).fetchone()
        if r[0]:
            print(f"{name}: n={r[0]} frames={r[1]:.1f} prompt_tokens={r[2]:.0f} seconds={r[3]:.1f} caption_chars={r[4]:.0f}")


if __name__ == "__main__":
    main()
