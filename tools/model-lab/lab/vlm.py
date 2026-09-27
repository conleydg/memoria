"""Caption + tags + OCR with Qwen3-VL through the local Ollama server
(127.0.0.1 only). The same prompt goes to every model so the outputs can
be compared side by side.

    python -m lab.vlm qwen3-vl:30b-a3b-instruct-q4_K_M
    python -m lab.vlm qwen3-vl:8b-instruct-q4_K_M

Photos send model.jpg; videos send their 4 keyframes in one request.
Output is constrained to a JSON schema (Ollama structured outputs). The
model is unloaded at the end (keep_alive 0), per ADR-0013: nothing large
stays resident after a batch.
"""

import base64
import json
import sys
import time

import requests

from .memwatch import OLLAMA, OllamaPeak
from .store import connect, media_dir, record_run

PROMPT_IMAGE = """You are indexing a personal photo library so it can be searched later.
Describe this photo.

Return JSON with:
- "caption": 1-3 plain sentences describing what is shown: the main subjects, what they are doing, the setting, and anything notable (season, event, weather, time of day). Do not guess names.
- "tags": 8-20 short lowercase search tags (objects, animals, activities, scene, event, style, e.g. "dog", "beach", "birthday cake", "screenshot").
- "ocr_text": any clearly legible text in the image, verbatim, or "" if there is none."""

PROMPT_VIDEO = """You are indexing a personal video library so it can be searched later.
These are 4 frames taken from one video, in order (at 10%, 35%, 60% and 85% of its length).
Describe the video as a whole.

Return JSON with:
- "caption": 1-3 plain sentences describing what happens in the video: the main subjects, what they are doing, the setting, and anything notable (season, event, weather, time of day). Do not guess names.
- "tags": 8-20 short lowercase search tags (objects, animals, activities, scene, event, style).
- "ocr_text": any clearly legible text in the frames, verbatim, or "" if there is none."""

SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ocr_text": {"type": "string"},
    },
    "required": ["caption", "tags", "ocr_text"],
}
OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 1500, "seed": 1}
# A few assets (text-heavy screenshots, busy scenes) send greedy decoding
# into a repetition loop until the token cap. The retry pass adds a
# repeat penalty and caps the tag list; those rows say so in vlm_results.
RETRY_OPTIONS = {**OPTIONS, "repeat_penalty": 1.15}
RETRY_SCHEMA = {**SCHEMA, "properties": {**SCHEMA["properties"],
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20}}}


def model_version(model: str) -> str:
    info = requests.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=30).json()
    digest = next((m["digest"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=10).json()["models"]
                   if m["name"] == model), "")
    d = info.get("details", {})
    return f"{model}@{digest[:12]} ({d.get('quantization_level', '?')}, {d.get('parameter_size', '?')})"


def model_size(model: str) -> int:
    return next((m["size"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=10).json()["models"]
                 if m["name"] == model), 0)


def unload(model: str):
    requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)


def images_for(uuid: str, kind: str) -> list[str]:
    d = media_dir(uuid)
    paths = sorted((d / "frames").glob("f*.jpg")) if kind == "video" else [d / "model.jpg"]
    return [base64.b64encode(p.read_bytes()).decode() for p in paths]


def describe(model: str, uuid: str, kind: str, keep_alive="10m", retry=False) -> dict:
    prompt = PROMPT_VIDEO if kind == "video" else PROMPT_IMAGE
    images = images_for(uuid, kind)
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": model, "stream": False, "format": RETRY_SCHEMA if retry else SCHEMA,
        "options": RETRY_OPTIONS if retry else OPTIONS, "keep_alive": keep_alive,
        "messages": [{"role": "user", "content": prompt, "images": images}],
    })
    r.raise_for_status()
    body = r.json()
    wall = time.time() - t0
    raw = body["message"]["content"]
    ns = 1e9
    if retry:
        prompt += f"\n\n[retry pass: options={json.dumps(RETRY_OPTIONS)}, tags capped at 20]"
    out = {
        "prompt": prompt, "raw": raw, "images_sent": len(images), "wall": wall,
        "load_seconds": body.get("load_duration", 0) / ns,
        "prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count"),
        "prompt_seconds": body.get("prompt_eval_duration", 0) / ns,
        "output_seconds": body.get("eval_duration", 0) / ns,
    }
    out["tps"] = out["output_tokens"] / out["output_seconds"] if out["output_seconds"] else None
    parsed = json.loads(raw)
    out["caption"] = parsed.get("caption", "").strip()
    out["tags"] = sorted({t.strip().lower() for t in parsed.get("tags", []) if t.strip()})
    out["ocr_text"] = (parsed.get("ocr_text") or "").strip()
    return out


def run(model: str, label: str, is_primary: bool, retry: bool = False):
    conn = connect()
    version = model_version(model)
    rows = conn.execute("SELECT a.uuid, a.kind FROM assets a WHERE a.status != 'failed' "
                        "AND a.uuid NOT IN (SELECT asset_id FROM vlm_results WHERE model=? AND error IS NULL) "
                        "ORDER BY a.date_created", (label,)).fetchall()
    unload(model)  # start cold, so the first call's load_duration is a real cold load
    time.sleep(2)
    done = failed = 0
    load_seconds = None
    started = time.time()
    with OllamaPeak() as peak:
        for i, r in enumerate(rows):
            try:
                o = describe(model, r["uuid"], r["kind"], retry=retry)
                if load_seconds is None:
                    load_seconds = o["load_seconds"]
                conn.execute(
                    "INSERT OR REPLACE INTO vlm_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["uuid"], label, version, o["caption"], json.dumps(o["tags"]), o["ocr_text"],
                     o["prompt"], o["raw"], o["images_sent"], o["prompt_tokens"], o["output_tokens"],
                     o["wall"] - (o["load_seconds"] if i == 0 else 0), o["prompt_seconds"],
                     o["output_seconds"], o["tps"], None, time.time()))
                conn.execute("INSERT OR REPLACE INTO timings VALUES (?,?,?,?)",
                             (r["uuid"], label, o["wall"] - (o["load_seconds"] if i == 0 else 0),
                              json.dumps({"prompt_tokens": o["prompt_tokens"], "output_tokens": o["output_tokens"],
                                          "tps": o["tps"], "images": o["images_sent"]})))
                if is_primary:
                    write_production(conn, r["uuid"], version, o)
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                conn.execute("INSERT OR REPLACE INTO vlm_results (asset_id, model, model_version, error, computed_at) "
                             "VALUES (?,?,?,?,?)", (r["uuid"], label, version, f"{type(e).__name__}: {str(e)[:200]}",
                                                    time.time()))
            conn.commit()
            if (i + 1) % 10 == 0:
                print(f"{label}: {i + 1}/{len(rows)} ({failed} failed), {time.time() - started:.0f}s", flush=True)
        total = time.time() - started
        unload(model)
    time.sleep(3)
    if retry:  # redo failed rows only; keep the original run's stats
        print(f"{label} retry: fixed {done}, still failing {failed}")
        return
    record_run(conn, label, model_version=version, role="caption + tags + OCR", runtime="Ollama (localhost)",
               license="Apache-2.0", disk_bytes=model_size(model), load_seconds=load_seconds,
               peak_memory_bytes=peak.peak,
               memory_note=f"max of Ollama process RSS ({peak.peak_rss / 1e9:.1f} GB) and /api/ps size ({peak.peak_ps / 1e9:.1f} GB)",
               assets_done=done, assets_failed=failed, total_seconds=total, started_at=started,
               finished_at=time.time(), notes=f"unloaded with keep_alive=0; options={json.dumps(OPTIONS)}")
    print(f"{label}: done {done}, failed {failed}, {total:.0f}s")


def write_production(conn, uuid, version, o):
    """The 30B output goes in memoria's production tables too (captions,
    tags); search_fts is rebuilt from all sources by lab.index."""
    now = time.time()
    conn.execute("INSERT OR REPLACE INTO captions VALUES (?,?,?,?)", (uuid, o["caption"], version, now))
    conn.execute("DELETE FROM tags WHERE asset_id=? AND source='qwen3-vl'", (uuid,))
    conn.executemany("INSERT OR IGNORE INTO tags VALUES (?,?,?,?,?)",
                     [(uuid, t, "qwen3-vl", version, now) for t in o["tags"]])


LABELS = {"qwen3-vl:30b-a3b-instruct-q4_K_M": "qwen3-vl-30b", "qwen3-vl:8b-instruct-q4_K_M": "qwen3-vl-8b",
          "qwen3-vl:32b-instruct-q4_K_M": "qwen3-vl-32b"}

if __name__ == "__main__":
    m = sys.argv[1]
    run(m, LABELS.get(m, m), is_primary=LABELS.get(m) == "qwen3-vl-30b", retry="--retry" in sys.argv)
