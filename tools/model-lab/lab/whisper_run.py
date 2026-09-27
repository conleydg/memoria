"""Whisper large-v3-turbo transcripts for videos (ADR-0009), via
mlx-whisper, once per video's whole audio track.

    python -m lab.whisper_run

Timestamped segments go in transcript_segments; the joined text goes in
memoria's `transcripts` table. A video counts as having speech when at
least one segment is confidently speech (no_speech_prob < 0.5 and
non-trivial text). Whisper tends to hallucinate short stock phrases on
silence, which is why the check isn't just "text is non-empty".
"""

import time
import wave

import mlx.core as mx
import numpy as np
import mlx_whisper

from .memwatch import ProcessPeak
from .store import connect, media_dir, record_run

REPO = "mlx-community/whisper-large-v3-turbo"
REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
MODEL_VERSION = f"{REPO}@{REVISION[:7]} (fp16, MLX)"


def load_wav(path) -> np.ndarray:
    """16 kHz mono PCM (written by lab.prepare) -> float32 in [-1, 1], so
    mlx-whisper doesn't need to find ffmpeg on PATH."""
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main():
    conn = connect()
    rows = conn.execute("SELECT a.uuid, l.duration FROM assets a JOIN lab_assets l ON l.uuid=a.uuid "
                        "WHERE a.kind='video' AND a.status != 'failed' AND l.has_audio=1").fetchall()
    started = time.time()
    done = failed = 0
    load_seconds = None
    audio_total = 0.0
    with ProcessPeak() as peak:
        for r in rows:
            wav = load_wav(media_dir(r["uuid"]) / "audio.wav")
            t0 = time.time()
            try:
                res = mlx_whisper.transcribe(wav, path_or_hf_repo=REPO, word_timestamps=False,
                                             condition_on_previous_text=False, verbose=None)
                secs = time.time() - t0
                if load_seconds is None:
                    # First call includes loading the weights; measure a warm
                    # call on the same file to split the two apart.
                    t1 = time.time()
                    mlx_whisper.transcribe(wav, path_or_hf_repo=REPO, verbose=None,
                                           condition_on_previous_text=False)
                    warm = time.time() - t1
                    load_seconds, secs = max(secs - warm, 0), warm
                segs = res.get("segments", [])
                speech = [s for s in segs if s.get("no_speech_prob", 1) < 0.5 and len(s["text"].strip()) > 3]
                conn.execute("DELETE FROM transcript_segments WHERE asset_id=?", (r["uuid"],))
                conn.executemany("INSERT INTO transcript_segments VALUES (?,?,?,?,?,?,?)",
                                 [(r["uuid"], i, s["start"], s["end"], s["text"].strip(), s.get("no_speech_prob"),
                                   s.get("avg_logprob")) for i, s in enumerate(segs)])
                text = " ".join(s["text"].strip() for s in speech)
                conn.execute("INSERT OR REPLACE INTO transcript_meta VALUES (?,?,?,?,?)",
                             (r["uuid"], res.get("language"), int(bool(speech)), r["duration"], MODEL_VERSION))
                conn.execute("DELETE FROM transcripts WHERE asset_id=?", (r["uuid"],))
                if speech:
                    conn.execute("INSERT INTO transcripts VALUES (?,?,?,?)", (r["uuid"], text, MODEL_VERSION, time.time()))
                conn.execute("INSERT OR REPLACE INTO timings VALUES (?,?,?,?)",
                             (r["uuid"], "whisper", secs, f'{{"audio_seconds": {r["duration"] or 0}}}'))
                audio_total += r["duration"] or 0
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("whisper failed:", r["uuid"], type(e).__name__, str(e)[:200])
            conn.commit()
    total = time.time() - started
    mlx_peak = mx.get_peak_memory()
    record_run(conn, "whisper", model_version=MODEL_VERSION, role="video transcripts", runtime="mlx-whisper (MLX)",
               license="MIT", disk_bytes=_hf_size(REPO), load_seconds=load_seconds,
               peak_memory_bytes=max(peak.peak, mlx_peak),
               memory_note=f"process RSS peak {peak.peak / 1e9:.1f} GB; MLX peak {mlx_peak / 1e9:.1f} GB",
               assets_done=done, assets_failed=failed, total_seconds=total, started_at=started,
               finished_at=time.time(), notes=f"{audio_total:.0f}s of audio total")
    print(f"whisper: done {done}, failed {failed}, {total:.0f}s for {audio_total:.0f}s audio")


def _hf_size(repo: str) -> int:
    from huggingface_hub import scan_cache_dir
    return next((r.size_on_disk for r in scan_cache_dir().repos if r.repo_id == repo), 0)


if __name__ == "__main__":
    main()
