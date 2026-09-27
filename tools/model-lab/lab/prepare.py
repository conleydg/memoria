"""Turn each cached original into the derivatives the models and the
dashboard need, all under data/model-lab/media/<uuid>/:

- images: model.jpg (long side 1024, what the VLMs and SigLIP2 see),
  full.jpg (1600, for the dashboard), thumb.jpg (360)
- videos: frames/f0..f3.jpg (4 keyframes at 10/35/60/85% of the clip),
  thumb.jpg, video.mp4 (720p H.264, so any browser can play it), and
  audio.wav (16 kHz mono, for Whisper) when there is an audio track

Everything is local (Pillow, pillow-heif, ffmpeg).
"""

import json
import subprocess

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .store import CACHE, connect, media_dir

register_heif_opener()
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
FRAME_POSITIONS = (0.10, 0.35, 0.60, 0.85)


def _save(img: Image.Image, path, long_side: int, quality=88):
    im = img.copy()
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    im.save(path, "JPEG", quality=quality)


def prepare_image(src, out):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        _save(im, out / "model.jpg", 1024)
        _save(im, out / "full.jpg", 1600)
        _save(im, out / "thumb.jpg", 360, quality=80)
    return {"width": w, "height": h, "has_audio": None, "frame_count": None}


def _probe(src):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(src)],
                       capture_output=True, check=True, text=True)
    return json.loads(r.stdout)


def prepare_video(src, out):
    info = _probe(src)
    duration = float(info["format"].get("duration") or 0)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    for i, pos in enumerate(FRAME_POSITIONS):
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{duration * pos:.3f}", "-i", str(src),
                        "-frames:v", "1", "-vf", "scale='min(1024,iw)':-2:force_original_aspect_ratio=decrease",
                        "-q:v", "3", str(frames / f"f{i}.jpg")], check=True)
    with Image.open(frames / "f1.jpg") as im:
        _save(im.convert("RGB"), out / "thumb.jpg", 360, quality=80)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-vf", "scale=-2:'min(720,ih)'",
                    "-c:v", "h264_videotoolbox", "-b:v", "3M", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(out / "video.mp4")], check=True)
    if has_audio:
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                        str(out / "audio.wav")], check=True)
    return {"width": v.get("width"), "height": v.get("height"), "has_audio": int(has_audio),
            "frame_count": len(FRAME_POSITIONS), "duration": duration}


def main():
    conn = connect()
    rows = conn.execute("SELECT a.uuid, a.kind, l.relpath FROM assets a JOIN lab_assets l ON l.uuid=a.uuid "
                        "WHERE a.status != 'failed'").fetchall()
    ok = failed = 0
    for r in rows:
        src = CACHE / r["relpath"]
        out = media_dir(r["uuid"])
        out.mkdir(parents=True, exist_ok=True)
        if (out / "thumb.jpg").exists():
            ok += 1
            continue
        try:
            if not src.exists():
                raise FileNotFoundError("original not cached")
            meta = (prepare_video if r["kind"] == "video" else prepare_image)(src, out)
            conn.execute("UPDATE lab_assets SET width=?, height=?, has_audio=?, frame_count=? WHERE uuid=?",
                         (meta["width"], meta["height"], meta["has_audio"], meta["frame_count"], r["uuid"]))
            if meta.get("duration"):
                conn.execute("UPDATE lab_assets SET duration=? WHERE uuid=?", (meta["duration"], r["uuid"]))
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            conn.execute("UPDATE assets SET status='failed', note=? WHERE uuid=?",
                         (f"prepare: {type(e).__name__}: {str(e)[:120]}", r["uuid"]))
        conn.commit()
    print(f"prepared {ok}, failed {failed}")


if __name__ == "__main__":
    main()
