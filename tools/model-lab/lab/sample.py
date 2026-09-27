"""Pick a ~200-asset test set from a Photos.sqlite snapshot, using
metadata only (no pixels are read here).

Strata, each drawn at random with a fixed seed so the set is reproducible:
year spread, people, pets, low light, screenshots, near-duplicate pairs
(Photos' own perceptual-duplicate grouping, plus burst stacks), and
videos across a range of durations. Nothing in Photos.sqlite marks
speech, so "with/without speech" is found after the fact by Whisper.

Writes data/model-lab/manifest.json (gitignored: it names real assets).
"""

import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHOTOS_DB = ROOT / "data" / "Photos.sqlite"
OUT = ROOT / "data" / "model-lab" / "manifest.json"
APPLE_EPOCH = 978307200  # 2001-01-01, Core Data's reference date
SEED = 20260926

IMAGE_UTIS = ("public.heic", "public.jpeg", "public.png")
VIDEO_UTIS = ("com.apple.quicktime-movie", "public.mpeg-4")
MAX_VIDEO_BYTES = 400 * 1024 * 1024

BASE = f"""
SELECT a.ZUUID, a.ZKIND, a.ZKINDSUBTYPE, a.ZDATECREATED + {APPLE_EPOCH},
       a.ZDIRECTORY, a.ZFILENAME, a.ZUNIFORMTYPEIDENTIFIER, a.ZDURATION,
       aa.ZORIGINALFILESIZE
FROM ZASSET a
LEFT JOIN ZADDITIONALASSETATTRIBUTES aa ON aa.ZASSET = a.Z_PK
LEFT JOIN ZEXTENDEDATTRIBUTES ea ON ea.ZASSET = a.Z_PK
LEFT JOIN ZMEDIAANALYSISASSETATTRIBUTES ma ON ma.ZASSET = a.Z_PK
WHERE a.ZTRASHEDSTATE = 0 AND a.ZHIDDEN = 0
  AND a.ZDIRECTORY IS NOT NULL AND a.ZFILENAME IS NOT NULL
"""
IMG = f" AND a.ZKIND = 0 AND a.ZUNIFORMTYPEIDENTIFIER IN {IMAGE_UTIS}"
VID = (f" AND a.ZKIND = 1 AND a.ZUNIFORMTYPEIDENTIFIER IN {VIDEO_UTIS}"
       f" AND COALESCE(aa.ZORIGINALFILESIZE, 0) < {MAX_VIDEO_BYTES}")


def main(photos_db=PHOTOS_DB, out=OUT):
    conn = sqlite3.connect(f"file:{photos_db}?mode=ro", uri=True)
    rng = random.Random(SEED)
    chosen: dict[str, dict] = {}

    def take(sql, n, stratum, params=()):
        rows = conn.execute(BASE + sql, params).fetchall()
        rows = [r for r in rows if r[0] not in chosen]
        for r in rng.sample(rows, min(n, len(rows))):
            add(r, stratum)

    def add(r, stratum):
        uuid, kind, subtype, ts, d, fn, uti, dur, size = r
        if uuid in chosen:
            chosen[uuid]["strata"].append(stratum)
            return
        chosen[uuid] = {
            "uuid": uuid,
            "kind": "video" if kind == 1 else "image",
            "subtype": subtype,
            "date_created": ts,
            "relpath": f"originals/{d}/{fn}",
            "uti": uti,
            "duration": dur if kind == 1 else None,
            "original_size": size,
            "strata": [stratum],
        }

    year = "CAST(strftime('%Y', a.ZDATECREATED + ?, 'unixepoch') AS INT)"
    years = [y for (y, n) in conn.execute(
        f"SELECT {year} y, COUNT(*) FROM ZASSET a WHERE ZTRASHEDSTATE=0 GROUP BY y",
        (APPLE_EPOCH,)) if n >= 20 and y >= 2000]
    # Year spread: 2 per year guaranteed, the rest proportional to volume.
    for y in years:
        take(f"{IMG} AND a.ZKINDSUBTYPE != 10 AND {year} = ?", 2, "year-spread", (APPLE_EPOCH, y))
    take(f"{IMG} AND a.ZKINDSUBTYPE != 10", 110 - 2 * len(years), "year-spread")

    take(f"{IMG} AND ma.ZFACECOUNT >= 2", 20, "people")
    take(f"{IMG} AND a.Z_PK IN (SELECT ZASSET FROM ZDETECTEDFACE WHERE ZDETECTIONTYPE IN (3, 4))",
         12, "pets")
    take(f"{IMG} AND ea.ZISO >= 1600", 12, "low-light")
    take(f"{IMG} AND a.ZKINDSUBTYPE = 10", 14, "screenshot")

    # Near-duplicates: 3 pairs from Photos' perceptual-duplicate albums,
    # 2 pairs from burst stacks.
    for col, npairs in (("ZDUPLICATEPERCEPTUALMATCHINGALBUM", 3), ("ZAVALANCHEUUID", 2)):
        groups = [g for (g,) in conn.execute(
            f"SELECT {col} FROM ZASSET a WHERE {col} IS NOT NULL AND ZKIND=0 "
            f"AND ZTRASHEDSTATE=0 AND ZUNIFORMTYPEIDENTIFIER IN {IMAGE_UTIS} "
            f"GROUP BY {col} HAVING COUNT(*) >= 2")]
        for g in rng.sample(groups, min(npairs, len(groups))):
            rows = conn.execute(BASE + IMG + f" AND a.{col} = ?", (g,)).fetchall()
            for r in rng.sample(rows, 2):
                add(r, "near-duplicate")
                chosen[r[0]]["dup_group"] = f"{col[1:4].lower()}:{g}"

    take(f"{VID} AND a.ZDURATION < 10 AND a.ZKINDSUBTYPE = 0", 7, "video-short")
    take(f"{VID} AND a.ZDURATION BETWEEN 10 AND 60 AND a.ZKINDSUBTYPE = 0", 10, "video-medium")
    take(f"{VID} AND a.ZDURATION BETWEEN 60 AND 240 AND a.ZKINDSUBTYPE = 0", 5, "video-long")
    take(f"{VID} AND a.ZKINDSUBTYPE IN (101, 102)", 2, "video-slomo-timelapse")

    add_events(conn, chosen, add)

    assets = sorted(chosen.values(), key=lambda a: a["date_created"] or 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": SEED, "source": str(photos_db), "assets": assets}, indent=1))
    return assets


def add_events(conn, chosen, add, n_events=4):
    """Whole events, so "more like this" has real neighbours to find: 4
    Photos moments (15-25 local photos each, from different years), taken
    complete, plus their short videos. Uses its own RNG so the strata
    above are unchanged. Local originals only (ZCLOUDLOCALSTATE = 1):
    iCloud-only items can't be fetched from the library Mac."""
    rng = random.Random(SEED + 1)
    year = f"CAST(strftime('%Y', MIN(ZDATECREATED) + {APPLE_EPOCH}, 'unixepoch') AS INT)"
    moments = conn.execute(f"""
        SELECT ZMOMENT, {year} FROM ZASSET
        WHERE ZTRASHEDSTATE = 0 AND ZCLOUDLOCALSTATE = 1 AND ZKIND = 0 AND ZKINDSUBTYPE != 10
          AND ZUNIFORMTYPEIDENTIFIER IN {IMAGE_UTIS} AND ZMOMENT IS NOT NULL
        GROUP BY ZMOMENT HAVING COUNT(*) BETWEEN 15 AND 25""").fetchall()
    rng.shuffle(moments)
    used_years = set()
    for moment, y in moments:
        if y in used_years or y < 2010:
            continue
        used_years.add(y)
        rows = conn.execute(BASE + f" AND a.ZMOMENT = ? AND a.ZCLOUDLOCALSTATE = 1 AND ("
                            f"(a.ZKIND = 0 AND a.ZUNIFORMTYPEIDENTIFIER IN {IMAGE_UTIS}) OR "
                            f"(a.ZKIND = 1 AND a.ZDURATION < 60 {VID}))", (moment,)).fetchall()
        for r in rows:
            add(r, "event")
            chosen[r[0]]["event"] = f"moment:{moment}"
        if len(used_years) >= n_events:
            break


if __name__ == "__main__":
    assets = main(*sys.argv[1:2])
    from collections import Counter
    print(len(assets), "assets")
    print(Counter(a["kind"] for a in assets))
    print(Counter(s for a in assets for s in a["strata"]))
