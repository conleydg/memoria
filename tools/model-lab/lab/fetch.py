"""Load the manifest into the lab store and pull each original from the
Mac that holds the library (ADR-0027), via memoria's own RemoteLibrary.

    python -m lab.fetch --host photos-mac --library "/Users/x/Pictures/Photos Library.photoslibrary"

Already-cached originals are not re-downloaded. A failed fetch marks the
asset 'failed' with the reason (ADR-0024) and the run moves on.
"""

import argparse
import json
import time

from memoria.remote_library import RemoteLibrary

from .store import CACHE, MANIFEST, connect


def load_manifest(conn) -> list[dict]:
    assets = json.loads(MANIFEST.read_text())["assets"]
    if "event" not in {r[1] for r in conn.execute("PRAGMA table_info(lab_assets)")}:
        conn.execute("ALTER TABLE lab_assets ADD COLUMN event TEXT")
    for a in assets:
        conn.execute(
            "INSERT OR IGNORE INTO assets (uuid, kind, original_filename, date_created) VALUES (?,?,?,?)",
            (a["uuid"], a["kind"], a["relpath"].rsplit("/", 1)[-1], a["date_created"]),
        )
        conn.execute(
            # Upsert, not REPLACE: REPLACE would wipe columns filled by
            # later stages (dimensions, has_audio, people data).
            "INSERT INTO lab_assets (uuid, relpath, uti, subtype, duration, original_size, strata, dup_group) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(uuid) DO UPDATE SET relpath=excluded.relpath, uti=excluded.uti, "
            "subtype=excluded.subtype, original_size=excluded.original_size, strata=excluded.strata, "
            "dup_group=excluded.dup_group",
            (a["uuid"], a["relpath"], a["uti"], a["subtype"], a["duration"], a["original_size"],
             ",".join(a["strata"]), a.get("dup_group")),
        )
        conn.execute("UPDATE lab_assets SET event=? WHERE uuid=?", (a.get("event"), a["uuid"]))
    conn.commit()
    return assets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="ssh target (alias or user@host)")
    ap.add_argument("--library", required=True, help="remote path to the .photoslibrary bundle")
    ap.add_argument("--max-cache-gb", type=float, default=8)
    args = ap.parse_args()

    conn = connect()
    assets = load_manifest(conn)
    lib = RemoteLibrary(args.host, args.library, CACHE, int(args.max_cache_gb * 1e9))
    ok = failed = 0
    t0 = time.time()
    for i, a in enumerate(assets, 1):
        try:
            lib.fetch(a["relpath"])
            ok += 1
        except Exception as e:  # noqa: BLE001 - recorded per asset, run continues
            failed += 1
            conn.execute("UPDATE assets SET status='failed', note=? WHERE uuid=?",
                         (f"fetch: {type(e).__name__}", a["uuid"]))
            conn.commit()
        if i % 20 == 0:
            print(f"{i}/{len(assets)} fetched, {failed} failed, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {ok} ok, {failed} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
