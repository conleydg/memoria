"""Copy people, local time and location for each test asset from the
Photos.sqlite snapshot into the lab store. No model involved: every value
here comes from Photos' own on-device analysis and metadata.

    python -m lab.people [path/to/Photos.sqlite snapshot to take names from]

The lab's assets are keyed by ZUUID from the library Mac's snapshot
(data/Photos.sqlite), but ZUUID is per-library: the same photo has a
different ZUUID in another Mac's library. When names come from a
different library (e.g. this Mac's own, where names were added), assets
are matched across libraries by ZCLOUDASSETGUID, the iCloud ID that
every synced copy of a photo shares. Person keys (ZPERSONUUID) are also
per-library, so asset_people keys follow the library the names came from.

- People: named people only (ADR-0015: identity comes from Photos' own
  face clustering, never from our models). Keyed by ZPERSONUUID, the
  stable identifier (ADR-0025), with the name cached at index time
  (ADR-0023) into memoria's asset_people table. Unnamed clusters are
  counted but not stored, since there is no name to search by.
- Time: Photos' time zone offset, so the dashboard can show local time
  of day (assets.date_created stays UTC).
- Location: raw GPS only. Turning coordinates into place names would
  need a geocoding service, which is off the table (nothing leaves the
  Mac), so no place names here yet.

Names end up in search_fts via lab.index, so a query like
"<name> at the beach" can match on the name by keyword.
"""

import sqlite3
import sys
import time

from .store import ROOT, connect

PHOTOS_DB = ROOT / "data" / "Photos.sqlite"
LAB_COLUMNS = {"tz_offset": "INTEGER", "tz_name": "TEXT", "latitude": "REAL", "longitude": "REAL",
               "face_count": "INTEGER", "unnamed_face_count": "INTEGER"}


def face_columns(photos) -> tuple[str, str]:
    """Photos renamed ZDETECTEDFACE's foreign keys across macOS versions
    (ZASSET/ZPERSON on older libraries, ZASSETFORFACE/ZPERSONFORFACE on
    newer ones). Returns (asset_col, person_col) for this snapshot."""
    cols = {r[1] for r in photos.execute("PRAGMA table_info(ZDETECTEDFACE)")}
    return ("ZASSETFORFACE", "ZPERSONFORFACE") if "ZASSETFORFACE" in cols else ("ZASSET", "ZPERSON")


def ensure_columns(conn):
    have = {r[1] for r in conn.execute("PRAGMA table_info(lab_assets)")}
    for col, typ in LAB_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE lab_assets ADD COLUMN {col} {typ}")


def main():
    conn = connect()
    ensure_columns(conn)
    photos_db = sys.argv[1] if len(sys.argv) > 1 else PHOTOS_DB
    photos = sqlite3.connect(f"file:{photos_db}?mode=ro", uri=True)
    fa, fp = face_columns(photos)
    ids = [r[0] for r in conn.execute("SELECT uuid FROM assets")]
    source = sqlite3.connect(f"file:{PHOTOS_DB}?mode=ro", uri=True)
    guid_of = dict(source.execute("SELECT ZUUID, ZCLOUDASSETGUID FROM ZASSET"))
    via_guid = missing = 0
    # The filename Photos shows (e.g. IMG_1234.HEIC), so an item can be
    # found in Photos by searching for it. ZFILENAME is Photos' internal
    # UUID-based name; ZORIGINALFILENAME is the one users see.
    for uuid, name in source.execute(
            "SELECT a.ZUUID, aa.ZORIGINALFILENAME FROM ZASSET a "
            "JOIN ZADDITIONALASSETATTRIBUTES aa ON aa.ZASSET = a.Z_PK WHERE aa.ZORIGINALFILENAME IS NOT NULL"):
        conn.execute("UPDATE assets SET original_filename=? WHERE uuid=?", (name, uuid))
    now = time.time()
    people_rows = located = 0
    for uuid in ids:
        sql = ("SELECT a.Z_PK, aa.ZTIMEZONEOFFSET, aa.ZTIMEZONENAME, a.ZLATITUDE, a.ZLONGITUDE "
               "FROM ZASSET a LEFT JOIN ZADDITIONALASSETATTRIBUTES aa ON aa.ZASSET = a.Z_PK WHERE a.")
        meta = photos.execute(sql + "ZUUID = ?", (uuid,)).fetchone()
        if not meta and guid_of.get(uuid):
            meta = photos.execute(sql + "ZCLOUDASSETGUID = ? AND a.ZTRASHEDSTATE = 0", (guid_of[uuid],)).fetchone()
            via_guid += bool(meta)
        if not meta:
            missing += 1
            continue
        pk, tz_offset, tz_name, lat, lon = meta
        # Photos uses -180/-180 (and sometimes 0/0) for "no location".
        if lat is None or not (-90 <= lat <= 90) or (lat == 0 and lon == 0) or lat == -180:
            lat = lon = None
        faces = photos.execute(
            "SELECT p.ZPERSONUUID, COALESCE(NULLIF(p.ZFULLNAME, ''), NULLIF(p.ZDISPLAYNAME, '')) "
            f"FROM ZDETECTEDFACE f LEFT JOIN ZPERSON p ON p.Z_PK = f.{fp} "
            f"WHERE f.{fa} = ? AND f.ZDETECTIONTYPE = 1 AND COALESCE(f.ZHIDDEN, 0) = 0", (pk,)).fetchall()
        named = {key: name for key, name in faces if key and name}
        conn.execute("UPDATE lab_assets SET tz_offset=?, tz_name=?, latitude=?, longitude=?, face_count=?, "
                     "unnamed_face_count=? WHERE uuid=?",
                     (tz_offset, tz_name, lat, lon, len(faces), sum(1 for k, n in faces if not n), uuid))
        conn.execute("DELETE FROM asset_people WHERE asset_id=?", (uuid,))
        conn.executemany("INSERT INTO asset_people VALUES (?,?,?,?)",
                         [(uuid, key, name, now) for key, name in named.items()])
        people_rows += len(named)
        located += lat is not None
    conn.commit()
    with_people = conn.execute("SELECT count(DISTINCT asset_id) FROM asset_people").fetchone()[0]
    distinct = conn.execute("SELECT count(DISTINCT person_key) FROM asset_people").fetchone()[0]
    with_faces = conn.execute("SELECT count(*) FROM lab_assets WHERE face_count > 0").fetchone()[0]
    with_tz = conn.execute("SELECT count(*) FROM lab_assets WHERE tz_offset IS NOT NULL").fetchone()[0]
    print(f"people: {people_rows} asset-person links, {distinct} distinct people, on {with_people} assets; "
          f"{with_faces} assets have faces; {located} with GPS; {with_tz} with a time zone offset; "
          f"{via_guid} matched by iCloud ID, {missing} not found in this library (kept their previous values)")


if __name__ == "__main__":
    main()
