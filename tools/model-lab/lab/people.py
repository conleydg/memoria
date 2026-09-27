"""Copy people, local time and location for each test asset from the
Photos.sqlite snapshot into the lab store. No model involved: every value
here comes from Photos' own on-device analysis and metadata.

    python -m lab.people

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
import time

from .store import ROOT, connect

PHOTOS_DB = ROOT / "data" / "Photos.sqlite"
LAB_COLUMNS = {"tz_offset": "INTEGER", "tz_name": "TEXT", "latitude": "REAL", "longitude": "REAL",
               "face_count": "INTEGER", "unnamed_face_count": "INTEGER"}


def ensure_columns(conn):
    have = {r[1] for r in conn.execute("PRAGMA table_info(lab_assets)")}
    for col, typ in LAB_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE lab_assets ADD COLUMN {col} {typ}")


def main():
    conn = connect()
    ensure_columns(conn)
    photos = sqlite3.connect(f"file:{PHOTOS_DB}?mode=ro", uri=True)
    ids = [r[0] for r in conn.execute("SELECT uuid FROM assets")]
    now = time.time()
    people_rows = located = 0
    for uuid in ids:
        meta = photos.execute(
            "SELECT a.Z_PK, aa.ZTIMEZONEOFFSET, aa.ZTIMEZONENAME, a.ZLATITUDE, a.ZLONGITUDE "
            "FROM ZASSET a LEFT JOIN ZADDITIONALASSETATTRIBUTES aa ON aa.ZASSET = a.Z_PK WHERE a.ZUUID = ?",
            (uuid,)).fetchone()
        if not meta:
            continue
        pk, tz_offset, tz_name, lat, lon = meta
        # Photos uses -180/-180 (and sometimes 0/0) for "no location".
        if lat is None or not (-90 <= lat <= 90) or (lat == 0 and lon == 0) or lat == -180:
            lat = lon = None
        faces = photos.execute(
            "SELECT p.ZPERSONUUID, COALESCE(NULLIF(p.ZFULLNAME, ''), NULLIF(p.ZDISPLAYNAME, '')) "
            "FROM ZDETECTEDFACE f LEFT JOIN ZPERSON p ON p.Z_PK = f.ZPERSON "
            "WHERE f.ZASSET = ? AND f.ZDETECTIONTYPE = 1 AND COALESCE(f.ZHIDDEN, 0) = 0", (pk,)).fetchall()
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
          f"{with_faces} assets have faces; {located} with GPS; {with_tz} with a time zone offset")


if __name__ == "__main__":
    main()
