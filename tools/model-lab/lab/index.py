"""Rebuild the keyword indexes and generate "how could this be found?"
example queries.

    python -m lab.index            # FTS only
    python -m lab.index --queries  # FTS + example queries (uses the 8B model)

search_fts (memoria's table) = 30B caption + tags + OCR + transcript
                               + named people (from Photos, lab.people).
search_fts_8b                = the same, from the 8B model's output.

Example queries are written by the 8B model from the 30B caption and
tags, phrased the way a person would search their own library. They are
derived from the caption, so they naturally favour keyword search on the
30B index. The dashboard says so.
"""

import json
import sys
import time

import requests

from .memwatch import OLLAMA
from .store import connect
from .vlm import model_version, unload

QUERY_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
QUERY_PROMPT = """Here is the description of one photo or video in someone's personal library.

Caption: {caption}
Tags: {tags}

Write 4 different short search queries (2-6 words each) that this person might type, months later, to find this item again. Vary them: one about the main subject, one about the setting or activity, one vaguer or more casual, and one that uses different words than the caption. Return JSON: {{"queries": [..4 strings..]}}"""
QSCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
           "required": ["queries"]}


def rebuild_fts(conn):
    for table, model in (("search_fts", "qwen3-vl-30b"), ("search_fts_8b", "qwen3-vl-8b")):
        conn.execute(f"DELETE FROM {table}")
        rows = conn.execute(
            "SELECT v.asset_id, v.caption, v.tags, v.ocr_text, t.text, "
            "(SELECT group_concat(person_name, ' ') FROM asset_people p WHERE p.asset_id = v.asset_id) AS people "
            "FROM vlm_results v LEFT JOIN transcripts t ON t.asset_id = v.asset_id WHERE v.model=? AND v.error IS NULL",
            (model,)).fetchall()
        for r in rows:
            text = " \n".join(x for x in (r["caption"], " ".join(json.loads(r["tags"] or "[]")),
                                          r["ocr_text"], r["text"], r["people"]) if x)
            conn.execute(f"INSERT INTO {table} (asset_id, text) VALUES (?,?)", (r["asset_id"], text))
        print(f"{table}: {len(rows)} rows")
    conn.execute("UPDATE assets SET status='indexed', indexed_at=? WHERE uuid IN "
                 "(SELECT asset_id FROM vlm_results WHERE model='qwen3-vl-30b' AND error IS NULL) "
                 "AND uuid IN (SELECT asset_id FROM embeddings)", (time.time(),))
    conn.commit()


def generate_queries(conn):
    version = model_version(QUERY_MODEL)
    rows = conn.execute("SELECT c.asset_id, c.caption, "
                        "(SELECT group_concat(tag, ', ') FROM tags WHERE asset_id=c.asset_id) AS tags "
                        "FROM captions c WHERE c.asset_id NOT IN (SELECT asset_id FROM example_queries)").fetchall()
    for r in rows:
        try:
            resp = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
                "model": QUERY_MODEL, "stream": False, "format": QSCHEMA, "keep_alive": "5m",
                "options": {"temperature": 0.3, "seed": 1, "num_predict": 200},
                "messages": [{"role": "user", "content": QUERY_PROMPT.format(caption=r["caption"], tags=r["tags"])}],
            }).json()
            qs = [q.strip() for q in json.loads(resp["message"]["content"])["queries"] if q.strip()][:5]
            conn.executemany("INSERT OR REPLACE INTO example_queries VALUES (?,?,?,?)",
                             [(r["asset_id"], i, q, version) for i, q in enumerate(qs)])
            conn.commit()
        except Exception as e:  # noqa: BLE001
            print("query gen failed:", r["asset_id"], type(e).__name__)
    unload(QUERY_MODEL)
    print("example queries:", conn.execute("SELECT count(*) FROM example_queries").fetchone()[0])


def add_people_queries(conn):
    """One extra query per named person in the asset: just the name.
    SigLIP2 has never seen these people, so this shows what only keyword
    search (fed by Photos' face recognition) can find."""
    conn.execute("DELETE FROM example_queries WHERE model_version='photos-people'")
    rows = conn.execute("SELECT asset_id, person_name FROM asset_people ORDER BY asset_id, person_name").fetchall()
    seen: dict[str, int] = {}
    for r in rows:
        n = seen[r["asset_id"]] = seen.get(r["asset_id"], 0) + 1
        conn.execute("INSERT OR REPLACE INTO example_queries VALUES (?,?,?,?)",
                     (r["asset_id"], 100 + n, r["person_name"], "photos-people"))
    conn.commit()
    print("people queries:", len(rows))


if __name__ == "__main__":
    c = connect()
    rebuild_fts(c)
    if "--queries" in sys.argv:
        generate_queries(c)
    add_people_queries(c)
