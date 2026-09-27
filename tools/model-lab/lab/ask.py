"""Ask: question answering over the indexed test set (retrieval-augmented
generation, a first cut at ADR-0005's query layer).

1. Filter (ADR-0022 "filter first, then rank"): a person's name in the
   question (matched against Photos' named people, ADR-0015) and/or a
   4-digit year narrows the candidate set with plain SQL.
2. Retrieve: hybrid search (FTS5 over the 30B captions + SigLIP2 vectors,
   RRF) over those candidates, top-k (k = 5-60, default 20). A bigger k
   helps questions that need many items ("every beach trip") but costs
   time, memory, and some precision: more loosely related items for the
   model to wade through.
3. Generate: the 8B model reads only those items' metadata (date, people,
   caption, tags, OCR, transcript) and answers, citing items as [n]. It
   never sees pixels here, and it is told to say so when the context
   doesn't contain the answer, not to guess.

Optional "facts" (e.g. birthdays) come from the user and are passed into
the prompt as given. They're never stored server-side.

Everything runs locally: Ollama on 127.0.0.1 and SigLIP2 in-process.
"""

import json
import re
import time

import requests

from . import searchlab
from .memwatch import OLLAMA

MODELS = {"8b": "qwen3-vl:8b-instruct-q4_K_M", "30b": "qwen3-vl:30b-a3b-instruct-q4_K_M"}
SYSTEM = """You answer questions about the user's own photo and video library.
You are given numbered items retrieved by a search. Each has its date, the people Photos recognised in it, and text other models wrote about it (caption, tags, text found in the image, speech transcript).

Rules:
- Use ONLY the items and facts provided. You cannot see the photos themselves, only these descriptions.
- Cite the items you used like [1] or [2][5].
- If the items don't contain the answer, say so plainly and suggest what to search for instead. Never guess names, ages or dates that aren't given.
- For ages, only calculate from a birth date given in the facts plus an item's date, and show the arithmetic.
- Be concise: a few sentences, or a short list."""


def detect_filters(conn, question: str):
    q = question.lower()
    people = [r[0] for r in conn.execute("SELECT DISTINCT person_name FROM asset_people")]
    matched = []
    for name in people:
        parts = [name.lower()] + [p for p in name.lower().split() if len(p) >= 3]
        if any(re.search(rf"\b{re.escape(p)}\b", q) for p in parts):
            matched.append(name)
    years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", question)]
    return matched, years


def candidates(conn, people, years):
    if not people and not years:
        return None
    clauses, params = [], []
    if people:
        clauses.append(f"a.uuid IN (SELECT asset_id FROM asset_people WHERE person_name IN ({','.join('?' * len(people))}))")
        params += people
    if years:
        clauses.append(f"CAST(strftime('%Y', a.date_created, 'unixepoch') AS INT) IN ({','.join('?' * len(years))})")
        params += years
    rows = conn.execute(f"SELECT a.uuid FROM assets a WHERE a.status != 'failed' AND {' AND '.join(clauses)}", params)
    return {r[0] for r in rows}


def item_context(conn, uuid, n):
    r = conn.execute("""
        SELECT a.kind, a.date_created, l.tz_offset, c.caption,
               (SELECT group_concat(tag, ', ') FROM tags t WHERE t.asset_id=a.uuid) AS tags,
               v.ocr_text, tr.text AS transcript,
               (SELECT group_concat(person_name, ', ') FROM asset_people p WHERE p.asset_id=a.uuid) AS people
        FROM assets a JOIN lab_assets l ON l.uuid=a.uuid
        LEFT JOIN captions c ON c.asset_id=a.uuid
        LEFT JOIN vlm_results v ON v.asset_id=a.uuid AND v.model='qwen3-vl-30b'
        LEFT JOIN transcripts tr ON tr.asset_id=a.uuid
        WHERE a.uuid=?""", (uuid,)).fetchone()
    date = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r["date_created"] + (r["tz_offset"] or 0)))
    lines = [f"[{n}] {r['kind']} taken {date}"]
    lines.append(f"    people: {r['people'] or 'none named'}")
    if r["caption"]:
        lines.append(f"    caption: {r['caption']}")
    if r["tags"]:
        lines.append(f"    tags: {r['tags']}")
    if r["ocr_text"]:
        lines.append(f"    text in image: {r['ocr_text'][:300]}")
    if r["transcript"]:
        lines.append(f"    speech: {r['transcript'][:500]}")
    return "\n".join(lines)


def answer(conn, question, embed_query, vec_index, facts="", model="8b", k=20):
    t0 = time.time()
    people, years = detect_filters(conn, question)
    cand = candidates(conn, people, years)
    total = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    _, kw = searchlab.keyword(conn, question, limit=total)
    vh = vec_index().search(embed_query(question), limit=total)
    if cand is not None:
        kw = [h for h in kw if h["id"] in cand]
        vh = [h for h in vh if h["id"] in cand]
    fused = searchlab.hybrid(kw, vh, limit=k)
    t_retrieve = time.time() - t0
    ids = [h["id"] for h in fused]
    context = "\n\n".join(item_context(conn, u, i) for i, u in enumerate(ids, 1))
    user = (f"Facts from the user:\n{facts.strip()}\n\n" if facts.strip() else "") + \
           f"Retrieved items:\n{context or '(no items matched)'}\n\nQuestion: {question}"
    t1 = time.time()
    body = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
        "model": MODELS[model], "stream": False, "keep_alive": "10m",
        # Each item is ~100-400 tokens of text, so the context window has
        # to grow with k (a bigger window also takes more memory).
        "options": {"temperature": 0.2, "num_ctx": 8192 if k <= 15 else 16384 if k <= 30 else 32768,
                    "num_predict": 600, "seed": 1},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    }).json()
    text = body.get("message", {}).get("content", "") or body.get("error", "")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(ids)})
    return {"question": question, "answer": text, "model": MODELS[model],
            "filters": {"people": people, "years": years,
                        "candidates": None if cand is None else len(cand)},
            "items": [{"n": i, "id": u, "score": h["score"], "parts": h["parts"]}
                      for i, (u, h) in enumerate(zip(ids, fused), 1)],
            "cited": cited, "prompt": {"system": SYSTEM, "user": user},
            "timing": {"retrieve_s": t_retrieve, "generate_s": time.time() - t1,
                       "load_s": body.get("load_duration", 0) / 1e9,
                       "prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count")}}
