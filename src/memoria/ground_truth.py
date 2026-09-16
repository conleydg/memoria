"""Ground truth collection for the retrieval eval (ADR-0006): runs
queries through Photos' own native search (already confirmed to work
well on object/scene terms) and records which assets come back as
cheap "silver" ground truth.

Photos' search returns asset references - an id and a filename - never
image content, so nothing in this file ever touches pixel data. The
AppleScript call (`run_photos_search`) is the one real I/O boundary,
kept thin and swappable (`collect_ground_truth` takes it as an injected
argument) so the parsing and aggregation logic can be fully tested
against fake AppleScript output, without a real Photos library.
See tests/test_ground_truth.py.

Output is personal data - which of the library's real photos match
which query - and must be saved only under the gitignored data/
directory, never committed.
"""

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    uuid: str
    filename: str


def run_photos_search(query: str) -> str:
    """The one real I/O call - runs `query` through Photos' own native
    search via AppleScript, returns its raw tab/newline-delimited text
    output (uuid<TAB>filename per line). Never touches image content -
    Photos hands back asset references, not pixels."""
    script = f'''
    tell application "Photos"
        set searchResults to search for "{_escape_applescript_string(query)}"
        set output to ""
        repeat with anItem in searchResults
            set output to output & (id of anItem) & "\t" & (filename of anItem) & "\n"
        end repeat
        return output
    end tell
    '''
    result = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, check=True
    )
    return result.stdout


def _escape_applescript_string(s: str) -> str:
    """Backslashes first, then quotes - reversing the order would
    double-escape the backslashes just inserted for the quotes."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def parse_search_output(raw_output: str) -> list[Match]:
    """Pure parsing logic, no I/O - turns run_photos_search's raw text
    into structured matches. A malformed line is skipped, not fatal -
    one bad line shouldn't crash an entire collection run across
    dozens of queries."""
    matches = []
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        uuid, filename = parts
        matches.append(Match(uuid=uuid, filename=filename))
    return matches


def collect_ground_truth(
    queries: list[str], search_fn=run_photos_search
) -> dict[str, list[Match]]:
    """Runs every query in `queries` through `search_fn` (defaults to
    the real AppleScript call; tests inject a fake one) and returns
    {query: [Match, ...]}. A query that legitimately returns nothing
    (ADR-0026's zero-result case) is recorded as an empty list, not
    skipped - "no results" is itself real, meaningful ground truth,
    not a failure to collect."""
    return {query: parse_search_output(search_fn(query)) for query in queries}


def save_ground_truth(ground_truth: dict[str, list[Match]], path: str) -> None:
    serializable = {
        query: [{"uuid": m.uuid, "filename": m.filename} for m in matches]
        for query, matches in ground_truth.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_ground_truth(path: str) -> dict[str, list[Match]]:
    with open(path) as f:
        raw = json.load(f)
    return {
        query: [Match(uuid=m["uuid"], filename=m["filename"]) for m in matches]
        for query, matches in raw.items()
    }
