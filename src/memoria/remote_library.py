"""On-demand access to a Photos library that lives on another Mac on the
same local network (ADR-0027, amending ADR-0017).

The indexing machine doesn't have room for every original, but another
Mac does, with "Download Originals to this Mac" on. Rather than copy
the whole library over, this pulls:

- a consistent snapshot of that Mac's Photos.sqlite (small), made on the
  remote side with sqlite3's own `.backup` so it's never a torn read of
  a live database mid-write, then read locally like any other
  Photos.sqlite (`db.connect_photos_readonly`);
- individual originals, one at a time as the pipeline needs them, into a
  size-bounded local cache that evicts least-recently-used files.

Transport is plain SSH (macOS "Remote Login") to a host alias the user
has already set up with key auth in ~/.ssh/config. Nothing here stores
or prompts for credentials, and nothing leaves the local network - no
third-party service is involved (ADR-0001). The remote library is only
ever read (`sqlite3 -readonly`, `cat`), never written.

`run` (the one real I/O boundary, `subprocess.run` by default) is
injected so every test drives a fake SSH - no test ever opens a network
connection. See tests/test_remote_library.py.
"""

import os
import shlex
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Where the remote snapshot is staged before it's pulled over; removed
# again right after.
REMOTE_SNAPSHOT_PATH = "/tmp/memoria-photos-snapshot.sqlite"


def original_relpath(directory: str, filename: str) -> str:
    """Path of an original inside the .photoslibrary bundle, from
    ZASSET.ZDIRECTORY and ZFILENAME (Photos 5+ layout:
    originals/<first hex char of the UUID>/<UUID>.<ext>). To be checked
    against the real library before the first full run - see ADR-0027."""
    return f"originals/{directory}/{filename}"


def original_relpaths(photos_conn: sqlite3.Connection) -> dict[str, str]:
    """{ZUUID: relpath} for every asset in a Photos.sqlite snapshot.
    Rows missing a directory or filename are skipped - there's nothing
    to fetch for them."""
    rows = photos_conn.execute(
        "SELECT ZUUID, ZDIRECTORY, ZFILENAME FROM ZASSET "
        "WHERE ZDIRECTORY IS NOT NULL AND ZFILENAME IS NOT NULL"
    )
    return {uuid: original_relpath(d, f) for uuid, d, f in rows}


@dataclass
class RemoteLibrary:
    host: str                    # an ~/.ssh/config alias, e.g. "photos-mac"
    library_path: str            # remote path to the .photoslibrary bundle
    cache_dir: Path
    max_cache_bytes: int
    run: Callable = field(default=subprocess.run, repr=False)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ssh(self, remote_command: str, **kwargs):
        return self.run(["ssh", self.host, remote_command], check=True, **kwargs)

    def _download(self, remote_path: str, dest: Path) -> None:
        """Streams one remote file to `dest`, via a temp file in the same
        directory so an interrupted transfer never leaves a truncated
        file where a complete one is expected."""
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".partial-")
        try:
            with os.fdopen(fd, "wb") as out:
                self._ssh(f"cat {shlex.quote(remote_path)}", stdout=out)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def snapshot_photos_db(self, dest: Path) -> Path:
        """Pulls a consistent copy of the remote Photos.sqlite to `dest`."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        db = f"{self.library_path}/database/Photos.sqlite"
        backup = f".backup {REMOTE_SNAPSHOT_PATH}"
        self._ssh(f"sqlite3 -readonly {shlex.quote(db)} {shlex.quote(backup)}")
        try:
            self._download(REMOTE_SNAPSHOT_PATH, dest)
        finally:
            self._ssh(f"rm -f {shlex.quote(REMOTE_SNAPSHOT_PATH)}")
        return dest

    def fetch(self, relpath: str) -> Path:
        """Local path to the original at `relpath`, downloading it only if
        it isn't cached already. A cache hit refreshes its recency."""
        local = self.cache_dir / relpath
        if local.exists():
            os.utime(local)
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        self._download(f"{self.library_path}/{relpath}", local)
        evict_to_budget(self.cache_dir, self.max_cache_bytes, keep=local)
        return local


def evict_to_budget(cache_dir: Path, max_bytes: int, keep: Path | None = None) -> list[Path]:
    """Deletes least-recently-used cached files until the cache fits in
    `max_bytes`. `keep` (the file just fetched) is never evicted, even if
    it alone is over budget - the caller is about to use it. Returns the
    evicted paths. Only ever touches files inside memoria's own cache."""
    files = [p for p in Path(cache_dir).rglob("*") if p.is_file() and not p.name.startswith(".partial-")]
    total = sum(p.stat().st_size for p in files)
    evicted = []
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= max_bytes:
            break
        if keep is not None and p == keep:
            continue
        total -= p.stat().st_size
        p.unlink()
        evicted.append(p)
    return evicted
