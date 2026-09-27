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

Transfers are slower than inference on some files (a big video over
Wi-Fi) and faster on others, so the batch job shouldn't fetch one file,
index it, then fetch the next. `RemoteLibrary.prefetch` keeps a window
of upcoming originals downloading on background threads while the
models work on the current one. Files in that window are pinned so the
cache's eviction never deletes a file the pipeline hasn't reached yet.
"""

import os
import shlex
import sqlite3
import subprocess
import tempfile
import threading
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
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


@dataclass(frozen=True)
class Prefetched:
    """One item from `RemoteLibrary.prefetch`. Exactly one of `path` and
    `error` is set: a failed fetch is handed to the caller rather than
    raised, so the batch job can mark that asset failed (ADR-0024) and
    carry on with the rest."""
    relpath: str
    path: Path | None
    error: Exception | None = None


@dataclass
class RemoteLibrary:
    host: str                    # an ~/.ssh/config alias, e.g. "photos-mac"
    library_path: str            # remote path to the .photoslibrary bundle
    cache_dir: Path
    max_cache_bytes: int
    run: Callable = field(default=subprocess.run, repr=False)
    # Files in use or waiting to be used, which eviction must skip. A
    # count, since the same file can be pinned by more than one caller.
    _pinned: Counter = field(default_factory=Counter, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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

    def fetch(self, relpath: str, pin: bool = False) -> Path:
        """Local path to the original at `relpath`, downloading it only if
        it isn't cached already. A cache hit refreshes its recency.

        With `pin=True` the file stays safe from eviction until `unpin` -
        used by `prefetch` for files fetched ahead of the pipeline."""
        local = self.cache_dir / relpath
        # Pinned before anything else, so a concurrent fetch's eviction
        # can't delete it between the download and the caller using it.
        with self._lock:
            self._pinned[local] += 1
        try:
            if local.exists():
                os.utime(local)
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                self._download(f"{self.library_path}/{relpath}", local)
                with self._lock:
                    evict_to_budget(self.cache_dir, self.max_cache_bytes, keep=set(self._pinned))
        except BaseException:
            self.unpin(local)
            raise
        if not pin:
            self.unpin(local)
        return local

    def unpin(self, path: Path) -> None:
        with self._lock:
            self._pinned[path] -= 1
            if self._pinned[path] <= 0:
                del self._pinned[path]

    def prefetch(
        self, relpaths: Iterable[str], workers: int = 4, ahead: int = 16
    ) -> Iterator[Prefetched]:
        """Yields every original in `relpaths`, in order, while up to
        `ahead` of the next ones download on `workers` background threads.

        Each yielded file stays pinned until the caller asks for the next
        one, so it's safe to use for as long as the loop body runs.
        Pinned files are kept even past `max_cache_bytes`, so the budget
        should be sized for roughly `ahead` of the largest originals.
        Stopping early (a `break`) cancels what hasn't started and
        unpins everything this call pinned."""
        it = iter(relpaths)
        pending = deque()
        with ThreadPoolExecutor(max_workers=workers) as pool:

            def top_up():
                while len(pending) < ahead:
                    relpath = next(it, None)
                    if relpath is None:
                        return
                    pending.append((relpath, pool.submit(self.fetch, relpath, pin=True)))

            try:
                top_up()
                while pending:
                    relpath, future = pending.popleft()
                    top_up()
                    try:
                        path = future.result()
                    except Exception as e:
                        yield Prefetched(relpath, None, e)
                        continue
                    try:
                        yield Prefetched(relpath, path)
                    finally:
                        self.unpin(path)
            finally:
                for _, future in pending:
                    future.cancel()
                # Downloads already running finish on their own; wait for
                # them, then release their pins.
                for _, future in pending:
                    if not future.cancelled() and future.exception() is None:
                        self.unpin(future.result())

def evict_to_budget(cache_dir: Path, max_bytes: int, keep: Iterable[Path] = ()) -> list[Path]:
    """Deletes least-recently-used cached files until the cache fits in
    `max_bytes`. Files in `keep` (in use, or fetched ahead and not used
    yet) are never evicted, even if they alone are over budget. Returns
    the evicted paths. Only ever touches files inside memoria's own cache."""
    keep = set(keep)
    files = [p for p in Path(cache_dir).rglob("*") if p.is_file() and not p.name.startswith(".partial-")]
    total = sum(p.stat().st_size for p in files)
    evicted = []
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= max_bytes:
            break
        if p in keep:
            continue
        total -= p.stat().st_size
        p.unlink()
        evicted.append(p)
    return evicted
