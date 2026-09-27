"""Tests for on-demand fetching from a Photos library on another Mac
(ADR-0027) - entirely against a fake SSH runner. No test opens a network
connection or touches a real Photos library: `run` is the one real I/O
boundary, and it's injected below.
"""

import os
import shlex
import sqlite3
import threading

import pytest

from memoria.remote_library import (
    REMOTE_SNAPSHOT_PATH,
    Prefetched,
    RemoteLibrary,
    evict_to_budget,
    original_relpath,
    original_relpaths,
)

LIBRARY = "/Users/me/Pictures/Photos Library.photoslibrary"


class FakeSSH:
    """Stands in for subprocess.run(["ssh", host, cmd], ...). Serves
    `cat` from an in-memory {remote_path: bytes} dict and records every
    remote command it was asked to run."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.commands = []

    def __call__(self, argv, check=False, stdout=None, **kwargs):
        assert argv[0] == "ssh"
        cmd = argv[2]
        self.commands.append(cmd)
        words = shlex.split(cmd)
        if words[0] == "cat":
            if words[1] not in self.files:
                raise FileNotFoundError(words[1])
            stdout.write(self.files[words[1]])
        elif words[0] == "sqlite3":
            self.files[REMOTE_SNAPSHOT_PATH] = b"snapshot-bytes"
        elif words[:2] == ["rm", "-f"]:
            self.files.pop(words[2], None)


def make_library(tmp_path, files=None, max_cache_bytes=1_000_000):
    ssh = FakeSSH(files)
    lib = RemoteLibrary(
        host="photos-mac",
        library_path=LIBRARY,
        cache_dir=tmp_path / "cache",
        max_cache_bytes=max_cache_bytes,
        run=ssh,
    )
    return lib, ssh


class TestOriginalRelpath:
    def test_follows_photos_originals_layout(self):
        assert original_relpath("A", "A1B2.heic") == "originals/A/A1B2.heic"

    def test_reads_every_asset_from_a_photos_snapshot(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE ZASSET (ZUUID TEXT, ZDIRECTORY TEXT, ZFILENAME TEXT)")
        conn.executemany(
            "INSERT INTO ZASSET VALUES (?, ?, ?)",
            [("u1", "A", "A1.heic"), ("u2", "0", "02.mov"), ("u3", None, None)],
        )
        assert original_relpaths(conn) == {
            "u1": "originals/A/A1.heic",
            "u2": "originals/0/02.mov",
        }


class TestFetch:
    def test_downloads_an_original_into_the_cache(self, tmp_path):
        lib, _ = make_library(tmp_path, {f"{LIBRARY}/originals/A/A1.heic": b"pixels"})
        local = lib.fetch("originals/A/A1.heic")
        assert local == tmp_path / "cache" / "originals/A/A1.heic"
        assert local.read_bytes() == b"pixels"

    def test_library_path_with_spaces_is_quoted_for_the_remote_shell(self, tmp_path):
        lib, ssh = make_library(tmp_path, {f"{LIBRARY}/originals/A/A1.heic": b"x"})
        lib.fetch("originals/A/A1.heic")
        assert shlex.split(ssh.commands[0]) == ["cat", f"{LIBRARY}/originals/A/A1.heic"]

    def test_cache_hit_does_not_touch_the_network(self, tmp_path):
        lib, ssh = make_library(tmp_path, {f"{LIBRARY}/originals/A/A1.heic": b"x"})
        lib.fetch("originals/A/A1.heic")
        lib.fetch("originals/A/A1.heic")
        assert len(ssh.commands) == 1

    def test_failed_transfer_leaves_nothing_in_the_cache(self, tmp_path):
        lib, _ = make_library(tmp_path)
        with pytest.raises(FileNotFoundError):
            lib.fetch("originals/A/missing.heic")
        assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []

    def test_fetching_past_the_budget_evicts_the_oldest(self, tmp_path):
        files = {f"{LIBRARY}/originals/{d}/{d}.heic": b"x" * 10 for d in "ABC"}
        lib, _ = make_library(tmp_path, files, max_cache_bytes=25)
        a = lib.fetch("originals/A/A.heic")
        os.utime(a, (1, 1))
        b = lib.fetch("originals/B/B.heic")
        os.utime(b, (2, 2))
        c = lib.fetch("originals/C/C.heic")
        assert not a.exists()
        assert b.exists() and c.exists()


class TestEvictToBudget:
    def test_under_budget_evicts_nothing(self, tmp_path):
        (tmp_path / "a").write_bytes(b"x" * 10)
        assert evict_to_budget(tmp_path, 100) == []

    def test_file_just_fetched_is_kept_even_if_alone_over_budget(self, tmp_path):
        big = tmp_path / "big.mov"
        big.write_bytes(b"x" * 50)
        old = tmp_path / "old.heic"
        old.write_bytes(b"x" * 10)
        os.utime(old, (1, 1))
        os.utime(big, (0, 0))  # oldest, but it's the one in use
        evicted = evict_to_budget(tmp_path, 20, keep={big})
        assert evicted == [old]
        assert big.exists()

    def test_in_progress_partial_downloads_are_left_alone(self, tmp_path):
        partial = tmp_path / ".partial-abc"
        partial.write_bytes(b"x" * 50)
        assert evict_to_budget(tmp_path, 0) == []
        assert partial.exists()


class TestSnapshotPhotosDb:
    def test_backs_up_remotely_then_pulls_and_cleans_up(self, tmp_path):
        lib, ssh = make_library(tmp_path)
        dest = lib.snapshot_photos_db(tmp_path / "snap" / "Photos.sqlite")
        assert dest.read_bytes() == b"snapshot-bytes"
        backup, pull, cleanup = (shlex.split(c) for c in ssh.commands)
        assert backup == [
            "sqlite3", "-readonly",
            f"{LIBRARY}/database/Photos.sqlite",
            f".backup {REMOTE_SNAPSHOT_PATH}",
        ]
        assert pull == ["cat", REMOTE_SNAPSHOT_PATH]
        assert cleanup == ["rm", "-f", REMOTE_SNAPSHOT_PATH]

    def test_remote_snapshot_is_cleaned_up_even_if_the_pull_fails(self, tmp_path):
        lib, ssh = make_library(tmp_path)

        def failing(argv, check=False, stdout=None, **kwargs):
            if argv[2].startswith("cat"):
                raise ConnectionError("dropped")
            return ssh(argv, check=check, stdout=stdout)

        lib.run = failing
        with pytest.raises(ConnectionError):
            lib.snapshot_photos_db(tmp_path / "Photos.sqlite")
        assert shlex.split(ssh.commands[-1]) == ["rm", "-f", REMOTE_SNAPSHOT_PATH]


def originals(n, size=10):
    return {f"{LIBRARY}/originals/{i % 10}/{i}.heic": bytes([i % 256]) * size for i in range(n)}


def relpaths(n):
    return [f"originals/{i % 10}/{i}.heic" for i in range(n)]


class TestPrefetch:
    def test_yields_every_original_in_order(self, tmp_path):
        lib, _ = make_library(tmp_path, originals(20))
        got = [(item.relpath, item.path.read_bytes()) for item in lib.prefetch(relpaths(20))]
        assert got == [(r, bytes([i]) * 10) for i, r in enumerate(relpaths(20))]

    def test_later_files_download_while_the_first_is_still_in_flight(self, tmp_path):
        # The fake blocks the first transfer until the second one has
        # started - which only happens if transfers run in parallel.
        lib, ssh = make_library(tmp_path, originals(2))
        second_started = threading.Event()

        def run(argv, check=False, stdout=None, **kwargs):
            if "/1.heic" in argv[2]:
                second_started.set()
            elif not second_started.wait(timeout=5):
                raise AssertionError("second download never started")
            return ssh(argv, check=check, stdout=stdout)

        lib.run = run
        assert [i.error for i in lib.prefetch(relpaths(2), workers=2)] == [None, None]

    def test_downloads_run_ahead_of_the_consumer(self, tmp_path):
        lib, ssh = make_library(tmp_path, originals(10))
        items = lib.prefetch(relpaths(10), workers=2, ahead=4)
        next(items)
        # Everything in the window is queued the moment the first item is
        # handed over; give the workers a moment to finish it.
        for _ in range(100):
            if len(ssh.commands) >= 5:
                break
            threading.Event().wait(0.01)
        assert len(ssh.commands) == 5  # the first, plus 4 ahead - no more
        items.close()

    def test_a_failed_fetch_is_reported_and_the_rest_continue(self, tmp_path):
        files = originals(3)
        del files[f"{LIBRARY}/originals/1/1.heic"]
        lib, _ = make_library(tmp_path, files)
        items = list(lib.prefetch(relpaths(3)))
        assert [i.path is not None for i in items] == [True, False, True]
        assert isinstance(items[1].error, FileNotFoundError)

    def test_files_fetched_ahead_are_never_evicted_before_use(self, tmp_path):
        # Budget fits one file, but the window holds up to eight: files
        # waiting their turn must survive the other downloads' eviction.
        lib, _ = make_library(tmp_path, originals(20), max_cache_bytes=10)
        for item in lib.prefetch(relpaths(20), workers=4, ahead=8):
            assert item.path.exists()

    def test_used_files_become_evictable_again(self, tmp_path):
        lib, _ = make_library(tmp_path, originals(20), max_cache_bytes=30)
        list(lib.prefetch(relpaths(20)))
        assert lib._pinned == {}
        lib.fetch("originals/0/0.heic")
        cached = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
        assert sum(p.stat().st_size for p in cached) <= 30

    def test_stopping_early_releases_every_pin(self, tmp_path):
        lib, _ = make_library(tmp_path, originals(20))
        for i, item in enumerate(lib.prefetch(relpaths(20), ahead=8)):
            if i == 2:
                break
        assert lib._pinned == {}

    def test_empty_input_yields_nothing(self, tmp_path):
        lib, ssh = make_library(tmp_path)
        assert list(lib.prefetch([])) == []
        assert ssh.commands == []


class TestFetchPinning:
    def test_plain_fetch_leaves_nothing_pinned(self, tmp_path):
        lib, _ = make_library(tmp_path, originals(1))
        lib.fetch("originals/0/0.heic")
        assert lib._pinned == {}

    def test_pinned_file_survives_eviction_until_unpinned(self, tmp_path):
        lib, _ = make_library(tmp_path, originals(3), max_cache_bytes=10)
        first = lib.fetch("originals/0/0.heic", pin=True)
        os.utime(first, (1, 1))
        lib.fetch("originals/1/1.heic")
        assert first.exists()
        lib.unpin(first)
        lib.fetch("originals/2/2.heic")
        assert not first.exists()
