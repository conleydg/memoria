# ADR-0027: Fetch originals on demand from another Mac on the local network

**Status:** Accepted (2026-09-26). Amends [ADR-0017](0017-require-full-local-originals.md).

## Context

ADR-0017 required every original to sit on the indexing machine's own disk. Now that the Mac Studio is here, that turns out to be uncomfortable: the full library (~750GB) doesn't fit alongside everything else. Another Mac in the house already holds the complete library with "Download Originals to this Mac" on. ADR-0017 rejected on-demand fetching *from iCloud*, because that brings in an internet dependency and a third-party service. Fetching from a machine the owner controls, on the same network, has neither problem.

## Decision

The indexing machine may read the library from another Mac on the local network, over SSH (macOS Remote Login, key-based auth set up by the user, referenced by a `~/.ssh/config` host alias). `src/memoria/remote_library.py`:

- pulls a snapshot of the remote `Photos.sqlite`, made on the remote side with `sqlite3 -readonly ... .backup`, so memoria never reads a live database over the network mid-write. The snapshot is then read locally exactly like a local Photos.sqlite (ADR-0001's read-only connection);
- fetches individual originals as the pipeline needs them, from `originals/<ZDIRECTORY>/<ZFILENAME>` inside the bundle, into a size-bounded local cache that evicts least-recently-used files.

The underlying requirement from ADR-0017 still holds: indexing must see full-resolution originals, never iCloud proxies. What changes is where they may live: on the indexing Mac, or on another Mac on the same network that has originals downloaded.

## Consequences

- No photo data goes to a third party, and no internet connection is needed; the only network hop is inside the home network (ADR-0001 unchanged).
- The other Mac must be awake and reachable for indexing to make progress. A fetch failure raises; the batch job should mark that asset `failed` with the reason (ADR-0024) and move on, not stop the run.
- Change detection (ADR-0018) runs against the latest snapshot, so it's only as fresh as the last `snapshot_photos_db` call; the watcher should take a new snapshot each cycle.
- AppleScript-driven pieces (the ground-truth collector) still talk to Photos.app on whichever machine runs them. They only read asset references, never pixels, so they work against an iCloud-synced library on the Studio too.
- Not yet verified against the real library: the `originals/<ZDIRECTORY>/<ZFILENAME>` layout (Photos 5+), and that the remote `sqlite3` accepts `-readonly` on the installed macOS version. Both should be checked on a handful of assets before the first full run.
- The models shouldn't sit idle waiting on the network. `RemoteLibrary.prefetch` keeps a window of upcoming originals (16 by default) downloading on background threads while the current one is being indexed. Files in that window are pinned, so eviction never deletes a file the pipeline hasn't reached. A failed fetch comes back as a result rather than stopping the run.
- Pinned files may push the cache over its budget. Size the budget for roughly the prefetch window times the largest originals: a window full of long videos needs far more room than one full of photos.

## Alternatives considered

- **Mount the other Mac's library over SMB and read it in place:** rejected. SQLite over a network filesystem, against a database Photos is actively writing, gives unreliable reads and locking; the remote `.backup` avoids that entirely.
- **Copy the whole library once:** rejected; it's the disk-space problem this ADR exists to solve, and the copy goes stale.
- **rsync instead of `ssh cat`:** not needed for one-file-at-a-time transfers, and macOS's bundled rsync differs across versions in how it handles the space in "Photos Library.photoslibrary". Worth revisiting if transfers ever need resuming.
