"""Unpack the Kinetics-700 clips the behavior set needs out of the 424 GB OpenDataLab archive.

The archive is four ~100 GiB pieces of ONE gzip stream, so there is no random access: getting a
single clip costs the same as getting all of them. Everything is therefore done in exactly one
pass — `cat part.* | tar -xz -T <the 24.5k members we want>` — and the ~625k clips we do not
want are decompressed and discarded rather than written. Measured 217 MB/s on this box, so the
pass is ~35 minutes regardless of how many members are selected.

Selection comes from `behavior_manifest.k700_rows(..., videos=False)`, which is also what the
cache builder and trainer use, so the three cannot disagree about which clips are in the study.

Re-running is safe and resumes: members already on disk are dropped from the list first. If the
list ends up empty the tar pass is skipped entirely.
"""
import argparse
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_manifest as BM                                   # noqa: E402

RAW = os.path.normpath(os.path.join(BM.ROOT, "..", "..",
                                    "OpenMMLab___Kinetics_700", "raw"))
SELECT = os.path.join(BM.MANIFESTS, "k700_select.txt")
MEMBER_PREFIX = "Kinetics_700/videos/"          # -> k700/videos/... after --strip-components=1


def selection():
    """-> [(archive_member, behavior, k700_class_slug, split)] over both splits."""
    out = []
    for split in ("train", "val"):
        for path, beh, slug, sp in BM.k700_rows(split, videos=False):
            rel = os.path.relpath(path, os.path.join(BM.K700_DIR, "videos"))
            out.append((MEMBER_PREFIX + rel, beh, slug, sp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW, help="directory holding Kinetics_700.tar.gz.000*")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the selection manifest and report, extract nothing")
    a = ap.parse_args()

    sel = selection()
    os.makedirs(BM.MANIFESTS, exist_ok=True)
    with open(SELECT, "w") as fh:
        for member, beh, slug, sp in sel:
            fh.write(f"{member}\t{beh}\t{slug}\t{sp}\n")
    c = Counter((sp, beh) for _, beh, _, sp in sel)
    for sp in ("train", "val"):
        print(f"[select] {sp:5s} n={sum(v for (s, _), v in c.items() if s == sp):6d}  " +
              "  ".join(f"{b}={c.get((sp, b), 0)}" for b in BM.BEHAVIORS), flush=True)
    print(f"[select] manifest -> {SELECT}", flush=True)

    dest = BM.K700_DIR
    todo = [m for m, _, _, _ in sel
            if not os.path.exists(os.path.join(dest, m[len("Kinetics_700/"):]))]
    print(f"[fetch] {len(sel) - len(todo)} already unpacked, {len(todo)} to go", flush=True)
    if a.dry_run or not todo:
        return

    parts = sorted(f for f in os.listdir(a.raw) if f.startswith("Kinetics_700.tar.gz."))
    assert parts, f"no archive parts under {a.raw}"
    print(f"[fetch] one streaming pass over {len(parts)} parts "
          f"({sum(os.path.getsize(os.path.join(a.raw, p)) for p in parts) / 1e9:.0f} GB), "
          f"~35 min", flush=True)

    listfile = os.path.join(BM.MANIFESTS, "k700_members.txt")
    with open(listfile, "w") as fh:
        fh.write("\n".join(todo) + "\n")
    os.makedirs(dest, exist_ok=True)
    # --strip-components=1 turns Kinetics_700/videos/vol_NN/x.mp4 into videos/vol_NN/x.mp4.
    # Members the archive does not contain (Kinetics has dead YouTube links) make tar exit 2
    # after listing them; that is expected, so the outcome is judged by counting files on disk.
    cmd = (f"cat {' '.join(os.path.join(a.raw, p) for p in parts)} | "
           f"tar -xzf - -C {dest} --strip-components=1 -T {listfile}")
    rc = subprocess.call(["bash", "-c", cmd])

    have = [m for m, _, _, _ in sel
            if os.path.exists(os.path.join(dest, m[len("Kinetics_700/"):]))]
    miss = len(sel) - len(have)
    print(f"[fetch] tar rc={rc}  unpacked={len(have)}/{len(sel)}  missing={miss}", flush=True)
    if miss:
        print(f"[fetch] {miss} selected clips are not in the archive (dead source links); "
              f"they are simply absent from the study", flush=True)


if __name__ == "__main__":
    main()
