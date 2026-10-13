"""Fetch only the HMDB51 classes needed for the 9-behavior label set.

Source: the `divm/hmdb51` dataset repo, which stores HMDB51 as individual mp4 files under
train/ test/ validation/ with the class encoded as the filename prefix. huggingface.co is
unreachable from this network, so everything goes through the hf-mirror.com mirror.

Layout written:  behavior_data/hmdb/<split>/<hmdb_class>/<file>.mp4
The HMDB class is kept (not the target behavior) so the source->target mapping stays a
pure data-side decision in behavior_manifest.py, and so the chase-curation pool
(run/walk) can be re-filtered without re-downloading.
"""
import argparse
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = "https://hf-mirror.com"
REPO = "divm/hmdb51"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "behavior_data")
ROOT = os.path.normpath(ROOT)
HMDB_DIR = os.path.join(ROOT, "hmdb")
LISTING = os.path.join(ROOT, "manifests", "hmdb51_listing.json")

# every HMDB51 class, longest-first so prefix matching is unambiguous
# (kick_ball before kick, climb_stairs before climb, sword_exercise before sword, ...)
HMDB_CLASSES = sorted([
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive",
    "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf",
    "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour",
    "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands",
    "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
    "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk",
    "wave"], key=len, reverse=True)

# classes carrying a target behavior -> downloaded in full
BEHAVIOR_SOURCES = ["wave", "throw", "fall_floor", "punch", "kick", "hit", "talk", "drink",
                    "pick", "shake_hands"]
# candidate pool the chase curator filters for 2+ people in pursuit -> downloaded in full
CHASE_POOL = ["run", "walk"]
# negatives for the `other` class -> subsampled, they only need to be diverse
OTHER_POOL = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs",
              "dive", "draw_sword", "dribble", "eat", "fencing", "flic_flac", "golf",
              "handstand", "hug", "jump", "kick_ball", "kiss", "laugh", "pour", "pullup",
              "push", "pushup", "ride_bike", "ride_horse", "shoot_ball", "shoot_bow",
              "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand",
              "swing_baseball", "sword", "sword_exercise", "turn"]
OTHER_PER_CLASS = 22   # 39 classes x 22 ~= 860 negatives, comparable to the largest target

# the mirror 403s requests without a browser-ish UA
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) darkpipe-behavior-fetch/1.0"}


def _open(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def hmdb_class_of(filename):
    for c in HMDB_CLASSES:
        if filename.startswith(c + "_"):
            return c
    return None


def load_listing():
    if os.path.exists(LISTING):
        return json.load(open(LISTING))
    url = f"{ENDPOINT}/api/datasets/{REPO}"
    print(f"[listing] {url}", flush=True)
    with _open(url, 120) as r:
        meta = json.load(r)
    files = [f["rfilename"] for f in meta.get("siblings", [])
             if f["rfilename"].endswith(".mp4")]
    os.makedirs(os.path.dirname(LISTING), exist_ok=True)
    json.dump(files, open(LISTING, "w"))
    print(f"[listing] {len(files)} mp4 entries -> {LISTING}", flush=True)
    return files


def fetch_one(rel):
    """rel = '<split>/<file>.mp4' in the repo -> local '<split>/<class>/<file>.mp4'."""
    split, base = rel.split("/", 1)
    cls = hmdb_class_of(base)
    out = os.path.join(HMDB_DIR, split, cls, base)
    if os.path.exists(out) and os.path.getsize(out) > 1024:
        return "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/{urllib.parse.quote(rel)}"
    tmp = out + ".part"
    for attempt in range(4):
        try:
            with _open(url, 90) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, out)
            return "ok"
        except Exception as e:
            if attempt == 3:
                print(f"  [fail] {rel}: {e}", flush=True)
                return "fail"
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = load_listing()
    by_class = {}
    for rel in files:
        cls = hmdb_class_of(rel.split("/", 1)[1])
        if cls is None:
            continue
        by_class.setdefault(cls, []).append(rel)

    wanted = []
    for c in BEHAVIOR_SOURCES + CHASE_POOL:
        wanted += by_class.get(c, [])
    rng = random.Random(args.seed)
    for c in OTHER_POOL:
        pool = sorted(by_class.get(c, []))
        rng.shuffle(pool)
        wanted += pool[:OTHER_PER_CLASS]

    print(f"[fetch] {len(wanted)} clips "
          f"(behavior={sum(len(by_class.get(c, [])) for c in BEHAVIOR_SOURCES)}, "
          f"chase_pool={sum(len(by_class.get(c, [])) for c in CHASE_POOL)}, "
          f"other={len(wanted) - sum(len(by_class.get(c, [])) for c in BEHAVIOR_SOURCES + CHASE_POOL)})",
          flush=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(fetch_one, wanted), 1):
            counts[res] += 1
            if i % 200 == 0:
                print(f"  {i}/{len(wanted)}  {counts}", flush=True)
    print(f"[fetch] done {counts} -> {HMDB_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
