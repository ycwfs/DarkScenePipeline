"""HMDB51 (+ARID, +Kinetics-700) -> the 10-label behavior set. Single source of truth here.

The mapping is data-side on purpose: `hmdb_fetch.py` stores clips under their HMDB class, so
re-deciding a mapping (or re-running chase curation) costs nothing.

`chase` is the exception — HMDB has no chase class, so its clips come from `chase_keep.txt`,
produced by `chase_curate.py` out of the run/walk pool. Those are WEAK labels and are reported
separately, never blended into the headline number.

Kinetics-700 is the third source and the reason the rare classes are trainable at all: HMDB
gives `chase` 28 weak clips, K700 gives it 550 labelled ones. K700 ships its labels as numeric
ids in MMAction2 list files, so the id -> name map (`label_map_k700.txt`, alphabetical,
id = line - 1) is stored next to them rather than fetched at run time.
"""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                     "behavior_data"))
HMDB_DIR = os.path.join(ROOT, "hmdb")
K700_DIR = os.path.join(ROOT, "k700")
DARKACT_DIR = os.path.normpath(os.path.join(ROOT, "..", "..", "darknet", "ARIM_v1"))
MANIFESTS = os.path.join(ROOT, "manifests")
CHASE_KEEP = os.path.join(MANIFESTS, "chase_keep.txt")
CHASE_DROP = os.path.join(MANIFESTS, "chase_ambiguous.txt")
K700_LABELS = os.path.join(MANIFESTS, "label_map_k700.txt")
K700_LISTS = {"train": os.path.join(K700_DIR, "kinetics700_train_list_videos.txt"),
              "val": os.path.join(K700_DIR, "kinetics700_val_list_videos.txt")}
# The `1` suffix matters: these are the 9040/3738 split the DarkAct README documents. The
# unsuffixed files next to them are an older 8769/4009 split.
DARKACT_LISTS = {"train": os.path.join(DARKACT_DIR, "annotations",
                                       "_rgb_train1_annotations.txt"),
                 "test": os.path.join(DARKACT_DIR, "annotations",
                                      "_rgb_test1_annotations.txt")}

BEHAVIORS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
             "shake_hands", "other"]
BID = {b: i for i, b in enumerate(BEHAVIORS)}

# HMDB class -> behavior. fight merges HMDB's three separate assault classes; everything not
# listed here (and not in the chase pool) is a negative and becomes `other`.
HMDB_TO_BEHAVIOR = {
    "wave": "wave", "throw": "throw", "fall_floor": "fall",
    "punch": "fight", "kick": "fight", "hit": "fight",
    "talk": "talk", "drink": "drink", "pick": "pick_up", "shake_hands": "shake_hands",
}
CHASE_POOL = ("run", "walk")

# ARID classes usable as real-dark supervision (the only overlap with the behavior set)
ARID_TO_BEHAVIOR = {"Drink": "drink", "Pick": "pick_up", "Wave": "wave",
                    "Jump": "other", "Pour": "other", "Push": "other", "Sit": "other",
                    "Stand": "other", "Turn": "other", "Walk": "other", "Run": "other"}

# Kinetics-700 class -> behavior. Every one of the nine behaviors has at least one named K700
# class, which is the whole reason for adding the dataset.
#
# Two exclusions worth stating, because both are plausible-looking and both are wrong:
#   * `pick_up` takes `person collecting garbage`, NOT `picking apples`/`picking blueberries`.
#     Collecting garbage is bend-down-and-lift; orchard picking is reaching upward.
#   * `throw` takes the everyday throws only. `hammer throw`, `javelin throw` and
#     `throwing discus` are whole-body athletic rotations that share almost nothing with
#     someone throwing an object in a corridor.
K700_TO_BEHAVIOR = {
    "waving hand": "wave",
    "throwing ball (not baseball or American football)": "throw",
    "throwing axe": "throw", "throwing knife": "throw", "throwing snowballs": "throw",
    "throwing water balloon": "throw", "card throwing": "throw",
    "chasing": "chase",
    "faceplanting": "fall", "falling off bike": "fall", "falling off chair": "fall",
    "slapping": "fight", "sword fighting": "fight", "pillow fight": "fight",
    "arm wrestling": "fight", "wrestling": "fight", "headbutting": "fight",
    "punching person (boxing)": "fight",
    "arguing": "talk", "talking on cell phone": "talk",
    "drinking shots": "drink", "sipping cup": "drink",
    "person collecting garbage": "pick_up",
    "shaking hands": "shake_hands",
}
# This mapping is what trained the shipped `videomamba_t_behavior_32f.pth`. Do not "tighten" it
# without new evidence -- that was tried and measured.
#
# fight and throw are the two broadest entries above (6197 + 4709 train clips, 49% of everything
# K700 adds), and on HMDB test they were also the two classes that dropped most against the
# HMDB+ARID-only checkpoint. The obvious inference is that the breadth caused the drop, so a
# second finetune ran with them cut to the visually-closest classes only -- throw to ball +
# snowballs, fight to slapping + headbutting + punching, the other eight excluded from the `other`
# pool as well. It did not help: HMDB macro-F1 went 0.5450 -> 0.5248 and ARID top-1 0.8761 ->
# 0.8722, both within noise of the broad mapping (McNemar p=0.44 and p=0.59).
#
# The reason the inference failed is that the per-class deltas it rested on were not real. HMDB
# test is 611 clips with 9-34 per class, so a 0.15 per-class F1 swing is three clips; in the
# tightened run `fall` and `shake_hands` moved just as far with their mappings untouched. A paired
# bootstrap puts every HMDB difference between the three checkpoints at p > 0.25. What does hold
# up is ARID top-1, +95/-53 clips, p=0.0007 -- K700 helps on real dark footage, and HMDB test is
# too small to say anything either way. Judge future mapping changes on ARID, not on HMDB per-class.
K700_EXCLUDE = set()                                   # kept as the hook; see the note above

# DarkAct (`darknet/ARIM_v1`) class -> behavior. Real low-light RGB at a median frame luma of
# 3.5/255, roughly 3x darker than ARID, so it is the only source that is NOT synthetically
# darkened *and* sits at the bottom of the exposure range the recognizer has to survive.
#
# Two entries are deliberately absent, and both look like obvious matches from the class name:
#   * `slap` is CLAPPING -- one person, hands meeting in front, no second person. The DarkAct
#     README lists the class as "clapping"; only the directory is named `slap`. Mapping it to
#     `fight` would teach applause as assault. Confirmed by rendering the thermal pair, which is
#     the only way to see the actor: the RGB is too dark and the subject is often a distant
#     silhouette of a few dozen pixels.
#   * `lift` is an OVERHEAD ARM RAISE, not bend-down-and-lift, so it is not `pick_up`. It stays
#     in `other`, where it doubles as a hard negative for `wave`.
# `pick` was checked the same way and IS bend-down-and-lift.
DARKACT_TO_BEHAVIOR = {"call": "talk", "wave": "wave", "drink": "drink", "pick": "pick_up"}
# DarkAct has no class for throw / chase / fall / fight / shake_hands. Those five get no new
# supervision from it and are watched for regression rather than improvement.
#
# `other` is capped for training only: 84% of DarkAct's train split is negatives, which uncapped
# would make it ~51% of the whole negative pool while supplying 10-25% of each positive class.
# Since DarkAct is also visibly the darkest source, that lopsidedness invites the shortcut
# "very dark -> other". 0 means uncapped -- the test split is never subsampled, because a capped
# test set is not the dataset's test set. The cap is a sorted prefix per source class (same rule
# as K700_OTHER_PER_CLASS) so it is reproducible, and raising it costs a retrain but no cache
# rebuild: `darkact_rows(cap=False)` is what the cache builder walks.
DARKACT_OTHER_CAP = {"train": 2000, "test": 0}
# K700 clips are a fixed 10 s where HMDB's median is 2.60 s and the live recognizer's window is
# ~1 s, so 32 frames spread over a whole K700 clip would be ~4x slower motion than the model ever
# sees. Every consumer (cache builder, evaluator) cuts the same contiguous window, and they must
# agree on its length or the evaluator scores frames the model was not trained on.
K700_SPAN_SEC = 3.0
# `other` is drawn from the ~600 classes that map to no behavior, a fixed number per class so the
# negative set is broad rather than deep. Deterministic (sorted prefix, no RNG) because the cache
# builder, the trainer and the evaluator each recompute this list independently and must agree.
K700_OTHER_PER_CLASS = {"train": 5, "val": 1}


def _read_list(path):
    if not os.path.exists(path):
        return set()
    return {ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")}


def hmdb_clips(split=None):
    """-> [(abs_path, hmdb_class, split)] for everything downloaded."""
    out = []
    for sp in sorted(os.listdir(HMDB_DIR)):
        if split and sp != split:
            continue
        d = os.path.join(HMDB_DIR, sp)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            for f in sorted(os.listdir(os.path.join(d, cls))):
                if f.endswith(".mp4"):
                    out.append((os.path.join(d, cls, f), cls, sp))
    return out


def label_of(path, cls, keep=None, ambiguous=None):
    """Behavior label for one HMDB clip, or None if it should be excluded.

    Chase pool: curated keeps -> `chase`; clips the curator flagged as multi-person but not
    clearly a pursuit -> None (excluded, they would poison `other`); the rest -> `other`.
    """
    if cls in HMDB_TO_BEHAVIOR:
        return HMDB_TO_BEHAVIOR[cls]
    if cls in CHASE_POOL:
        base = os.path.basename(path)
        if keep and base in keep:
            return "chase"
        if ambiguous and base in ambiguous:
            return None
        return "other"
    return "other"


def build(split=None, with_chase=True):
    """-> [(path, behavior, hmdb_class, split)] over the downloaded HMDB clips."""
    keep = _read_list(CHASE_KEEP) if with_chase else set()
    amb = _read_list(CHASE_DROP) if with_chase else set()
    rows = []
    for path, cls, sp in hmdb_clips(split):
        lab = label_of(path, cls, keep, amb)
        if lab is not None:
            rows.append((path, lab, cls, sp))
    return rows


def arid_rows(split="test"):
    """Real-dark ARID clips in the behavior vocabulary -> [(path, behavior, arid_class, split)].

    Same tuple shape as `build()`, so a caller can swap simulated-dark HMDB for real-dark ARID
    without changing anything else. Only drink / pick_up / wave / other are reachable — that is
    the whole overlap between ARID-11 and the behavior set.
    """
    root = os.path.join(ROOT, "..", "arid", f"{split}_clips")
    rows = []
    for cls in sorted(os.listdir(root)):
        if cls not in ARID_TO_BEHAVIOR:
            continue
        for f in sorted(os.listdir(os.path.join(root, cls))):
            if f.endswith(".mp4"):
                rows.append((os.path.join(root, cls, f), ARID_TO_BEHAVIOR[cls], cls, split))
    return rows


def darkact_rows(split="test", cap=True):
    """Real-dark DarkAct clips -> [(path, behavior, darkact_class, split)], as `build()`.

    The list files carry `RGB/<class>/<file>.mp4 <id>`; the class comes from the path, and the
    id (an index into `sorted(videos/RGB)`) is checked against it so a re-released split with a
    different label order fails loudly instead of silently mislabelling every clip.

    `cap=True` applies `DARKACT_OTHER_CAP` to the negatives — training only. The cache builder
    passes `cap=False` so raising the cap later needs no rebuild.
    """
    names = sorted(os.listdir(os.path.join(DARKACT_DIR, "videos", "RGB")))
    root = os.path.join(DARKACT_DIR, "videos")
    lim = DARKACT_OTHER_CAP.get(split, 0) if cap else 0
    rows, other_pool = [], {}
    for ln in open(DARKACT_LISTS[split]):
        ln = ln.strip()
        if not ln:
            continue
        rel, lid = ln.rsplit(" ", 1)
        cls = rel.split("/")[1]
        if names[int(lid)] != cls:
            raise ValueError(f"DarkAct label id {lid} is {names[int(lid)]!r}, path says {cls!r}")
        p = os.path.join(root, rel)
        beh = DARKACT_TO_BEHAVIOR.get(cls)
        if beh:
            rows.append((p, beh, cls, split))
        else:
            other_pool.setdefault(cls, []).append(p)

    neg = [(p, "other", cls, split) for cls in sorted(other_pool)
           for p in sorted(other_pool[cls])]
    if lim and len(neg) > lim:                         # round-robin over classes, then prefix:
        by_cls = {}                                    # a flat prefix would keep only `call`..
        for r in neg:
            by_cls.setdefault(r[2], []).append(r)
        neg, i = [], 0
        while len(neg) < lim:
            row = [by_cls[c][i] for c in sorted(by_cls) if i < len(by_cls[c])]
            if not row:
                break
            neg.extend(row[:lim - len(neg)])
            i += 1
    return rows + neg


def k700_names():
    """-> [class name], index == the label id used in the K700 list files."""
    return [ln.rstrip("\n") for ln in open(K700_LABELS)]


def k700_slug(name):
    """Class name -> a directory-safe token. Kept reversible enough to read at a glance."""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


def k700_rows(split="train", videos=True):
    """-> [(path, behavior, k700_class_slug, split)], same tuple shape as `build()`.

    `videos=False` returns rows for clips the archive has not been unpacked for yet — that is
    what `k700_fetch.py` needs to decide which tar members to extract. With `videos=True` (the
    default) rows whose mp4 is missing are dropped, so every caller downstream of the fetch
    sees only clips that actually exist.
    """
    names = k700_names()
    per_cls = K700_OTHER_PER_CLASS[split]
    wanted, other_pool = [], {}
    for ln in open(K700_LISTS[split]):
        ln = ln.strip()
        if not ln:
            continue
        rel, lid = ln.rsplit(" ", 1)
        name = names[int(lid)]
        if name in K700_EXCLUDE:                       # neither a behavior nor a negative
            continue
        beh = K700_TO_BEHAVIOR.get(name)
        if beh:
            wanted.append((rel, beh, name))
        else:
            other_pool.setdefault(name, []).append(rel)
    for name in sorted(other_pool):                    # deterministic `other` draw
        for rel in sorted(other_pool[name])[:per_cls]:
            wanted.append((rel, "other", name))

    root = os.path.join(K700_DIR, "videos")
    rows = []
    for rel, beh, name in wanted:
        p = os.path.join(root, rel)
        if not videos or os.path.exists(p):
            rows.append((p, beh, k700_slug(name), split))
    return rows


def summarize(rows):
    from collections import Counter
    c = Counter(r[1] for r in rows)
    return {b: c.get(b, 0) for b in BEHAVIORS}


if __name__ == "__main__":
    for sp in ("train", "validation", "test", None):
        rows = build(sp)
        name = sp or "ALL"
        print(f"{name:12s} n={len(rows):5d}  " +
              "  ".join(f"{b}={n}" for b, n in summarize(rows).items()))
    for sp in ("train", "val"):
        if not os.path.exists(K700_LISTS[sp]):
            continue
        sel = k700_rows(sp, videos=False)
        have = k700_rows(sp)
        print(f"k700 {sp:7s} n={len(have):5d}/{len(sel):5d} unpacked  " +
              "  ".join(f"{b}={n}" for b, n in summarize(have).items()))
    for sp in ("train", "test"):
        if not os.path.exists(DARKACT_LISTS[sp]):
            continue
        rows = darkact_rows(sp)
        print(f"darkact {sp:4s} n={len(rows):5d}/{len(darkact_rows(sp, cap=False)):5d} uncapped  " +
              "  ".join(f"{b}={n}" for b, n in summarize(rows).items()))
