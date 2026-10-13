"""One video per behavior class: the four-stage grid, rendered for all ten labels.

`../stage_video.py` shows the four pipeline stages on a single clip. This shows them once per
behavior, so the reel answers "what does the system do on *each* action class" rather than
"what does it do on one clip". Every panel comes from the shipped stage classes in the shipped
order (`stage_video.run_clip`, which mirrors `darkpipe/pipeline.py:process_chunk`) -- enhance
-> recognizer tap (post-enhance, PRE-SR) -> bicubic x2 -> label bar.

`--source hmdb` (default) uses the HMDB test split, darkened by the same calibrated simulation
the training cache used (`darken.py`, per-clip exposure derived from the filename, so a clip is
as dark here as it was in training). The two sources are never mixed inside one reel: ARID
covers only 4 of the 10 labels, and a sheet where some classes are real dark and others are
simulated would invite exactly the comparison it cannot support.

`--source arid` renders the drink / pick_up / wave / other subset from genuinely dark ARID
footage with no simulation at all. That reel exists because the simulation and real darkness
differ where it is most visible: the simulated frames carry ~13x ARID's real per-pixel noise
(BEHAVIOR_REPORT.md 2.2), which the enhancer's 6-11x gain turns into coloured speckle. Run both
and the difference is the point -- the HMDB reel shows the harder, pessimistic domain, the ARID
reel shows what the shipped pipeline does on real dark video.

Clip choice, and why it is stated on every frame: `--pick correct` walks the class's test clips
in sorted order and keeps the first one the recognizer gets right, falling back to the first
clip if none of `--tries` works. That is a demo reel, not a metric -- so the banner always
shows TRUE and PREDICTED side by side, the frame counter says which candidate was used, and a
class where no candidate worked is rendered anyway and marked MISS in red. `--pick first`
disables the search and takes the sorted-first clip, matching `behavior_demo.py`.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.dirname(HERE)
DP = os.path.dirname(COMPARE)
sys.path.insert(0, HERE)
sys.path.insert(0, COMPARE)
sys.path.insert(0, DP)

import behavior_manifest as BM                                          # noqa: E402
import darken as DK                                                     # noqa: E402
import stage_video as SV                                                # noqa: E402
from darkpipe.constants import BEHAVIOR_DISPLAY                         # noqa: E402

RESULTS = os.path.join(COMPARE, "results", "class_video")
BAN_H = 46
OK_C, MISS_C = (120, 235, 140), (120, 140, 245)      # BGR: green ok, red miss


def banner(w, lab, pred, conf, correct, pick_note):
    """Title strip above the 2x2 grid: the class, what the pipeline said, and the clip used."""
    s = np.full((BAN_H, w, 3), SV.BG, np.uint8)
    left = f"{BEHAVIOR_DISPLAY.get(lab, lab)}  ({lab})"
    right = f"predicted: {pred} {conf:.2f}" if pred else "predicted: (no event)"
    cv2.putText(s, left, (SV.PAD, 31), SV.FONT, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right, SV.FONT, 0.62, 1)
    cv2.putText(s, right, (w - rw - SV.PAD - 74, 30), SV.FONT, 0.62,
                OK_C if correct else MISS_C, 1, cv2.LINE_AA)
    cv2.putText(s, "OK" if correct else "MISS", (w - SV.PAD - 46, 30), SV.FONT, 0.62,
                OK_C if correct else MISS_C, 2, cv2.LINE_AA)
    cv2.putText(s, pick_note, (SV.PAD, 43), SV.FONT, 0.36, (140, 140, 140), 1, cv2.LINE_AA)
    return s


def majority(events):
    """The label the reel is judged on: most frequent event label, mean confidence for it."""
    if not events:
        return "", 0.0
    labs = [e.label for e in events]
    top = max(set(labs), key=labs.count)
    return top, float(np.mean([e.confidence for e in events if e.label == top]))


def sheet_frame(r):
    """The annotated frame to put on the contact sheet.

    Late in the clip (the recognizer needs a full window before it says anything), but pinned
    to a frame where the visible label bar shows the clip's majority label -- otherwise a cell
    captioned `other` can carry a bar reading `Waving 73%` from one dissenting window, and the
    sheet contradicts its own caption.
    """
    ann, ch = r["ann"], r["chunk"]
    # The bar is per CHUNK and shows that chunk's LAST event (pipeline.py:process_chunk), so
    # indexing by the event's own frame is not enough -- a later event in the same chunk wins.
    last = {}
    for e in r["events"]:
        last[e.frame_index // ch] = e.label
    hit = [c for c, lab in last.items() if lab == r["pred"]]
    i = max(hit) * ch + ch - 1 if hit else 3 * len(ann) // 4
    return ann[min(max(i, 0), len(ann) - 1)]


def render(src, frame_stages, recognizer, max_frames, chunk, darken=True,
           full_well=DK.FULL_WELL, panel="stage"):
    """Run one clip through the shipped stages, return everything the grid and index need.

    `darken` simulates the training exposure for a bright HMDB clip; ARID is already dark, and
    darkening it again would stack a second sensor's noise on top of the real one.

    `full_well` is the simulated sensor's photo-electron capacity: it sets SNR, and the shipped
    1200 e- is ~13x noisier than a real sensor at the same brightness (BEHAVIOR_REPORT.md 2.2).
    Raising it for a demo makes the enhancer's input realistic instead of speckled.

    `panel="original"` puts the undarkened source in the enhancement slot and runs SR and
    recognition on it. Panel 1 is still the dark frame, so the grid reads dark -> clear, but
    the enhancer never ran: the caption says so, and nothing here may claim otherwise.
    """
    raw, fps = SV.read_frames(src, max_frames)
    if not raw:
        return None
    k, orig = None, raw
    if darken:
        rng = DK.clip_rng(src)
        k = DK.clip_k(rng)
        raw = list(DK.darken_frames(np.stack(raw), k, rng, full_well=full_well))
    if recognizer:
        recognizer.reset()          # independent clips through one loaded recognizer
    if panel == "original":
        # Drop the enhancer and feed the bright source: run_clip's "enhanced" output is then
        # the original, and SR + the recognizer tap see exactly what a viewer sees.
        stages = [s for s in frame_stages if not s.name.startswith("enhance")]
        enh, sr, ann, events, per_chunk = SV.run_clip(orig, stages, recognizer, fps, chunk)
    else:
        enh, sr, ann, events, per_chunk = SV.run_clip(raw, frame_stages, recognizer, fps,
                                                      chunk)
    pred, conf = majority(events)
    return dict(raw=raw, enh=enh, sr=sr, ann=ann, events=events, per_chunk=per_chunk,
                fps=fps, k=k, pred=pred, conf=conf, src=src, chunk=chunk, panel=panel)


def write_grid(out, r, lab, correct, pick_note, args, scale):
    """The per-class video: banner + the same 2x2 grid stage_video.py writes."""
    H, W = r["raw"][0].shape[:2]
    SH, SW = r["sr"][0].shape[:2]
    ms, _ = SV.steady_state(r["per_chunk"])
    four = [("input", r["raw"]), ("enhance", r["enh"]), ("sr", r["sr"]), ("reco", r["ann"])]
    # Panel 2 must say what actually made it. Under --enhance-panel original the enhancer did
    # not run, and the caption says "original source (enhancer NOT run)" rather than naming a
    # model that had no part in the pixels -- a demo may be flattering, not false.
    ref = r["panel"] == "original"
    titles = [
        (f"1. input ({'real dark, ARID' if args.source == 'arid' else 'simulated dark'})",
         f"{W}x{H}"),
        ("2. original source (enhancer NOT run)" if ref else
         f"2. + enhancement: {args.enhance}",
         f"{W}x{H}   reference panel" if ref else f"{W}x{H}   {ms['enh']:.1f} ms/frame"),
        (f"3. + super-resolution: {args.sr}", f"{SW}x{SH}   {ms['sr']:.1f} ms/frame"),
        (f"4. + recognition: {args.recognize}", f"{SW}x{SH}+bar   {ms['reco']:.1f} ms/frame"),
    ]
    g0 = SV.grid_frame(four, titles, 0)
    gw = SV._even(g0.shape[1])
    gh = SV._even(g0.shape[0] + BAN_H)
    dst = (int(gw * scale) // 2 * 2, int(gh * scale) // 2 * 2)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), r["fps"], dst)
    top = banner(gw, lab, r["pred"], r["conf"], correct, pick_note)
    for i in range(len(r["raw"])):
        g = np.vstack([top, SV.grid_frame(four, titles, i)])
        g = cv2.copyMakeBorder(g, 0, gh - g.shape[0], 0, gw - g.shape[1],
                               cv2.BORDER_CONSTANT, value=(SV.BG, SV.BG, SV.BG))
        if scale != 1.0:
            g = cv2.resize(g, dst, interpolation=cv2.INTER_AREA)
        wr.write(g)
    wr.release()
    return dst if scale != 1.0 else (gw, gh)


def contact_sheet(shots, path, cols=5, scale=1.0):
    """One annotated frame per class in a single png -- the reel's table of contents."""
    h = max(s.shape[0] for _, s in shots)
    w = max(s.shape[1] for _, s in shots)
    rows = []
    for i in range(0, len(shots), cols):
        cells = []
        for lab, im in shots[i:i + cols]:
            c = np.full((h, w, 3), SV.BG, np.uint8)
            y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
            c[y:y + im.shape[0], x:x + im.shape[1]] = im
            cells.append(np.vstack([SV._caption(w, lab, "", 0.62), c]))
        while len(cells) < cols:
            cells.append(np.full_like(cells[0], SV.BG))
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    if scale != 1.0:
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, sheet)
    return path, sheet.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhance", default="retinexformer")
    ap.add_argument("--sr", default="bicubic_x2")
    ap.add_argument("--recognize", default="behavior", choices=["behavior", "xclip"])
    ap.add_argument("--ckpt-dir", default=os.path.join(DP, "ckpts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--source", default="hmdb", choices=["hmdb", "arid"],
                    help="hmdb = bright clips darkened by the calibrated simulation (all 10 "
                         "labels); arid = real dark footage (drink/pick_up/wave/other only)")
    ap.add_argument("--enhance-panel", default="stage", choices=["stage", "original"],
                    help="stage = the enhancer's real output; original = the bright source in "
                         "the enhancement slot, with SR and recognition run on it (demo mode; "
                         "the panel is captioned 'enhancer NOT run')")
    ap.add_argument("--full-well", type=float, default=DK.FULL_WELL,
                    help="simulated sensor capacity in e-. The trained default 1200 is ~13x "
                         "noisier than a real sensor; ~30000 matches ARID (report 2.2)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=120,
                    help="cap per clip: keeps the ten videos comparable in length")
    ap.add_argument("--pick", default="correct", choices=["correct", "first"])
    ap.add_argument("--tries", type=int, default=6,
                    help="candidates per class under --pick correct")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sheet-scale", type=float, default=0.5,
                    help="the contact sheet is 5 x 640px wide at 1.0; halved it still reads")
    ap.add_argument("--outdir", default=RESULTS)
    a = ap.parse_args()

    rows = BM.arid_rows(a.split) if a.source == "arid" else BM.build(a.split)
    by_lab = {}
    for path, lab, cls, _ in sorted(rows, key=lambda r: os.path.basename(r[0])):
        by_lab.setdefault(lab, []).append((path, cls))

    os.makedirs(a.outdir, exist_ok=True)
    frame_stages, recognizer = SV.build_stages(a.enhance, a.sr, a.recognize,
                                               a.ckpt_dir, a.device)
    tag = ("" if a.source == "hmdb" else f"_{a.source}") \
        + ("_ref" if a.enhance_panel == "original" else "") \
        + ("" if a.full_well == DK.FULL_WELL else f"_fw{int(a.full_well)}")
    index, shots, hits = [], [], 0
    for lab in BM.BEHAVIORS:
        cands = by_lab.get(lab, [])[: (a.tries if a.pick == "correct" else 1)]
        if not cands:
            # Expected for 6 of the 10 labels under --source arid: ARID has no footage of
            # them at all, which is why the simulated reel exists in the first place.
            print(f"  {lab:<12} no clip in {a.source}/{a.split}")
            continue
        want = BEHAVIOR_DISPLAY.get(lab, lab)
        best, tried = None, 0
        for src, cls in cands:
            r = render(src, frame_stages, recognizer, a.max_frames, a.chunk,
                       darken=a.source == "hmdb", full_well=a.full_well,
                       panel=a.enhance_panel)
            if r is None:
                continue
            tried += 1
            r["cls"] = cls
            best = best or r                      # fallback: the first readable candidate
            if r["pred"] == want:
                best = r
                break
        if best is None:
            print(f"  {lab:<12} no readable clip")
            continue
        correct = best["pred"] == want
        hits += correct
        # The exposure is on-screen because it varies ~20x across clips (each keeps the k it
        # was trained at), and it is the reason some panels are far noisier than others.
        dark = (f"exposure k={best['k']:.5f} (simulated"
                f"{'' if a.full_well == DK.FULL_WELL else ', %g e- full well' % a.full_well})"
                if best["k"] else "real dark (no simulation)")
        if a.enhance_panel == "original":
            dark += "  |  panels 2-4 from the ORIGINAL source, enhancer not run"
        note = (f"{'clip %d/%d tried' % (tried, len(cands)) if a.pick == 'correct' else 'first clip'}"
                f"  |  {dark}  |  {os.path.basename(best['src'])[:70]}")
        out = os.path.join(a.outdir, f"class_{lab}_{a.recognize}{tag}.mp4")
        gw, gh = write_grid(out, best, lab, correct, note, a, a.scale)
        shots.append((f"{lab}{'' if correct else '  (MISS)'}", sheet_frame(best)))
        index.append(dict(behavior=lab, display=want, src_class=best["cls"],
                          source=os.path.basename(best["src"]), frames=len(best["raw"]),
                          darken_k=round(best["k"], 5) if best["k"] else None,
                          events=len(best["events"]),
                          predicted=best["pred"], confidence=round(best["conf"], 3),
                          correct=bool(correct), candidates_tried=tried,
                          grid=f"{gw}x{gh}", video=os.path.relpath(out, a.outdir)))
        print(f"  {lab:<12} {len(best['raw']):>3}f  "
              f"k={('%-8.5f' % best['k']) if best['k'] else 'real    '} "
              f"{len(best['events']):>2} ev  -> {best['pred'] or '(none)':<14}"
              f"{best['conf']:.2f}  {'ok' if correct else 'MISS'}  ({tried} tried)  "
              f"{os.path.basename(out)}")

    for s in frame_stages:
        s.close()
    if recognizer:
        recognizer.close()

    sheet, shape = contact_sheet(shots, os.path.join(a.outdir,
                                                     f"class_contact_{a.recognize}{tag}.png"),
                                 cols=min(5, max(1, len(shots))), scale=a.sheet_scale)
    p = os.path.join(a.outdir, f"class_index_{a.recognize}{tag}.json")
    json.dump(dict(source=a.source, enhance_panel=a.enhance_panel, full_well=a.full_well,
                   pick=a.pick, tries=a.tries, split=a.split,
                   enhance=a.enhance, sr=a.sr,
                   recognize=a.recognize, correct=hits, n=len(index), clips=index),
              open(p, "w"), indent=1)
    print(f"\n  {hits}/{len(index)} classes shown with a correct label"
          f"{' (best of %d candidates each)' % a.tries if a.pick == 'correct' else ''}"
          f" -- a demo reel, not a metric; the numbers are in BEHAVIOR_REPORT.md")
    print(f"  -> {sheet}  ({shape[1]}x{shape[0]})\n  -> {p}")


if __name__ == "__main__":
    main()
