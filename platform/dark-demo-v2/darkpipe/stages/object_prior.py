"""Object-detection prior gate: the meeting "three-step" decision's step 3 (二次判断).

Step 1+2 (per-branch classification, take the higher-confidence branch) live in
skeleton_aux.SkeletonAux.analyze + recognize.BehaviorRecognizer; this module is step 3 --
necessary conditions that a preliminary result must satisfy to be reported:

  pick_up  some non-person object sits on the ground next to a skeleton
           (box bottom in the ankle zone, box bottom-center near an ankle).
  wave     the highest arm point (best visible wrist OR elbow) stays above the torso
           middle line (midpoint of shoulder-mid <-> hip-mid) -- skeleton geometry only,
           no detector involved. The 2026-10-14 revision moved the line DOWN from the
           heart line (shoulder midpoint) and added the elbows, so a low one-arm wave or
           a raised elbow counts -- the 10-07 overhead rule had killed real low waves.
  drink    a container (bottle / wine glass / cup / bowl) is near an arm
           (box center within DRINK_RADIUS torso-lengths of a wrist).
  shake_hands  two qualifying skeletons' wrists come within SHAKE_RADIUS (mean of the two
           persons') torso-lengths of each other -- hand contact needs two people; a
           single-person shake cannot pass. Geometry only, no detector.

Every spatial threshold is expressed in units of the person's own torso length
(shoulder-mid to hip-mid, normalized coordinates), so the gate is scale- and
resolution-free.

The detector is the YOLO26m COCO-80 head exported to ONNX end-to-end (the export already
contains the top-300 + scores, no NMS on our side); it runs through the SAME onnxruntime
the pose model uses, so the image ships no new Python dependencies. Fail policy is the
skeleton branch's: load() reports by returning a reason string, and a missing/over-budget
detector leaves the OBJECT priors fail-OPEN (cannot verify -> do not reject) while the
wave prior (pure geometry) keeps working. A detector that would break the <=1 s latency
spec disables itself at load, exactly like SkeletonAux.POSE_MS_BUDGET.

The gate is only consulted when the preliminary result is one of PRIOR_ACTIONS and the
window HAS a qualifying skeleton (skeleton quality < TAU means "illumination killed the
skeleton branch" -> the meeting rule says output the RGB result directly, no second
judgment -- enforced by the caller, which simply never calls check() in that case).
"""
import os
import time

import cv2
import numpy as np

# Weights live in ckpts/ next to the pose/ST-GCN weights. The m model measured ~10 ms/frame
# on a RTX 3090 at 640; the gate runs OBJ_SAMPLE_FRAMES per firing window and only for
# pick_up/drink preliminary results, i.e. ~60 ms on those windows, amortized over a 16-frame
# stride. If a site finds it too slow: export yolo26s-detect.pt the same way and swap the
# file (same COCO-80 classes, same end-to-end output).
YOLO_ONNX = "yolo26m_detect.onnx"
DET_INPUT = 640
DET_MS_BUDGET = 30.0          # per frame, steady state (after the session-init call)
DET_CONF = 0.25

# COCO-80 ids: bottle=39, wine glass=40, cup=41, bowl=45; person=0.
CONTAINER_CLS = (39, 40, 41, 45)
PERSON_CLS = 0

OBJ_SAMPLE_FRAMES = 6         # window frames the detector sees (evenly spaced)

# Gate constants, in torso-length units unless said otherwise.
DRINK_RADIUS = 0.9            # container center <-> wrist
PICKUP_RADIUS = 1.0           # object bottom-center <-> nearest ankle
PICKUP_GROUND_SLACK = 0.5     # box bottom may sit this far above the ankle line and
                              # still count as "on the ground"
SHAKE_RADIUS = 0.8            # wrist <-> wrist across TWO persons (10-14 meeting)
SHAKE_MIN_FRAC = 0.3          # fraction of the pair's common frames needing contact
KPT_CONF = 0.3                # a keypoint counts as visible above this
COMP_MIN_VIS = 10             # meeting wording "骨骼完整": >=10/17 keypoints visible
COMP_FRAC = 0.5               # ... on >=50% of the person's frames (skeleton_aux side)

# Keypoint indices (COCO-17, same numbering as skeleton_aux).
L_SHO, R_SHO, L_ELB, R_ELB = 5, 6, 7, 8
L_WRI, R_WRI, L_HIP, R_HIP, L_ANK, R_ANK = 9, 10, 11, 12, 15, 16
WAVE_MIN_FRAC = 0.3           # fraction of the person's frames needing an arm point
                              # above the torso middle line
WAVE_MARGIN = 0.02            # normalized-y margin above that line

#: preliminary labels that face a second judgment. Anything else passes unchecked.
PRIOR_ACTIONS = ("pick_up", "wave", "drink", "shake_hands")


def _torso(kp):
    """Normalized torso length from one (17,3) keypoint frame, or None."""
    sho, hip = kp[[L_SHO, R_SHO]], kp[[L_HIP, R_HIP]]
    if (sho[:, 2] > KPT_CONF).all() and (hip[:, 2] > KPT_CONF).all():
        return float(np.linalg.norm(sho.mean(0)[:2] - hip.mean(0)[:2]))
    return None


class YoloOnnxDetector:
    """COCO-80 detector over the stack's own onnxruntime. Same fail-soft contract as
    SkeletonAux: load() -> "" or a reason; a failed detector just leaves the object
    priors fail-open."""

    def __init__(self, ckpt_dir, device):
        self.path = os.path.join(ckpt_dir, YOLO_ONNX)
        self.device = "cuda:0" if str(device) == "cuda" else str(device)
        self.sess = None
        self.det_ms = 0.0

    def load(self):
        if not os.path.exists(self.path):
            return (f"[recognize] 物体先验关闭：检测权重缺失 {self.path}。"
                    "捡东西/喝水的先验改为不否决（挥手先验不受影响）")
        try:
            import onnxruntime as ort
            prov = ["CUDAExecutionProvider"] if str(self.device).startswith("cuda") \
                else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(self.path, providers=prov)
        except Exception as e:                           # noqa: BLE001 -- fail-soft by contract
            self.sess = None
            return (f"[recognize] 物体先验关闭：检测模型不可用（{type(e).__name__}: {e}）。"
                    "捡东西/喝水的先验改为不否决（挥手先验不受影响）")
        dummy = np.zeros((472, 840, 3), np.uint8)
        self._frames_to_dets([dummy, dummy])             # session init, untimed
        self._frames_to_dets([dummy, dummy])             # clock/pool ramp, untimed
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            self._frames_to_dets([dummy, dummy])
            times.append((time.perf_counter() - t0) * 1000 / 2)
        self.det_ms = min(times)          # jitter only adds time; min = the card's real cost
        if self.det_ms > DET_MS_BUDGET:
            self.sess = None
            return (f"[recognize] 物体先验关闭：检测推理实测 {self.det_ms:.0f} ms/帧，"
                    f"超过 {DET_MS_BUDGET:.0f} ms 预算（onnxruntime 没用上 GPU？）。"
                    "捡东西/喝水的先验改为不否决（挥手先验不受影响）")
        return ""

    @staticmethod
    def _letterbox(img, new=DET_INPUT, pad=114):
        """Aspect-preserving resize onto a gray `new`x`new` canvas (ultralytics-style).
        Distorting to a square instead measurably costs detections (probed 3->1 boxes on a
        real platform clip), so the geometry is worth the extra line."""
        h, w = img.shape[:2]
        r = min(new / h, new / w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((new, new, 3), pad, np.uint8)
        top, left = (new - nh) // 2, (new - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def _frames_to_dets(self, frames):
        """BGR frames -> list of (n_i, 6) arrays [x1 y1 x2 y2 conf cls], letterbox divided
        out: boxes are in each frame's OWN pixel coordinates. One session run per frame --
        the exported model is fixed-batch-1 (dynamic=False), and 6 single frames inside one
        call still sit far under the per-fire budget."""
        outs = []
        for f in frames:
            canvas, r, left, top = self._letterbox(f)
            x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            row_all = self.sess.run(
                None, {self.sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]
            row_all = row_all[row_all[:, 4] > DET_CONF]
            if len(row_all):
                box = row_all[:, :4]
                box[:, [0, 2]] = (box[:, [0, 2]] - left) / r
                box[:, [1, 3]] = (box[:, [1, 3]] - top) / r
                np.clip(box, [0, 0, 0, 0], [f.shape[1], f.shape[0]] * 2, out=box)
                row_all[:, :4] = box
            outs.append(row_all)
        return outs

    def __call__(self, frames):
        """-> list of (n_i, 6) [x1 y1 x2 y2 conf cls] per frame, or None when unavailable."""
        if self.sess is None:
            return None
        return self._frames_to_dets(frames)


class PriorGate:
    """Step-3 checker. `check()` never raises and never rejects on missing evidence: no
    detector, no visible keypoints of the kind the prior needs -> pass with a reason."""

    def __init__(self, ckpt_dir, device):
        self.det = YoloOnnxDetector(ckpt_dir, device)

    def load(self):
        return self.det.load()

    def warmup(self):
        if self.det.sess is not None:
            self.det._frames_to_dets([np.zeros((472, 840, 3), np.uint8)])

    # ---- priors -------------------------------------------------------------
    def _wave(self, sk, valid):
        """The person's highest visible arm point (wrist or elbow) above the torso middle
        line on >= WAVE_MIN_FRAC of their frames, for any qualifying person. Frames with a
        measurable line but no visible arm point still count toward the denominator -- the
        arm being out of view is evidence of not-waving, not of a skipped frame."""
        for m in np.flatnonzero(valid):
            seen, above = 0, 0
            for t in range(sk.shape[0]):
                kp = sk[t, m]
                sho, hip = kp[[L_SHO, R_SHO]], kp[[L_HIP, R_HIP]]
                sho_ok = bool((sho[:, 2] > KPT_CONF).all())
                hip_ok = bool((hip[:, 2] > KPT_CONF).all())
                if sho_ok and hip_ok:
                    line = float((sho[:, 1].mean() + hip[:, 1].mean()) / 2)
                elif sho_ok:
                    line = float(sho[:, 1].mean())   # stricter than the spec's line
                else:
                    continue
                arms = kp[[L_WRI, R_WRI, L_ELB, R_ELB]]
                vis = arms[arms[:, 2] > KPT_CONF]
                if not len(vis):
                    continue
                seen += 1
                if float(vis[:, 1].min()) < line - WAVE_MARGIN:
                    above += 1
            if seen and above / seen >= WAVE_MIN_FRAC:
                return True, f"wave ok ({above}/{seen} 帧手臂过躯干中线)"
        return False, "没有骨架的手臂过躯干中部"

    def _shake(self, sk, valid):
        """Wrist-to-wrist contact between two qualifying persons on >= SHAKE_MIN_FRAC of
        their common frames. Distance is normalized by the pair's mean torso length of
        that frame, so the radius is scale-free like every other gate constant."""
        ms = np.flatnonzero(valid)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                seen, above = 0, 0
                for t in range(sk.shape[0]):
                    torso = (_torso(sk[t, ms[i]]), _torso(sk[t, ms[j]]))
                    if None in torso:
                        continue
                    wa = [w for w in (L_WRI, R_WRI) if sk[t, ms[i], w, 2] > KPT_CONF]
                    wb = [w for w in (L_WRI, R_WRI) if sk[t, ms[j], w, 2] > KPT_CONF]
                    if not wa or not wb:
                        continue
                    seen += 1
                    d = min(float(np.hypot(*(sk[t, ms[i], a, :2] - sk[t, ms[j], b, :2])))
                            for a in wa for b in wb)
                    if d < SHAKE_RADIUS * (torso[0] + torso[1]) / 2:
                        above += 1
                if seen and above / seen >= SHAKE_MIN_FRAC:
                    return True, (f"shake ok (两人腕距 <{SHAKE_RADIUS:g} 躯干长, "
                                  f"{above}/{seen} 帧)")
        return False, "没有两个人骨架的手部交互"

    def _drink(self, sk, valid, idx, sampled, dets, H):
        """Container near a wrist of a qualifying person, on any sampled frame."""
        if dets is None:
            return True, "检测器不可用，喝水先验放行"
        for t, det in zip(idx, dets):
            cont = det[np.isin(det[:, 5].astype(int), CONTAINER_CLS)]
            if not len(cont):
                continue
            W = sampled[0].shape[1]
            for m in np.flatnonzero(valid):
                kp = sk[t, m]
                torso = _torso(kp)
                if torso is None:
                    continue
                for c in cont:
                    cx, cy = (c[0] + c[2]) / 2 / W, (c[1] + c[3]) / 2 / H
                    for w in (L_WRI, R_WRI):
                        if kp[w, 2] > KPT_CONF and \
                           np.hypot(cx - kp[w, 0], cy - kp[w, 1]) < DRINK_RADIUS * torso:
                            return True, f"drink ok (容器距手腕 {np.hypot(cx - kp[w, 0], cy - kp[w, 1]) / torso:.2f} 躯干长)"
        return False, "手部附近没有杯子/容器"

    def _pickup(self, sk, valid, idx, sampled, dets, H):
        """Non-person object on the ground next to a skeleton (near the ankle zone)."""
        if dets is None:
            return True, "检测器不可用，捡东西先验放行"
        for t, det in zip(idx, dets):
            obj = det[det[:, 5].astype(int) != PERSON_CLS]
            if not len(obj):
                continue
            W = sampled[0].shape[1]
            for m in np.flatnonzero(valid):
                kp = sk[t, m]
                torso = _torso(kp)
                if torso is None:
                    continue
                for a in (L_ANK, R_ANK):
                    if kp[a, 2] <= KPT_CONF:
                        continue
                    for o in obj:
                        bx, by = (o[0] + o[2]) / 2 / W, o[3] / H
                        d = np.hypot(bx - kp[a, 0], by - kp[a, 1])
                        if d < PICKUP_RADIUS * torso and \
                           by >= kp[a, 1] - PICKUP_GROUND_SLACK * torso:
                            return True, f"pickup ok (物体距脚踝 {d / torso:.2f} 躯干长, 贴地)"
        return False, "脚边地面没有物体"

    def check(self, label, sk, valid, frames):
        """-> (passed: bool, reason: str). `sk` (T,M,17,3) normalized xy+conf from
        SkeletonAux.analyze, `valid` the per-slot qualification mask, `frames` the raw
        window. The detector only ever sees OBJ_SAMPLE_FRAMES evenly spaced frames of it."""
        H = frames[0].shape[0]
        if label == "wave":
            return self._wave(sk, valid)
        if label == "shake_hands":
            return self._shake(sk, valid)
        idx = np.linspace(0, len(frames) - 1,
                          min(OBJ_SAMPLE_FRAMES, len(frames))).round().astype(int)
        sampled = [frames[i] for i in idx]
        dets = self.det(sampled)
        if label == "drink":
            return self._drink(sk, valid, idx, sampled, dets, H)
        if label == "pick_up":
            return self._pickup(sk, valid, idx, sampled, dets, H)
        return True, "该行为没有先验条件"
