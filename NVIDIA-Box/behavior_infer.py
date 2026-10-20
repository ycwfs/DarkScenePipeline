"""动作识别模型 · 独立推理脚本（VideoMamba-Tiny / 32 帧，权重
weights/videomamba_t_behavior_32f_realactall_full_pn2.pth，外加可选 COCO-17 骨架分支）。

与 DarkScenePipeline 平台链路（recognize=behavior，v2）完全一致的加载与推理：

    帧预处理（逐帧，由权重自带的 preprocess 元数据自动选择，--weights 换权重即自动切换）：
      fullfov<H>x<W>（当前默认权重 realactall_full_pn2，2026-10-19 负样本续训）：
        BGR uint8 -> 整帧等比缩放到 WxH（400x224，INTER_AREA，不裁剪）
        -> RGB、/255，按 ImageNet mean/std 归一化
      resize256_centercrop224（旧权重 realact10_all 等裁剪链权重，或无元数据的更早权重）：
        BGR uint8 -> 短边缩放到 256（缩放比 < 0.5 用 INTER_AREA，否则 INTER_LINEAR）
        -> 中心裁剪 224x224 -> RGB、/255 归一化
      （读元数据而不是加开关：权重和它的输入管线是同一个事实，分开可配就可能配错——
       裁剪链喂整帧模型会静默丢掉画面四周一大圈。）
    推理（整窗）：
      32 帧均匀采样 -> (1,3,32,H,W) fp32 送 GPU -> autocast fp16 前向 -> softmax
    骨架分支（v2，默认开启；--skel-alpha 1.0 或权重/依赖缺失时自动退回纯 RGB）：
      与 RGB 完全相同的 32 帧 -> RTMO-s 姿态（onnxruntime CUDA EP，每帧取前 2 人 COCO-17）
      -> ST-GCN -> p_skel 与 presence（关键点得分兼任“画面里有没有人”的门控信号）
        presence >= 0.5：p = α·p_rgb + (1-α)·p_skel（α 默认 0.7）
        presence <  0.5：按 --skel-no-person 策略（默认 soft：空骨架仍以 (1-α) 权重投
        other 票；默认阈值 0.75 > α=0.7，无人窗无法触发具名动作）
    照度门（“两部曲”第一步，--lum-gate，默认 75，0=关）：
      对提亮前原图的 32 帧窗求平均 BT.601 亮度（--enhance 时量的是增强前那些帧）。
      低于门限：只用 RGB 结果，骨架分支连姿态都不跑——暗场骨架不可靠，全量现场视频
      实测提亮前亮度 ≥75 后姿态可用率才到 1.0 平台期；提亮后的画面亮度被增强模型
      拉平到固定亮度带，量它等于瞎。不低于门限：α 加权融合照常。
      与平台 main.py 的 --lum_gate 同语义同默认（75 = 现场亮度拐点）。
    输出：
      10 类概率：9 类具名行为 + other（兜底类）。

线上判定规则：融合后概率里最高分的具名行为低于阈值（平台默认 0.75）时报为 other——本脚本
同时给出纯 RGB 分支的原始 argmax（便于单/双分支对比）和线上规则后的最终结果。

骨架分支 fail-soft：缺 rtmlib/onnxruntime-gpu/权重文件时自动降级纯 RGB 并打印中文原因。
权重说明见 README.md。用法：

    python behavior_infer.py --input sample_data/sample_drink.mp4
    python behavior_infer.py --input sample_drink.mp4 --enhance   # 先增强再识别（线上顺序）
    python behavior_infer.py --input clip.mp4 --skel-alpha 1.0    # 关骨架分支（纯 RGB 对比）
    python behavior_infer.py --input clip.mp4 --lum-gate 0        # 关照度门（退回纯加权融合）
    python behavior_infer.py --input clip.mp4 --json out/probs.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from video_io import downscale_long_side, is_image, read_image, read_video, sample_uniform

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "weights",
                            "videomamba_t_behavior_32f_realactall_full_pn2.pth")
DEFAULT_ENHANCE_CKPT = os.path.join(HERE, "weights", "NTIRE.pth")

# 9 类具名行为 + other。顺序就是输出概率向量的维度顺序，不可改动（与 skeleton_aux.BEHAVIORS
# 逐字一致）。
LABELS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
          "shake_hands", "other"]
LABEL_ZH = {"wave": "挥手", "throw": "投掷物品", "chase": "追逐", "fall": "跌倒",
            "fight": "打斗", "talk": "交谈", "drink": "喝水",
            "pick_up": "捡拾物品", "shake_hands": "握手", "other": "其他"}

# 裁剪链输入规格（与平台 RECO_CFG["behavior"] 一致）：T 帧窗口、短边 resize、中心裁剪、归一化。
# fullfov 权重不走这两个裁剪常量，整帧缩到元数据给的 WxH。
WINDOW_T = 32
RESIZE_SHORT = 256
CROP_SIZE = 224
IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], np.float32)
# 线上默认判定阈值：具名行为最高分低于它时改报 other（平台 suanzi.json 的 reco_min_conf）
RECO_MIN_CONF = 0.75
# 骨架分支默认值：与平台 main.py 的 skel_fusion_alpha / skel_no_person 默认一致
SKEL_ALPHA_DEFAULT = 0.7
# 照度门默认值：与平台 main.py 的 lum_gate 默认一致（全量现场视频实测的提亮前亮度拐点）
LUM_GATE_DEFAULT = 75.0


def frame_luma(frame_bgr):
    """单帧 BGR uint8 的平均 BT.601 亮度（0-255）——照度门量的量，必须是提亮前的原图。

    提亮会毁掉亮度信息：Retinexformer 把不管多暗的场景都拉到固定亮度带（现场四套件的
    提亮后窗口亮度全落在 66-180，最该被拦的平台误报切片反而最亮），所以门只能设在
    增强之前。逐字取自平台 recognize.frame_luma：先求通道均值（cv2.mean，840p 上
    0.1 ms，numpy 轴归约要 10 ms）再加权求和，与逐像素加权结果一致。
    """
    b, g, r = cv2.mean(frame_bgr)[:3]
    return float(0.114 * b + 0.587 * g + 0.299 * r)


def gate_luma(luma_window, lum_gate):
    """照度门（平台 recognize.gate_luma 的逐字移植，alpha/max_prior 两种决策模式共用）：
    触发时返回挡下的窗口平均亮度，不触发返回 None（门关 lum_gate<=0、窗口没有亮度
    数据 fail-open、或不低于门限）。低于门限 = 纯 RGB，骨架分支不运行。"""
    if not lum_gate or lum_gate <= 0 or not luma_window:
        return None
    vals = [v for v in luma_window if v is not None]
    if not vals:
        return None
    lum = float(np.mean(vals))
    return None if lum >= lum_gate else lum


def preprocess_frame(frame_bgr, fov=None):
    """BGR uint8 HxWx3 -> 归一化 HWC float32，与平台 recognize.preprocess_frame 逐行一致。

    fov=(H, W) 走整帧管线：等比缩放到 WxH（INTER_AREA，与训练侧一致），不裁剪。
    fov=None 走历史裁剪链：短边 256 -> 中心裁剪 224。
    """
    if fov:
        r = cv2.resize(frame_bgr, (fov[1], fov[0]), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - IN_MEAN) / IN_STD
    h, w = frame_bgr.shape[:2]
    s = RESIZE_SHORT / min(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    interp = cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR
    r = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
    top, left = (nh - CROP_SIZE) // 2, (nw - CROP_SIZE) // 2
    crop = r[top:top + CROP_SIZE, left:left + CROP_SIZE]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - IN_MEAN) / IN_STD


def reject_probs(p, tau=RECO_MIN_CONF):
    """线上开集规则：具名行为最高分 < tau 时把结果改报为 other。返回 (改报后 argmax, 概率)。
    tau<=0 表示关闭规则，直接取原始 argmax。"""
    p = np.asarray(p, np.float32)
    named = np.delete(p, LABELS.index("other"))
    if tau > 0 and named.max() < tau and p.argmax() != LABELS.index("other"):
        q = p.copy()
        q[LABELS.index("other")] = max(1.0 - float(named.max()), float(named.max()) + 1e-6)
        return int(q.argmax()), q
    return int(p.argmax()), p


def fov_from_meta(sd):
    """权重字典 -> (H, W) 或 None。realact_train.py 给每份权重盖了 "preprocess" 元数据：
    "fullfov<H>x<W>" = 整帧 400x224 谱系；"resize256_centercrop224" 或无该键 = 裁剪链。"""
    meta = str(sd.get("preprocess", "")) if isinstance(sd, dict) else ""
    if meta.startswith("fullfov"):
        h, w = (int(x) for x in meta[len("fullfov"):].split("x", 1))
        return (h, w)
    return None


class BehaviorRecognizer:
    """VideoMamba-T 行为识别模型封装：构造一次，predict() 可反复调用。

    参数
      ckpt            权重路径（键名为 model；其 preprocess 元数据决定输入管线）
      device          必须是 CUDA —— VideoMamba 依赖 mamba_ssm 的 CUDA 算子，没有 CPU 实现
      skel_alpha      骨架融合权重 α（p=α·RGB+(1-α)·骨架）；>= 0.999 关闭骨架分支（纯 RGB）
      skel_no_person  无人窗策略 soft/other/rgb，语义见 skeleton_aux.NO_PERSON_MODES
    """

    def __init__(self, ckpt: str = DEFAULT_CKPT, device: str = "cuda",
                 skel_alpha: float = SKEL_ALPHA_DEFAULT, skel_no_person: str = "soft"):
        if torch.device(device).type != "cuda":
            raise SystemExit("error: 行为识别模型只支持 CUDA，请使用 --device cuda 或 cuda:N")
        if not torch.cuda.is_available():
            raise SystemExit("error: 行为识别模型依赖 mamba_ssm 的 CUDA 扩展，必须有 NVIDIA GPU")
        self.device = device
        self.skel_alpha = float(skel_alpha)
        sd = torch.load(ckpt, map_location="cpu")
        self.fov = fov_from_meta(sd)          # (H, W) 整帧输入；None = 裁剪链
        from models.videomamba import videomamba_tiny
        # fov 是 (H, W)=(224, 400)：vendor 模型的 pos_embed 按元组插值，与平台 _build 传法一致
        net = videomamba_tiny(num_classes=len(LABELS), num_frames=WINDOW_T,
                              img_size=self.fov or CROP_SIZE)
        net.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
        self.net = net.to(device).eval()

        # 骨架分支（fail-soft：起不来就纯 RGB，绝不影响 RGB 路径本身）
        self.aux = None
        self.aux_reason = ""
        if self.skel_alpha >= 0.999:
            print("[load] skel_alpha>=0.999，骨架分支关闭，纯 RGB 识别")
            return
        from skeleton_aux import SkeletonAux
        aux = SkeletonAux(os.path.dirname(os.path.abspath(ckpt)), device, self.skel_alpha,
                          no_person=skel_no_person)
        self.aux_reason = aux.load()
        if self.aux_reason:
            print(f"[load] [warn] {self.aux_reason}")
            return
        self.aux = aux
        from skeleton_aux import NO_PERSON_ZH, TAU
        print(f"[load] 骨架分支已启用 RTMO-s+ST-GCN 融合 α={self.skel_alpha:g} "
              f"(无人窗策略={NO_PERSON_ZH.get(skel_no_person, skel_no_person)}，"
              f"TAU={TAU:g}，姿态 {aux.pose_ms:.1f} ms/帧)")

    @property
    def input_pipeline(self) -> str:
        if self.fov:
            return f"整帧 fullfov{self.fov[0]}x{self.fov[1]}（不裁剪）"
        return "裁剪链 resize256_centercrop224"

    @torch.no_grad()
    def predict(self, frames_bgr: list, raw_frames_bgr: list | None = None,
                lum_gate: float = LUM_GATE_DEFAULT) -> dict:
        """list[BGR uint8]（任意帧数，内部均匀采样到 32 帧）-> 类别与概率。

        参数
          raw_frames_bgr  提亮前原图的帧列表，照度门量它。须与 frames_bgr 等长同序
                         （采样下标相同）；None = 没做增强，frames_bgr 本身就是原图。
          lum_gate        照度门门限（提亮前原图 32 帧窗平均 BT.601 亮度，0-255）；
                          0 = 关门。默认 75，与平台 --lum_gate 默认一致。

        返回 dict：
          label / label_zh              纯 RGB 分支的原始 argmax（单/双分支对比用）
          confidence                    其概率
          label_reco / confidence_reco  线上链路（照度门 -> 骨架融合 -> 0.75 阈值规则）
                                        后的最终结果
          probs                         纯 RGB 的 10 类完整概率 {label: p}
          probs_final                   实际送进阈值规则的那份概率（融合后；无骨架时=probs）
          presence                      骨架门控信号：窗内最自信人的平均关键点得分；
                                        骨架分支未启用或被照度门挡下时为 None
          lum / lum_gate / gate_blocked 照度门：窗内提亮前原图平均亮度、门限取值、
                                        本窗是否被门挡下（挡下 = 纯 RGB，骨架未运行）
        """
        if not frames_bgr:
            raise ValueError("识别输入不能为空")
        if raw_frames_bgr is not None and len(raw_frames_bgr) != len(frames_bgr):
            raise ValueError("原图与识别帧必须等长同序，避免照度门使用错误的窗口")
        win = sample_uniform(frames_bgr, WINDOW_T)
        arr = np.stack([preprocess_frame(f, self.fov) for f in win])
        x = torch.from_numpy(arr).to(self.device).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = self.net(x)
        p = logits.softmax(1)[0].float().cpu().numpy()

        # 照度门量提亮前原图：与识别窗同一套采样下标（sample_uniform 对等长列表确定一致），
        # --enhance 时 raw_frames_bgr 是增强前的那些帧。
        raw_win = sample_uniform(frames_bgr if raw_frames_bgr is None else raw_frames_bgr,
                                 WINDOW_T)
        lumas = [frame_luma(f) for f in raw_win]
        lum_mean = float(np.mean(lumas)) if lumas else None
        lum = gate_luma(lumas, lum_gate)

        pres = None
        p_final = p
        if self.aux is not None and lum is None:
            # 未被照度门挡下才跑骨架；挡下时骨架（含姿态）整体不运行，与平台线上同语义
            ps, pres = self.aux.probs(win)       # 与 RGB 逐帧相同的 32 帧（线上同语义）
            p_final = self.aux.fuse(p, ps, pres)

        i_raw = int(p.argmax())
        i_reco, p_reco = reject_probs(p_final)
        return dict(
            label=LABELS[i_raw], label_zh=LABEL_ZH[LABELS[i_raw]], confidence=float(p[i_raw]),
            label_reco=LABELS[i_reco], label_zh_reco=LABEL_ZH[LABELS[i_reco]],
            confidence_reco=float(p_reco[i_reco]),
            probs={lb: float(p[j]) for j, lb in enumerate(LABELS)},
            probs_final={lb: float(p_final[j]) for j, lb in enumerate(LABELS)},
            presence=pres,
            lum=lum_mean,
            lum_gate=float(lum_gate or 0.0),
            gate_blocked=lum is not None)


def select_window(frames, fps, policy, window_sec):
    """选出一个识别窗口（32 帧由 predict 内部统一采样）。

    last  取视频最后 window_sec 秒（线上流式语义：reco_span_sec=1.0，判定依据最近 1 秒）
    whole 全片均匀取窗（评测算子语义：对整段已剪好的动作片段均匀抽 32 帧）
    """
    if policy == "whole" or fps <= 0 or len(frames) <= 2:
        return frames
    n = max(2, int(round(window_sec * fps)))
    return frames[-n:]


def main():
    ap = argparse.ArgumentParser(description="VideoMamba 动作识别 · 独立推理")
    ap.add_argument("--input", required=True, help="视频路径（或单张图片，会复制成 32 帧）")
    ap.add_argument("--weights", default=DEFAULT_CKPT,
                    help="识别权重；换回 weights/videomamba_t_behavior_32f_realact10_all.pth "
                         "即回滚到 9 月裁剪链版本（预处理随权重元数据自动切换）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enhance", action="store_true",
                    help="识别前先跑一遍低照度增强模型（线上链路为 下采样 -> 增强 -> 识别）")
    ap.add_argument("--enhance-weights", default=DEFAULT_ENHANCE_CKPT)
    ap.add_argument("--proc-max-side", type=int, default=720,
                    help="进入链路前先把长边缩到该值（当前部署服务链路默认 720）；0 = 不缩放")
    ap.add_argument("--window", choices=["last", "whole"], default="last",
                    help="last=取最后 --window-sec 秒（线上流式语义，默认）；whole=全片均匀取 32 帧（评测语义）")
    ap.add_argument("--window-sec", type=float, default=1.0,
                    help="window=last 时的窗口时长（线上默认 1.0 秒）")
    ap.add_argument("--skel-alpha", type=float, default=SKEL_ALPHA_DEFAULT,
                    help="骨架融合权重 α：p=α·RGB+(1-α)·骨架；>=0.999 关闭骨架分支。默认 0.7（线上默认）")
    ap.add_argument("--skel-no-person", choices=["soft", "other", "rgb"], default="soft",
                    help="无人窗策略：soft=骨架仍以 (1-α) 投 other 票（默认，线上默认）；"
                         "other=强制 other；rgb=骨架弃权按纯 RGB")
    ap.add_argument("--lum-gate", type=float, default=LUM_GATE_DEFAULT,
                    help="照度门门限（两部曲第一步）：提亮前原图 32 帧窗平均亮度低于它时"
                         "只用 RGB、骨架分支不运行（两种决策模式共用；--enhance 时量的是"
                         "增强前的帧）。默认 75（现场亮度拐点），0 = 关门退回纯加权融合")
    ap.add_argument("--json", default=None, help="把完整概率写到该 JSON 文件")
    a = ap.parse_args()
    if torch.device(a.device).type != "cuda":
        ap.error("行为识别只支持 CUDA，请使用 --device cuda 或 cuda:N")

    enhancer = None
    if a.enhance:
        from lowlight_infer import LowLightEnhancer
        enhancer = LowLightEnhancer(a.enhance_weights, device=a.device)

    model = BehaviorRecognizer(a.weights, a.device, skel_alpha=a.skel_alpha,
                               skel_no_person=a.skel_no_person)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"[load] {a.weights}  device={model.device}  参数量 {n_params / 1e6:.2f}M")
    print(f"[load] 输入管线: {model.input_pipeline}")

    if is_image(a.input):
        frames, fps = [read_image(a.input)] * WINDOW_T, 0.0
    else:
        frames, fps = read_video(a.input)
    if a.proc_max_side:
        frames = [downscale_long_side(f, a.proc_max_side) for f in frames]
    frames = select_window(frames, fps, a.window, a.window_sec)
    raw_window = frames                # 提亮前原图，照度门量它（增强是逐帧 1:1，等长同序）

    t0 = time.time()
    if enhancer is not None:
        frames = enhancer.enhance(frames)
    r = model.predict(frames, raw_frames_bgr=raw_window, lum_gate=a.lum_gate)
    dt = time.time() - t0

    chain = "增强 -> 识别" if enhancer is not None else "识别"
    top3 = sorted(r["probs_final"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"[done] {os.path.basename(a.input)}  ({chain}, {dt:.2f}s)")
    print(f"  判定(最终): {r['label_zh_reco']} ({r['label_reco']})  "
          f"置信度 {r['confidence_reco']:.4f}")
    if model.aux is not None:
        if r["gate_blocked"]:
            print(f"  照度门: 提亮前原图亮度 {r['lum']:.1f} < 门限 {r['lum_gate']:.0f}，"
                  f"仅用RGB结果（两部曲第一步，骨架分支未运行）")
        else:
            pres = r["presence"]
            gate = "有人" if pres >= 0.5 else "无人(策略生效)"
            print(f"  照度门: 提亮前原图亮度 {r['lum']:.1f} ≥ 门限 {r['lum_gate']:.0f}，"
                  f"加权融合照常")
            print(f"  骨架分支: α={model.skel_alpha:g}  presence={pres:.3f} ({gate})")
    print(f"  纯RGB对比: {r['label_zh']} ({r['label']})  置信度 {r['confidence']:.4f}")
    print("  Top-3(融合后): " + "  ".join(f"{LABEL_ZH[k]} {v:.4f}" for k, v in top3))

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"input": os.path.abspath(a.input), "labels": LABELS,
                       "top1": r["label"], "confidence": r["confidence"],
                       "top1_reco": r["label_reco"], "confidence_reco": r["confidence_reco"],
                       "probs_rgb": r["probs"], "probs_final": r["probs_final"],
                       "presence": r["presence"],
                       "lum": r["lum"], "lum_gate": r["lum_gate"],
                       "gate_blocked": r["gate_blocked"],
                       "skel_alpha": model.skel_alpha,
                       "skel_enabled": model.aux is not None,
                       "input_pipeline": model.input_pipeline,
                       "proc_max_side": a.proc_max_side, "device": a.device,
                       "weights": os.path.abspath(a.weights),
                       "enhance": a.enhance, "window": a.window,
                       "window_sec": a.window_sec,
                       "seconds": round(dt, 3)}, f, ensure_ascii=False, indent=2)
        print(f"  概率已写入 {a.json}")


if __name__ == "__main__":
    main()
