"""读入示例数据：图片 / 图片目录 / 视频，统一为 BGR uint8 帧列表。"""
import os

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMG_EXT


def read_image(path):
    """图片路径 -> BGR uint8 HxWx3；读不到时给出明确报错。"""
    f = cv2.imread(path, cv2.IMREAD_COLOR)
    if f is None:
        raise SystemExit(f"error: 读不到图片 {path}")
    return f


def list_images(folder):
    fs = sorted(os.path.join(folder, f) for f in os.listdir(folder) if is_image(f))
    if not fs:
        raise SystemExit(f"error: 目录里没有图片 {folder}")
    return fs


def read_video(path):
    """视频路径 -> (BGR uint8 帧列表, fps)；解不开时给出明确报错。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"error: 打不开视频 {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"error: 视频解不出任何帧 {path}")
    return frames, fps


def sample_uniform(frames, n):
    """任意帧数 -> 均匀取 n 帧（不足 n 帧时循环复用），与平台评测/识别同一条采样规则：
    np.linspace(0, n-1, n).round() —— 首尾帧必在，短片段按比例重复取帧。"""
    k = len(frames)
    idx = np_linspace_idx(k, n)
    return [frames[i] for i in idx]


def np_linspace_idx(k, n):
    import numpy as np
    return np.linspace(0, k - 1, n).round().astype(int)


def downscale_long_side(frame, max_side):
    """长边缩到 max_side（从不上采样），偶数对齐，INTER_AREA —— 与部署的
    DownscaleStage（--proc_max_side）逐行一致。max_side<=0 原样返回。"""
    h, w = frame.shape[:2]
    if max_side is None or max_side <= 0 or max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    nw, nh = (int(round(w * s)) // 2) * 2, (int(round(h * s)) // 2) * 2
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def encode_video(frames, path, fps):
    """BGR uint8 帧列表 -> mp4（mp4v 编码，仅用于示例结果预览）。"""
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise SystemExit(f"error: 打不开输出视频 {path}")
    for f in frames:
        vw.write(f)
    vw.release()
