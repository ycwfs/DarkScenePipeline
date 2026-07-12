"""Shared constants: class labels, recognizer input configs, checkpoint map, perf table."""
import numpy as np

CLASSES = ["Drink", "Jump", "Pick", "Pour", "Push", "Run", "Sit", "Stand", "Turn", "Walk", "Wave"]

KIN_MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
KIN_STD = np.array([0.22803, 0.22145, 0.216989], np.float32)
IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], np.float32)

# per-recognizer input pipeline (frames sampled T, short-side resize, center crop, norm)
RECO_CFG = {
    "r3d": dict(T=16, size=112, resize=128, mean=KIN_MEAN, std=KIN_STD),
    "videomamba": dict(T=32, size=224, resize=256, mean=IN_MEAN, std=IN_STD),
}

REALRESTORER_PROMPT = ("Please restore this low-quality image, recovering its normal "
                       "brightness and clarity.")

CKPT_FILES = {
    "retinexformer": "NTIRE.pth",
    "lightsr_x2": "mambairv2_lightSR_x2.pth",
    "r3d": "r2plus1d_arid.pth",
    "videomamba": "videomamba_t_arid_32f.pth",
    "realrestorer": "realrestorer",  # HF bundle directory
}

# measured on a single RTX 4090 at 320x240 input (see README performance section)
EXPECTED_FPS = {
    ("retinexformer", "off"): 175.0,
    ("retinexformer", "lightsr_x2"): 6.6,
    ("off", "lightsr_x2"): 6.9,
    ("off", "off"): 1000.0,
    ("realrestorer", "off"): 1 / 45.0,
    ("realrestorer", "lightsr_x2"): 1 / 45.0,
}
