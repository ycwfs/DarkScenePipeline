"""Shared constants: class labels, recognizer input configs, checkpoint map, perf table."""
import numpy as np

CLASSES = ["Drink", "Jump", "Pick", "Pour", "Push", "Run", "Sit", "Stand", "Turn", "Walk", "Wave"]

# General-behavior label set (--recognize behavior | xclip). The nine required behaviors plus
# an explicit `other`: without it every window is forced into one of the nine, so ordinary
# activity gets reported as a behavior at some confidence.
BEHAVIORS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
             "shake_hands", "other"]

BEHAVIOR_DISPLAY = {
    "wave": "Waving", "throw": "Throwing object", "chase": "Chasing", "fall": "Falling",
    "fight": "Fighting", "talk": "Talking", "drink": "Drinking water",
    "pick_up": "Picking up object", "shake_hands": "Shaking hands", "other": "Other",
}

# Prompt ensemble for the open-vocabulary recognizer: several phrasings per behavior, scored
# independently and averaged. Wording follows CLIP-style captions ("a person ...ing"), which
# X-CLIP's text tower was trained on.
BEHAVIOR_PROMPTS = {
    "wave": ["a person waving their hand",
             "someone waving hello with a raised arm",
             "a person waving goodbye",
             "a man or woman waving an arm in the air"],
    "throw": ["a person throwing an object",
              "someone tossing something through the air",
              "a person hurling an item away",
              "throwing a ball or an object"],
    "chase": ["one person chasing another person",
              "a person running after someone",
              "two people running, one pursuing the other",
              "a foot chase between two people"],
    "fall": ["a person falling down to the ground",
             "someone collapsing onto the floor",
             "a person losing balance and falling over",
             "a man or woman falling down"],
    "fight": ["two people fighting",
              "a person punching another person",
              "people kicking and hitting each other",
              "a physical fight or brawl between people"],
    "talk": ["people talking to each other",
             "two persons having a conversation",
             "a person speaking with someone",
             "people chatting face to face"],
    "drink": ["a person drinking water",
              "someone drinking from a cup or a bottle",
              "a person taking a sip of a drink",
              "drinking a beverage"],
    "pick_up": ["a person picking something up from the ground",
                "someone bending down to pick up an object",
                "a person lifting an item off the floor",
                "picking up an object"],
    "shake_hands": ["two people shaking hands",
                    "a handshake between two persons",
                    "people greeting each other with a handshake",
                    "shaking hands"],
    # `other` must be described as concrete ordinary activity, never as negation: CLIP cannot
    # ground "no notable behavior" and such a class simply never wins the argmax (measured:
    # 0.076 recall). Generic everyday actions, not an enumeration of the eval set's negatives.
    "other": ["a person walking normally",
              "a person standing still",
              "a person sitting down",
              "a person running alone",
              "people eating a meal",
              "a person smiling or laughing",
              "a person riding a bicycle",
              "a person exercising",
              "a person climbing",
              "a person clapping their hands",
              "an ordinary everyday activity"],
}

# fallback ensemble for open-vocabulary labels passed via --labels, which have no entry in
# BEHAVIOR_PROMPTS. "{}" is filled with the label text verbatim.
PROMPT_TEMPLATES = ["a video of {}", "a person {}", "footage of {}", "{}"]

# Open-set rejection for the zero-shot recognizer: report `other` when no named behavior wins
# with at least this probability. Even with concrete prompts a text `other` class is too weak
# on its own, so ordinary footage gets forced into one of the nine.
# Calibrated on the HMDB VALIDATION split (never test): tau=0.4 maximised macro-F1
# (0.53 vs 0.43 for plain argmax; `other` recall 0.59 vs 0.08, top-1 0.61 vs 0.35).
XCLIP_REJECT_TAU = 0.4

KIN_MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
KIN_STD = np.array([0.22803, 0.22145, 0.216989], np.float32)
IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], np.float32)

# per-recognizer input pipeline (frames sampled T, short-side resize, center crop, norm).
# xclip mirrors its shipped preprocessor_config.json exactly: short side -> 224, centre crop
# 224, ImageNet statistics (X-CLIP uses ImageNet norm, not CLIP's).
RECO_CFG = {
    "r3d": dict(T=16, size=112, resize=128, mean=KIN_MEAN, std=KIN_STD),
    "videomamba": dict(T=32, size=224, resize=256, mean=IN_MEAN, std=IN_STD),
    "behavior": dict(T=32, size=224, resize=256, mean=IN_MEAN, std=IN_STD),
    "xclip": dict(T=32, size=224, resize=224, mean=IN_MEAN, std=IN_STD),
}

# Recognizers whose vocabulary is BEHAVIORS rather than the ARID CLASSES
BEHAVIOR_RECOGNIZERS = ("behavior", "xclip")

REALRESTORER_PROMPT = ("Please restore this low-quality image, recovering its normal "
                       "brightness and clarity.")

# Super-resolution factor. All three backends are scale-generic: bicubic takes any integer,
# and both neural arches build their upsampler from `upscale` (pixelshuffle for lightSR,
# conv+PixelShuffle / two-stage for CATANet at x4). What is NOT generic is the weights — a
# checkpoint is trained for one factor — so x3/x4 need the matching file from the same
# upstream release, named by SR_CKPT_FILES below. `bicubic` needs no weights at any scale.
SR_SCALES = (2, 3, 4)
SR_BACKENDS = ("bicubic", "lightsr", "catanet")
# Legacy CLI values kept working: each pins its scale, so `--sr lightsr_x2 --sr-scale 3` is
# rejected as self-contradictory rather than silently loading x3.
SR_ALIASES = {"bicubic_x2": ("bicubic", 2), "lightsr_x2": ("lightsr", 2),
              "catanet_x2": ("catanet", 2)}
SR_CKPT_FILES = {"lightsr": "mambairv2_lightSR_x{s}.pth", "catanet": "catanet_x{s}.pth"}


def sr_ckpt_file(backend: str, scale: int) -> str:
    """Checkpoint filename for a (backend, scale) pair; "" for weightless backends."""
    return SR_CKPT_FILES.get(backend, "").format(s=scale)


CKPT_FILES = {
    "retinexformer": "NTIRE.pth",
    "cidnet": "CIDNet_generalization.pth",
    "lightsr_x2": "mambairv2_lightSR_x2.pth",
    "catanet_x2": "catanet_x2.pth",
    "r3d": "r2plus1d_arid.pth",
    "videomamba": "videomamba_t_arid_32f.pth",
    "behavior": "videomamba_t_behavior_32f.pth",
    "xclip": "xclip-base-patch16-zero-shot",  # HF snapshot directory
    "realrestorer": "realrestorer",  # HF bundle directory
}

# Measured end-to-end on ONE RTX 3090 at 640x480 with a recognizer attached
# (`compare/behavior/bench_pipeline.py`, 300 frames, steady state after model load).
# 640x480 rather than the old 320x240 figures: throughput is resolution-bound, and quoting
# the smallest resolution made the pipeline look ~4x faster than a deployed camera would see.
# Scale roughly: x2.0 at 320x240, x0.4 at 1280x720 (where --gpus recovers the loss).
EXPECTED_FPS = {
    ("retinexformer", "off"): 22.0,
    ("retinexformer", "bicubic_x2"): 21.5,   # -2% vs sr=off in a paired run: cv2.resize is
    ("retinexformer", "lightsr_x2"): 1.2,    # ~1 ms and the 2x-size encode is on its own
    ("retinexformer", "catanet_x2"): 1.2,    # thread, so x2 output is nearly free
    ("cidnet", "off"): 19.5,
    ("cidnet", "bicubic_x2"): 18.7,
    ("cidnet", "lightsr_x2"): 1.0,
    ("cidnet", "catanet_x2"): 1.1,
    ("off", "bicubic_x2"): 98.0,  # decode + resize + encode only, measured on 150 frames
    ("off", "lightsr_x2"): 1.3,
    ("off", "catanet_x2"): 1.2,   # 828 ms/frame at batch 1 is the ceiling (chunk 1 by default)
    ("off", "off"): 300.0,
    ("realrestorer", "off"): 1 / 45.0,
    ("realrestorer", "lightsr_x2"): 1 / 45.0,
}
