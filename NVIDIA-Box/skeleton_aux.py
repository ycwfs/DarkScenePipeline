"""Compatibility import for the standalone model CLI.

Use the same skeleton association, per-person classification and winner selection as
run.py and the platform. Runtime and weights are included by package_nvidia_box.py.
"""
from runtime.darkpipe.stages.skeleton_aux import (  # noqa: F401
    BEHAVIORS, COCO_EDGES, COMP_FRAC, COMP_MIN_VIS, MAX_PERSONS, NO_PERSON_MODES,
    NO_PERSON_ZH, POSE_ONNX, POSE_SCORE_THR, SKEL_CKPT, TAU, STGCN, STGCNBlock,
    SkeletonAux, coco_adjacency, completeness, duplicate_pose, merge_persons,
    normalize_window,
)
