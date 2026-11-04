"""Preprocessing for already enhanced platform recordings, without old label bars."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OP = ROOT / 'platform/dark-demo-v2'
SOURCE = ROOT / 'online-negative-sample/20261012_083305_78'
REQUESTED = SOURCE / 'drinking_water/20261012_095918_drinking_water_0060.mp4'


def content_height(height):
    candidates = [h for h in range(height - 180, height) if h % 2 == 0
                  and h + max(48, round(.08 * h)) + (max(48, round(.08 * h)) & 1) == height]
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous old label bar: {height=} {candidates=}')
    return candidates[0]
