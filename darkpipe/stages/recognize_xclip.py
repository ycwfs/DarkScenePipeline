"""Open-vocabulary recognizer: X-CLIP (`microsoft/xclip-base-patch16-zero-shot`).

Labels are text, not a trained head, so any vocabulary works without retraining — the nine
behaviors by default, anything else via `--labels "a,b,c"`. This is what covers `chase`,
which no public action dataset contains.

Speed note: in X-CLIP only `prompts_generator` is video-conditioned; the text tower's output
(`text_projection(text_model(ids))`) does not depend on the frames. So the tokenizer + 12-layer
text transformer run ONCE in load() and only the small cross-attention prompt generator runs
per window. `_infer` below is the exact decomposition of `XCLIPModel.forward` around that
cached tensor — `tests/test_xclip.py` asserts it matches the stock forward numerically.

Prompt ensembling: each label carries several phrasings; all of them go through the prompt
generator and their cosine similarities are averaged per label before the softmax.
"""
import os

import numpy as np
import torch

from ..constants import (BEHAVIOR_PROMPTS, BEHAVIORS, CKPT_FILES, PROMPT_TEMPLATES,
                         XCLIP_REJECT_TAU)
from .recognize import _WindowRecognizer


def prompts_for(label: str) -> list:
    """Curated ensemble when we have one, generic templates for open-vocabulary labels."""
    return BEHAVIOR_PROMPTS.get(label) or [t.format(label) for t in PROMPT_TEMPLATES]


def reject_probs(p, labels, tau):
    """Open-set rule: unless a NAMED behavior clears `tau`, call it `other`.

    Reported `other` confidence is 1 - max(named), i.e. "probability it is none of the nine" —
    the quantity the rule actually decided on. The untouched distribution still reaches the
    caller through the top-k list.

    Module-level so the calibration script can sweep tau over stored raw probabilities instead
    of re-running the vision tower, without reimplementing (and drifting from) the rule.
    """
    if "other" not in labels or tau <= 0:
        return p
    o = labels.index("other")
    named = np.delete(p, o)
    if named.max() < tau and p.argmax() != o:
        p = p.copy()
        # max() keeps `other` the argmax even for a tau above 0.5
        p[o] = max(1.0 - float(named.max()), float(named.max()) + 1e-6)
    return p


class XCLIPRecognizer(_WindowRecognizer):
    kind = "xclip"

    def __init__(self, ckpt_dir: str, stride: int | None = None,
                 span_sec: float | None = None, labels=None, model_dir: str = "",
                 reject_tau: float = XCLIP_REJECT_TAU):
        super().__init__(ckpt_dir, stride=stride, span_sec=span_sec)
        self.labels = list(labels) if labels else list(BEHAVIORS)
        self.reject_tau = reject_tau
        self.ckpt = model_dir or os.path.join(ckpt_dir, CKPT_FILES["xclip"])
        self._text = None    # [P, D] cached, video-independent text embeddings
        self._group = None   # [P] label index of each prompt
        self._gcount = None  # [L] prompts per label

    def _build(self):
        from transformers import XCLIPModel
        return XCLIPModel.from_pretrained(self.ckpt)

    def load(self, device: str) -> None:
        super().load(device)
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.ckpt)
        self.set_labels(self.labels)

    def set_labels(self, labels, prompts=None):
        """(Re)build the cached text embeddings. Cheap — no vision work — so the eval scripts
        can sweep prompt sets against already-encoded clips."""
        self.labels = list(labels)
        texts, group = [], []
        for i, lab in enumerate(self.labels):
            for p in (prompts or {}).get(lab, prompts_for(lab)):
                texts.append(p)
                group.append(i)
        batch = self._tok(texts, padding=True, return_tensors="pt").to(self.device)
        # no_grad, not inference_mode: the cached tensor outlives this call and inference
        # tensors cannot be reused outside an inference context (e.g. from the eval scripts).
        with torch.no_grad():
            self._text = self.net.get_text_features(**batch).float()
        self._group = torch.tensor(group, device=self.device)
        self._gcount = torch.bincount(self._group, minlength=len(self.labels)).float()
        return self

    @torch.no_grad()
    def encode_video(self, arr):
        """arr: (T,H,W,C) float32 -> (video_embeds (1,D), img_features (1,patches,D)).

        This is the expensive half (the ViT over T frames) and it does not depend on the
        labels — so the calibration/eval scripts encode a split once and then score any
        number of prompt sets against it for free.
        """
        m = self.net
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(self.device)  # (T,C,H,W)
        T = x.shape[0]
        with torch.autocast("cuda", dtype=torch.float16):
            vout = m.vision_model(pixel_values=x)
            video = m.visual_projection(vout[1])               # (T, D)
            video = m.mit(video.unsqueeze(0))[1]               # (1, D)
            img = m.prompts_visual_layernorm(vout[0][:, 1:, :])
            img = img @ m.prompts_visual_projection
            img = img.view(1, T, -1, video.shape[-1]).mean(1)   # (1, patches, D)
        return video.float(), img.float()

    @torch.no_grad()
    def score(self, video, img, text=None):
        """Cosine similarity of one clip against each prompt -> (P,) float32."""
        m = self.net
        text = self._text if text is None else text
        with torch.autocast("cuda", dtype=torch.float16):
            t = text.unsqueeze(0)                              # (1, P, D)
            t = t + m.prompts_generator(t, img)
            v = video / video.norm(p=2, dim=-1, keepdim=True)
            t = t / t.norm(p=2, dim=-1, keepdim=True)
            return torch.einsum("bd,bkd->bk", v, t)[0].float()

    def _infer(self, arr):
        """arr: (T,H,W,C) float32 -> probability vector over self.labels."""
        p = self.pool(self.score(*self.encode_video(arr))).cpu().numpy()
        return self.reject(p)

    def reject(self, p):
        return reject_probs(p, self.labels, self.reject_tau)

    @torch.no_grad()
    def pool(self, sim):
        """Prompt ensemble -> per-label probabilities (mean cosine per label, then softmax)."""
        per_label = torch.zeros_like(self._gcount).index_add_(0, self._group, sim) / self._gcount
        return (per_label * self.net.logit_scale.exp().float()).softmax(0)

    def close(self) -> None:
        self._text = None
        super().close()


def zero_shot_probs(model, frames_bchw, text_embeds, group, gcount, device):
    """Same math as XCLIPRecognizer._infer, exposed for the offline evaluation scripts.

    frames_bchw: (B,T,C,H,W) normalized float tensor. Returns (B, L) probabilities.
    """
    B, T = frames_bchw.shape[:2]
    with torch.autocast("cuda", dtype=torch.float16):
        vout = model.vision_model(pixel_values=frames_bchw.flatten(0, 1).to(device))
        video = model.visual_projection(vout[1])
        video = model.mit(video.view(B, T, -1))[1]
        img = model.prompts_visual_layernorm(vout[0][:, 1:, :])
        img = img @ model.prompts_visual_projection
        img = img.view(B, T, -1, video.shape[-1]).mean(1)
        text = text_embeds.unsqueeze(0).expand(B, -1, -1)
        text = text + model.prompts_generator(text, img)
        video = video / video.norm(p=2, dim=-1, keepdim=True)
        text = text / text.norm(p=2, dim=-1, keepdim=True)
        sim = torch.einsum("bd,bkd->bk", video, text).float()  # (B, P)
    per_label = torch.zeros(B, len(gcount), device=sim.device)
    per_label.index_add_(1, group, sim)
    per_label = per_label / gcount
    return (per_label * model.logit_scale.exp().float()).softmax(1)


__all__ = ["XCLIPRecognizer", "prompts_for", "reject_probs", "zero_shot_probs"]
