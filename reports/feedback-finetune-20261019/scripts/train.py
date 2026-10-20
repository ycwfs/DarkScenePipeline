"""Low-LR VideoMamba continuation with old-data replay and a sealed session holdout."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'reports/feedback-finetune-20261019'
sys.path.insert(0, str(ROOT/'platform/dark-demo-v2'))
from darkpipe.constants import BEHAVIORS, IN_MEAN, IN_STD
from darkpipe.stages.recognize import BehaviorRecognizer


class Windows(Dataset):
    def __init__(self, items: list, augment: bool):
        self.items, self.augment = items, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple:
        path, label, is_new = self.items[i]
        x = np.load(path)
        if x.shape[1:3] != (224, 400):
            x = np.stack([cv2.resize(f, (400, 224), interpolation=cv2.INTER_AREA) for f in x])
        if self.augment and random.random() < .5:
            x = x[:, :, ::-1]
        x = np.ascontiguousarray(x).astype(np.float32)
        x /= 255.
        x -= IN_MEAN
        x /= IN_STD
        return torch.from_numpy(x).permute(3, 0, 1, 2), label, is_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=19)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    m = json.loads((OUT/'data-manifest.json').read_text())
    base = [(str(p), BEHAVIORS.index(p.parent.name), False)
            for p in sorted(Path(m['base_cache']).glob('*/*.npy'))]
    new = [(r['path'], 9, True) for r in m['windows'] if r['split'] == 'train']
    assert new and not any(Path(p).parent.name == 'holdout' for p, _, _ in new)
    items = base + new
    counts = Counter(y for _, y, _ in base)
    weights = [.8 / len(counts) / counts[y] for _, y, _ in base] + [.2/len(new)] * len(new)
    sampler = WeightedRandomSampler(weights, num_samples=len(items), replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(Windows(items, True), batch_size=a.batch, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)
    ckpt = OUT/'before/platform/dark-demo-v2/ckpts/videomamba_t_behavior_32f_realactall_full_pn.pth'
    rec = BehaviorRecognizer(str(ckpt.parent), skel_fusion_alpha=1.)
    rec.ckpt = str(ckpt)
    rec.load('cuda:0')
    net = rec.net
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([
        {'params': [p for n, p in net.named_parameters() if not n.startswith('head.')], 'lr': a.lr*.1},
        {'params': [p for n, p in net.named_parameters() if n.startswith('head.')], 'lr': a.lr},
    ], weight_decay=.05)
    total = a.epochs * len(loader)
    warm = max(1, len(loader)//2)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step:
        (step+1)/warm if step < warm else .5*(1+np.cos(np.pi*min(1.,(step-warm)/max(1,total-warm)))))
    scaler = torch.amp.GradScaler('cuda')
    meta = torch.load(ckpt, map_location='cpu', weights_only=False)
    config = dict(epochs=a.epochs, head_lr=a.lr, backbone_lr=a.lr*.1, batch=a.batch,
                  seed=a.seed, old_count=len(base), new_count=len(new), new_sampling_mass=.2,
                  lwf_old_only=.2, label_smoothing=.1, steps_per_epoch=len(loader),
                  init_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  heldout_sessions=m['heldout_sessions'], holdout_used_for_training=False,
                  selection='ABCP regression and training negatives; sealed session holdout only after selection')
    (OUT/'training-config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('[config]', json.dumps(config), flush=True)
    hist = []
    for epoch in range(1, a.epochs+1):
        net.train()
        started = time.monotonic()
        sums = Counter()
        for step, (x, y, is_new) in enumerate(loader, 1):
            x, y, is_new = x.cuda(non_blocking=True), y.cuda(non_blocking=True), is_new.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = net(x)
                ce = F.cross_entropy(logits, y, label_smoothing=.1)
                old = ~is_new
                kl = logits.new_tensor(0.)
                if old.any():
                    with torch.no_grad():
                        target = teacher(x[old])
                    kl = F.kl_div(F.log_softmax(logits[old]/2,1), F.log_softmax(target/2,1),
                                  log_target=True,reduction='batchmean')*4
                loss = ce + .2*kl
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch} step {step}')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            scaler.step(opt)
            scaler.update()
            schedule.step()
            sums.update(loss=float(loss), ce=float(ce), kl=float(kl),
                        correct=int((logits.argmax(1)==y).sum()), examples=len(y),
                        new_examples=int(is_new.sum()))
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f'[train] epoch={epoch} step={step}/{len(loader)} loss={sums["loss"]/step:.4f} '
                      f'acc={sums["correct"]/sums["examples"]:.4f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        record = dict(epoch=epoch, loss=sums['loss']/len(loader), ce=sums['ce']/len(loader),
                      kl=sums['kl']/len(loader), train_top1=sums['correct']/sums['examples'],
                      examples=sums['examples'], new_examples=sums['new_examples'],
                      seconds=time.monotonic()-started)
        hist.append(record)
        save = {k:v for k,v in meta.items() if k not in ('model','realact_top1','realact_macro_f1','arid_top1','arid_macro_f1')}
        save.update(model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                    epoch=epoch, init=ckpt.name, source='realact_all_pn + platform_feedback_20261019',
                    preprocess='fullfov224x400', training=config, training_metrics=record,
                    archive_sha256=m['archive_sha256'], lwf=.2, lwf_scope='old replay only')
        dest=OUT/'checkpoints'/f'epoch-{epoch:02d}.pth'
        torch.save(save,dest)
        (OUT/'training-history.json').write_text(json.dumps(hist,indent=2)+'\n')
        print('[checkpoint]',dest,flush=True)


if __name__ == '__main__':
    main()
