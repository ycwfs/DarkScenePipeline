"""Create an isolated runnable operator candidate, leaving official payloads untouched."""
import ast
import json
import shutil
import sys
from common import OP, OUT, ROOT, save, digest


def replace(path, old, new, count=1):
    text=path.read_text()
    assert text.count(old)==count,(path,old,text.count(old))
    path.write_text(text.replace(old,new))


def main():
    dest=OUT/'operator/dark-demo-v2'
    assert not dest.exists()
    shutil.copytree(OP,dest,ignore=shutil.ignore_patterns('ckpts','__pycache__','*.pyc','.cache'))
    (dest/'ckpts').symlink_to(OP/'ckpts',target_is_directory=True)
    shutil.copy2(OUT/'implementation/wave_prior.py',dest/'darkpipe/person/wave_prior.py')
    config=dest/'darkpipe/config.py'
    replace(config,'    object_min_conf: float = 0.581','    object_min_conf: float = 0.581\n    wave_pose_min_frames: int = 3  # distinct source frames; 0 disables only wave prior')
    replace(config,'    if not 0.0 <= cfg.object_min_conf <= 1.0:',
        '    if isinstance(cfg.wave_pose_min_frames, bool) or not isinstance(cfg.wave_pose_min_frames, int) or not 0 <= cfg.wave_pose_min_frames <= 32:\n        _die("--wave_pose_min_frames must be an integer within [0, 32]")\n\n    if not 0.0 <= cfg.object_min_conf <= 1.0:')
    action=dest/'darkpipe/person/action_engine.py'
    replace(action,'from .core import fuse_person','from .core import fuse_person\nfrom .wave_prior import apply_wave_prior')
    replace(action,"                    row['object_prior'] = dict(applied=False, reason='disabled', label=row['label'])",
        "                    row['object_prior'] = dict(applied=False, reason='disabled', label=row['label'])\n                apply_wave_prior(task, row, min_frames=self.options.get('wave_pose_min_frames', 3),\n                    lum_gate=self.options['lum_gate'], pose_enabled=self.options['alpha'] < .999)\n                self.stats['wave_'+row['wave_prior']['reason']] += 1")
    runtime=dest/'darkpipe/person/runtime.py'
    replace(runtime,'person_object_prior=cfg.person_object_prior, object_min_conf=cfg.object_min_conf)',
        'person_object_prior=cfg.person_object_prior, object_min_conf=cfg.object_min_conf,\n                wave_pose_min_frames=cfg.wave_pose_min_frames)')
    replace(runtime,"f'物体先验={cfg.person_object_prior} 物体阈值={cfg.object_min_conf:g}', flush=True)",
        "f'物体先验={cfg.person_object_prior} 物体阈值={cfg.object_min_conf:g} '\n                  f'挥手骨架最少帧={cfg.wave_pose_min_frames}', flush=True)")
    main=dest/'main.py'
    replace(main,'    p.add_argument("--pose_overlay", type=parse_bool, default=True,',
        '    p.add_argument("--wave_pose_min_frames", type=int, default=3,\n                   help="逐人挥手需要同一腕关节高于双肩的独立源帧数，默认3，0关闭；关键点置信度至少0.5，低亮或原亮度未知时跳过")\n    p.add_argument("--pose_overlay", type=parse_bool, default=True,')
    replace(main,'        person_object_prior=getattr(args, "person_object_prior", True),',
        '        person_object_prior=getattr(args, "person_object_prior", True),\n        wave_pose_min_frames=getattr(args, "wave_pose_min_frames", 3),',count=2)
    replace(main,'                   "pose_overlay": cfg.pose_overlay,',
        '                   "pose_overlay": cfg.pose_overlay, "wave_pose_min_frames": cfg.wave_pose_min_frames,')
    manifest=dest/'suanzi.json';d=json.loads(manifest.read_text())
    d['inputs'].append(dict(name='wave_pose_min_frames',type='Int',default=3,
        description='挥手候选需同一只腕在当前1秒窗口内至少3个不同源帧高于双肩；腕及双肩置信度均≥0.5。原图低亮/亮度未知/纯RGB时跳过，亮场证据不足报其他。0关闭，范围0–32。',
        annotations=dict(show='挥手骨架先验最少帧数',component='intInput',parameter=1)))
    d['implementation']['container']['args']+=['--wave_pose_min_frames',dict(inputValue='wave_pose_min_frames')]
    manifest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    readme=dest/'README.md'
    readme.write_text('# 挥手骨架先验实验候选（未替换正式包）\n\n新增 `--wave_pose_min_frames 3`，0关闭；同一只腕高于双肩，腕和双肩置信度均至少0.5，源帧去重，在现有1秒窗口内累计。照度门35约束，低亮/原亮度未知/纯RGB时跳过。明亮场景姿态证据不足报其他，可能漏检。其他模型与参数不变。ckpts在本机链接到原正式权重；本目录尚非可直接上传的归档包。\n\n下面保留原交付说明作为基础流程参考。\n\n'+readme.read_text())
    sys.path.insert(0,str(ROOT/'platform'))
    from package_v2 import validate_contract
    validate_contract(dest)
    files={str(p.relative_to(dest)):digest(p)for p in dest.rglob('*')if p.is_file()and 'ckpts'not in p.parts and '__pycache__'not in p.parts}
    for p in dest.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    save(OUT/'candidate-code-manifest.json',dict(files=files,contract_passed=True,
        same_models=True,source=str(dest.relative_to(ROOT)),weight_directory=str(OP/'ckpts'),wave_min_frames=3))
    print('Runnable isolated candidate built; CLI/manifest validated; official operator untouched')


if __name__=='__main__':main()
