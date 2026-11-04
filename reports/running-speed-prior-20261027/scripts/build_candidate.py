"""Add a selectable speed prior to accepted wave candidate, isolated from delivery."""
import ast
import json
import shutil
import sys
from common import OUT, ROOT, OP, WAVE, save, digest


def replace(path, old, new, count=1):
    content=path.read_text();assert content.count(old)==count,(path,old,content.count(old))
    path.write_text(content.replace(old,new))


def main():
    sel=json.loads((OUT/'selection.json').read_text());tau=sel['threshold'];policy=sel['missing_policy']
    dest=OUT/'operator/dark-demo-v2';assert not dest.exists()
    shutil.copytree(WAVE/'operator/dark-demo-v2',dest,ignore=shutil.ignore_patterns('ckpts','__pycache__','*.pyc'))
    (dest/'ckpts').symlink_to(OP/'ckpts',target_is_directory=True)
    shutil.copy2(OUT/'implementation/running_prior.py',dest/'darkpipe/person/running_prior.py')
    cfg=dest/'darkpipe/config.py'
    replace(cfg,'    wave_pose_min_frames: int = 3',f'    running_pose_min_speed: float = {tau}  # body-box heights/sec; zero disables\n    wave_pose_min_frames: int = 3')
    replace(cfg,'    if isinstance(cfg.wave_pose_min_frames, bool)',
        '    if not math.isfinite(cfg.running_pose_min_speed) or cfg.running_pose_min_speed < 0:\n        _die("--running_pose_min_speed must be finite and nonnegative")\n\n    if isinstance(cfg.wave_pose_min_frames, bool)')
    if 'import math' not in cfg.read_text():replace(cfg,'from dataclasses import', 'import math\nfrom dataclasses import')
    core=dest/'darkpipe/person/core.py'
    replace(core,'    binding_valid: bool = False\n','    binding_valid: bool = False\n    frame_shape: tuple | None = None\n')
    replace(core,'                                   self.box.copy(), binding_valid))','                                   self.box.copy(), binding_valid, frame.shape[:2]))')
    action=dest/'darkpipe/person/action_engine.py'
    replace(action,'from .wave_prior import apply_wave_prior','from .wave_prior import apply_wave_prior\nfrom .running_prior import apply_running_prior')
    replace(action,"                self.stats['wave_'+row['wave_prior']['reason']] += 1",
        "                self.stats['wave_'+row['wave_prior']['reason']] += 1\n                apply_running_prior(task, row, self.options.get('running_pose_min_speed', "+str(tau)+"),\n                    self.options['lum_gate'], pose_enabled=self.options['alpha'] < .999, missing_policy="+repr(policy)+")\n                self.stats['running_'+row['running_prior']['reason']] += 1")
    runtime=dest/'darkpipe/person/runtime.py'
    replace(runtime,'wave_pose_min_frames=cfg.wave_pose_min_frames)',
        'wave_pose_min_frames=cfg.wave_pose_min_frames, running_pose_min_speed=cfg.running_pose_min_speed)')
    replace(runtime,"f'挥手骨架最少帧={cfg.wave_pose_min_frames}', flush=True)",
        "f'挥手骨架最少帧={cfg.wave_pose_min_frames} 跑动速度阈值={cfg.running_pose_min_speed:g}', flush=True)")
    main=dest/'main.py'
    replace(main,'    p.add_argument("--wave_pose_min_frames",',
        f'    p.add_argument("--running_pose_min_speed", type=float, default={tau},\n                   help="跑动骨架平均速度下限，单位人体框高度/秒；0关闭，受照度门约束")\n    p.add_argument("--wave_pose_min_frames",')
    replace(main,'        wave_pose_min_frames=getattr(args, "wave_pose_min_frames", 3),',
        f'        wave_pose_min_frames=getattr(args, "wave_pose_min_frames", 3),\n        running_pose_min_speed=getattr(args, "running_pose_min_speed", {tau}),',count=2)
    replace(main,'"wave_pose_min_frames": cfg.wave_pose_min_frames,','"wave_pose_min_frames": cfg.wave_pose_min_frames, "running_pose_min_speed": cfg.running_pose_min_speed,')
    manifest=dest/'suanzi.json';d=json.loads(manifest.read_text())
    d['inputs'].append(dict(name='running_pose_min_speed',type='Float',default=tau,
        description=f'跑动骨架速度下限，平均关节位移除以人体框高及秒数；默认{tau}，0关闭。低亮或亮度未知跳过；姿态不足策略：{policy}。',
        annotations=dict(show='跑动骨架速度阈值',component='floatInput',parameter=1)))
    d['implementation']['container']['args']+=['--running_pose_min_speed',dict(inputValue='running_pose_min_speed')]
    manifest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    readme=dest/'README.md'
    readme.write_text(f'# 跑动速度先验实验候选（未替换正式交付）\n\n新增 `--running_pose_min_speed {tau}`，人体框高度/秒，0关闭。同人现有1秒窗口，约0.2秒的关键点位移，至少5个可靠共同关节、3个观测对且覆盖0.3秒，关键点置信度≥0.5。源帧去重、不平滑。低亮/原亮度未知/骨架关闭跳过；姿态不足策略：{policy}。过滤后标签统一用于显示、事件和动作片段，原模型概率保留。\n\n保留挥手3帧与正式7类物体先验。权重本机符号链接，尚非可直接上传归档；验证见上级报告。\n\n'+readme.read_text())
    for p in dest.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    sys.path.insert(0,str(ROOT/'platform'));from package_v2 import validate_contract
    validate_contract(dest)
    save(OUT/'candidate-code-manifest.json',dict(threshold=tau,missing_policy=policy,contract_passed=True,
        files={str(p.relative_to(dest)):digest(p)for p in dest.rglob('*')if p.is_file()and 'ckpts'not in p.parts and '__pycache__'not in p.parts}))
    print('Isolated runnable candidate built',tau,policy)

if __name__=='__main__':main()
