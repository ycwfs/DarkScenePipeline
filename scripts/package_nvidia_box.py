#!/usr/bin/env python3
"""Sync the platform runtime and build a self-contained native NVIDIA Box ZIP."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platform'))
from package_v2 import ASSETS, WEIGHTS, package_files, validate_contract

VERSION = '20261103-orin-box-candidate'
BOX_FILES = ('README.md', 'README_legacy_20261022.md', 'models.json', 'MODEL_TOOLS.md', 'VERIFICATION.md', 'requirements.txt', 'constraints.txt',
             'run.py', 'run.sh', 'setup_env.sh', 'check_env.py', 'check_models.py',
             'behavior_infer.py', 'lowlight_infer.py', 'skeleton_aux.py', 'video_io.py',
             'runtime-overrides.json', 'JETSON.md', 'requirements-jetson.txt',
             'setup_jetson.sh', 'build_engines.py', 'benchmark_box.py', 'box-models.json')
SAMPLES = ('dark_frame.png', 'sample_drink.mp4', 'sample_wave.mp4')
LEGACY_WEIGHT = 'videomamba_t_behavior_32f_realact10_all.pth'
BOX_WEIGHTS = ('rtmo-s_person_20261024_box.onnx',)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def runtime_sources(platform: Path) -> dict[Path, Path]:
    validate_contract(platform)
    files = package_files(platform)
    return {Path('runtime') / p: p for p in files
            if p in (Path('main.py'), Path('oputil.py')) or
            (p.parts[0] == 'darkpipe' and (p.suffix == '.py' or str(p) in ASSETS))}


def build(box: Path, platform: Path, output: Path, sync: bool = False) -> dict:
    box, platform, output = box.resolve(), platform.resolve(), output.resolve()
    if box in output.parents or output.suffix != '.zip':
        raise ValueError('output must be a .zip outside the Box tree')
    mapping = runtime_sources(platform)
    overrides = json.loads((box / 'runtime-overrides.json').read_text())
    modified = {Path('runtime') / p: record for p, record in overrides['modified'].items()}
    added = {Path('runtime') / p for p in overrides['added']}
    if not set(modified) <= set(mapping) or added & set(mapping):
        raise ValueError('Box overrides do not match platform/runtime file boundaries')
    for rel in set(modified) | added:
        if '..' in rel.parts or rel.is_absolute() or rel.suffix != '.py':
            raise ValueError(f'Invalid runtime override: {rel}')
    for dest, record in modified.items():
        if digest(platform / mapping[dest]) != record['platform_sha256']:
            raise ValueError(f'Platform changed beneath Box override; review/rebase before sync: {dest}')
    mapping[Path('models.json')] = Path('models.json')
    mapping.update({Path('weights') / name: Path('ckpts') / name for name in WEIGHTS})
    mapping[Path('weights') / LEGACY_WEIGHT] = Path('ckpts/videomamba_t_behavior_32f.pth')
    for dest, source in mapping.items():
        if sync and dest not in modified:
            (box / dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(platform / source, box / dest)
        if not (box / dest).is_file() or (dest not in modified and digest(box / dest) != digest(platform / source)):
            raise ValueError(f'{dest} differs from platform {source}; use --sync after review')
    actual_runtime = {p.relative_to(box) for p in (box / 'runtime').rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts}
    expected_runtime = {p for p in mapping if p.parts[0] == 'runtime'} | added
    if actual_runtime != expected_runtime:
        raise ValueError(f'unexpected runtime files: {actual_runtime ^ expected_runtime}')
    files = set(mapping) | added | {Path(p) for p in BOX_FILES}
    files.update(Path('weights') / name for name in BOX_WEIGHTS)
    files.update(p.relative_to(box) for p in (box / 'models').rglob('*.py'))
    files.update(Path('sample_data') / p for p in SAMPLES)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.box-build-', dir=output.parent) as tmp:
        stage = Path(tmp)
        records = {}
        for rel in sorted(files):
            src, dest = box / rel, stage / 'NVIDIA-Box' / rel
            if not src.is_file() or src.is_symlink():
                raise ValueError(f'missing or linked package file: {rel}')
            if src.suffix == '.py':
                compile(src.read_bytes(), str(src), 'exec')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            record = {'bytes': dest.stat().st_size, 'sha256': digest(dest)}
            if rel in mapping:
                record['platform_source'] = mapping[rel].as_posix()
                if rel in modified:
                    record['box_override'] = True
                    record['platform_sha256'] = modified[rel]['platform_sha256']
                elif record['sha256'] != digest(platform / mapping[rel]):
                    raise ValueError(f'source changed during snapshot: {rel}')
            records[rel.as_posix()] = record
        manifest = {'version': VERSION, 'platform_source': 'platform/dark-demo-v2',
                    'validation_scope': 'Single RTX3090 tested; physical Orin 4GB and JetPack pending',
                    'default_config': {'box_profile': 'balanced', 'box_backend': 'auto', 'box_action_batch': 2,
                                       'sr': 'off', 'cuda_graphs': False,
                                       'proc_max_side': 720, 'reco_min_conf': 0.925,
                                       'candidate_recheck_enabled': True, 'candidate_recheck_lower': 0.85,
                                       'action_display_hold_sec': 0.8,
                                       'lum_gate': 35, 'object_min_conf': 0.581, 'person_object_prior': True, 'fuse_mode': 'alpha', 'person_mode': True, 'clip_denoise': 'off',
                                       'skel_fusion_alpha': 0.972, 'pose_overlay': True,
                                       'wave_pose_min_frames': 3, 'running_pose_min_speed': 0.35,
                                       'clip_pre_sec': 2, 'clip_post_sec': 2, 'gpu_ids': '0'},
                    'model': 'weights/videomamba_t_behavior_32f_person_roi_20261023.pth',
                    'files': records}
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
        (stage / 'NVIDIA-Box/manifest.json').write_text(manifest_text)
        archive = stage / 'delivery.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for rel in sorted(files | {Path('manifest.json')}):
                info = zipfile.ZipInfo('NVIDIA-Box/' + rel.as_posix(), (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if rel.suffix == '.sh' else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, (stage / 'NVIDIA-Box' / rel).read_bytes())
        archive.replace(output)
        (box / 'manifest.json').write_text(manifest_text)
    archive_hash = digest(output)
    sidecar = output.with_suffix(output.suffix + '.sha256')
    sidecar.write_text(f'{archive_hash}  {output.name}\n')
    output.with_suffix(output.suffix + '.manifest.json').write_text(json.dumps(
        {'archive': output.name, 'sha256': archive_hash, 'size': output.stat().st_size,
         'contents': manifest}, ensure_ascii=False, indent=2) + '\n')
    return {'output': str(output), 'bytes': output.stat().st_size,
            'sha256': archive_hash, 'files': len(files) + 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'NVIDIA-Box')
    parser.add_argument('--platform', type=Path, default=ROOT / 'platform/dark-demo-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'NVIDIA-Box.zip')
    parser.add_argument('--sync', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.platform, args.output, args.sync), indent=2))
