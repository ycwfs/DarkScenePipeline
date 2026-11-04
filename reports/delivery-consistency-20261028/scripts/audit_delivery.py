"""Read-only archive/source/default/diagram audit; writes JSON and checksum report.

Pass --image-id from a current read-only `docker image inspect` call.
Historical releases are not republished and no default aliases are changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OP = ROOT / 'platform/dark-demo-v2'
BOX = ROOT / 'NVIDIA-Box'
FLOW = ROOT / 'reports/algorithm-flow-20261027'
sys.path.insert(0, str(ROOT / 'platform'))
from package_v2 import package_files, parser_of, validate_contract


def digest(stream):
    sha = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        sha.update(block)
    return sha.hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return digest(stream)


def load(path):
    return json.loads(path.read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_archive(path):
    side = load(Path(str(path) + '.manifest.json'))
    sha = file_hash(path)
    assert sha == side['sha256'] and path.stat().st_size == side['size'], path
    assert side['archive'] == path.name, path
    checksum = Path(str(path) + '.sha256')
    if checksum.exists():
        recorded_sha, recorded_name = checksum.read_text().strip().split(maxsplit=1)
        assert recorded_sha == sha and recorded_name.lstrip('*') == path.name, checksum
    return dict(path=str(path.relative_to(ROOT)), sha256=sha, bytes=path.stat().st_size,
                manifest_matches=True, checksum_sidecar_verified=checksum.exists()), side


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--image-id', required=True)
    args = cli.parse_args()
    contract = validate_contract(OP)
    files = package_files(OP)
    source = {str(p): file_hash(OP / p) for p in files}
    archive_records = []
    for ext in ('tar.gz', 'zip'):
        path = ROOT / f'platform/dark-demo-v2_full_priors_20261027.{ext}'
        record, side = verify_archive(path)
        payload = {}
        if ext == 'zip':
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    assert not item.is_dir() and item.filename not in payload
                    with archive.open(item) as stream:
                        payload[item.filename] = digest(stream)
        else:
            with tarfile.open(path, 'r|gz') as archive:
                for item in archive:
                    assert item.isfile() and not item.issym()
                    rel = str(Path(item.name).relative_to('dark-demo-v2'))
                    assert rel not in payload
                    with archive.extractfile(item) as stream:
                        payload[rel] = digest(stream)
        assert payload == source, path
        assert set(side['files']) == set(source)
        for rel, item in side['files'].items():
            assert item == {'sha256': source[rel], 'size': (OP / rel).stat().st_size}, rel
        assert side['image'] == contract['implementation']['container']['image']
        record['payload_files_verified'] = len(payload)
        archive_records.append(record)
        alias, alias_side = verify_archive(ROOT / f'platform/dark-demo-v2.{ext}')
        assert alias['sha256'] == record['sha256']
        assert alias_side['files'] == side['files'] and alias_side['image'] == side['image']
        alias['identical_to'] = record['path']
        archive_records.append(alias)
    print('Platform: archive payloads, source, manifests and aliases verified', flush=True)

    box_path = ROOT / 'NVIDIA-Box_Orin4GB_candidate_20261028.zip'
    box_record, box_side = verify_archive(box_path)
    with zipfile.ZipFile(box_path) as archive:
        manifest = json.loads(archive.read('NVIDIA-Box/manifest.json'))
        assert manifest == box_side['contents'] == load(BOX / 'manifest.json')
        expected = {'NVIDIA-Box/' + p for p in manifest['files']} | {'NVIDIA-Box/manifest.json'}
        assert len(archive.namelist()) == len(expected) and set(archive.namelist()) == expected
        for rel, item in manifest['files'].items():
            with archive.open('NVIDIA-Box/' + rel) as stream:
                assert digest(stream) == item['sha256'] == file_hash(BOX / rel), rel
            assert archive.getinfo('NVIDIA-Box/' + rel).file_size == item['bytes'] == (BOX / rel).stat().st_size
            assert not rel.endswith('.engine') and '/.cache/' not in rel
    packager = module('box_packager_audit', ROOT / 'scripts/package_nvidia_box.py')
    mapping = packager.runtime_sources(OP)
    expected_sources = set(mapping) | {Path(p) for p in packager.BOX_FILES}
    expected_sources |= {Path('weights') / p for p in packager.WEIGHTS + packager.BOX_WEIGHTS + (packager.LEGACY_WEIGHT,)}
    expected_sources |= {p.relative_to(BOX) for p in (BOX / 'models').rglob('*.py')}
    expected_sources |= {Path('sample_data') / p for p in packager.SAMPLES}
    overrides = load(BOX / 'runtime-overrides.json')
    expected_sources |= {Path('runtime') / p for p in overrides['added']}
    assert {str(p) for p in expected_sources} == set(manifest['files'])
    modified = {f'runtime/{p}' for p in overrides['modified']}
    for rel, item in manifest['files'].items():
        if 'platform_source' in item:
            base_sha = source[item['platform_source']]
            if rel in modified:
                assert item['box_override'] is True
                assert item['platform_sha256'] == base_sha == overrides['modified'][rel.removeprefix('runtime/')]['platform_sha256']
            else:
                assert item['sha256'] == base_sha, rel
    actual_runtime = {str(p.relative_to(BOX)) for p in (BOX / 'runtime').rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts}
    assert actual_runtime == {p for p in manifest['files'] if p.startswith('runtime/')}
    box_record['payload_files_verified'] = len(manifest['files'])
    box_record['archive_files_verified'] = len(expected)
    archive_records.append(box_record)
    alias, alias_side = verify_archive(ROOT / 'NVIDIA-Box.zip')
    assert alias['sha256'] == box_record['sha256'] and alias_side['contents'] == manifest
    alias['identical_to'] = box_record['path']
    archive_records.append(alias)
    previous = load(ROOT / 'reports/box-single-gpu-20261028/release-verification.json')
    assert previous['passed'] and previous['sha256'] == box_record['sha256']
    for earlier in previous['platform_unchanged']['archives']:
        now = next(r for r in archive_records if r['path'] == earlier['path'])
        assert now['sha256'] == earlier['sha256']
    print('Box: all payloads, allowed runtime differences and default alias verified', flush=True)

    models = load(OP / 'models.json')
    assert models == load(BOX / 'models.json')
    for model in models['models'].values():
        assert source['ckpts/' + model['file']] == model['sha256']
        assert manifest['files']['weights/' + model['file']]['sha256'] == model['sha256']
    derived = load(BOX / 'box-models.json')['derived_pose']
    assert source['ckpts/' + derived['source']] == derived['source_sha256']
    assert manifest['files']['weights/' + derived['output']]['sha256'] == derived['sha256']
    platform_defaults = vars(parser_of(OP).parse_args(['--video_path', 'probe', '--session_json', 'probe.json']))
    launcher = module('box_launcher_audit', BOX / 'run.py')
    box_defaults = vars(launcher.build_parser().parse_args(['--video_path', 'probe']))
    for key, value in models['defaults'].items():
        assert platform_defaults[key] == box_defaults[key] == value, key
    for key, value in manifest['default_config'].items():
        assert box_defaults['box_cuda_graphs' if key == 'cuda_graphs' else key] == value, key
    differences = {key: {'platform': value, 'box': box_defaults[key]}
                   for key, value in platform_defaults.items() if value != box_defaults[key]}
    assert set(differences) == {'ckpt_dir', 'gpu_ids', 'clip_dir', 'extra_hosts', 'session_json', 'sr'}, differences
    for key, value in {'proc_max_side': 720, 'clip_pre_sec': 2, 'clip_post_sec': 2,
                       'person_mode': True, 'pose_overlay': True, 'fuse_mode': 'alpha'}.items():
        assert platform_defaults[key] == box_defaults[key] == value, key

    base = '逐人动作识别_完整先验方案_20261027'
    ppt = FLOW / f'{base}_可编辑.pptx'
    drawio = FLOW / f'{base}.drawio'
    with zipfile.ZipFile(ppt) as archive:
        slide_names = sorted(n for n in archive.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml'))
        assert len(slide_names) == 2 and not any(n.startswith('ppt/media/') for n in archive.namelist())
        slides = [ET.fromstring(archive.read(n)) for n in slide_names]
        texts = ['\n'.join(x.text or '' for x in slide.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'))
                 for slide in slides]
    diagram = ET.parse(drawio).getroot()
    assert len(diagram.findall('diagram')) == 2
    for page in diagram.findall('diagram'):
        ids = {node.get('id') for node in page.iter('mxCell')}
        for node in page.iter('mxCell'):
            if node.get('edge'):
                assert node.get('source') in ids and node.get('target') in ids
    draw_text = '\n'.join(node.get('value', '') for node in diagram.iter('mxCell'))
    for token in ('0.972', '0.028', '0.925', '0.581', '35', '720', '0.35',
                  '挥手至少 3 帧', 'YOLO26m', 'VideoMamba-Tiny', 'ST-GCN', 'RTMO-s', 'Retinexformer'):
        assert token in texts[0] and token in draw_text, token
    for token in ('现场微调', '公开 NTIRE', '535', '319', '72', '1626',
                  'Box', 'TensorRT', '1440', '真机验收'):
        assert token in texts[1] and token in draw_text, token
    assert file_hash(drawio) == file_hash(ROOT / 'platform/模型运行流程_v2.drawio')
    qa = load(FLOW / 'editable-qa.json')
    assert qa['defaults'] == models['defaults'] and qa['models_sha256'] == file_hash(OP / 'models.json')
    assert qa['box_release'] == box_path.stem and qa['physical_orin_tested'] is False
    pdf_text = subprocess.check_output(['pdftotext', str(FLOW / f'{base}_可编辑.pdf'), '-'], text=True)
    for token in ('0.972', '0.925', '0.581', 'TensorRT', '1440', '2026-10-28'):
        assert token in pdf_text, ('PDF export mismatch', token)
    cpu_log = (OUT / 'cpu-checks.log').read_text()
    assert '13 passed' in cpu_log and 'failed' not in cpu_log
    artifacts = {p.name: file_hash(p) for p in [ppt, drawio, FLOW / f'{base}_可编辑.pdf',
                                              FLOW / 'preview-1.png', FLOW / 'preview-2.png', FLOW / '讲解备注.md']}
    print('Parser defaults, six models and actual editable diagrams verified', flush=True)

    historical_image = load(ROOT / 'reports/performance-confirmation-20261015/image-archive-integrity.json')
    image_path = ROOT / 'platform/darkpipe-operator-0.3.0.tar.gz'
    image_sha = file_hash(image_path)
    assert image_sha == historical_image['sha256']
    assert image_path.stat().st_size == historical_image['bytes']
    assert args.image_id == historical_image['image_id']
    result = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
                  scope='Current platform release and Box candidate plus aliases; historical archives retained',
                  archives=archive_records, platform_payload_files=len(source),
                  shared_original_weights=len(packager.WEIGHTS), models=models['models'],
                  shared_defaults=models['defaults'], launcher_default_differences=differences,
                  box_runtime_overrides=overrides, box_derived_pose=derived,
                  flow=dict(slides=2, native_editable=True, pdf_text_verified=True, artifacts=artifacts),
                  cpu_checks=dict(passed=13, log='cpu-checks.log'),
                  image=dict(path=str(image_path.relative_to(ROOT)), bytes=image_path.stat().st_size,
                             sha256=image_sha, loaded_image_id=args.image_id,
                             id_source='Current docker image inspect supplied by --image-id', unchanged=True),
                  matches_previous_inference_tested_archives=True, new_gpu_inference=False,
                  physical_orin_tested=False)
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in archive_records)
                                  + f'{image_sha}  {image_path.relative_to(ROOT)}\n')
    print(json.dumps(dict(passed=True, platform_files=len(source), box_payload_files=len(manifest['files']),
                          archives=len(archive_records), image_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
