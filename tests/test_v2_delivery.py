"""Deployed v2 clip quality, defaults, and reproducible delivery contracts; no GPU."""
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OP = Path(os.environ.get("DARKPIPE_V2_OPERATOR_DIR", ROOT / "platform/dark-demo-v2"))
sys.path.insert(0, str(ROOT / "platform"))
import package_v2 as delivery


@pytest.fixture(scope="module")
def clips():
    name = "_darkpipe_v2_delivery"
    spec = importlib.util.spec_from_file_location(
        name, OP / "darkpipe/__init__.py", submodule_search_locations=[str(OP / "darkpipe")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{name}.clips")


class Event:
    label = "Waving"
    confidence = 0.9

    def to_dict(self):
        return {"label": self.label, "confidence": self.confidence}


@pytest.mark.parametrize("mode", ["off", "fast", "quality", "quality_high"])
@pytest.mark.parametrize("count", [10, 50])
def test_every_source_frame_is_cleaned_once_before_resampling(clips, monkeypatch, tmp_path,
                                                            mode, count):
    cleaned, written = [], []

    def denoise(frame, selected):
        assert selected == mode
        cleaned.append(int(frame[0, 0, 0]))
        return frame + 100

    class Writer:
        def __init__(self, path, fps):
            Path(path).touch()
            self.count = 0

        def write(self, frame):
            written.append(int(frame[0, 0, 0]))
            self.count += 1

        def close(self):
            pass

    util = importlib.import_module(f"{clips.__package__}.denoise_util")
    monkeypatch.setattr(util, "denoise_frame", denoise)
    monkeypatch.setattr(clips, "VideoWriter", Writer)
    rec = clips.ClipRecorder(str(tmp_path), pre_sec=0.4, post_sec=100, max_sec=100,
                             denoise=mode, realtime=False, stage_dir=str(tmp_path / "stage"))
    frames = [np.full((16, 16, 3), i, np.uint8) for i in range(count)]
    try:
        for i, frame in enumerate(frames):
            # Slow down after the encoder opens, forcing repeated output frames.
            ts = i / 25 if i < 25 else 1 + (i - 25) * 0.15
            rec.push(frame, Event() if i == 5 else None, ts)
    finally:
        rec.close()
    assert not rec._thread.is_alive() and rec.saved == 1
    assert cleaned == ([] if mode == "off" else list(range(count)))
    offset = 0 if mode == "off" else 100
    assert written[0] == offset and written[-1] == count - 1 + offset
    assert all(offset <= value < count + offset for value in written)
    if count == 50:
        assert len(written) > count, "test must exercise duplicate frames"
    assert all(int(f[0, 0, 0]) == i for i, f in enumerate(frames)), "shared frames mutated"


def test_platform_and_cli_defaults_and_choices_agree():
    delivery.validate_contract(OP)
    delivery.package_files(OP.resolve())


def test_default_clip_covers_two_seconds_before_and_after_trigger(clips, tmp_path):
    args = delivery.parser_of(OP).parse_args(["--video_path", "probe", "--session_json", "out"])
    rec = clips.ClipRecorder(str(tmp_path), pre_sec=args.clip_pre_sec, post_sec=args.clip_post_sec,
                             denoise="off", realtime=False, naming="range",
                             stage_dir=str(tmp_path / "stage"))
    try:
        for i in range(70):
            rec.push(np.full((32, 32, 3), i, np.uint8), Event() if i == 30 else None, i / 10)
    finally:
        rec.close()
    assert not rec._thread.is_alive() and rec.saved == 1
    meta = json.loads(next(Path(rec.root).rglob("*.json")).read_text())
    assert meta["trigger_stream_time"] == 3.0
    assert meta["start_seconds"] == pytest.approx(1.0)
    assert 5.0 <= meta["end_seconds"] <= 5.2
    assert 4.0 <= meta["duration_seconds"] <= 4.2


def test_default_threshold_rejects_between_old_cli_and_platform_values(clips):
    args = delivery.parser_of(OP).parse_args(["--video_path", "probe", "--session_json", "out"])
    recognize = importlib.import_module(f"{clips.__package__}.stages.recognize")
    model = recognize.BehaviorRecognizer("unused", reject_tau=args.reco_min_conf)
    probs = np.full(10, 0.4 / 9)
    probs[0] = 0.6
    assert model.labels[int(recognize.reject_probs(probs, model.labels, model.reject_tau).argmax())] == "other"
    assert args.reco_min_conf == 0.925
    probs = np.full(10, 0.16 / 9)
    probs[0] = 0.84
    assert model.labels[int(recognize.reject_probs(probs, model.labels, model.reject_tau).argmax())] == "other"
    probs = np.full(10, (1-args.reco_min_conf) / 9)
    probs[0] = args.reco_min_conf
    assert model.labels[int(recognize.reject_probs(probs, model.labels, model.reject_tau).argmax())] == "wave"


@pytest.fixture
def source(tmp_path):
    """Use real code/assets and tiny stand-ins for weights to exercise packaging quickly."""
    target = tmp_path / "operator"
    for rel in delivery.package_files(OP.resolve()):
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts[0] == "ckpts":
            dst.write_bytes(rel.name.encode())
        else:
            shutil.copyfile(OP / rel, dst)
    return target


@pytest.mark.parametrize("fault", ["default", "choices", "reference", "weight", "active_weight"])
def test_packaging_rejects_drift_before_producing_output(source, tmp_path, fault):
    manifest = source / "suanzi.json"
    doc = json.loads(manifest.read_text())
    if fault == "default":
        next(i for i in doc["inputs"] if i["name"] == "reco_min_conf")["default"] = 0.5
    elif fault == "choices":
        next(i for i in doc["inputs"] if i["name"] == "sr")["annotations"]["choice"] = "['bicubic']"
    elif fault == "reference":
        doc["implementation"]["container"]["args"][0] = "--reco_min_conf"
    elif fault == "weight":
        (source / "ckpts" / delivery.WEIGHTS[1]).unlink()
    else:
        p = source / "darkpipe/constants.py"
        tree = ast.parse(p.read_text())
        mapping = next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n, ast.Assign) and any(
                           isinstance(t, ast.Name) and t.id == "CKPT_FILES" for t in n.targets))
        current = mapping['behavior']
        assert current in delivery.WEIGHTS
        p.write_text(p.read_text().replace(f'"behavior": "{current}"',
                                          '"behavior": "missing.pth"'))
    manifest.write_text(json.dumps(doc))
    output = tmp_path / "bad.tar.gz"
    with pytest.raises(ValueError):
        delivery.build(source, output)
    assert not output.exists()


@pytest.mark.parametrize("suffix", [".tar.gz", ".zip"])
def test_archives_are_reproducible_and_exclude_runtime_files(source, tmp_path, suffix):
    (source / "darkpipe/__pycache__").mkdir()
    (source / "darkpipe/__pycache__/stale.pyc").write_bytes(b"stale")
    (source / "debug.log").write_text("not for delivery")
    first, second = tmp_path / ("first" + suffix), tmp_path / ("second" + suffix)
    a = delivery.build(source, first)
    for rel in delivery.package_files(source):
        os.utime(source / rel, (123456, 123456))
    b = delivery.build(source, second)
    assert a["sha256"] == b["sha256"]
    for path, report in ((first, a), (second, b)):
        assert Path(str(path) + ".sha256").read_text() == (
            f"{report['sha256']}  {path.name}\n")
    assert a["files"] == b["files"]
    assert {name.removeprefix("ckpts/") for name in a["files"] if name.startswith("ckpts/")} == set(delivery.WEIGHTS)
    if suffix == ".zip":
        with zipfile.ZipFile(first) as z:
            assert set(z.namelist()) == set(a["files"])
            assert z.read("main.py") == (source / "main.py").read_bytes()
    else:
        with tarfile.open(first) as tar:
            assert set(tar.getnames()) == {"dark-demo-v2/" + name for name in a["files"]}
            assert tar.extractfile("dark-demo-v2/main.py").read() == (source / "main.py").read_bytes()


def test_image_build_default_matches_v2_manifest(tmp_path):
    platform = tmp_path / "platform"
    platform.mkdir()
    shutil.copyfile(ROOT / "platform/build_image.sh", platform / "build_image.sh")
    (platform / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "ckpts").mkdir()
    for name in ("NTIRE.pth", "videomamba_t_behavior_32f.pth"):
        (tmp_path / "ckpts" / name).write_bytes(b"test")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_TEST_LOG"\n')
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DOCKER_TEST_LOG=str(log))
    env.pop("IMAGE", None)
    subprocess.run(["bash", str(platform / "build_image.sh")], env=env, check=True,
                   capture_output=True, text=True, timeout=10)
    image = json.loads((OP / "suanzi.json").read_text())["implementation"]["container"]["image"]
    assert log.read_text().splitlines()[0].startswith(f"build -t {image} ")
