#!/usr/bin/env python3
"""Build the deployed v2 package from its own tree, using only the standard library."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import tarfile
import tempfile
import zipfile

from validate_suanzi import check

HERE = Path(__file__).resolve().parent
ROOT_FILES = ("main.py", "oputil.py", "suanzi.json", "README.md", "CLAUDE.md",
              "requirements-skeleton.txt", ".gitignore", "models.json", "README_legacy_20261022.md")
WEIGHTS = (
    "NTIRE.pth", "videomamba_t_behavior_32f_realactall_full_pn2.pth",
    "videomamba_t_behavior_32f_realactall_full_pn.pth",
    "videomamba_t_behavior_32f_realactall_full.pth", "videomamba_t_behavior_32f.pth",
    "videomamba_t_behavior_32f_person_roi_20261023.pth", "stgcn_person_20261024.pth",
    "yolo26m_objects_20261026.onnx",
    "yolo26m_person_20261024.onnx", "rtmo-s_person_20261024.onnx",
    "rtmo-s_end2end.onnx", "stgcn_behavior_32f_realactall_full.pth", "yolo26m_detect.onnx",
)
ASSETS = ("darkpipe/assets/NotoSansCJK-subset.ttf", "darkpipe/assets/NotoSansCJK-LICENSE.txt")
REQUIRED_CODE = (
    "darkpipe/__init__.py", "darkpipe/config.py", "darkpipe/constants.py",
    "darkpipe/server.py", "darkpipe/pipeline.py", "darkpipe/clips.py", "darkpipe/media.py",
    "darkpipe/streams.py", "darkpipe/shard.py", "darkpipe/shard_worker.py",
    "darkpipe/stages/recognize.py",
    "darkpipe/stages/skeleton_aux.py", "darkpipe/stages/object_prior.py",
    "darkpipe/person/runtime.py", "darkpipe/person/action_engine.py",
    "darkpipe/person/recheck.py",
    "darkpipe/person/object_prior.py", "darkpipe/person/object_runtime.py", "darkpipe/person/object_overlay.py",
    "darkpipe/person/wave_prior.py", "darkpipe/person/running_prior.py",
)


def parser_of(source: Path) -> argparse.ArgumentParser:
    """Execute the real parser function without importing CUDA/model modules."""
    tree = ast.parse((source / "main.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "build_parser")
    namespace = {"argparse": argparse,
                 "parse_bool": runpy.run_path(str(source / "oputil.py"))["parse_bool"]}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(source / "main.py"), "exec"), namespace)
    return namespace["build_parser"]()


def validate_contract(source: Path) -> dict:
    errors = check(str(source / "suanzi.json"))
    if errors:
        raise ValueError("\n".join(errors))
    doc = json.loads((source / "suanzi.json").read_text(encoding="utf-8"))
    parser = parser_of(source)
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    declared = {i["name"]: i for i in doc["inputs"] + doc["outputs"]}
    if set(actions) != set(declared):
        raise ValueError("CLI flags and manifest inputs/outputs differ")
    for item in doc["inputs"]:
        name = item["name"]
        action = actions[name]
        if "default" in item:
            value = item["default"]
            if action.required or value != action.default:
                raise ValueError(f"{name}: manifest default {value!r} != CLI {action.default!r}")
        elif not action.required and action.default != "":
            raise ValueError(f"{name}: optional input must accept the platform's empty value")
        choice = item["annotations"].get("choice")
        options = ast.literal_eval(choice) if choice is not None else None
        if (options is None) != (action.choices is None) or (
                options is not None and set(options) != set(action.choices)):
            raise ValueError(f"{name}: manifest choices differ from CLI")

    args = doc["implementation"]["container"]["args"]
    if len(args) != 2 * len(declared):
        raise ValueError("container.args must pass each declared flag exactly once")
    seen = set()
    argv = []
    inputs = {i["name"] for i in doc["inputs"]}
    for flag, ref in zip(args[::2], args[1::2]):
        if not isinstance(ref, dict) or len(ref) != 1:
            raise ValueError(f"invalid argument reference: {ref!r}")
        kind, name = next(iter(ref.items()))
        if name in seen or name not in declared or flag != f"--{name}":
            raise ValueError(f"flag/reference mismatch or duplicate: {flag!r} {ref!r}")
        seen.add(name)
        item = declared[name]
        component = item.get("annotations", {}).get("component")
        expected = ("inputPath" if component in {"fileRead", "modelRead", "fileLoad"}
                    else "inputValue") if name in inputs else "outputPath"
        if kind != expected:
            raise ValueError(f"{name}: expected {expected}, got {kind}")
        value = item.get("default", "probe" if actions[name].required else "")
        argv.extend([flag, str(value)])
    parser.parse_args(argv)  # Also exercise conversion and choices, including booleans.
    return doc


def package_files(source: Path) -> list[Path]:
    files = {Path(p) for p in ROOT_FILES + ASSETS}
    files.update(Path("ckpts") / name for name in WEIGHTS)
    files.update(p.relative_to(source) for p in (source / "darkpipe").rglob("*.py")
                 if "__pycache__" not in p.parts)
    for rel in files | {Path(p) for p in REQUIRED_CODE}:
        path = source / rel
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty package file: {rel}")
        if path.is_symlink() or source not in path.resolve().parents:
            raise ValueError(f"package file must be inside source tree: {rel}")
        if path.suffix == ".py":
            compile(path.read_bytes(), str(rel), "exec")
    tree = ast.parse((source / "darkpipe/constants.py").read_text(encoding="utf-8"))
    mapping = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and any(
                       isinstance(t, ast.Name) and t.id == "CKPT_FILES" for t in n.targets))
    for key in ("retinexformer", "behavior"):
        if Path("ckpts") / mapping[key] not in files:
            raise ValueError(f"active {key} checkpoint is not included: {mapping[key]}")
    return sorted(files)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("output must be outside the operator source tree")
    is_zip = output.name.endswith(".zip")
    if not is_zip and not output.name.endswith(".tar.gz"):
        raise ValueError("output must end in .tar.gz or .zip")
    doc = validate_contract(source)
    files = package_files(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage a snapshot so a concurrent source edit cannot disagree with the hash report.
    with tempfile.TemporaryDirectory(prefix=".v2-package-", dir=output.parent) as temp:
        stage = Path(temp)
        snapshot = stage / "source"
        records = {}
        for rel in files:
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / rel).read_bytes())
            records[rel.as_posix()] = {"size": target.stat().st_size, "sha256": sha256(target)}
        doc = validate_contract(snapshot)
        package_files(snapshot)
        archive = stage / "archive"
        if is_zip:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for rel in files:
                    info = zipfile.ZipInfo(rel.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, (snapshot / rel).read_bytes())
        else:
            with archive.open("wb") as raw, gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for rel in files:
                        path = snapshot / rel
                        info = tarfile.TarInfo(f"dark-demo-v2/{rel.as_posix()}")
                        info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                        with path.open("rb") as f:
                            tar.addfile(info, f)
        report = {"archive": output.name, "sha256": sha256(archive),
                  "size": archive.stat().st_size,
                  "image": doc["implementation"]["container"]["image"], "files": records}
        sidecar = stage / "manifest.json"
        sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksum = stage / "sha256"
        checksum.write_text(f"{report['sha256']}  {output.name}\n", encoding="utf-8")
        archive.replace(output)
        sidecar.replace(Path(str(output) + ".manifest.json"))
        checksum.replace(Path(str(output) + ".sha256"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "dark-demo-v2")
    parser.add_argument("--output", type=Path, default=HERE / "dist/dark-demo-v2.tar.gz")
    parser.add_argument("--check", action="store_true", help="validate without producing an archive")
    args = parser.parse_args()
    try:
        if args.check:
            validate_contract(args.source.resolve())
            files = package_files(args.source.resolve())
            print(f"[ok] v2 manifest/CLI/source/weights: {len(files)} files")
        else:
            report = build(args.source, args.output)
            print(f"[package] {args.output} ({report['size']} bytes)")
            print(f"[sha256] {report['sha256']}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
