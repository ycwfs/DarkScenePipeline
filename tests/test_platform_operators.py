"""Platform operator packaging: the manifest and the code must not drift apart.

`suanzi.json` is the only thing the platform reads to build the operator's command line, and
`main.py` is the only thing that has to accept it. Nothing at runtime checks that the two agree
-- the framework just runs the command and the operator dies on an unrecognised flag, in a
container, in a scheduler, where the traceback is least convenient to read. These tests are
that check, at the cost of a few milliseconds.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORM = os.path.join(ROOT, "platform")
OPS = ["op_dark_behavior", "op_dark_behavior_eval"]

sys.path.insert(0, ROOT)


def manifest(op):
    with open(os.path.join(PLATFORM, op, "suanzi.json"), encoding="utf-8") as f:
        return json.load(f)


def parser_of(op):
    """Import the operator's main.py in isolation and hand back its argparse parser."""
    path = os.path.join(PLATFORM, op, "main.py")
    spec = importlib.util.spec_from_file_location(f"_op_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, PLATFORM)                     # oputil.py lives one level up
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(PLATFORM)
    return mod.build_parser()


def flags_of(parser):
    """-> {"--video_path": action}, excluding -h."""
    out = {}
    for a in parser._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        for opt in a.option_strings:
            out[opt] = a
    return out


@pytest.mark.parametrize("op", OPS)
def test_manifest_conforms_to_spec(op):
    r = subprocess.run([sys.executable, os.path.join(PLATFORM, "validate_suanzi.py"),
                        os.path.join(PLATFORM, op, "suanzi.json")],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"validate_suanzi.py rejected {op}:\n{r.stdout}"


@pytest.mark.parametrize("op", OPS)
def test_every_declared_parameter_is_accepted_by_main(op):
    doc, flags = manifest(op), flags_of(parser_of(op))
    for group in ("inputs", "outputs"):
        for item in doc[group]:
            assert f"--{item['name']}" in flags, \
                f"{op}: suanzi.json declares {group} {item['name']!r} but main.py has no " \
                f"--{item['name']} flag"


@pytest.mark.parametrize("op", OPS)
def test_args_order_matches_code_order(op):
    """The spec requires args to be ordered as the parameters appear in the operator code."""
    args = manifest(op)["implementation"]["container"]["args"]
    in_args = [list(a.values())[0] for a in args if isinstance(a, dict)]
    parser = parser_of(op)
    order = [a.option_strings[0].lstrip("-") for a in parser._actions
             if a.option_strings and not isinstance(a, argparse._HelpAction)]
    ranks = [order.index(n) for n in in_args if n in order]
    assert ranks == sorted(ranks), \
        f"{op}: args order {in_args} does not follow main.py's parameter order {order}"


@pytest.mark.parametrize("op", OPS)
def test_manifest_defaults_match_code_defaults(op):
    """A default in two places is a default that will disagree in one of them."""
    doc, flags = manifest(op), flags_of(parser_of(op))
    for item in doc["inputs"]:
        if "default" not in item:
            continue
        action = flags[f"--{item['name']}"]
        assert str(action.default) == str(item["default"]), \
            f"{op}: {item['name']} default is {item['default']!r} in suanzi.json but " \
            f"{action.default!r} in main.py"


@pytest.mark.parametrize("op", OPS)
def test_manifest_choices_match_code_choices(op):
    doc, flags = manifest(op), flags_of(parser_of(op))
    for item in doc["inputs"]:
        ann = item.get("annotations") or {}
        if ann.get("component") != "choice":
            continue
        action = flags[f"--{item['name']}"]
        assert action.choices is not None, \
            f"{op}: {item['name']} is a choice in suanzi.json but unconstrained in main.py"
        declared = json.loads(ann["choice"].replace("'", '"'))
        assert sorted(map(str, declared)) == sorted(map(str, action.choices)), \
            f"{op}: {item['name']} choices differ -- suanzi.json {declared} vs " \
            f"main.py {list(action.choices)}"


@pytest.mark.parametrize("op", OPS)
def test_required_inputs_have_no_default(op):
    """规范: 有 default 即代表必填且不能为空——所以真正由用户提供的参数不应带 default。"""
    doc, flags = manifest(op), flags_of(parser_of(op))
    for item in doc["inputs"]:
        action = flags[f"--{item['name']}"]
        if action.required:
            assert "default" not in item, \
                f"{op}: {item['name']} is required in main.py, so suanzi.json must not " \
                f"give it a default"


@pytest.mark.parametrize("op", OPS)
def test_package_files_present(op):
    """These four are what make_operator_zip.sh puts at the zip root."""
    for f in ("main.py", "suanzi.json", "README.md", "zip_name"):
        assert os.path.exists(os.path.join(PLATFORM, op, f)), f"{op} is missing {f}"
    name = open(os.path.join(PLATFORM, op, "zip_name"), encoding="utf-8").read().strip()
    assert name.count("_") >= 2, \
        f"{op}: zip_name {name!r} must be 一级类别_二级类别_算子名称"


@pytest.mark.parametrize("op", OPS)
def test_entrypoint_is_main_py(op):
    cmd = manifest(op)["implementation"]["container"]["command"]
    assert cmd[0].startswith("/"), "command[0] must be an absolute interpreter path"
    assert os.path.basename(cmd[-1]) == "main.py"
