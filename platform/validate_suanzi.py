#!/usr/bin/env python3
"""suanzi.json 规范校验器（docs/第三方算法开封接入规范1.docx）。

用法：
    python platform/validate_suanzi.py platform/op_*/suanzi.json

只做规范能判定的静态检查——字段齐全性、取值域、名称正则、args 与 inputs/outputs 的
一致性。不做的：不检查镜像是否存在、不检查权重是否就位（那是构建与运行期的事）。
每条错误都指出违反了规范中的哪一条，便于对照文档修改。
"""
import json
import os
import re
import sys

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ARCHS = ("arm64", "amd64")
TYPES = ("String", "Int", "Float", "Bool", "Binary")
GPU_TYPES = ("nvidia.com/gpu", "huawei.com/Ascend910B")
COMPONENTS = ("fileRead", "dirRead", "modelRead", "fileLoad", "randomGen", "range", "bool",
              "intInput", "stringInput", "floatInput", "choice", "resourcechoice",
              "databasechoice", "password", "datetime")
# 单位后缀见规范“memory”一节
MEM_RE = re.compile(r"^\d+(Ki|Mi|Gi|Ti|k|M|G|T)?$")


def check(path):
    errs = []

    def bad(msg):
        errs.append(msg)

    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:                                   # noqa: BLE001
        return [f"文件无法解析为 JSON: {e}"]

    for key in ("name", "description", "metadata", "inputs", "outputs", "implementation"):
        if key not in doc:
            bad(f"缺少必填字段 {key}")
    if not str(doc.get("name", "")).strip():
        bad("name 不能为空（算子名称）")
    if not str(doc.get("description", "")).strip():
        bad("description 不能为空（规范要求细致填写，便于他人理解与调用）")

    meta = doc.get("metadata") or {}
    if meta.get("arch") not in ARCHS:
        bad(f"metadata.arch 必填且只能是 {ARCHS}，当前为 {meta.get('arch')!r}")
    for block in ("limits", "request"):
        b = meta.get(block)
        if not isinstance(b, dict):
            bad(f"metadata.{block} 必填，需为对象")
            continue
        cpu, mem = b.get("cpu"), b.get("memory")
        if not isinstance(cpu, str) or not cpu.isdigit():
            bad(f"metadata.{block}.cpu 必填，且为整数的 String 类型，如 \"2\"（当前 {cpu!r}）")
        if not isinstance(mem, str) or not MEM_RE.match(mem):
            bad(f"metadata.{block}.memory 必填，String 类型，如 \"2Gi\"/\"300M\"/\"500\""
                f"（当前 {mem!r}）")
    if isinstance(meta.get("limits"), dict) and isinstance(meta.get("request"), dict):
        lc, rc = meta["limits"].get("cpu"), meta["request"].get("cpu")
        if str(lc).isdigit() and str(rc).isdigit() and int(rc) > int(lc):
            bad(f"metadata.request.cpu ({rc}) 大于 limits.cpu ({lc})：最小需求不应超过最大限制")
    gpu = meta.get("gpu")
    if gpu is not None:
        if gpu.get("type") not in GPU_TYPES:
            bad(f"metadata.gpu.type 只能是 {GPU_TYPES}，当前 {gpu.get('type')!r}")
        if not isinstance(gpu.get("count"), int) or gpu["count"] < 1:
            bad(f"metadata.gpu.count 需为正整数（当前 {gpu.get('count')!r}）")
        if "memory" in gpu and not MEM_RE.match(str(gpu["memory"])):
            bad(f"metadata.gpu.memory 单位只支持 Ki/Mi/Gi（当前 {gpu['memory']!r}）")

    def check_iface(items, kind):
        names = []
        if not isinstance(items, list):
            bad(f"{kind} 必填，需为数组（可为空数组）")
            return names
        for i, it in enumerate(items):
            where = f"{kind}[{i}]"
            name = it.get("name")
            if not isinstance(name, str) or not NAME_RE.match(name):
                bad(f"{where}.name 必填，只允许小写字母/数字/下划线且以小写字母开头"
                    f"（当前 {name!r}）")
            else:
                if name in names:
                    bad(f"{where}.name 重复：{name!r}")
                names.append(name)
            if it.get("type") not in TYPES:
                bad(f"{where}.type 必填，只能是 {TYPES}（当前 {it.get('type')!r}）")
            if not str(it.get("description", "")).strip():
                bad(f"{where}.description 为空——规范建议详细填写，页面上就是使用者看到的说明")
            ann = it.get("annotations") or {}
            if not isinstance(ann, dict):
                bad(f"{where}.annotations 需为对象")
                continue
            comp = ann.get("component")
            if kind == "inputs":
                if comp is None:
                    bad(f"{where}.annotations.component 未填，页面将无法渲染该参数")
                elif comp not in COMPONENTS:
                    bad(f"{where}.annotations.component 取值 {comp!r} 不在规范列表中")
                if "choice" in ann and comp != "choice":
                    bad(f"{where}: choice 仅在 component 为 choice 时有效（当前 {comp!r}）")
                if comp == "choice":
                    ch = ann.get("choice")
                    if not isinstance(ch, str) or not ch.strip().startswith("["):
                        bad(f"{where}: component 为 choice 时必须给出 choice，且为形如 "
                            f"\"['a', 'b']\" 的字符串（当前 {ch!r}）")
                for k in ("min", "max"):
                    if k in ann and comp != "range":
                        bad(f"{where}: {k} 仅在 component 为 range 时有效（当前 {comp!r}）")
                if "parameter" in ann and ann["parameter"] not in (0, 1):
                    bad(f"{where}.annotations.parameter 只能是 0 或 1"
                        f"（当前 {ann['parameter']!r}）")
                # 规范：“如果有default字段，则表示该参数是必填项，不能为空值”
                if "default" in it:
                    d = it["default"]
                    if d is None or (isinstance(d, str) and not d.strip()):
                        bad(f"{where}.default 存在即代表该参数必填、不能为空值（当前 {d!r}）")
                    if isinstance(d, str) and re.search(r"[一-鿿]", d):
                        bad(f"{where}.default 不支持中文（当前 {d!r}）")
                    if ann.get("component") == "choice" and isinstance(ann.get("choice"), str):
                        opts = re.findall(r"'([^']*)'", ann["choice"]) or \
                            re.findall(r"[-\d.]+", ann["choice"])
                        if opts and str(d) not in opts:
                            bad(f"{where}.default {d!r} 不在 choice {ann['choice']} 之内")
            else:
                if comp is not None:
                    bad(f"{where}: 输出接口的 annotations 只需要 show，不应带 component")
        return names

    in_names = check_iface(doc.get("inputs"), "inputs")
    out_names = check_iface(doc.get("outputs"), "outputs")

    impl = (doc.get("implementation") or {}).get("container")
    if not isinstance(impl, dict):
        bad("implementation.container 必填")
        return errs
    cmd = impl.get("command")
    if not isinstance(cmd, list) or not cmd:
        bad("implementation.container.command 必填（容器 ENTRYPOINT）")
    else:
        py = str(cmd[0])
        if not py.startswith("/"):
            bad(f"command[0] 必须是解释器的绝对路径（规范：使用 conda 环境一定要指定 python "
                f"所在路径），当前 {py!r}")
        entry = next((c for c in cmd if str(c).endswith((".py", ".jar"))), None)
        if entry is None:
            bad("command 中没有出现 main.py / main.jar 主入口")
        else:
            entry = os.path.basename(str(entry))
            if entry not in ("main.py", "main.jar"):
                bad(f"规范要求主入口为 main.py 或 main.jar，当前 {entry!r}")
            src = os.path.join(os.path.dirname(os.path.abspath(path)), entry)
            if entry.endswith(".py") and not os.path.exists(src):
                bad(f"command 指定的主入口 {entry} 在 {os.path.dirname(src)} 下不存在——"
                    f"规范要求主函数源码文件名与启动命令中的文件名一致")

    args = impl.get("args")
    if args is None:
        args = []
    if not isinstance(args, list):
        bad("implementation.container.args 需为数组")
        args = []
    used_in, used_out = set(), set()
    for a in args:
        if isinstance(a, str):
            continue
        if not isinstance(a, dict) or len(a) != 1:
            bad(f"args 中的引用需形如 {{\"inputValue\": \"x\"}}（当前 {a!r}）")
            continue
        (kind, ref), = a.items()
        if kind in ("inputValue", "inputPath"):
            if ref not in in_names:
                bad(f"args 引用了未声明的输入 {ref!r}（inputs 中没有同名 name）")
            used_in.add(ref)
        elif kind == "outputPath":
            if ref not in out_names:
                bad(f"args 引用了未声明的输出 {ref!r}（outputs 中没有同名 name）")
            used_out.add(ref)
        else:
            bad(f"args 中的引用类型只能是 inputValue / inputPath / outputPath（当前 {kind!r}）")
    for n in in_names:
        if n not in used_in:
            bad(f"输入 {n!r} 已声明但未出现在 args 中——规范要求所有输入都必须作为参数传给算子")
    for n in out_names:
        if n not in used_out:
            bad(f"输出 {n!r} 已声明但未出现在 args 中")
    if out_names and not used_out:
        bad("算子之间的数据传递必须以文件形式且至少有一个文件，args 中没有任何 outputPath")

    env = impl.get("env")
    if env is not None and not isinstance(env, dict):
        bad("implementation.container.env 需为对象，如 {\"key\": \"value\"}")

    # README.md 与 suanzi.json 都必须位于 zip 根目录，这里只能确认它们在同一目录下
    readme = os.path.join(os.path.dirname(os.path.abspath(path)), "README.md")
    if not os.path.exists(readme):
        bad("同目录下缺少 README.md（算子使用说明文件，规范要求与 suanzi.json 一同置于包根）")
    return errs


def main(argv):
    if not argv:
        sys.exit(f"用法: {sys.argv[0]} <suanzi.json> [suanzi.json ...]")
    total = 0
    for p in argv:
        errs = check(p)
        total += len(errs)
        if errs:
            print(f"[FAIL] {p}")
            for e in errs:
                print(f"   - {e}")
        else:
            print(f"[ OK ] {p}")
    if total:
        print(f"\n共 {total} 处不符合规范")
        return 1
    print(f"\n{len(argv)} 个 suanzi.json 全部符合规范")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
