#!/usr/bin/env python3
"""按 suanzi.json 组装框架真实会下发的 argv，把算子当子进程跑一遍，检查产物。

这不是单元测试的替代品，而是回答一个单元测试回答不了的问题：*框架照着 suanzi.json 拼出来的
命令行，算子到底认不认*。因此参数全部由 suanzi.json 的 args 顺序与 default 生成，不在这里
另写一份，manifest 与代码一旦不同步就会在这里暴露。

用法：
    python platform/dryrun_test.py --video /tmp/in.mp4 --workdir /tmp/dryrun
    python platform/dryrun_test.py --video /tmp/in.mp4 --set sr_scale=3 --set max_frames=60
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build_argv(manifest, values, outdir):
    """把 suanzi.json 的 args 展开成框架下发的命令行；返回 (argv, 输出文件路径表)。"""
    args = manifest["implementation"]["container"].get("args", [])
    inputs = {i["name"]: i for i in manifest["inputs"]}
    argv, outputs = [], {}
    for a in args:
        if isinstance(a, str):
            argv.append(a)
            continue
        (kind, ref), = a.items()
        if kind == "outputPath":
            # 框架给的是绝对路径，且父目录不保证存在——算子必须自己创建，这里刻意用一个
            # 尚不存在的子目录，专门验证这条规范
            p = os.path.join(outdir, "framework_created", f"{ref}{_suffix(ref, manifest)}")
            outputs[ref] = p
            argv.append(p)
        else:
            if ref in values:
                argv.append(str(values[ref]))
            elif "default" in inputs[ref]:
                d = inputs[ref]["default"]
                argv.append(str(d).lower() if isinstance(d, bool) else str(d))
            else:
                sys.exit(f"error: 输入 {ref!r} 既没有 default 也没有通过 --set 提供")
    return argv, outputs


def _suffix(name, manifest):
    """框架并不保证输出路径带扩展名，这里按名称/描述猜一个，猜不到就故意不带扩展名——
    算子必须能应付这种路径（视频输出会先写临时 .mp4 再改名到目标路径）。"""
    desc = next((o.get("description", "") for o in manifest["outputs"] if o["name"] == name), "")
    hay = (name + " " + desc).lower()
    for ext in (".mp4", ".json", ".csv"):
        if ext[1:] in hay:
            return ext
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default=os.path.join(HERE, "op_dark_behavior"))
    ap.add_argument("--video", required=True, help="喂给算子的输入文件（视频或评测清单）")
    ap.add_argument("--input-name", default="", help="接收 --video 的输入参数名，默认取第一个")
    ap.add_argument("--workdir", default="/tmp/darkpipe_dryrun")
    ap.add_argument("--python", default=sys.executable, help="跑 main.py 的解释器")
    ap.add_argument("--set", action="append", default=[], metavar="k=v",
                    help="覆盖某个输入的默认值，可重复")
    a = ap.parse_args()

    with open(os.path.join(a.op, "suanzi.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    values = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        values[k] = v
    values[a.input_name or manifest["inputs"][0]["name"]] = a.video

    shutil.rmtree(a.workdir, ignore_errors=True)
    os.makedirs(a.workdir, exist_ok=True)
    argv, outputs = build_argv(manifest, values, a.workdir)
    entry = os.path.join(a.op, "main.py")
    print(f"[dryrun] {os.path.basename(a.op)}: " + " ".join([a.python, "-u", "main.py"] + argv))

    r = subprocess.run([a.python, "-u", entry] + argv, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = (r.stdout or "").strip().splitlines()
    for ln in tail[-25:]:
        print("   |", ln)
    if r.returncode != 0:
        sys.exit(f"[dryrun] FAIL: 算子退出码 {r.returncode}")

    ok = True
    for name, p in outputs.items():
        if not os.path.exists(p):
            print(f"[dryrun] FAIL: 声明的输出 {name} 未生成: {p}")
            ok = False
            continue
        size = os.path.getsize(p)
        note = ""
        if p.endswith(".json"):
            try:
                doc = json.load(open(p, encoding="utf-8"))
                note = f", JSON 可解析（{len(doc)} 项）"
            except Exception as e:                            # noqa: BLE001
                print(f"[dryrun] FAIL: {name} 不是合法 JSON: {e}")
                ok = False
        elif p.endswith(".mp4"):
            import cv2
            cap = cv2.VideoCapture(p)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            note = f", {w}x{h}, {n} 帧"
            if n <= 0:
                print(f"[dryrun] FAIL: {name} 解码不出帧")
                ok = False
        print(f"[dryrun] OK  {name}: {p} ({size} 字节{note})")
    if not ok:
        sys.exit("[dryrun] 有输出未通过检查")
    print("[dryrun] PASS")


if __name__ == "__main__":
    main()
