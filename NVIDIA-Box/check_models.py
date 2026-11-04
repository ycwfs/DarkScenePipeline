"""环境自检：确认这台 NVIDIA 盒子能跑包内两个模型（v2 含骨架分支）。

    python check_env.py

检查项：Python 版本、torch/CUDA、mamba_ssm 与 causal_conv1d 的 CUDA 扩展、其余依赖、
权重文件齐全、两个模型 dummy 前向、骨架分支（onnxruntime 的 CUDA EP + RTMO-s 时延探针 +
ST-GCN 前向 + 融合通路）、照度门（两部曲：暗窗纯 RGB / 亮窗融合）。全部通过会打印 [OK]。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

fail = []


def check(name, fn):
    try:
        info = fn()
        print(f"[OK] {name}: {info}")
    except Exception as e:  # noqa: BLE001
        fail.append(name)
        print(f"[FAIL] {name}: {e}")


def c_python():
    v = sys.version_info
    assert v[:2] == (3, 10), f"需要 Python 3.10（mamba_ssm 预编译 wheel 只发 cp310），当前 {v.major}.{v.minor}"
    return sys.version.split()[0]


def c_torch():
    import torch
    assert torch.cuda.is_available(), "torch 检测不到 CUDA"
    name = torch.cuda.get_device_name(0)
    return f"{torch.__version__} (cuda {torch.version.cuda}) / GPU {name}"


def c_cuda_ext():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update  # noqa: F401
    return "mamba_ssm CUDA 扩展可加载（执行在「两个模型 dummy 前向」里验证）"


def c_deps():
    import cv2
    import einops  # noqa: F401
    import numpy
    import timm  # noqa: F401
    extra = []
    try:
        import onnxruntime
        provs = onnxruntime.get_available_providers()
        extra.append(f"onnxruntime {onnxruntime.__version__} providers={provs}")
    except Exception as e:  # noqa: BLE001
        extra.append(f"onnxruntime 不可用（{e}）")
    try:
        import rtmlib  # noqa: F401
        extra.append("rtmlib 可用")
    except Exception as e:  # noqa: BLE001
        extra.append(f"rtmlib 不可用（{e}）")
    return f"opencv {cv2.__version__} / numpy {numpy.__version__}; " + "; ".join(extra)


def c_weights():
    need = [os.path.join("weights", "NTIRE.pth"),
            os.path.join("weights", "videomamba_t_behavior_32f_realactall_full_pn2.pth"),
            os.path.join("weights", "videomamba_t_behavior_32f_realactall_full_pn.pth"),
            os.path.join("weights", "stgcn_behavior_32f_realactall_full.pth"),
            os.path.join("weights", "rtmo-s_end2end.onnx"),
            os.path.join("weights", "videomamba_t_behavior_32f_realactall_full.pth"),
            os.path.join("weights", "videomamba_t_behavior_32f_realact10_all.pth")]
    for p in need:
        assert os.path.isfile(p), f"缺权重 {p}"
    return ", ".join(f"{os.path.basename(p)} ({os.path.getsize(p) / 2**20:.1f}MB)" for p in need)


def _frames_for_skel():
    """骨架检查的输入：优先用包内样例视频（真人，走 presence>=0.5 的常规通路）；
    没有样例就用随机噪声（无人，走 soft 策略通路）。两条通路都要能出合法概率。"""
    import cv2
    import numpy as np
    sample = os.path.join("sample_data", "sample_drink.mp4")
    if os.path.isfile(sample):
        cap = cv2.VideoCapture(sample)
        frames = []
        while len(frames) < 32:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) >= 32:
            return frames[:32], f"样例 {os.path.basename(sample)}"
    return [np.random.RandomState(2).randint(0, 255, (270, 480, 3), dtype=np.uint8)
            for _ in range(32)], "随机噪声（无样例视频）"


def c_forward():
    """RGB 主模型 dummy 前向（骨架关掉，纯查 VideoMamba/mamba_ssm 链路）。"""
    import numpy as np
    from behavior_infer import BehaviorRecognizer, WINDOW_T
    from lowlight_infer import LowLightEnhancer
    enh = LowLightEnhancer()
    dark = (np.random.RandomState(0).rand(270, 480, 3) * 30).astype(np.uint8)
    lit = enh.enhance([dark])[0]
    assert lit.shape == dark.shape and lit.mean() > dark.mean(), "增强结果异常"
    rec = BehaviorRecognizer(skel_alpha=1.0)
    assert rec.fov == (224, 400), f"默认权重的 fullfov 元数据未识别: {rec.fov}"
    frames = [np.random.RandomState(1).randint(0, 255, (270, 480, 3), dtype=np.uint8)
              for _ in range(WINDOW_T)]
    r = rec.predict(frames)
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-3, "概率和不为 1"
    return (f"增强 270x480 平均亮度 {dark.mean():.0f}->{lit.mean():.0f}; "
            f"识别(整帧 400x224 输入) top1={r['label']}({r['confidence']:.3f})")


def c_skeleton():
    """骨架分支端到端：ORT CUDA EP -> RTMO-s 时延探针 -> ST-GCN -> 融合 -> 阈值规则。"""
    import onnxruntime
    provs = onnxruntime.get_available_providers()
    assert "CUDAExecutionProvider" in provs, f"onnxruntime 没有 CUDA EP（装成 CPU 版了？）: {provs}"
    from behavior_infer import BehaviorRecognizer
    rec = BehaviorRecognizer()          # 默认 α=0.7，默认权重（pn2）
    if rec.aux is None:
        raise RuntimeError(rec.aux_reason or "骨架分支未启用")
    frames, src = _frames_for_skel()
    # lum_gate=0：这一项验的就是骨架通路本身，不让照度门挡它（ARID 暗样例的提亮前
    # 亮度低于门限 75，默认门下会按两部曲走纯 RGB，骨架根本不跑）。
    r = rec.predict(frames, lum_gate=0.0)
    assert r["presence"] is not None and abs(sum(r["probs_final"].values()) - 1.0) < 1e-3, \
        "融合后概率异常"
    gate = f"presence={r['presence']:.2f}({src})"
    return (f"姿态 {rec.aux.pose_ms:.1f} ms/帧, {gate}, "
            f"最终判定 {r['label_reco']}({r['confidence_reco']:.3f})")


def c_gate():
    """照度门（两部曲第一步）端到端：暗窗被挡（纯 RGB，骨架不跑），亮窗融合照常。"""
    import numpy as np
    from behavior_infer import BehaviorRecognizer, WINDOW_T
    rec = BehaviorRecognizer()          # 默认权重 + α=0.7 + 门限 75
    if rec.aux is None:
        raise RuntimeError(rec.aux_reason or "骨架分支未启用，照度门无从验证")
    dark = [(np.random.RandomState(3).rand(270, 480, 3) * 40).astype(np.uint8)
            for _ in range(WINDOW_T)]   # 平均亮度约 20，远低于门限 75
    rd = rec.predict(dark)
    assert rd["gate_blocked"] and rd["presence"] is None, \
        f"暗窗未被照度门挡下: lum={rd['lum']:.1f} blocked={rd['gate_blocked']}"
    bright = [np.full((270, 480, 3), 150, np.uint8) for _ in range(WINDOW_T)]  # 均值 150
    rb = rec.predict(bright)
    assert not rb["gate_blocked"], f"亮窗被照度门误挡: lum={rb['lum']:.1f}"
    return (f"暗窗 lum={rd['lum']:.1f} 被挡(纯RGB, {rd['label_reco']})；"
            f"亮窗 lum={rb['lum']:.1f} 融合照常(presence={rb['presence']:.2f})")


if __name__ == "__main__":
    check("Python", c_python)
    check("torch/CUDA", c_torch)
    check("mamba_ssm CUDA 扩展", c_cuda_ext)
    check("其余依赖", c_deps)
    check("权重文件", c_weights)
    check("两个模型 dummy 前向", c_forward)
    check("骨架分支（RTMO-s+ST-GCN+融合）", c_skeleton)
    check("照度门（两部曲：暗窗纯RGB/亮窗融合）", c_gate)
    if fail:
        print(f"\n{len(fail)} 项未通过: {fail} —— 处理建议见 README.md「环境要求」")
        sys.exit(1)
    print("\n全部通过，可以运行:")
    print("  python lowlight_infer.py --input sample_data/dark_frame.png")
    print("  python behavior_infer.py --input sample_data/sample_drink.mp4")
    print("  # 关骨架分支做纯 RGB 对比：加 --skel-alpha 1.0")
    print("  # 关照度门退回纯加权融合：加 --lum-gate 0")
