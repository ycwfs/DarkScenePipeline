# 2026-10-20 Git 整理与提交前验证

整理范围为此前尚未提交的 v2 生命周期/录制修复、720与0.75默认值、候选人物骨架、pn2换装、
盒子推理源码、测试、复现脚本与交付证据。没有重新训练或改变已验收的算子/盒子包和镜像。

- 根`.gitignore`继续排除模型权重、视频和归档；NVIDIA Box只纳入源码/说明/manifest。
- `reports/.gitignore`保留报告、复现脚本、紧凑统计、逐窗概率和可编辑会议材料，排除原始输入、缓存、
  检查点、解包副本、测试临时目录和日志。所有本地文件保留。
- `.gitattributes`将模型上游源码标记为vendored、生成数据标记为generated；保留这些文件原始字节，
  不为清理空白破坏交付哈希或CSV格式。维护源码仍执行空白检查。
- pytest默认只收集`tests/`，报告中的盒子检查需显式运行；默认收集402项，不代表402项全部执行。
- README和报告索引说明新克隆如何从交付包恢复模型/示例，以及本地归档链接的用途。

验证在原`darkpipe-operator:0.3.0`镜像完成：

| 检查 | 结果 |
|---|---|
| 当前算子6个v2测试文件 | 111项通过；包括3项重复拥塞/行为切换/主动停止测试 |
| 当前盒子合约测试 | 31项通过 |
| `platform/package_v2.py --check` | 47个包内文件、参数清单与权重契约通过 |
| Python/JSON/JSONL/XML/draw.io与shell语法 | 通过 |
| `git diff --cached --check` | 通过 |
| 新提交内容与文件大小 | 无新增模型、视频、缓存或大型归档，无单文件超过5 MiB |
| 凭据扫描 | 未发现真实密钥；两个RTSP匹配是历史文档中的密码占位符 |
| 交付源码与pn2包manifest核对 | 一致；整理只改变版本管理配置和仓库级说明 |

测试命令：

```bash
python -m pytest -q tests/test_v2_remaining_fixes.py tests/test_v2_serve_lifecycle.py \
  tests/test_v2_delivery.py tests/test_v2_pose_overlay.py tests/test_v2_platform_feedback.py \
  tests/test_v2_clip_sequence.py
python -m pytest -q reports/feedback-finetune-20261019/scripts/test_box_contract.py
python -m pytest --collect-only -q
python3 platform/package_v2.py --check
bash -n platform/build_image.sh platform/make_operator_zip.sh
```

测试使用CPU替身验证线程/协议，真实编解码仍执行FFmpeg；未重跑GPU训练或模型评估。
真实模型与独立交付验证沿用[pn2报告](feedback-finetune-20261019.md)，本次不增加泛化、速度或长期稳定性声明。
