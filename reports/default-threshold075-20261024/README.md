# 默认动作阈值0.75与重新打包核验（2026-10-24）

按用户决定将默认 `reco_min_conf` 从0.91改为 **0.75**，后续由用户根据线上效果调整。亮度门30、RGB权重0.95、当前模型、逐人融合及保存流程保持。

- [最新提交说明与归档哈希](../../platform/提交说明_20261024.md)
- [归档与默认别名逐项核验](archive-verification.json)
- [48项回归日志](regression.log)
- [Box独立解包文件自检](box-files.log)

入口CLI、平台清单、PipelineConfig、逐人融合/绘制默认阈值、Box运行时、自检和模型清单已同步。模型清单保留原参数选择锁的SHA256，并明确标注0.75是后续用户覆盖值。平台须重新导入suanzi.json，已有任务显式传入的0.91也须改为0.75，启动日志核对实际值。

平台tar与ZIP的63个载荷文件逐字节一致；与上版相比仅main.py、config.py、person/core.py、suanzi.json、models.json、README.md、CLAUDE.md改变。12份模型权重哈希完全一致。Box内82个清单文件（另含manifest.json）全部通过校验，同源运行时和模型与平台一致。日期包、默认别名及manifest/sha256均匹配，旧0.91日期包哈希复核未变。原0.3.0镜像未修改，无需重传。

48项现有CPU回归全部通过：平台清单/CLI默认和选项一致、可复现打包与漂移拒绝；Box参数透传及路径处理；逐人融合亮度门、低阈值拒识、并发人物/动作H.264保存、重叠上下文、长片段分段和停止边界。其中平台/Box合约针对新归档的独立解包目录；逐人片段测试使用已验证与归档同字节的源码。

复验命令（仓库根目录）：

```bash
DARKPIPE_V2_OPERATOR_DIR="$PWD/reports/default-threshold075-20261024/release-extracted/dark-demo-v2" \
NVIDIA_BOX_ROOT="$PWD/reports/default-threshold075-20261024/release-extracted/NVIDIA-Box" \
.venv/bin/python -m pytest -q tests/test_v2_delivery.py tests/test_nvidia_box_native.py tests/test_person_release.py
.venv/bin/python reports/default-threshold075-20261024/release-extracted/NVIDIA-Box/check_env.py --files-only
```

此次未重新训练、重做GPU推理/实时压力/准确率实验，也未重新渲染已有demo/PPT。[此前0.91版本报告](../person-release-20261024/README.md)及素材保留原参数；其准确率、误报和录制负载数字不能直接当作0.75的结果。当前0.75为线上试用默认值。
