# 逐人方案性能验证

结论：**当前独立逐人实现未满足 ≥15 fps、识别时延≤1秒。**
最新ROI VideoMamba、α0.7、阈值0.75、照度门75；不用物体先验。单张RTX3090、六段视频、每段三次吞吐复测及一次节奏回放。

- [详细结论与精确数表](analysis-report.md)
- [兼容单卡、多卡的优化方案](optimization-plan.md)
- [统计口径与限制](stats-appendix.md)
- [图表说明](figure-catalog.md)
- [结构化汇总](summary.json)、[CSV](summary.csv)
- [输入/代码/模型哈希与环境](evidence/protocol.json)
- [视频解码和算法等价核验](evidence/verification.json)

原始日志在benchmark.log，完整运行与视频在runs/（大文件本地保留）。生产算子、权重和交付包未改。

复现（仓库根目录，选择空闲GPU；默认runs/已存在时脚本拒绝覆盖）：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib" \
.venv/bin/python -u reports/person-performance-20261023/scripts/benchmark.py --output /tmp/person-performance-replay
.venv/bin/python reports/person-performance-20261023/scripts/analyze.py
```

分析脚本读取本目录runs/，复测到其他目录时需明确选择对应分析输入，不能将两批结果混用。
