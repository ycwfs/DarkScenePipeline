# 复现与产物

在DarkScenePipeline目录下使用`.venv/bin/python`。GPU需主机CUDA访问；物理GPU编号不写入算法协议。本轮分别使用空闲GPU1、4分配独立原视频，未占用其他计算进程所在的卡。

```bash
# 首次冻结；已有protocol时主动报错，避免覆盖来源与协议。
.venv/bin/python reports/onsite-object-prior-20261026/scripts/evaluate.py freeze

# 两个独立进程，可恢复已经完成且协议哈希一致的逐视频结果。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 .venv/bin/python -u reports/onsite-object-prior-20261026/scripts/evaluate.py infer --shard 0/2
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 .venv/bin/python -u reports/onsite-object-prior-20261026/scripts/evaluate.py infer --shard 1/2

.venv/bin/python reports/onsite-object-prior-20261026/scripts/analyze.py
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 .venv/bin/python reports/onsite-object-prior-20261026/scripts/visualize.py
.venv/bin/python -m pytest tests/test_person_object_prior.py -q
```

`per-video/*.json`覆盖535历史来源，407现场和128线上分开。每个文件保留来源、动作概率、标签出处、窗口帧号、原图亮度、实际裁剪像素SHA256、物体检测输出、融合和过滤原因。动作、人体、骨架使用固定缓存；本轮新执行的是候选帧增强与物体推理，图中人体、动作为离线缓存回放。

`demo-selection.json`预先记录按结果类型选择的8个错误分析案例；`demo-results.json`给出完整来源视频帧数和严格解码检查。演示是定性取例，不代表随机样本准确率。`.mp4`按原速，无音频；原视频和旧推理记录不改。

`verification.json`核对全部来源/窗口数量、低亮和未知亮度不过滤、仅喝水/捡物可变other、汇总一致性，以及与前一轮选定m720方案55个重叠窗的决策复现。

未接入生产PersonEngine，未打平台包、Box包，未重建镜像；本实验不是实时性能或保存器完整性验收。
