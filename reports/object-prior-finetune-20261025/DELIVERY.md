# 本次交付文件

正式平台：`DarkScenePipeline/platform/dark-demo-v2_person_tau085_20261025.tar.gz` 和同名ZIP。

正式盒子：`DarkScenePipeline/NVIDIA-Box_tau085_20261025.zip`。

这三份包默认动作阈值0.85，“追逐”显示“跑动”，新片段名running。原图亮度门30、RGB权重0.95、soft、720、前后2秒保留。12份正式权重未改变，原`darkpipe-operator-0.3.0.tar.gz`镜像不需要重新上传。已有平台任务显式填入的阈值需要改为0.85，并重新导入suanzi.json。

| 正式包 | SHA256 |
|---|---|
| 平台tar.gz | `389de87c814ae9f01e7177bdf01b7d39fcc946bcebbb64cfd57ae51d199cfaeb` |
| 平台zip | `3eda18b17ebff8ea81afca36cc8e386b2b68c37bc90854a381a8f4c41daeb7b2` |
| Box zip | `27c0c6648b33809684051b143d618726284ec552ded240ec29f4515c3455e6c9` |

物体实验另交两份便携ZIP，均位于本实验目录：

- `object-prior-review_20261026.zip`：浏览入口、8段视频、标注审核页、图表、完整报告和454片段映射；适合看结果。
- `object-prior-candidate_20261026.zip`：额外包含候选PT/ONNX、实际397张YOLO数据集图／标签、407个已接受原图／gamma视图、审核坐标与决定、原始评估输出、复现脚本；适合继续修正标注和研究。

SHA256和逐文件清单见同名`.sha256`与`.manifest.json`；`delivery-verification.json`记录独立解包校验。数据集另附可搬运的`dataset-portable.json`，采用包内相对路径。原始训练manifest和审核记录保留当时的绝对路径，作为来源追溯；不应直接拿这些旧绝对路径在别机启动训练。

**这两份物体实验包都不是可直接上传运行的新平台算子。** 候选未达到纠错要求，正式包没有启用物体先验；没有对新物体支路做实时输入输出／保存／15fps验收。原9段长视频和454个MP4本体未重复装进实验ZIP；其文件对应关系完整保存在CSV/JSON中。

单图探针 `scripts/infer_image.py`可在独立解包后运行，默认CPU。训练／整批动作回放依赖完整项目及原数据，具体见REPRODUCE.md。`scripts/package_experiment.py`只复制manifest允许的样本，不把早期排除的遗留图混入数据集。
