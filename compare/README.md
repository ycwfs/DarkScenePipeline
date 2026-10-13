# compare/ — 开发与对比实验工具（入仓副本）

本目录是仓库外开发工具树 `/ssd1/wfs/project/lowlight/compare/` 的副本（2026-10 整理入仓），
仓库根相对路径常量已改指本仓库。三层结构：

- `*.py` — 增强模型对比（HVI-CIDNet / Retinexformer，见仓库根 dark-skeleton-detection.md
  与 compare/results/ 报告）、超分对比与测速、ARID20 训练评测、管线分段可视化
- `behavior/` — 行为识别训练线：realact 缓存与训练（realact_cache / realact_train，含
  fullfov 224×400、LwF、ARID floor 门控协议）、X-CLIP 对照线、负样本缓存
  （negatives_cache / plat_neg_cache——后者处理平台回传误报片段：裁标签条、不二次增强）、
  演示/评测/可视化工具
- `behavior/skeleton/` — 骨架分支研究线：三套件评测（fusion_eval：val / fp65 /
  positives）、α 重放（alpha_resweep_online——融合对 α 线性，可从 CSV 精确重算）、
  ST-GCN 训练（skel_train）、几何先验研究（skel_prior_*，实测不增益已否决，实现存档）、
  DA-LLPose 教师实验（dark_infer_dallpose）等

**运行期缓存与数据不入仓**：`behavior/cache/`（realact/skel/platneg 缓存，数十 GB）、
ARID v1.5、real-actiondata、平台负样本回传、RTMO-s onnx（根 `models/`）都留在仓库外
`/ssd1/wfs/project/lowlight/` 原位；各脚本的默认路径指向那里（绝对路径常量或仓库外相对
引用），在本机这份克隆里可直接运行；换机需按相同布局重建数据。

**权重不入 git**（GitHub 单文件 100MB 上限）：部署权重在根 `ckpts/`（白名单方式，
见根 .gitignore 注释）；RTMO-s 姿态权重随算子包交付，解包 zip 即得；
`platform/dark-demo-v2/ckpts/` 是打包时从 `ckpts/` + `models/` 拷出的，可重建。
