# 多人骨架动作展示与最高分人物融合（2026-10-22）

已完成平台算子和 NVIDIA Box 同步。骨架分支激活时，每个有效人物展示自己的 ST-GCN 类别及置信度；黄色骨架和 `*` 标记被选中参与 alpha 融合的人物，其余人物为绿色。人物标签包括“其他”，不再受视频底栏最终类别限制。

直接打开 [可视化预览页](index.html)，或查看 [双人挥手3](gate-off-demo/videos/挥手3_多人骨架.mp4)、[双人挥手4](gate-off-demo/videos/挥手4_多人骨架.mp4)。可拷贝整个目录，或解压同目录的 `multiperson-previews-20261022.zip` 后打开 `index.html`。

**这两段双人视频单独设置了 `lum_gate=0`**：源图平均亮度低于默认门限75，按生产默认配置其骨架分支不会激活。默认75的同源视频也在预览页中，未更改生产照度门。视频开头仍需约1秒积累识别窗口，首次识别前不画骨架。

## 决策与显示约定

对每个有效人物得到10类概率 `p_person[i]`：

```text
selected = argmax_i max_c p_person[i, c]       # 包括 other
p_skeleton = p_person[selected, :]            # 该人的完整10类向量
p_fused = 0.7 * p_rgb + 0.3 * p_skeleton
bottom_label = reject_probs(p_fused, threshold=0.75)
```

选人依据是**动作分类概率**，而不是姿态关键点质量分。每人的类别单独取其向量最大项；人物标签展示原始骨架分类概率，0.75拒识阈值仍作用于最终融合结果。例如挥手3的4.8秒事件：后方人物“挥手98.85%”、前方人物“其他46.05%”，选择后方人物，底栏为“挥手94.03%”。当 other 人物分数最高时，同样使用其完整向量。

- 窗内最多6个人。对同一人的序列做倒序关键点位置关联，以减少每帧检测置信度排序变化造成的换人；近乎重合的重复姿态在分类之前去重。
- 有效人物须满足已有姿态质量/完整度要求，并在窗口末仍可见。逐帧关键点平均分的窗口最大值至少0.5，出现帧中至少一半有至少10个关节分数超过0.3；末帧也须满足平均分0.5及至少10个可见关节。
- 槽位只在识别窗口内有意义，不是跨窗口身份ID。长遮挡、交叉运动、缺少有效姿态的人员仍可能无法正确关联或显示。
- 无合格人物时保留既有无人策略。低于照度门时只走RGB，无骨架；显示开关不影响融合计算。
- 识别间隙仍用干净增强帧做光流跟随，过期、跟随失败或骨架分支关闭时隐藏。不会把骨架文字送回识别模型。
- 事件JSON新增 `skeleton_selected_slot`，每个 `actor_poses` 元素包含自己的 `label`、`confidence` 和 `selected_for_fusion`。

本次默认仍为 pn2、处理长边720、`reco_min_conf=0.75`、`lum_gate=75`、`skel_fusion_alpha=0.7`、`pose_overlay=true`、片段前后各2秒。未重训或替换模型。可选 max_prior 仍按其已有决策规则选择具名动作候选；本轮要求的最高分人物完整向量融合适用于默认 alpha。

## 可视化实测

在原 `darkpipe-operator:0.3.0` 镜像、物理GPU1/5完成真实 Retinexformer、VideoMamba pn2、RTMO-s、ST-GCN 推理。共14个不同源视频、16次运行、1,689输出帧、88条事件；全部H.264完整解码，输出帧数与输入一致。每条事件独立核对逐人类别、最高分人物、完整融合向量和阈值后的最终结果。视频分组不能当作独立精度测试集。

| 样例/配置 | 事件数 | 激活骨架事件 | 双骨架事件 | 同时画双骨架的帧数 | 说明 |
|---|---:|---:|---:|---:|---|
| 挥手3，默认门75 | 9 | 0 | 0 | 0 | 原图在门下，正确保持纯RGB |
| 挥手4，默认门75 | 10 | 0 | 0 | 0 | 同上 |
| 挥手3，仅展示关门 | 9 | 9 | 9 | 136/160 | 真实同框两人；可显示不同类别 |
| 挥手4，仅展示关门 | 10 | 10 | 10 | 145/169 | 真实同框两人；底栏9次挥手、1次其他 |
| 追逐24，默认门75 | 7 | 7 | 7 | 102/137 | 包含被误检为人的固定人形展品 |
| 追逐25，默认门75 | 7 | 7 | 7 | 112/141 | 同上，底栏全部其他 |
| 追逐26，默认门75 | 5 | 5 | 4 | 57/102 | 同上 |
| other80，默认门75 | 4 | 4 | 0 | 0 | 骨架可报挥手，但底栏融合后全部其他 |

其余样例为挥手5/20/18、追逐15/16、喝水16、捡东西2、other1；逐条结果见 [combined-summary.json](combined-summary.json)，解码记录见 [media-verification.json](media-verification.json)。

**模型限制在视频中原样保留**：中心屏视角的人形展品会被RTMO误检为人，并可能参与最高分选择；不能把这些样例中的两个骨架算作两名真实人员。挥手视频中的未挥手者也有被骨架模型低置信度判成挥手的帧。此次验证的是逐人展示和融合实现，不代表检测/动作类别均正确，也不代表所有场景误报改善。旧ABCP准确率来自旧人物合并方式，本轮没有重跑全量ABCP，不可直接沿用为新规则成绩。

## 最终包验证

- 源码相关回归155项通过；随后在原镜像内，针对**独立解包的最终平台/盒子文件**同样155项通过，见 [package-regression.xml](package-regression.xml)。覆盖多人选择/other/人员离开、排序换人、光流、底栏与片段事件一致性、拥塞收尾、服务生命周期和Box入口。镜像未安装pytest，测试依赖通过本地目录挂载，镜像没有修改。
- TAR、ZIP各47个文件；Box清单66个文件加内置manifest。归档SHA256、ZIP CRC、全部成员与源码哈希、Box runtime/权重与平台来源、默认别名字节一致性均通过，见 [archive-verification.json](archive-verification.json)。
- 实际平台TAR解包入口在原镜像/GPU1运行；实际Box ZIP解包入口在开发机原生环境/GPU5运行，均使用挥手3和 `lum_gate=0`，其余模型/输出参数保持默认，包括片段quality去噪。
- 两端各输出160帧/6.40秒、1440×874 H.264整段视频；各9条事件全部带两个人物及唯一最高分标记；视频级标签、选中槽位两端一致。各保存1段159帧/6.36秒动作片段，片段时长差1帧来自既有时间网格重采样。完整视频与片段全部完整解码。
- 两端退出码0、会话 `finished`，`clips_forced/clips_unfinished/clips_failed/write_errors/frames_dropped/clips_abandoned` 全部为0，见 [entry-verification.json](entry-verification.json)；实际保存画面见 [平台截图](platform-entry.jpg)、[Box截图](box-entry.jpg)。这次没有连接实体Box，也没有做线上长期稳定性或速度验收。

## 交付

| 包 | SHA256 |
|---|---|
| [平台TAR](../../platform/dark-demo-v2_multiperson_20261022.tar.gz) | `7b9b9afd51e72e851d51625eb0908214b3f23582ec4a906ff4513c583636fcf5` |
| [平台ZIP](../../platform/dark-demo-v2_multiperson_20261022.zip) | `5dfe0e15b607088d615474f13216891350588af587cafcbaae775454c6383e18` |
| [NVIDIA Box ZIP](../../NVIDIA-Box_20261022.zip) | `e8de2f71cc0ab6c65e3349c194463838ea24d6ca80cdec61ee1597afee8f4dfb` |

默认别名 `platform/dark-demo-v2.tar.gz`、`platform/dark-demo-v2.zip`、`NVIDIA-Box.zip` 已同步，旧日期包保留。各包附逐文件manifest和SHA256文件。平台重新导入包内`suanzi.json`，确认实际任务参数为`alpha/0.7/0.75/pose_overlay=true`，生产默认照度门保持75。

平台8份权重和Box对应权重逐字节同源，现役pn2 SHA256为 `d30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362`。原镜像ID与归档SHA256均核实未变，详见 [image-verification.json](image-verification.json)，**不需要重传镜像**。

复现见 [RUNBOOK.md](RUNBOOK.md)。
