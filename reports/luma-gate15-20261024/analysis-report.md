# 原图照度门75与15配对验证

**门限15可以启用更暗场景的骨架，但不能视为无损下调：实测同时出现误报减少与正确动作被压为其他。当前证据不支持仅改门限就直接替换生产方案。**

完成264段、27,623原视频帧回放；每段独立运行两套跟踪/动作状态。
固定清理展品后的YOLO26m、暗场微调RTMO-s、逐人ROI VideoMamba epoch07、原ST-GCN、Retinexformer；α=0.7，动作阈值0.75，无骨架平滑。
此处检验用户指定的15，不扫描阈值、不训练、不改变当前生产默认75或交付包。

| 数据 | 配对已标注窗 | 正动作命中：75 → 15 | other误报：75 → 15 | 全类正确：75 → 15 |
|---|---:|---:|---:|---:|
| 验证集 · 全部亮度 | 198 | 25 → 25 / 36 | 6 → 6 / 162 | 181 → 181 / 198 |
| 验证集 · 15≤L<75 | 18 | 4 → 4 / 11 | 0 → 0 / 7 | 11 → 11 / 18 |
| 测试集 · 全部亮度 | 214 | 33 → 33 / 51 | 5 → 2 / 163 | 191 → 194 / 214 |
| 测试集 · 15≤L<75 | 16 | 0 → 0 / 0 | 3 → 0 / 16 | 13 → 16 / 16 |
| 训练内诊断 · 全部亮度 | 550 | 347 → 288 / 348 | 12 → 0 / 202 | 537 → 490 / 550 |
| 训练内诊断 · 15≤L<75 | 515 | 317 → 258 / 317 | 12 → 0 / 198 | 503 → 456 / 515 |

配对窗口中共181个最终标签变化；已标注部分修正15个、损失59个，其中4个损失来自无合格骨架的soft退化。
受影响亮度区间的测试集16窗全部为other，没有正动作；其100%正确率只表示这16个负窗未误报，不能证明暗场正动作召回。验证集此区间仅11个有标签的正动作窗，两档均只命中4个。
注意：能看见骨架、允许调用姿态、最终融合识别正确是不同指标。无合格骨架时，现有规则以other向量混合；非other最高只剩0.7，达不到0.75动作阈值。

## 具体变化示例

| 来源 | 帧/原图L | 真值 | 75结果 | 15结果 | 15分支 | 骨架真类概率 |
|---|---|---|---|---|---|---:|
| real-actiondata-all/喝水10.mp4 | 50 / 20.96 | drink | drink 96.3% | other | rgb_skeleton | 11.2% |
| real-actiondata-all/喝水11.mp4 | 24 / 19.41 | drink | drink 95.9% | other | rgb_skeleton | 5.8% |
| real-actiondata-all/喝水12.mp4 | 50 / 23.15 | drink | drink 96.1% | other | rgb_skeleton | 4.5% |
| real-actiondata-all/喝水13.mp4 | 63 / 24.67 | drink | drink 96.2% | other | rgb_soft_no_pose | 无 |
| real-actiondata-all/喝水2.mp4 | 50 / 32.52 | drink | drink 95.9% | other | rgb_skeleton | 7.5% |
| real-actiondata-all/喝水42.mp4 | 63 / 31.09 | drink | drink 95.8% | other | rgb_skeleton | 13.4% |
| real-actiondata-all/喝水5.mp4 | 37 / 15.01 | drink | drink 96.1% | other | rgb_skeleton | 0.9% |
| real-actiondata-all/挥手12.mp4 | 37 / 21.48 | wave | wave 95.4% | other | rgb_skeleton | 11.3% |
| real-actiondata-all/other/other200.mp4 | 64 / 31.58 | other | drink 85.5% | other | rgb_skeleton | 8.7% |
| real-actiondata-all/other/other225.mp4 | 37 / 31.02 | other | wave 94.8% | other | rgb_soft_no_pose | 无 |
| real-actiondata-all/other/other189.mp4 | 69 / 31.21 | other | drink 84.8% | other | rgb_soft_no_pose | 无 |
| real-actiondata-all/other/other220.mp4 | 63 / 31.13 | other | wave 91.8% | other | rgb_skeleton | 61.3% |
| real-actiondata-all/other/other224.mp4 | 63 / 31.18 | other | drink 95.0% | other | rgb_skeleton | 69.4% |
| real-actiondata-all/other/other78.mp4 | 24 / 34.97 | other | fall 84.0% | other | rgb_soft_no_pose | 无 |
| real-actiondata-all/other/other79.mp4 | 74 / 35.37 | other | chase 89.3% | other | rgb_soft_no_pose | 无 |
| real-actiondata/抛物.mp4 | 505 / 38.86 | other | throw 84.5% | other | rgb_skeleton | 54.0% |

有合格骨架也可能退化：固定ST-GCN对真动作分数不够时，会拉低RGB正确类别。例如RGB=0.96，骨架真类概率必须至少0.26，才能使0.7×RGB+0.3×骨架达到0.75。姿态模型微调并未同步适配骨架动作分类器。

[喝水12实际漏报帧](evidence/drink12-f50.jpg)：原图L≈23.15，RGB喝水0.96066、ST-GCN喝水0.04477，融合0.68589低于0.75。两档使用同一人体裁剪，右侧仍显示骨架。

## 覆盖与比较边界

| 划分 | 来源数 | 窗口75/15 | 未标注75/15 | 无窗口来源75/15 |
|---|---:|---:|---:|---:|
| val | 60 | 226/226 | 28/28 | 16/16 |
| test | 58 | 252/252 | 38/38 | 6/6 |
| train | 146 | 946/946 | 396/396 | 31/31 |

主要表格只比较同源、完全相同32个取帧索引和RGB裁剪字节的窗口；两档RGB概率逐位相同。跟踪因姿态改变造成的未配对窗口在results.json另列，不混入主配对表。
旧逐人动作标注通过人物框IoU与时间段迁移，至少80%采样帧关联一致；新增正样本人物不能按视频名自动赋标签，未确认窗口保持unknown。额外人工视觉核对如有，保留在label-review.json。
视频级other沿用已有数据标注，包含可能的误检候选；无检测或无完整窗口不能当作正确other。
验证/测试沿用冻结划分；历史初始化模型见过许多现场来源，不能称全新场景泛化。训练内暗场只用于找退化原因。
本次是全帧离线动作/门控验证；共享前端和RGB计算减少重复工作，不能将运行耗时当作双流生产FPS或实时平台验收。
门限使用增强前原图的整帧均值，原始黑边参与；回传片段缺少原图L，门限保持跳过。窗口均值与逐帧是否提取姿态分开判断。

113个全程L<15、L≥75或原L未知来源，两档输入窗口与融合概率完全一致。
已处理所有列入协议的来源，未删除漏检或识别失败的视频。

## 证据

- [可播放对照页](index.html) · [视频预览ZIP](gate15-review.zip)
- [逐窗CSV](paired-windows.csv) · [完整统计](results.json) · [统计附录](stats-appendix.md)
- [图表与解释](figure-catalog.md) · [冻结协议及模型SHA256](protocol.json)
- data目录保留每帧框/姿态、两档事件和完整概率。scripts目录提供回放、分析、发布与验证。
