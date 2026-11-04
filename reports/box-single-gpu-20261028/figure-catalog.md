# 图表目录

- `figures/01-live-comparison.png/.pdf`：原版与候选的300秒实际吞吐、health中位FPS、采集后延迟与动作年龄P95。来源live-baseline/final-verification.json。无误差线：每组只有一次运行。观察吞吐是否提高及动作延迟是否同时恶化；不能把编码15FPS当算法速度，不能把P95当置信区间。输出尺寸变化属于配置比较。
- `figures/02-scene-memory.png/.pdf`：分场景处理FPS及随时间的主进程RSS。来源by-scene.json与frames.jsonl。用来识别拥挤瓶颈和内存压力，指导板端验收；曲线不包含所有子进程，也不等于Jetson统一内存。每段帧是相关观测，不作独立显著性样本。
- `analysis-report.md`内精确数表与`summary-table.csv`：对应每次运行原始值。完整演示是模型推理输出，无生成式替代图片。
