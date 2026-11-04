# 低分候选复核实验

结果入口：[analysis-report.md](analysis-report.md)。

复现顺序：prepare.py → launch_frontend.py 与 prepare_objects.py/launch_objects.py → test_policy.py → evaluate.py → verify.py → report.py。GPU步骤使用已有0.3.0镜像，只读挂载正式源码与模型；所有产物写本目录。已完成步骤可复用features/object-evidence。preliminary-invalid不参与分析。
