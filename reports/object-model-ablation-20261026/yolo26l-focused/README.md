# YOLO26l 720增强裁剪复测

- 权重 SHA256：`9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`
- 设备：CPU（当前会话 NVIDIA 驱动不可用）
- 输入：small_enhanced (long side 720, Retinexformer, person/ROI crop)；置信度 `0.35`；GT IoU `0.5`；原图亮度门 `30`。
- 测试 `78` 个带辅助物体框的val/test裁剪，以及真喝水窗口 `new-01_1935` 的 `5` 个采样帧人物ROI。

## 确认的真喝水窗口

用户确认样本为原片01/T317、149.92秒，原始亮度 `115.94`。检测结果只在同人ROI内统计瓶、酒杯、杯、碗（COCO类39/40/41/45）；物体类别框没有独立人工逐帧确认。

|帧|绑定有效|同人ROI容器检出|类别/置信度|
|---:|:---:|:---:|---|
|3723|True|否|无|
|3729|True|否|无|
|3736|True|否|无|
|3742|True|否|无|
|3748|True|否|无|

窗口内5帧至少一帧有同人ROI容器支持：**否**。

## 其他物品标注集

按旧辅助框逐类评估；模型是COCO 80类，没有camera、paper、portable_object、robot_or_exhibit这些目标类别。

### val

|类别|TP|FP|FN|GT|检测器支持该类|
|---|---:|---:|---:|---:|:---:|
|bottle|0|0|3|3|是|
|cup|0|0|0|0|是|
|phone|0|0|1|1|是|
|camera|0|0|0|0|否|
|paper|0|0|15|15|否|
|portable_object|0|0|0|0|否|
|robot_or_exhibit|0|0|4|4|否|

### test

|类别|TP|FP|FN|GT|检测器支持该类|
|---|---:|---:|---:|---:|:---:|
|bottle|0|0|12|12|是|
|cup|0|0|0|0|是|
|phone|0|1|3|3|是|
|camera|0|0|0|0|否|
|paper|0|0|12|12|否|
|portable_object|0|0|1|1|否|
|robot_or_exhibit|0|0|7|7|否|

逐图记录见 `holdout.csv`；同人喝水窗口见 `true-drink-frames.csv`。这一轮只验证模型是否能检测到物体，不据此更改平台配置或宣称动作先验已上线。
