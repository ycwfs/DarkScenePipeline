# 逐图核对记录

已查看40组对照图。以下为助手视觉核对，未绘制全量物体真值框，不计算检测mAP。
候选出现的位置、语义与动作支持分开记录；图号对应index.html。

| 图号 | 视频/帧/人物 | 观察 |
|---|---|---|
| [001](visual/001_2f8105910668ee63_000044_T001.jpg) | real-actiondata-all/喝水1.mp4 / 44 / T01 | 原图ROI框位于人物嘴边的蓝色瓶子；增强ROI仍未过阈值。 |
| [002](visual/002_2f8105910668ee63_000058_T001.jpg) | real-actiondata-all/喝水1.mp4 / 58 / T01 | 两种人物ROI都框中嘴边蓝色瓶子。 |
| [003](visual/003_a10a2ba1533ca770_000046_T001.jpg) | real-actiondata-all/喝水12.mp4 / 46 / T01 | 新增bottle框位于人物头上方的背景展品区域，不是其手中容器；无可用手腕先验。 |
| [004](visual/004_a10a2ba1533ca770_000068_T001.jpg) | real-actiondata-all/喝水12.mp4 / 68 / T01 | cup候选在人物头上方背景；另有人物旁handbag候选，不构成喝水物体证据。 |
| [005](visual/005_a8b212b367a7594f_000024_T001.jpg) | real-actiondata-all/喝水17.mp4 / 24 / T01 | 原图ROI小框落在嘴边所持透明容器，bottle/cup细分类不可靠。 |
| [006](visual/006_0c6f7f8f83b0ee00_000035_T001.jpg) | real-actiondata-all/喝水2.mp4 / 35 / T01 | 整帧框中蓝色瓶子0.265；增强人物ROI降至0.236并把手臂附近误报为skateboard，原图ROI漏检。 |
| [007](visual/007_0c6f7f8f83b0ee00_000053_T001.jpg) | real-actiondata-all/喝水2.mp4 / 53 / T01 | 整帧和增强ROI都框中蓝色瓶子；原图ROI未检出。 |
| [008](visual/008_a677b36824b2473f_000022_T001.jpg) | real-actiondata-all/喝水35.mp4 / 22 / T01 | 喝水35胸前容器：整帧0.067，增强ROI cup0.738，原图ROI cup0.451。 |
| [009](visual/009_a677b36824b2473f_000089_T001.jpg) | real-actiondata-all/喝水35.mp4 / 89 / T01 | 喝水35嘴边容器被两种ROI检出；椅子也进入脚边先验，表明物体存在不代表捡物。 |
| [010](visual/010_6d410d75543e1fbd_000032_T001.jpg) | real-actiondata-all/捡东西14.mp4 / 32 / T01 | 原图ROI框落在被拾起的纸片/薄片上，但类别误报skateboard0.426；只能算弱定位线索。 |
| [011](visual/011_e89e6fbe81384ec5_000032_T001.jpg) | real-actiondata-all/捡东西22.mp4 / 32 / T01 | 原图ROI把弯腰者身体误报dog/cow，框并非所捡纸片。 |
| [012](visual/012_7ce9e264c4e2c972_000024_T001.jpg) | real-actiondata-all/捡东西23.mp4 / 24 / T01 | 原图ROI dog0.436包住弯腰者身体，纸片并未被正确检出。 |
| [013](visual/013_182e3a30ac1a8d7b_000014_T001.jpg) | real-actiondata-all/捡东西24.mp4 / 14 / T01 | 增强ROI skateboard0.584落在手臂/裤腿附近，未框中地上的纸片。 |
| [014](visual/014_8594e05e8a56819b_000012_T001.jpg) | real-actiondata-all/捡东西25.mp4 / 12 / T01 | 增强ROI dog0.852、原图ROI dog0.409包住弯腰的人，属于明确假物体。 |
| [015](visual/015_8594e05e8a56819b_000036_T001.jpg) | real-actiondata-all/捡东西25.mp4 / 36 / T01 | 增强ROI dog0.898、原图ROI dog0.915包住弯腰的人，分数高但类别错误。 |
| [016](visual/016_fd54d3f8ee4adff7_000030_T001.jpg) | real-actiondata-all/捡东西27.mp4 / 30 / T01 | 增强ROI tennis racket0.316对应人物裤腿，不是其手边纸片。 |
| [017](visual/017_314915317e94aa69_000065_T001.jpg) | real-actiondata/喝水1.mp4 / 65 / T01 | 旧目录喝水1，瓶子位于嘴边，两种ROI检出；增强ROI还把头部小区域误报tennis racket。 |
| [018](visual/018_e7e805c9abfc755e_000046_T001.jpg) | real-actiondata/喝水12.mp4 / 46 / T01 | 旧目录喝水12，增强ROI bottle框在头上方背景物品，缺少本人手边关联。 |
| [019](visual/019_35f30d6b84db9fa2_000053_T001.jpg) | real-actiondata/喝水3.mp4 / 53 / T01 | 旧目录喝水3，原图ROI bottle框位于嘴边蓝色容器。 |
| [020](visual/020_e00e00e05307c3e5_000035_T008.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_095756_drinking_water_0054.mp4 / 35 / T08 | other0054人物手部附近cup0.260；小物体外观不足以确认杯子，单帧也不能证明喝水。 |
| [021](visual/021_ad87568563d7ab31_000047_T009.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_101141_drinking_water_0098.mp4 / 47 / T09 | other0098的bottle0.272落在前景遮挡人物/衣物边缘；物体类别和归属无法可靠确认。 |
| [022](visual/022_ad87568563d7ab31_000059_T009.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_101141_drinking_water_0098.mp4 / 59 / T09 | 同一other0098后续帧bottle0.275仍在前景衣物边缘，重复检测并不自动证明正确。 |
| [023](visual/023_1b03574a847c85fe_000000_T004.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_102443_drinking_water_0137.mp4 / 0 / T04 | other0137的cup0.329在手持物附近；画面模糊，类别不确定，既有动作真值仍为other。 |
| [024](visual/024_53daffaa9ef9103a_000000_T009.jpg) | reports/feedback-finetune-20261019/data/videos/20261015_161630_77/drinking_water/20261015_162326_drinking_water_0001.mp4 / 0 / T09 | 新回传0001中，cup候选位于人物腰前手持物附近，不能从外观确认；动作标签为other。 |
| [025](visual/025_53daffaa9ef9103a_000112_T009.jpg) | reports/feedback-finetune-20261019/data/videos/20261015_161630_77/drinking_water/20261015_162326_drinking_water_0001.mp4 / 112 / T09 | 同一新回传0001的4秒处仍出现cup候选；静态手边物体无法证明喝水。 |
| [026](visual/026_1d5e40daf952ed1b_000000_T002.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_083425_drinking_water_0002.mp4 / 0 / T02 | 整帧在人物腿侧边缘给tennis racket，裁剪后消失；非人框不是捡物证据。 |
| [027](visual/027_2d5b0f004cf5b221_000068_T001.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_091756_drinking_water_0009.mp4 / 68 / T01 | 原图ROI把脚部/鞋附近误报skis0.416。 |
| [028](visual/028_431d51a1bdfa6168_000022_T003.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_091922_drinking_water_0010.mp4 / 22 / T03 | 人体框实际是展品机器人；整帧脚边支持来自ROI之外的邻近框，裁剪后被移除。 |
| [029](visual/029_a6dc542d4e2f385a_000011_T003.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_091930_drinking_water_0012.mp4 / 11 / T03 | 同一机器人候选，整帧脚边支持在可见人物ROI之外；裁剪减少干扰但未解决人体误检。 |
| [030](visual/030_b0994c2f308d7657_000051_T003.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_092839_drinking_water_0014.mp4 / 51 / T03 | 机器人轨迹，整帧非人先验来自局部画面之外；裁剪后没有支持。 |
| [031](visual/031_a6e8375246cb3a7c_000000_T003.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_093135_drinking_water_0016.mp4 / 0 / T03 | 两种ROI在行走者鞋边给skateboard候选，没有可见滑板或拾取动作。 |
| [032](visual/032_bba3333c7bc0ab8b_000044_T002.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_094227_drinking_water_0018.mp4 / 44 / T02 | 机器人轨迹；整帧高脚边分数并不对应真实人物动作。 |
| [033](visual/033_82c4d886d2db744e_000000_T004.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_094245_drinking_water_0019.mp4 / 0 / T04 | 原图ROI在脚边把鞋/地面局部误报skateboard。 |
| [034](visual/034_da6e3389c18aa09c_000024_T005.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_094436_drinking_water_0021.mp4 / 24 / T05 | 画面确有随身包，但它与走路者同行，不是正在捡起的地面物；多人重叠也使归属不明确。 |
| [035](visual/035_bad35584ef5e9ffc_000000_T005.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_094525_drinking_water_0022.mp4 / 0 / T05 | 原图ROI在鞋边给skateboard0.336，人物在站立/行走。 |
| [036](visual/036_47535b47f6aa9641_000014_T007.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_094714_drinking_water_0024.mp4 / 14 / T07 | 机器人候选；整帧脚边支持来自ROI之外，裁剪移除了背景证据。 |
| [037](visual/037_033b456513837833_000047_T002.jpg) | online-negative-sample/20261012_083305_78/drinking_water/20261012_094721_drinking_water_0025.mp4 / 47 / T02 | 增强ROI把鞋部附近误报skateboard0.339。 |
| [038](visual/038_8f780ff0eefc1fcb_000044_T001.jpg) | real-actiondata-all/喝水16.mp4 / 44 / T01 | 喝水16嘴边容器在三种输入下都未过阈值；整帧还输出无关train框。 |
| [039](visual/039_ad442c59b37f1b2f_000060_T001.jpg) | real-actiondata-all/喝水34.mp4 / 60 / T01 | 喝水34嘴边容器仍未检出；ROI检出/误分类前景椅子，不能修复RGB把喝水判挥手的问题。 |
| [040](visual/040_1f153eddcd6d673e_000028_T001.jpg) | real-actiondata-all/捡东西2.mp4 / 28 / T01 | 捡东西2手边纸片在三种输入下都未检出；COCO80没有通用纸张类别。 |
