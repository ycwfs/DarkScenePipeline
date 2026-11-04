# 10-29/30 人工标注多人场景 + 历史 ROI 动作模型微调

用户要求关闭8766审核服务，完全不用XCLIP，融合最新人工标注、预测other和旧少人ROI，微调VideoMamba及ST-GCN，然后测试、选择参数。8766已验证停止，8765未动。生产包/模型未修改。

上传原件：`../manypeople-action-review-20261029/10-25_动作审核.json`，SHA256 `9847f18d5e6474926e855bb38b87aa67a9e260c039785bdb6378377278678655`。473条，385确认/47排除/41不确定。用户明确同意排除两处时间冲突窗口；见annotations.json。12条确认other还与同轨迹排除实体冲突，整轨隔离，不训练/评分。

数据计划、协议在dataset-plan.json/protocol.json；原协议冻结，另有训练前补充：alpha=1只作消融，不可成为正式完整方案候选（保留挥手/跑动先验）。旧535源原划分不变，保留所有此前有标注ROI的源视频；当前前端同人同帧优先，保留旧版独有窗口。旧5504窗(5009/233/262)，新1516窗(598/275/643)，其中人工736/弱标签780。新train=05/07/09、val=04/06/08、test=01/02/03，跨视角按采集时段一起分配。新talk仅在train，没有独立talk召回证据。新other弱标签只限已经人工确认是真人的同轨迹，高other分且远离审核/不确定时段；权重.25，从不作测试真值。

运行容器：mp-ft29-extract-new-{0,1,2}在GPU0/1/2，mp-ft29-extract-old-{0,1,2}在GPU3/4/5。原darkpipe-operator:0.3.0，镜像未改。GPU此前全空闲。提取只读原片，写本报告下缓存；新完整顺序前端的轨迹/代次/32源帧/人物框全部必须对齐审核原运行，旧增强裁剪逐字节SHA校验。源03按24.999fps。禁止使用带字审核预览训练。

宿主协调器：scripts/orchestrate.py，PID见orchestrator-state.json，日志orchestrator.log；容器名/命令/阶段见pipeline-jobs.json。协调器先等待旧提取，GPU4/5补全val/test每个窗口的两种物体动作证据；再等待全部提取，组装及SHA校验，启动GPU0 RGB六轮、GPU1 STGCN二十轮，GPU2/3原模型同输入推理；选轮后推理候选，最后仅val联合搜索，写selection-lock.json后才计算test指标。

阶段检查：读取`*.log`尾部、`progress/*.json`、`pipeline-error.json`（如存在）。用授权宿主docker inspect检查确切容器exit状态；不要把运行中视为完成。脚本路径以当前报告定位，不需要覆盖旧证据。协调器失败后不要直接整轮重启同名容器/历史文件；查明已完成的阶段后继续缺失阶段。当前脚本已py_compile。extract初版以root启动，已有features/extraction/data/progress所有权已chown给1032，后续容器以1032运行；末尾可统一整理本报告目录归属。

后续仍必须完成：实际两个模型训练/验证/测试/阈值选择；确认脚本运行错误及时修复；选择结果的完整前端真实推理短视频（原图/亮暗/新多人，固定ROI指标不能冒充端到端准确率）；产生analysis-report.md、stats-appendix.md、figure-catalog.md及真实对比图/训练曲线、推荐模型及配置清单。新test只有1个采集时段，历史数据已反复查看，不能宣称独立盲测/显著泛化。不得把pseudo other当真值。若候选退化，明确保留旧分支。未授权新外部部署；本轮无需擅改平台生产包或提交Git。

## 更新：全部提取完成，训练已启动

7020窗全部像素/骨架/时间轴审核通过，dataset.json和skeleton-inputs.npz完成；old提取535/535、新9/9，所选1516新窗原轨迹/框/源帧精确对齐。物体证据两分片全部完成，覆盖全部canonical val/test；旧194训练窗骨架重建与历史normalized数组最大差0。

ST-GCN已完成20轮，验证选epoch1，权重`skeleton/epoch-01.pth` SHA2985edc2e3a20b96ff36e7166f965ea365c7550dc3153788a1dfa406139723b8。分支单独阈值.75：旧val FP20→5，新valFP18→9；此非最终融合成绩。骨架基线/候选全部窗口推理已完成。RGB目前两GPU0/1 DDP独立进程训练6轮、batch8（每卡4），约.6秒/step/700step每轮，原模型GPU2推理7020窗。查看rgb/history.json/train-rgb.log；必须等待真实完成，不能把epoch0当训练完成。

修复训练执行问题：thread DataParallel Triton autotuner NoneType竞争，独立DDP两轮梯度smoke通过；随后fork DataLoader在CUDA/NCCL后停在step1，保留attempts/fork-loader-stalled，改spawn workers重启同seed同init，已连续step70+通过。细节execution-corrections.json。停止的失败容器改名后缀-fork-stalled，别把其失败当现活动进程失败。当前协调器PID查orchestrator-state.json；finish.py也在后台等pipeline-complete，PID见finish-state.json。

finish.py已串好后续：verify_replay.py用实际生产融合+三先验逐窗校准一致性，report.py生成统计/图表，build_candidate.py生成本地隔离模型/配置及7个label-defined输入，demo.py三分片GPU0/1/2运行baseline与selected实际service输出/动作clips，verify_demos.py严格解码所有视频、检查事件阈值/先验/写盘统计。进度finish.log、错误finish-error.json、最终completion.json。这些脚本已写但尚未执行，后续仍需亲自核对结果/图/视频、处理任何真实报错并补充报告/AGENTS结论。

训练前grid补充已冻结在protocol-amendment-before-training.json：alpha .5/.6/.7/.8/.9/.95/.972/.99/1，object .01/.03/.05/.1/.25/.35/.5/.581/.65/.75；gate30/35/40/50，action .7/.75/.8/.85/.875/.9/.925/.95/.975。alpha1只作diagnostic，不可选为完整方案。训练期后不再改选择约束或拿test调参。

## 更新：联合选轮取代“先单分支选轮、再调参”

RGB第1轮验证：旧FP8/动作TP28保持；新硬负FP97→47，但动作TP14→9（宏召回.75→.4875）。按单分支原限制该轮不合格，不能只看误报下降就晋升。用户已获说明：最终检查完整融合可能与单分支不同，因此保留单分支选择作诊断，最终RGB全部7检查点（原模型+6轮）×STGCN全部21×四参数一起仅val选优。`joint-selection-protocol.json`记录是在第1轮val可见、test指标从未查看时补充；全方案的旧每类召回-5pp／新正类宏召回-10pp约束及FP优先不变，不是偷偷用test改规则。

当前coordinator PID3247252（以orchestrator-state.json为准），RGB独立DDP+spawn已持续正常，第2轮step590+；每轮约7.8分钟，6轮尚需等待。train-rgb容器未因协调器重启而中断。原RGB全部7020输入推理已完成；train/val与历史cache最大概率差旧.002812/新.002756、argmax0变化，见baseline-reproduction-audit.json。完整dataset合成就绪，STGCN已完成。

协调器后续现在为：候选单分支推理完成→joint_select.py（只读val，各骨架检查点补4gate推理，约423k组合）→写selection-lock.json→predict.py --which selected两分支→evaluate_selected.py第一次计算test→pipeline-complete。原tune.py保留但不再执行。`finish.py`等待后自动verify_replay/report/build_candidate/7输入×2模型实际demo/verify_demos。相关脚本已适配selected概率键与联合epoch。最终导出models/videomamba_t_person_roi_crowded_20261030.pth及stgcn_person_crowded_20261030.pth（若epoch0则明确原模型保留），candidate/ckpts本地链接按正式文件名加载；不是可上传完整平台包。

report.py新增两个分支自身test对照（.75/门35、骨架只pose-valid窗口），审核条目any-trigger FP/TP汇总，避免只以重复窗口说明误报改善；不把审核组数当实际clip数。新talk/其他无独立支持类别的局限明确。未生成最终报告，仍须等待/检查/完成任务。

## 2026-10-30 恢复进度与当前阻塞

RGB六轮训练、骨架20轮训练及两分支baseline/candidate全量推理均已落盘。`rgb/selection.json`为epoch0，即VideoMamba沿用原始权重；`predictions/rgb-candidate.npz`已完成7020×10，和baseline逐元素完全一致（max abs diff=0）。ST-GCN候选是epoch1，哈希与上方记录一致。XCLIP未使用，未查看测试成绩，生产权重/包未改。

联合选模、`selection-lock.json`、测试评分、生产融合回放、对比报告和实际离线服务demo尚未完成；协调器PID3247252和finish PID3241066已不存在。`predict-rgb-candidate.log`有完整`PREDICTED rgb candidate (7020, 10)`标记。`report.py`训练曲线虚线曾只对RGB读取联合锁定轮次，已修正为RGB和骨架都从`selection-lock.json`取轮次；脚本`py_compile`通过。

当前受控运行环境普通权限访问Docker API返回permission denied；授权提权检查/恢复调用均被automatic approval review以HTTP 503拒绝，调用未执行。宿主Python无PyTorch，NVIDIA驱动也不可用，因此不能在宿主替代CUDA联合选轮。不要换用其他Docker socket或重训。待审批服务恢复后，先检查`pipeline-jobs.json`记录的容器状态，再运行`orchestrate.py`接续联合验证；脚本会复用已登记的训练/预测任务。继续遵守测试标签只在选择锁生成后读取。

## 2026-10-30 最新恢复：评分已完成，实际服务演示验证中

更正上段进程判断：受限环境的`/proc`不可见不能证明宿主协调器已退出。原后台任务随后完成了联合选模及selected全量预测，锁定RGB0/ST-GCN2、alpha .972/门35/动作.925/物体.25；423360组，98172组符合验证约束。授权Docker查询已恢复，确认这些容器退出0。

`evaluate_selected.py`覆盖检查误迭代dict键导致`zero-dimensional arrays cannot be concatenated`，已改为`.values()`；828个真值val/test窗都有物体证据、伪标签不计分，保留失败日志与新任务`evaluate-selected-fixed`。锁定选择未改，测试结果已生成：旧test原方案FP2/163、TP34/51，候选FP2/163、TP36/51；新test原方案FP5/271、TP10/28，候选FP11/271、TP11/28。多出的6窗全是误喝水，因此不推荐晋升候选参数，生产保持原版。预列的“选定权重＋原参数”新test FP仍5/271、TP11/28，只作为诊断，不重新用test选参。

生产融合与三个先验2484次回放一致；6个生产主模型哈希仍与协议相同，旧新采集组各自train/val/test无交叉。报告脚本因原镜像没有Matplotlib初次失败，已把纯NumPy指标移到evaluation_metrics.py（20组存储指标全量复现），宿主绘图；模型推理环境不变。analysis-report/stats/figures已生成。

实际服务demo初次载入候选骨架时，训练checkpoint的TorchVersion元数据被生产weights_only加载器拒绝。导出脚本现仅保存相同张量与基本类型元数据，原始训练文件和选择锁不变；weights_only=True加载与逐张量相等断言通过。导出骨架SHA20761f5f…baddd，原始epoch2 SHA8be7f3e5…f8f48。原baseline完成部分保留，重跑任务名demos-{0,1,2}-exportfix。GPU0/1/4演示，GPU2/3已有其他任务；控制任务GPU5。

七个实际输入：新test四个人工类别上下文片（other/drink/chase/wave），旧暗场val(L30.99)、微暗train(L35.26)、亮场test(L112.43)。旧test没有L<75正例，暗/微暗仅展示功能，不作为额外测试真值。完成状态须看`finish-exportfix.log`和`completion.json`，视频检查尚待结束。

## 最终状态：实验完成，保持正式版本

`completion.json`已生成，`finish-exportfix.log`为FINISHED。7输入×2方案全部完成：2518帧、922事件、14完整输出视频＋8动作片段，共22个H.264视频严格完整解码通过；写错误/强制释放/unfinished/片段队列丢帧均0。461对同人同源帧事件标签0变化，短片喝水/跑动漏检保留，不能以演示宣称识别改善。三组抽帧对照已目视检查，见figures/demo-frame-check.jpg。

分析/统计/真实图表、独立模型导出与候选配置、视频索引均已生成，入口README.md、analysis-report.md、DEMO_VIDEOS.md。adoption-decision.json结论为retain_production：联合候选多人误报增加，因此不替换正式模型、参数或交付包。所有本轮授权的训练、选参、测试和有限离线演示工作已完成；新的实时平台部署/打包不是本轮已完成项，候选本身也未达到采用目标。没有Git提交。
