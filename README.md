# Bayes Final Project

公开仓库：[mmei0114/Bayes-Final-Project](https://github.com/mmei0114/Bayes-Final-Project)（2026-11-11经用户确认保持公开）。

离散纵向数据的混合 GLMM 聚类。

课程大作业基础修订版：独立重要性抽样MCEM、高斯变分近似（GVA）和一维积分ML参照（QML），覆盖二元、Poisson、负二项响应。

**最新报告：[11页PDF](report.pdf) · [LaTeX正文](report.tex) · [教师要求对应清单](requirements.md)**

正式实验为14个场景、70份不同模拟数据、155次拟合，每次3起点。QML 70/70、GVA 70/70、MCEM 15/15满足各自数值停止标准；28次参数边界、4次内部求解器历史非成功状态均保留并在报告中说明。每场景5次重复，结论限于基础版本；小Q增量停止不等于高精度似然最优。

教师最新补充允许以整体似然的稳定性评价收敛，不强求每个参数都达到高精度稳定；另要求写清具体积分和抽样技术。该要求已记入 [任务清单](requirements.md)。正文已加入积分技术表，区分MCEM独立重要性抽样、观测似然的自适应GH、GVA的解析/GH期望，以及条件核验的联合独立MH。

收敛补充已完成21/21条：`src/convergence.py`复用原数值内核，重跑所选初始化并记录每轮参数和观测log似然；MCEM必要时最多延长30轮，原始155次结果不覆盖。预先固定似然相对变化阈值为10⁻⁶、连续3轮，MCEM末窗至少有一次接受更新；参数缩放变化10⁻³仅作辅助。15次MCEM中，二元5/5、Poisson0/5、负二项5/5通过本次严格似然口径；参数平稳分别为0/5、5/5、2/5。Poisson未通过者含一个全拒绝窗，不能凭视觉平直直接改记通过；其余4条末窗最大相对变化约1.03×10⁻⁶至1.91×10⁻⁶。

另6条QML/GVA代表性轨迹均通过观测似然窗口检查；15条MCEM在原轮数处均与原保存结果达到预设数值一致性。原轮数处0/15通过新增严格似然口径，与原算法停止标记分开报告。新记录在 [data/convergence/](data/convergence/README.md)，本地yellow完成11条、新服务器base完成10条；[逐条汇总](results/convergence_summary.csv)、[审计](qa/convergence_audit.json)及参数/似然图均随仓库保存。阈值不因结果而修改，数值稳定也不证明全局最优。

## 目录

| 路径 | 内容 |
|---|---|
| [src/](src/) | 当前统计代码，所有方法在同一套目录中维护 |
| [results/](results/) | 原155次拟合指标、汇总表及新增21条收敛诊断 |
| [figures/](figures/) | 报告图件；三类响应各自的参数图及整体观测似然图 |
| [qa/](qa/) | 数值检查、正式审计与交付核验 |
| [docs/methods.md](docs/methods.md) | 方法细节、实验设计和复现说明 |
| [data/](data/README.md) | 已完成正式拟合、原始模拟数据、原始采样链、实际执行源码副本及SHA256，随仓库保存 |
| [docs/reading_notes.md](docs/reading_notes.md) | 三篇课程论文阅读笔记；v1取舍与当前修订范围已区分 |
| [requirements.md](requirements.md)、[Project.pdf](Project.pdf) | 老师任务清单、九问对应与原始要求 |
| [scripts/collect_results.py](scripts/collect_results.py) | 从本地完整备份提取报告输入，逐文件核对，不运行实验 |
| [scripts/collect_convergence.py](scripts/collect_convergence.py) | 从本地回传副本汇入已完成的新诊断，保留原始结果 |
| [cloud/](cloud/README.md) | 可复用的上传、回传和校验脚本及配置模板 |
| [requirements.txt](requirements.txt)、[Makefile](Makefile) | 实际Python依赖及当前版本统一入口 |
| [MANIFEST.sha256](MANIFEST.sha256) | 仓库交付文件的SHA256；不包含清单自身 |
| `archive/`（本地） | 历史初稿、探索预检、重复参照及整理记录，完整保留，不上传GitHub |
| `cloud/received/`、`cloud/snapshots/`（本地） | 完整云端回传与历史快照，不重复上传GitHub |

报告正文与PDF直接位于项目根目录；代码、结果和图按用途分别维护于src、results和figures，不再按版本分目录。版本历史交给Git管理；原始旧备份仍在本地archive中保留。

## 本地复现

以下命令在本仓库根目录执行。本地Python使用已配置的Conda `yellow` 环境；远程实验当时使用用户指定的 `base`。依赖为NumPy、SciPy、Matplotlib；编译报告另需XeLaTeX、ctex/Fandol，提取PDF文本需Poppler。

```sh
conda run -n yellow python -m pip install -r requirements.txt
make verify       # 核对交付文件及正式数据
make check        # 小规模梯度、解析期望和MCSE核验
make summarize    # 从已保存结果重建表图；不重新拟合
make convergence  # 核验21条补充记录并重建收敛表图；不重新拟合
make report       # 双次XeLaTeX编译当前报告
```

从GitHub完整下载后即可重建汇总和报告，不依赖原服务器、本地历史归档或完整云端备份。`src/` 为日常编辑入口；`data/runs/*/work/` 为不可变的实际执行副本，供源码哈希核验，勿混用。

已用仅含待上传文件的独立副本完成 `make verify`、`make summarize`、`make convergence` 与强制双次报告编译；17份重新生成的指标、表格、审计和报告文本与当前交付一致。新增轨迹的来源哈希、接受逻辑、变化量和高阶积分核对通过；原155次拟合的JSON指标与审计保持一致。记录见 [目录核验](qa/repository_check.json) 和 [收敛审计](qa/convergence_audit.json)。

完整重算设置见 [方法与复现](docs/methods.md) 及 [云端正式入口](cloud/remote_revision_main.sh)。重算应使用新的输出目录（如被忽略的 `local_runs/`），不覆盖已完成结果。此次补充新增逐轮诊断和方法说明，没有修改原统计内核、种子或155次原始拟合；延长后的估计不替换原比较表。

## 保存与上传范围

原服务器的所有任务已完成，1430份原始运行文件曾逐一与本地SHA256核对。完整备份继续留在原电脑；仓库另保存280份报告实际需要的原始文件，复制时已逐一核对，其中包含155次正式拟合及报告使用的全部保留采样样本。

新服务器的补充任务也已完成，398份运行文件与本地SHA256逐一一致，含代码工作副本、环境、日志、检查点和最终输出；见 [回传核验](qa/cloud_convergence_verification.json)。定期同步进程已停止，可以释放服务器。两台服务器的完整备份和校验清单分开保留，旧备份未覆盖。补充前的报告副本另保存在本地 `archive/convergence_20261111/`。

GitHub仓库当前公开。服务器连接配置、SSH私有目录、历史压缩包、重复工作树、编译缓存和逐页PNG预览由 `.gitignore` 排除，仍在本地保留。重新租用服务器时按 `cloud/config.env.example` 建立本机配置；正常阅读、检查和生成报告不需要服务器。

维护约定：新增交付文件、实验批次或目录变更时同步更新相应README及校验清单。当前整理日期为2026-11-11。
