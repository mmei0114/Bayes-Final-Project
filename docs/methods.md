# 方法与复现

**最新交付：[report.pdf](../report.pdf)（11页）与 [report.tex](../report.tex)。** 旧版整体归档于原电脑的 `../archive/v1/`，未删除，也不进入新版汇总。

## 方法与结果

1. MCEM最大化单一加权Q，不平均两次M步参数；独立验证样本评估Q增量，分层自归一化IS的delta方差包含责任度归一化误差。参考Booth–Hobert及Caffo等的误差控制/上升思想，具体适配与有限样本局限在正文说明。
2. GVA保留标签与随机效应的条件依赖，局部二维严格凹优化，外层用剖面ELBO和包络梯度；Poisson采用解析高斯期望。它是参数点估计，不是全贝叶斯参数VB。
3. 随机截距一维，以61点自适应GH的积分ML作局部优化参照，101点复核；固定同一QML参数，用联合独立MH和积分核对潜变量近似。

### 积分与抽样技术

有限类别直接求和；随机截距积分为一维。MCEM在每个个体/簇下以条件众数和逆曲率建立t₅提议，独立抽样，以重要性比率估计积分、归一化权重估计条件期望；候选更新的Q增量及MCSE另用独立验证样本。这里没有使用参数Gibbs，也不以MCMC代替E步。

观测log似然及QML使用条件众数/曲率中心化的61点自适应Gauss–Hermite求积，101点复核。GVA中Poisson的高斯指数矩解析计算，二元和负二项的高斯期望用31点标准GH。固定参数的潜变量核验另外使用联合独立MH；正文给出提议、接受率及实际链长。这些技术对应不同计算目标，不混称为同一种积分方法。

### 参数与观测似然轨迹

`src/convergence.py`仅重跑原选中的一个初始化：三类响应各5条MCEM，另各取第一次重复的QML/GVA，共21条。每轮记录原尺度beta、D、pi和61点观测log似然；初始点、原轮数处与终点用101点核对。MCEM原Q增量、MCSE、抽样量和接受标记也完整保存。GVA仍优化ELBO，其观测似然轨迹仅作额外诊断。

协议在运行前固定：连续3轮 `abs(ell_t-ell_prev)/(1+abs(ell_prev)) <= 1e-6`，且MCEM末窗至少有一次接受更新；参数最大缩放变化不超过1e-3仅作辅助。MCEM至少运行原轮数，必要时最多多30轮；全拒绝窗和预算耗尽不算通过。阈值不是教师指定的唯一标准，检查不证明全局最优。新轨迹不替换155次原拟合参数或耗时；原轮数处与保存结果的数值一致性另行报告。

正式设计冻结为：GVA/QML在14场景各5次重复，共140次拟合；MCEM在基准二元、Poisson、负二项各5次，共15次拟合；每拟合3起点。合计70份不同数据、155次拟合，不声称所有方法覆盖所有场景。

155次所选解均满足各自数值停止条件；28次触及参数边界，4次所选MCEM起点历史出现内部求解器非成功状态。MCEM最终观测投影得分/个体约0.00122–0.0576，小Q增量停止不等于高精度似然驻点。GVA基准平均ARI为0.885，分离减半和作业建议参数下为0.411、0.363。GVA实测耗时小于MCEM，但三个响应场景的一维QML又均快于GVA；5次重复不支持普遍排序。

原Poisson分开条件更新的链接受率约0.877，却有最大split诊断约36，未混合。正式对照采用标准标签—随机效应联合独立MH，目标不变、提议和接受率见正文；原链保留为诊断。三个响应场景各取第一次重复的前30人，4链，各2000次预热、8000次保留；最大经典split R-hat为1.0021、1.0038、1.0016。保存全部样本，不称为参数后验或精确有限样本真值。

## 文件

下表路径相对项目根目录，项目按用途维护，版本历史由Git记录。

| 路径 | 用途 |
|---|---|
| `src/inference.py` | 三类响应一致的目标与导数、GVA、积分ML、独立验证MCEM |
| `src/legacy_model.py` | 旧代码只读复用副本；当前仅复用生成机制、初始化与评价定义 |
| `src/validate.py` | 梯度、Poisson解析期望、极端预测器及IS标准误核验 |
| `src/experiment.py` | 场景、种子、多起点、原子结果与检查点、源码/配置校验 |
| `src/conditional_mcmc.py` | 固定参数的分开条件更新与联合独立MH；正式对照使用 `--blocked` |
| `src/summarize.py` | 检查正式输入与源码，生成表图和一致性审计 |
| `src/convergence.py` | 固定协议的参数/观测似然补充轨迹，保留逐轮记录与来源哈希 |
| `src/summarize_convergence.py` | 核验21条补充轨迹、积分阶数及接受逻辑，生成新表图和审计 |
| [theory.md](../theory.md)、[sources.md](../sources.md) | 推导、补充来源和适用边界 |
| `results/` | 155次逐拟合指标、31组汇总以及报告LaTeX表格 |
| `figures/` | 条件轨迹、目标/增量/增样诊断、MCMC轨迹及新参数/观测似然图PDF |
| `qa/` | 数值验证、正式审计 `audit.json`、积分阶数核对、服务器资源与PDF文本 |
| `report.tex`、`report.pdf` | 当前LaTeX正文及最终PDF |
| `Makefile` | 数值核验、汇总及双次XeLaTeX编译 |
| `build/`、`qa/final/`（本地） | 编译中间文件和逐页PNG，不上传GitHub |

本地早期探索预检已移入 `../archive/pilot_local/`，未计入正式结果；旧报告参照副本在 `../archive/v1/reference_copies/`。旧交付清单按旧布局保存在本地整理记录中，当前仓库使用根目录 `MANIFEST.sha256`。

## 无需服务器的重建

在仓库根目录运行：

```sh
make verify
make check
make summarize
make convergence
make report
```

汇总默认使用随仓库保存的 `../data/runs/revision_v2_main_20261110/` 及 `revision_v2_diagnostic_20261110/`。它核对140+15份完成结果、实际执行源码SHA256、种子、三起点、15组同数据比较、三份MCMC目标及样本。

`results/replicates.csv` 为155次拟合指标，`summary.csv` 为31组均值/标准差。详细参数、全部起点及数据在 `../data/runs/revision_v2_main_20261110/{robustness,comparison}/`；联合MH的保留样本在独立诊断目录的 `mcmc/*.chains.npz`。主运行 `probe_mcmc/poisson.*` 为报告图3的旧链。输入文件说明和校验见 [data/README.md](../data/README.md)。

## 实验来源与完整重算

实际拟合源码在 `../data/runs/revision_v2_main_20261110/work/revision_v2/src/`，联合采样执行源码在独立诊断运行的对应 `work/` 中；两者保持原样。冻结运行器的 `failed_in_this_run` 曾把数值错分率 `error` 误计为失败，不影响拟合；当前版本已修正这一计数。正式完成判据是155份结果、无异常文件和源码/数据审计。

完整重算参数在 `../cloud/remote_revision_main.sh`、`remote_revision_diagnostic.sh`，服务器当时使用base，本地使用yellow。重算用新目录，例如：

```sh
conda run --no-capture-output -n yellow python src/experiment.py --output local_runs/robustness --reps 5 --starts 3 --jobs 8 --maxiter 300 --methods QML GVA
conda run --no-capture-output -n yellow python src/experiment.py --output local_runs/comparison --reps 5 --starts 3 --jobs 8 --maxiter 250 --methods MCEM --scenarios baseline poisson nb
```

上述命令在仓库根目录执行，会重新拟合；它们不是重建现有报告所必需的步骤。补充轨迹的重算使用 `src/convergence.py --output local_runs/convergence --jobs 8`，远程入口为 `../cloud/remote_convergence.sh`。补充运行本地yellow完成11条，新服务器base完成10条，执行环境与对应文件哈希见 `../data/convergence/`。完整云端工作副本、检查点、预检与快照仍在原电脑的 `../cloud/received/`、`snapshots/`，未重复上传。原统计内核和155次原始结果未修改。
