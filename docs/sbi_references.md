# SBI论文与实现核对记录

核对日期：2026-11-18。本页是调研记录，不是实验结果；主计划见 [sbi_plan.md](sbi_plan.md)。

## 主要来源

| 来源 | 已核对内容与项目用途 |
|---|---|
| Kucharský & Bürkner, *Amortized Bayesian Mixture Models*, [arXiv:2501.10229v3](https://arxiv.org/abs/2501.10229v3)，2025初稿、2026-03-16修订 | 核心依据。§2.3–2.4式(11)、(14)–(17)、(20)与图2–3：连续参数NPE、条件标签分类、模拟训练及后验抽样后分类；§3.1–3.3：校准、Stan对比、约束与训练规模。本项目采用独立个体分解，不引入其HMM filtering/smoothing |
| Radev et al., *BayesFlow: Learning complex stochastic models with invertible neural networks*, [arXiv:2003.06281](https://arxiv.org/abs/2003.06281)，IEEE TNNLS 33(4), 1452–1466 (2022) | 核对§2的条件可逆网络、换元密度、先验预测训练损失及学习摘要；理论极限不等于有限训练已正确 |
| Habermann et al., *Amortized Bayesian Multilevel Models*, [arXiv:2408.13230v3](https://arxiv.org/abs/2408.13230v3)，[Bayesian Analysis DOI](https://doi.org/10.1214/25-BA1570) (2025) | 核对§2.3的全局/局部后验分解、两级摘要和条件采样。支持保留纵向个体层级；本轮不复现完整随机效应神经后验 |
| Kühmichel et al., *BayesFlow 2.0: Multi-Backend Amortized Bayesian Inference in Python*, [arXiv:2602.07098v1](https://arxiv.org/abs/2602.07098v1) (2026) | 软件工作流及先验模拟、adapter、神经后验和校准模块的背景。论文2.0与当前2.1接口需区分，不能混用旧示例 |
| Talts et al., *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*, [arXiv:1804.06788](https://arxiv.org/abs/1804.06788) | 核对§4 Algorithm 1：独立先验预测数据、后验抽样、真值秩统计与参考波动；SBC只检验所选模型/量下的计算，不是模型合理性或全部后验正确的证明 |
| Modrák et al., *Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity*, [论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12490788/)，[arXiv](https://arxiv.org/abs/2211.02383) | 补充核对SBC对检验量选择的敏感性；需要结合后验收缩、预测和同目标参照，不能只看边缘秩图 |
| Radev et al., *BayesFlow: Amortized Bayesian Workflows With Neural Networks*, [JOSS 8(89), 5702](https://joss.theoj.org/papers/10.21105/joss.05702) (2023) | 软件书目信息核对；这是JOSS而非JSS。本文不据此声称完成了软件的全部示例复现 |

## 公开代码与接口

- [混合模型作者仓库](https://github.com/Kucharssim/bayesflow-amortized-mixtures)：已读取README、requirements、`src/networks.py`、`src/Constraints.py`。作者以`AmortizedPosteriorMixture`连接连续后验与条件分类，分类输入包含后验θ样本；交叉熵由logits计算。约束实现包括正值、ordered及K−1维stick-breaking simplex变换。仅阅读，未执行。
- [作者requirements](https://github.com/Kucharssim/bayesflow-amortized-mixtures/blob/main/requirements.txt)：BayesFlow 1.1.6、TensorFlow 2.15.1、Keras 2.15.0；与BayesFlow 2.x不是即插即用关系。论文另给出[OSF代码入口](https://osf.io/7wvyk/)，本次没有逐文件核对OSF归档，不声称与GitHub快照完全一致。
- [BayesFlow官方仓库](https://github.com/bayesflow-org/bayesflow)与[PyPI元数据](https://pypi.org/pypi/bayesflow/json)：本次读取正式版2.1.0（2026-10-23发布），Python要求`>=3.12,<3.14`，基于Keras 3且需另配计算后端；元数据原文已在本地归档，不将本机旧Python 3.11直接当作兼容。
- [CouplingFlow API 2.1.0](https://bayesflow.org/v2.1.0/api/bayesflow.networks.inference.coupling.CouplingFlow.html)：核对normalizing-flow与spline选项。另读取了[2.0.13 inference guide](https://bayesflow.org/v2.0.13/user_guide/inference_networks.html)、[summary guide](https://bayesflow.org/v2.0.13/user_guide/summary_networks.html)、[adapter guide](https://bayesflow.org/v2.0.13/user_guide/data_processing.html)和[诊断说明](https://bayesflow.org/v2.0.13/user_guide/diagnostics.html)。这些只作架构/流程参考；对应2.1用户指南入口本次未成功取得，正式实现前须以2.1源码/API再核对，尚未验证可执行调用。
- [Stan离散变量边缘化](https://mc-stan.org/docs/stan-users-guide/latent-discrete.html)与[标准约束变换](https://mc-stan.org/docs/reference-manual/transforms.html)：核对对U有限求和和可逆变换/Jacobian。新后验参照仍须对本题连续随机效应完整建模，不能拿旧固定θ的条件链代替。
- [NVIDIA RTX 4090规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)：24GB显存。计划推荐该级别单卡是工程估算，不是论文对GLMM的显存实测。

## 本题适配与未决项

论文提供推断分解和标准训练目标，没有直接验证我们的三类离散GLMM。参数化模拟器、先验尺度、个体内编码、训练预算与通过标准属于本项目需明示并检验的设置。尤其不能照搬论文高斯混合实验的先验、网络规模、训练量或加速倍数。

已下载5篇核心论文PDF、提取全文，并保存上面两份作者源码及PyPI元数据到本地`archive/sbi_literature_20261118/`；路径/URL/校验值见该目录`SOURCE_MANIFEST.json`。公开仓库保存本计划与来源链接，不把第三方源码当成本项目实现，也不把未运行的SBI写进现有报告的结果。
