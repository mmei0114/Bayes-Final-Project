# 修订版补充来源与适用范围

原始三篇课程论文的阅读笔记在 `docs/reading_notes.md`，全文提取的历史副本保存在本地 `archive/v1/sources/`；原始PDF只读引用。以下补充来源于2026-11-10核对。

- Ormerod–Wand (2012), *Gaussian Variational Approximate Inference for Generalized Linear Mixed Models*, JCGS 21:2–17：[作者版全文](https://matt-p-wand.net/publicns/Ormerod12.pdf)，[DOI](https://doi.org/10.1198/jcgs.2011.09118)。重点是第3节的高斯族/Jensen目标、Poisson解析期望，以及第4节一致性结果的适用条件。本文把这一标准GVA目标用于混合随机截距模型；责任度、局部严格凹性和剖面梯度在正文直接推导，不将原文单GLMM定理移植成混合模型的证明。
- Caffo–Jank–Jones (2005), *Ascent-based Monte Carlo expectation–maximization*, JRSS-B 67:235–251：[出版社页面](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00499.x)，[DOI](https://doi.org/10.1111/j.1467-9868.2005.00499.x)。核对书目信息及误差控制、资源分配和高概率上升思想。在线出版社页面本次仅取得摘要；不声称已经从该入口读到完整正文。本文的独立验证、分层比率delta方差和具体停止阈值是明示的本题实现，并非原文完整算法或定理的复现。方差推导在 `theory.md` 和报告中给出，并用80次独立抽样验证。

条件分布核验采用标准独立Metropolis–Hastings公式，按本题联合目标和提议直接代入推得接受率，不将它包装成新方法。原分开条件更新的链混合不良，保留其数据与诊断；正式近似精度表使用相同参数下的联合更新。
