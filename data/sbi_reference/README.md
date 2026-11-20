# 同先验 NUTS 与分类积分核验

使用原三类响应各自第一次重复的数据。Stan 模型沿用 BayesFlow 的先验与有序截距约定。对簇类别直接求和，抽样全局参数和每人的标准化随机效应。每族 4 条链各预热和保留 1000 次，目标接受率 0.95，最大树深度 12。两链从扰动的 QML 参数起步，另两链从先验抽初值。

`global_draws.npz` 保存 12 个全局分量的四链样本，三族各为 `[4, 1000, 12]`。`sbi_classifier_same_draws.npz` 保存固定原有 1024 个 BayesFlow 参数样本时的两套分类概率。每族前缀为 `binary`、`poisson` 或 `nb`，后缀 `_gh` 与 `_neural` 分别对应自适应 GH 和条件神经分类器。诊断见 [NUTS 审计](../../results/sbi_reference_audit.json)、[随机截距方差轨迹](../../figures/sbi_reference_trace.pdf) 和 [条件分类核验](../../results/sbi_classifier_same_draws.json)。Stan 与独立 Python 计算在相同参数点对比的是 log 后验核差值，最大误差约 $5.46\times10^{-12}$。见 [目标函数检查](../../qa/sbi_reference_target_check.json)。

三族参数链均未通过全局分量的混合筛查。原始 CSV 含全部连续变量和采样诊断，连同初始化、输入数据、运行日志和 `SHA256SUMS` 保存在本地 `local_runs/sbi_reference_20261120/`。三族各一次的四链采样用于识别链间差异，合并结果仅作诊断。报告中两套条件分类的比较固定了相同的参数样本，因此不会混入更换参数样本的影响。

CPU 使用本地 Apple arm64、CmdStan 2.38.0，Stan 源码为 `src/sbi_reference.stan`。从 [官方发布](https://github.com/stan-dev/cmdstan/releases/tag/v2.38.0) 将 CmdStan 解压至 `build/stan/cmdstan-2.38.0/` 后，在项目根目录运行：

```sh
cp src/sbi_reference.stan build/stan/sbi_reference.stan
make -C build/stan/cmdstan-2.38.0 -j4 arch=arm64 build ../sbi_reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run -n yellow python scripts/run_sbi_reference.py \
  --model build/stan/sbi_reference --output local_runs/sbi_reference_repeat
conda run -n yellow python scripts/summarize_sbi_reference.py \
  local_runs/sbi_reference_repeat build/stan/cmdstan-2.38.0/bin/stansummary
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run -n yellow python scripts/check_sbi_classifier.py
```

`arch=arm64` 用于 Apple Silicon 的 TBB 构建。复现汇总会写入项目结果路径，比较时可使用独立工作副本。
