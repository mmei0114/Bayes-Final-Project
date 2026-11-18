# 三类响应 SBI 计算产物

方法、冻结数据分区和运行范围见 [三类响应的 SBI 计算](../../docs/sbi_extension.md)。

| 位置 | 内容 |
|---|---|
| `models/poisson/`、`models/nb/` | 各计数模型选中的全局后验网络和分类器，连同标准化及CPU重载检查 |
| [二元权重](../sbi_pilot/README.md) | 预检归档中的二元完整检查点 |
| `comparison/` | 原15份数据的后验样本、分类概率及输入文件SHA256 |
| `datasets/` | 三族的生成配置、分区和数据分块哈希 |
| `cloud/received/runs/sbi_extension_*` | 原始运行记录的本地恢复位置，含训练数据、每500步的检查点和全部校准抽样样本 |

完整原始档案已核对并保存在矩池私有网盘 `bayes-sbi-20261118/`。大文件名为 `sbi_extension_data_20261118.tar`、`sbi_extension_train_20261118.tar`、`sbi_extension_predict_20261118.tar`。下载至项目的 `cloud/received/matpool_sbi_20261118/` 后，可以在项目根目录校验并恢复。模型权重和本页列出的比较产物随仓库保存。

```sh
conda run -n yellow python scripts/restore_sbi_extension.py cloud/received/matpool_sbi_20261118
conda run -n yellow python src/summarize_sbi_extension.py
conda run -n yellow python scripts/plot_sbi_extension.py
```

恢复逐包校验SHA256，汇总代码只依赖基础数值库。CPU重载需使用与训练对应的BayesFlow和TensorFlow版本。现有指标的重建只需基础数值库。
