# 三类响应 SBI 计算产物

方法、冻结数据分区和运行范围见 [三类响应的 SBI 计算](../../docs/sbi_extension.md)。

| 位置 | 内容 |
|---|---|
| `models/poisson/`、`models/nb/` | 各计数模型选中的全局后验网络和分类器，连同标准化及CPU重载检查 |
| [二元权重](../sbi_pilot/README.md) | 预检归档中的二元完整检查点 |
| `comparison/` | 原15份数据的后验样本、分类概率及输入文件SHA256 |
| `datasets/` | 三族的生成配置、分区和数据分块哈希 |
| `cloud/received/runs/sbi_extension_data_20261118/` | Poisson、负二项各 10 万份训练数据和独立验证、校准、分类核验输入 |
| `cloud/received/runs/sbi_extension_train_20261118/` | 两个计数模型各 20 个完整训练检查点、日志和训练环境 |
| `cloud/received/runs/sbi_extension_predict_20261118/` | 三族 1,500 份数据的校准抽样、原有 15 份数据的预测和执行源码 |
| `cloud/received/matpool_sbi_20261118/` | 网盘下载的原始 tar 包、哈希、精简模型包和汇总档案 |

2026-11-20，六个下载包和解包后的 445 个运行文件均通过 SHA256 校验。用恢复的文件在本地重建了校准表和两张比较表，数值与已保存结果一致到浮点精度，校准秩直方图完全一致。完整档案和数据保存在本地 `cloud/received/`。仓库保存精简模型、正式比较样本、生成协议、诊断汇总和 [本地恢复核验](../../qa/sbi_extension/local_restore.json)。

在项目根目录运行下列命令，可从已经恢复的本地文件重建供核对的结果。

```sh
conda run -n yellow python src/summarize_sbi_extension.py --destination build/sbi_extension_restored
```

完整文件可用 `cloud/received/SHA256SUMS_sbi_extension_20261118` 核对。如需从原始 tar 重新恢复，运行 `conda run -n yellow python scripts/restore_sbi_extension.py cloud/received/matpool_sbi_20261118`。汇总代码只依赖基础数值库。
