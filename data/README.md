# 正式实验数据

这里是从本地完整云端备份中复制并逐文件核对的报告输入，随Git仓库保存。共280份原始文件，约30.7 MiB；155次正式拟合使用70份不同模拟数据。目录保留原运行编号和报告程序需要的路径。

| 路径 | 内容 |
|---|---|
| `runs/revision_v2_main_20261110/robustness/` | 14场景×5重复×QML/GVA，140次拟合、70份数据及元数据 |
| `runs/revision_v2_main_20261110/comparison/` | 基准、Poisson、负二项各5次MCEM；15份数据与上表对应数据相同，供审计核对 |
| `runs/revision_v2_main_20261110/probe_mcmc/` | 报告图3使用的旧Poisson分开更新链及诊断 |
| `runs/revision_v2_diagnostic_20261110/mcmc/` | 三类响应的正式联合独立MH诊断与全部保留样本 |
| `runs/*/work/revision_v2/src/` | 当时实际执行的不可变源码副本，仅保留统计代码，供元数据SHA256核对 |
| `runs/*/work/revision_v2/qa/validation.json` | 对应运行的小规模数值验证结果 |
| `runs/*/{STATE,EXIT_CODE,source_snapshot.txt,source_archive.sha256}` | 完成标记、退出码、来源快照和归档校验值 |
| `environment.json` | 实际远程Python与数值库版本 |
| `inventory.json`、`SHA256SUMS` | 上述280份原始文件的路径、大小与SHA256 |

在本目录运行 `shasum -a 256 -c SHA256SUMS` 可核对原始文件。项目根目录的 `make summarize` 使用这些输入，不需要服务器或未上传的本地备份。

`work/` 是为复现保留的原始执行代码，不是项目的日常编辑入口；当前可维护代码在 `../src/`。冻结运行器曾将数值错分率 `error` 误计进 `failed_in_this_run`，不影响拟合；核验使用155份完成文件、无异常文件及源码哈希，当前运行器已修正此计数。

完整云端备份还包含预检、检查点、其他旧采样链和重复工作树，仍保存在原电脑的 `cloud/received/`，未重复放入仓库。提取过程见 `../scripts/collect_results.py`；重复执行只接受字节一致的已有文件，不覆盖不同内容。
