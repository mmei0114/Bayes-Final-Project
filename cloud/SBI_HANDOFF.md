# SBI关机接续记录（2026-11-18）

本批已完成并回传。613个服务器运行文件逐一通过SHA256校验，可以释放服务器。回传与预检结果见 [服务器状态](STATUS.md) 和 [预检记录](../docs/sbi_pilot.md)。以下保留关机时的接续记录。

关机前核实主脚本PID709的父进程为1、无终端，标准输入为`/dev/null`，输出写入独立`run.log`。服务器随后继续后台运行。

远程根目录：`/root/bayes-project`。数据批次`runs/sbi_data_20261118`已完成并在本地完整保存。当前任务`runs/sbi_pilot_20261118`使用冻结的`work/`副本；检查时训练到550步，500步完整检查点已生成。软件、GPU、模拟器、变换及高斯解析后验检查均通过；GLMM训练和独立校准尚未结束。

主脚本自动执行：GPU批量测试→解析高斯检查→二元训练（最多10000步或90分钟，预设验证早停）→独立校准与分类积分核验（最多25分钟）→写入最终STATE/EXIT_CODE。任务完成后自动结束计算，不自动释放服务器；达到训练预算不代表校准合格。

关机后，本地每分钟回传进程无法继续运行，新文件会先保存在服务器。**不要在恢复回传并完成最终SHA256核验前释放机器**，否则关机期间产生且尚未回传的文件可能丢失。服务器磁盘不等于本地备份。

当时记录的接续步骤如下，现已执行。

1. 使用本机`cloud/config.env`与`manage.sh`检查远程STATE、PHASE、run.log和EXIT_CODE；不重启已有任务。
2. `bash cloud/manage.sh pull`回传新文件。全部写入停止后执行`bash cloud/manage.sh verify sbi_pilot_20261118`，保留旧服务器校验清单。
3. 用本地yellow运行`scripts/collect_sbi_pilot.py`和`src/summarize_sbi.py`，整理独立诊断、选中权重、环境和来源记录。
4. 更新预检说明、README、根MANIFEST，核验图件及原报告未改；完成后同步公开GitHub。是否进入正式三族训练，按预检结果与用户预算另定。

本地源码、协议与处理脚本保存在`src/`、`scripts/`、`docs/`和`cloud/`。完整回传位于`cloud/received/runs/sbi_*`。
