# 云端计算与本地保存

本地 `project/` 是主副本。服务器仅作为临时计算节点；旧初稿的 `src/`、`results/`、报告不被云端回传覆盖。

2026-11-11整理后，旧初稿整体位于本地 `../archive/v1/`，当前报告为 `../report.pdf`，代码位于 `../src/`。服务器任务已完成，无需连接服务器即可使用随仓库保存的 `../data/` 重建正式汇总。完整回传、历史快照及私有连接配置留在原电脑，不上传GitHub。

## 目录与文件

| 路径 | 内容 |
|---|---|
| `config.env.example` | 不含真实连接信息的配置模板，随仓库保存 |
| `config.env`（本地） | SSH地址、端口、远程路径、Conda环境；无密码，不上传GitHub |
| `manage.sh` | 上传快照、回传运行目录、逐文件校验、前台定期同步 |
| `remote_check.sh` | 服务器 base 环境元数据保存与小规模数值检查；不启动模拟实验 |
| `remote_revision.sh`、`remote_revision_main.sh` | 修订版预检、正式拟合和旧条件采样诊断的冻结入口 |
| `remote_revision_diagnostic.sh` | 改进条件混合的标准联合独立MH核验入口；使用独立快照 |
| `remote_convergence.sh` | 补齐参数/观测似然逐轮轨迹，复用已完成的本地11项，仅计算剩余10项 |
| `probe_convergence_env.py` | 用指定base环境读取依赖和CPU/内存信息 |
| `progress.py` | 通过 `manage.sh status` 只读显示各批次完成数、异常文件与活动检查点 |
| `STATUS.md` | 本次接入、依赖安装、检查与回传的完成情况 |
| `snapshots/<编号>/project.tar.gz`（本地） | 上传前先在本地生成的项目快照，包含当时源码、现有数据和输出 |
| `snapshots/<编号>/SHA256SUMS` | 快照校验值；传输后在服务器核对 |
| `received/runs/<运行编号>/`（本地） | 从服务器回传的完整运行目录，不覆盖主项目结果；报告所需子集另复制到 `../data/` 上传 |
| `received/SHA256SUMS` | 停止写入后获取的远程运行文件清单与 SHA256 |
| `logs/verification_latest.log` | 最近一次本地逐文件校验日志 |
| `received/SHA256SUMS_convergence_20261111`、`logs/verification_convergence_20261111.log`（本地） | 新服务器398份文件的完整校验，与原服务器1430份旧清单分别保留 |
| `private/` | 主机指纹、临时 SSH 控制套接字与本机连接配置；禁止上传、打包或提交 |

服务器根目录为 `/root/bayes-project/`：`releases/<快照编号>/project/` 存放上传版本，`runs/<运行编号>/work/` 存放独立计算副本和该次输出。每次启动使用新运行编号，记录源码快照、命令、环境版本和日志，避免覆盖之前结果。

## 常用命令（在本地父目录执行）

```sh
bash project/cloud/manage.sh push
bash project/cloud/manage.sh pull
bash project/cloud/manage.sh watch 60
bash project/cloud/manage.sh verify
bash project/cloud/manage.sh status
bash project/cloud/manage.sh ssh
```

SSH 使用已有连接或终端密码提示，不把密码写进脚本。`watch 60` 需保持本地终端运行，每 60 秒回传已落盘文件；这不是永久后台服务，本地休眠或断网时不能继续同步。所有同步均不使用 `--delete`。

新服务器的收敛诊断独立保存为 `runs/convergence_20261111/`。结束后用 `manage.sh verify convergence_20261111` 生成带批次名的校验清单，保留原服务器1430份文件的旧清单。回传使用压缩传输；源码快照排除本地archive、检查点与临时文件。

该批次现已21/21完成、退出码0，398份运行文件全部回传并逐文件校验一致，定期回传已停止，可以释放服务器。报告输入已汇入 `../data/convergence/`；计算完成与通过收敛阈值分别报告，不将预算耗尽改记通过。

上述命令仅在重新租用服务器时使用。新克隆先复制 `config.env.example` 为 `config.env` 并填写新服务器信息；配置默认远程base，不在此保存密码。日常整理和复现不会连接原服务器。新的上传归档也排除Git内部目录及本机连接配置。

原子保存中的 `*.tmp` 不回传、不纳入最终文件校验；临时文件改名为正式结果/检查点后才同步，避免读到半写文件或原子改名导致的同步竞态。

## 每次计算与释放前

1. 本地修改方法并检查后，`push` 生成新源码快照；再在服务器新运行目录执行对应版本。大规模运行前应先加入逐次实验的原子写入和必要的中间检查点。
2. 计算期间保持本地同步终端；每个批次结束立即 `pull`。长任务还要记录进程状态和完成/失败标记，不能只依赖 SSH 窗口。
3. **释放前先等待任务结束或保存检查点并停止写入，再执行 `verify`。** 只有校验通过、预期文件数量与完成标记核对后，才释放服务器。
4. 本地快照和回传目录不随服务器释放而消失。若在未同步/未保存检查点时直接释放，尚在远程内存或未回传的数据仍可能丢失；不能以服务器目录代替本地备份。

## 环境约定

- 本地 Python 继续用 `yellow`；用户于 2026-11-10 明确指定服务器使用 `base`。
- 服务器 Conda 路径 `/root/miniconda3/bin/conda`。先检查依赖，再按项目固定版本安装；不为安装依赖自动更换 Python 环境。
- 依赖安装使用有限超时/重试，若下载慢或卡住，暂停并报告用户，不反复等待或自行切换来源。
- 首版方法保留存档；经修订和验证的正式实现位于项目根目录的 `src/`，不将旧结果冒充新方法的结果。
