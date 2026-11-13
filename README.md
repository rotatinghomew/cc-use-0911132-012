# 供给受限记录中的资源请求量预测

项目使用 Python 3.11 标准库与 SQLite，现有逐案处理入口为 `emergency_resource_dispatch.control.command`。保留原有处理器、持久化与测试；分析业务使用本地合成资料，不连接生产系统。无需 pip 安装、下载模型或启动额外服务。

## 现有程序

在仓库根目录运行 `bash scripts/demo.sh`，依次完成现有产品回归、分析输入核对、语法构建和真实业务入口调用。所有运行文件位于临时目录，结束后清理；运行失败立即退出，不使用预存输出代替执行。

单独测试：`python3 -m unittest discover -s tests/resource_schedules -p 'test_*.py' -v`。

输入检查：`python3 -m unittest discover -s tests -p test_analysis_inputs.py -v`。

构建检查：`python3 -m compileall -q emergency_resource_dispatch tests`。

业务调用：`python3 -m emergency_resource_dispatch.control.command fixtures/demo.json --database /tmp/emergency_resource_dispatch-demo.sqlite3`。

## 分析资料与联调

`fixtures/analysis/input.json` 保存业务演练数据，字段解释在 `fixtures/analysis/fields.md`。以下已核对的数据形态用于分析回归：censored_service、reassigned_task、unfinished_tasks。现有输入测试检查这些形态是否真实存在，不代替分析方法或数值正确性的验证。

分析入口接通后执行 `bash scripts/feature_demo.sh`：先重跑现有产品与分析行为测试，再从干净数据库计算分析、同输入重放、核对输出结构并打印完整 JSON。分析 ID 为 `plan_id`，其余输出包括 `demand_reconciliation, forecasts, backtests, support, assumptions`。方法正确性由 `tests/test_demand_planning.py` 覆盖业务说明中的反例，结构核验不能代替统计验证。

单独调用分析入口：

```
python3 -m emergency_resource_dispatch.demand_planning \
    --input fixtures/analysis/input.json \
    --database /tmp/emergency_resource_dispatch-analysis.sqlite3 \
    --output /tmp/demand_plan.json
```

分析行为单独测试：`python3 -m unittest discover -s tests -p test_demand_planning.py -v`。

### 分析口径

- **预测目标**：到达需求（`requested_hours`，按请求到达日计量），不是完工量。`served_hours` 受容量约束且部分删失，只作为证据，不作为预测对象。
- **一个请求一次需求**：`tasks.replaces` 是同请求的重新派工，终态行取未被替换的最新一行，重复行不计入请求量。
- **放弃不是零需求**：终态为 `abandoned` 的请求保留其申报工时；未结案请求的申报工时计入到达需求，但其 `served_hours` 不进入训练，避免泄漏未结案结果。
- **无法观察的放弃量**：只作显式假设与上下界（`withdrawal_bounds`），不声称能从完工日志恢复真实总体。
- **时间前推**：训练窗口严格早于 `final_training_cutoff`；每个滚动起点只用 `available_at` 已到达的事实；切分点之后的首个窗口作为独立 holdout，仅用于验证、不参与选型。
- **模型与区间**：基线为 14 日移动均值；主模型为按父级（同资源兄弟地区或上一层级）收缩的 28 日比率，用于稀薄或长期零需求单元；周期候选仅在尾部窗口能可靠识别周期时可用。区间为 gamma 矩匹配的近似负二项预测区间，覆盖率由回测核验。
- **支撑判定**：逐单元按回测 WAPE 与区间覆盖率给出 `supported / limited / unsupported`。无历史的 `new-region`、`rare-boat` 明确标为 `unsupported` 且不输出点预测，不借用不存在的数据，也不硬凑排班表。
- **产能诊断**：`capacity_diagnostics` 对连续容量档位比较到达需求、申报服务与观测服务，并给出到达需求排名与完工量排名的位次差，说明完工量下降不等于需求减少；同时记录账本在 `as_of` 前的尾部静默属于覆盖缺口而非零需求。

`fixtures/analysis/capacity_drop_illustration.json` 是补充反例：容量从 24 降到 4 工时/日，而到达需求保持不变。分析在该样例上给出“完工量下降来自容量而非需求”的结论，预测仍停在需求水平，并由 `tests/test_demand_planning.py::CapacityDropIllustrationTests` 回归。

分析只读运行账本（`runs/events/idempotency/failures`），结果写入独立的 `demand_plans` 表，不执行任何调度。

## 离线环境

使用 Python 3.11.2 的 Linux 环境即可复现。容器环境预置解释器后，以只读方式挂载仓库，将文件复制到容器临时目录再运行脚本；运行时关闭网络。基础测试和小规模本地数据适合在 90 秒内展示，正式展示命令为 `bash scripts/feature_demo.sh`。不要在展示脚本里安装依赖或等待外部服务。
