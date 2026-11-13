import json
import sys
from pathlib import Path

first = json.loads(Path(sys.argv[1]).read_text())
replay = json.loads(Path(sys.argv[2]).read_text())
required = ['plan_id', 'demand_reconciliation', 'forecasts', 'backtests', 'support', 'assumptions']
missing = [field for field in required if field not in first]
if missing:
    raise SystemExit("缺少分析字段：" + ", ".join(missing))
if not first['plan_id'] or first['plan_id'] != replay.get('plan_id'):
    raise SystemExit("同一输入重放未引用同一分析记录")
balance = first['demand_reconciliation']['balance']
if not balance['accounted_equals_requested'] or balance['unaccounted_hours'] != 0:
    raise SystemExit("请求工时未与需求对账平衡")
if first['demand_reconciliation']['double_count_prevented']['duplicate_hours_not_counted'] <= 0:
    raise SystemExit("未识别并剔除重派重复工时")
if not any(item['status'] == 'unsupported' for item in first['support']):
    raise SystemExit("无历史地区/资源未被标为数据不足")
if first['scope']['target_metric'] != 'requested_hours':
    raise SystemExit("预测目标必须是到达需求而不是完工量")
print("分析输出结构、重放标识、需求对账与支撑判定核验完成")
