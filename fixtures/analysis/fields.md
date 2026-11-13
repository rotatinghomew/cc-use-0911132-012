# 业务演练数据口径

requested_hours 为请求方申报工时，不是经事后审定的真实工作量；served_hours 是受容量约束的实现服务。tasks.replaces 指同请求重派，不能双计请求量。calendar 已知未来属性可用，未来任务结案不可用。新地区和 rare-boat 无历史，不能借不存在的数据验证。

各 ID 均为本地合成标识，不包含真实人员或生产数据。时刻使用带时区 ISO 8601；as_of 是分析可知信息的截止时刻，不是业务发生时间上界。晚到事实允许保留在输入中，但只能在 available_at 到达之后采用。source_versions 标识输入版本；baseline_case_id 关联同库既有演示运行。空值表示未知，不应未经说明转成零。
