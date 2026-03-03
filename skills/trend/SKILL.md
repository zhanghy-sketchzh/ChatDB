---
id: trend
label: 趋势分析
description: 按时间聚合，分析指标随时间的变化走势（逐年/逐月/逐日）
keywords: [趋势, 走势, 逐月, 逐年, 逐季, 变化趋势, 历年, 近几年, 月度, 年度, 季度, 逐日, 每月, 每年]
priority_boost: 0
sql_hints:
  summary: 按时间 GROUP BY + ASC → 适合"看随时间变化走势"
  order_by_column: 时间列
  order_by_direction: ASC
meta:
  requires_time: true
  default_time_granularity: year
  intent_hint_template: "按{granularity}聚合{metric}，观察时间变化趋势，需要 GROUP BY 时间列并按时间排序"
  stats_fields: [available_years, year_count, growth_rate]
  issue_fields: [only_single_year, missing_time_range]
---

# When to Activate

观察指标随时间的变化走势，包括但不限于：
- 按时间粒度（年/季/月/周/日）聚合指标，查看其随时间的变化规律
- 分析历史数据的增减趋势，识别上升/下降/平稳阶段
- 将同一指标在多个时间点上排列，形成时间序列

**典型表述**：
- "看看近几年销售额的趋势"
- "2020到2025年的收入变化走势"
- "各月的订单量趋势"
- "逐季度分析利润变化"

**关键特征**：按时间维度展开数据，形成时间序列，重点在于**观察走势**而非计算具体差值。

# Not For

- 需要**计算两期具体差值或增长率**的分析 → 用 `comparison`（如"同比增长多少""哪年下降最多"）
- 按非时间维度拆解构成 → 用 `source`（如"按部门拆解销售额"）
- 找极值/排名 → 用 `ranking`（如"销售额最高的年份"，除非是要看完整趋势线）

# Disambiguation

- 用户只说"看看近几年销售额" → trend（看趋势走势，不需要计算变化量）
- 用户说"近几年销售额变化，哪年下降最多" → 第一步 trend，第二步 comparison
- 单纯看走势不需要计算差值 → trend；需要计算涨跌幅/增长率 → comparison
- "各月销售额趋势" → trend；"1月比2月增长了多少" → comparison
- "历年收入变化" → trend（展示时间序列）；"今年比去年增长率" → comparison（计算差值）

# Planner Prompt

## 趋势分析 (trend)

**适用场景**：观察指标随时间的变化走势（逐年/逐月/逐日），形成时间序列

**不适用**：
- 对比两期差异、计算增长率 → `comparison`
- 按非时间维度拆解 → `source`
- 找极值 → `ranking`

**SQL 模式**：`GROUP BY 时间列 + ORDER BY 时间 ASC`

**核心概念**：
趋势分析的本质是将指标在时间轴上展开，让用户观察其变化规律。关键决策点是**时间粒度的选择**：
- 跨年范围（3年以上）→ 通常用年粒度
- 单年内 → 通常用月粒度
- 单月内 → 通常用日粒度
- 用户明确指定粒度时，以用户为准

**注意事项**：
1. 趋势任务**只负责展示时间序列**，不要在同一任务中计算差值、增长率
2. 如果用户同时想看趋势和计算变化量，应拆为两个任务：先 trend 再 comparison
3. 结果必须按时间升序排列，确保可视化时时间轴正确
4. 时间列的 GROUP BY 粒度必须与用户意图匹配

**任务示例**：
```json
{
  "id": "trend_1",
  "type": "trend",
  "description": "分析2020-2025年销售额的年度变化趋势",
  "depends_on": [],
  "notes": ["时间粒度: 年", "排序: 按时间升序"],
  "virtual_fields": []
}
```

# SQL Instruction

### 趋势分析 SQL 生成规则

**基本结构**：
```
SELECT 时间列, 聚合函数(指标列) AS 指标别名
FROM 表
WHERE 时间/业务筛选条件
GROUP BY 时间列
ORDER BY 时间列 ASC
```

**详细规则**：
1. **时间粒度**：根据任务描述选择 GROUP BY 的时间粒度
   - 年粒度：直接 GROUP BY 年份列
   - 月粒度：GROUP BY 年份, 月份（或年月组合列）
   - 日粒度：GROUP BY 日期列
2. **排序**：通常按时间升序排列（`ORDER BY 时间列 ASC`），具体排序方向以任务描述为准
3. **单 SQL 原则**：只生成 1 个 SQL，不要用 CTE 串联多个查询
4. **不要计算差值**：趋势 SQL 只返回各时间点的聚合值，不要在 SQL 中加入 LAG/LEAD 计算同比环比
5. **时间列格式**：确保返回的时间列可读

**常见错误**：
- ❌ 在 trend SQL 中使用 LAG() 计算增长率（这是 comparison 的职责）
- ❌ ORDER BY 指标列 DESC（趋势应该按时间排序，不是按值排序）
- ❌ 使用 LIMIT 截断时间序列（趋势需要完整的时间线）
