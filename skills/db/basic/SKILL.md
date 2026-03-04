---
id: basic
label: 基础查询
description: 简单聚合查询，获取单一数值（总计/合计/计数/平均），结果为单行，不涉及维度拆解
keywords: [查询, 总共, 多少, 合计, 是多少, 总计, 总额, 总量, 一共, 平均]
priority_boost: 0
sql_hints:
  summary: 简单聚合（SUM/COUNT/AVG）→ 适合"XX是多少/总共多少"，结果为单行
meta:
  intent_hint_template: "执行基础查询，获取核心指标"
  stats_fields: [total_value]
  issue_fields: []
---

# When to Activate

简单聚合查询，获取**单行**数值结果，包括但不限于：
- 求总和（SUM）、计数（COUNT）、平均值（AVG）、最大/最小值（MAX/MIN）
- 回答"XX 是多少""总共多少""有几个"类问题
- 无需按维度分组、排序、对比、趋势的简单指标查询

**典型表述**：
- "总销售额是多少"
- "一共有多少条记录"
- "平均订单金额是多少"
- "最大单笔交易金额"

**关键特征**：结果是**单行**（一个或少数几个聚合值），**不需要 GROUP BY**，**不需要按维度拆解**。

# Not For

以下场景**不应使用 basic**，请务必选择对应的专项类型：
- 出现"分别""各""按XX分""每个XX"等暗示分组的表述 → 用 `source`（按维度 GROUP BY）
- 需要同时查看多个分组的数值（如"A和B分别是多少"） → 用 `source`
- 需要排序/TopN → 用 `ranking`
- 需要时间序列/走势 → 用 `trend`
- 需要对比变化/增速 → 用 `comparison`
- 需要计算百分比/占比 → 用 `ratio`

# Disambiguation

**核心判断规则**：如果用户想看的是"**一个总数**"用 basic；如果想看"**多个分组值**"用 source。

- "总销售额是多少" → basic（单一聚合值）
- "各部门的销售额" → ❌ 不是 basic → source（"各"暗示分组）
- "A地区和B地区销售额分别是多少" → ❌ 不是 basic → source（"分别"暗示按维度分组）
- "2023年、2024年、2025年的销售额分别是多少" → ❌ 不是 basic → source（多年+分别 = 按年分组）
- "销售额最高的产品" → ❌ 不是 basic → ranking（排序+LIMIT）
- "XX是多少""总共多少" → basic
- "各XX分别是多少" → ❌ source（"各"+"分别"明确要求分组）

**易错情形**：
- "A区和B区销售额分别是多少" — 虽然只有两项，但本质是按维度分组查询，应用 source
- "2023年、2024年的销售额" — 本质是按年份分组，应用 source（而不是拆成多个 basic）

# Planner Prompt

## 基础查询 (basic)

**适用场景**：简单聚合，获取单一数值，结果为**单行**

**不适用（非常重要，请严格遵守）**：
- 包含"分别""各""每个""按XX看"等表述 → 必须用 `source`
- 需要同时返回多个维度值 → 必须用 `source`
- 涉及多年/多期且用户想看各期数据 → 用 `source`（按年/月分组）或 `trend`
- 需要排序、对比、趋势等复杂分析 → 使用对应的专项类型

**SQL 模式**：`SELECT SUM/COUNT/AVG(指标) FROM 表 WHERE 条件`

**核心概念**：
basic 是最简单的分析类型，**结果必须是单行**。只需要一个（或少量）聚合函数和 WHERE 筛选条件。如果你发现需要 GROUP BY，那说明应该使用 source 而非 basic。

**注意事项**：
1. ⚠️ 如果用户的问题包含"分别""各""每个"或列举了多个分组值，**绝对不要用 basic**，应该用 source
2. ⚠️ 如果一个问题需要拆成多个 basic 分别查不同年/不同分组，这通常意味着应该用**一个 source**（按年/按分组 GROUP BY）更合理
3. basic 是其他类型都不匹配时的兜底选项，但在有分组语义时 basic 绝不是正确选择

**任务示例**：
```json
{
  "id": "basic_1",
  "type": "basic",
  "description": "查询2025年总销售额",
  "depends_on": [],
  "notes": [],
  "virtual_fields": []
}
```

# SQL Instruction

### 基础查询 SQL 生成规则

**基本结构**：
```
SELECT 聚合函数(指标列) AS 指标别名
FROM 表
WHERE 时间/业务筛选条件
```

**详细规则**：
1. **聚合函数**：根据语义选择 SUM/COUNT/AVG/MAX/MIN
   - "总XX" → SUM
   - "有多少个" → COUNT(DISTINCT ...)
   - "平均" → AVG
   - "最大/最小" → MAX/MIN
2. **不需要 GROUP BY**：basic 的结果是单行，如果发现需要 GROUP BY，请确认任务类型是否正确
3. **只生成 1 个 SQL**
4. **结果列名要有意义**（用有含义的别名，如 `SUM("amount") AS total_sales`）

**常见错误**：
- ❌ 把需要分组的查询用 basic 处理（应该用 source）
- ❌ 返回多行结果（basic 的结果应该是单行）
- ❌ 结果列没有别名，可读性差
