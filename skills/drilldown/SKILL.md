---
id: drilldown
label: 下钻分析
description: 对上游 Top 结果进一步细分，增加更细粒度的维度
keywords: [下钻, 细分, 进一步, 明细, 深入, 展开, 详细看看, 具体到]
priority_boost: 0
sql_hints:
  summary: 在上游筛选基础上增加更细粒度维度 → 适合"进一步细分/下钻"
meta:
  requires_previous_results: true
  intent_hint_template: "在上一步结果基础上进一步细分，增加筛选条件或更细粒度的维度"
  stats_fields: [parent_top_ratio, drilldown_depth]
  issue_fields: [no_previous_result, drilldown_exhausted]
---

# When to Activate

基于上一步 Top 结果进一步细分，包括但不限于：
- 对排名/来源分析的结果进行维度下钻
- 在上游结果的基础上增加更细粒度的维度
- 对某个特定项进一步展开明细

**典型表述**：
- "对这个结果按月份下钻"
- "进一步看看这个产品的渠道明细"
- "把第一名展开到月度"
- "深入分析一下这个部门"

**关键特征**：
1. 必须有上游任务结果作为基础（通过 `depends_on` 关联）
2. 保留上游的筛选条件，在此基础上增加新维度或更细粒度

# Not For

- 首次分析（没有上游结果）→ 用其他首发类型
- 上游结果和下钻之间没有父子关系的独立分析 → 用对应的专项类型

# Disambiguation

- "对第一名进一步下钻" → drilldown（有上游结果，在此基础上细分）
- "按产品看销售额" → 不是 drilldown → source（首次拆解，没有上游依赖）
- 有 depends_on 且是在上游结果上做细分 → drilldown
- 没有上游依赖的独立分析 → 不是 drilldown

# Planner Prompt

## 下钻分析 (drilldown)

**适用场景**：基于上一步 Top 结果进一步细分

**不适用**：
- 首次分析（没有上游结果）
- 独立的分析需求

**SQL 模式**：保留上游筛选条件 + 增加更细粒度维度或新的 GROUP BY

**核心概念**：
下钻分析的本质是**在已有分析结果上深入一层**，形成"总览→聚焦→细节"的分析链路。关键决策点：
1. **继承条件**：上游结果中的筛选条件必须保留
2. **新增维度**：选择比上游更细的粒度（如产品→月份，部门→产品线）
3. **分析深度**：避免过度下钻（通常2-3层就够了）

**注意事项**：
1. 下钻任务**必须**有 `depends_on` 字段指向上游任务
2. 上游结果中的筛选条件会通过 `parent_results_summary` 传递
3. description 中应明确说明下钻的对象和新维度
4. notes 中应包含继承的筛选条件

**任务示例**：
```json
{
  "id": "drilldown_1",
  "type": "drilldown",
  "description": "对排名第一的产品按月份下钻，查看各月明细",
  "depends_on": ["ranking_1"],
  "notes": ["下钻维度: month"],
  "virtual_fields": []
}
```

# SQL Instruction

### 下钻分析 SQL 生成规则

**基本结构**：
```
SELECT 新维度列, 聚合函数(指标列) AS 指标别名
FROM 表
WHERE 上游继承的筛选条件 AND 新的筛选条件
GROUP BY 新维度列
ORDER BY 新维度列/指标别名
```

**详细规则**：
1. **继承上游条件**：`WHERE` 中必须包含上游任务传递的筛选条件
2. **新增维度**：`GROUP BY` 使用比上游更细粒度的维度
3. **排序**：
   - 如果新维度是时间 → 按时间升序
   - 如果新维度是非时间 → 按指标降序
4. **上游结果来源**：从 `parent_results_summary` 或 `transition_context` 中获取筛选值
5. **只生成 1 个 SQL**

**常见错误**：
- ❌ 遗漏上游的筛选条件
- ❌ GROUP BY 维度粒度比上游还粗（下钻应该更细）
- ❌ 没有 depends_on 的情况下使用 drilldown 类型
- ❌ 使用上游结果中不存在的筛选值

**★ 与虚拟字段（指标型）配合的规则**：
- 指标型虚拟字段（如 `total_flow`）本身已含 `SUM(CASE WHEN ...)` 聚合表达式
- **禁止**在任何位置（包括子查询中）对指标虚拟字段再套聚合函数：
  - ❌ `SUM(total_flow)` → 展开后变为 `SUM(SUM(CASE WHEN ...))` → **报错**
- **正确做法**：直接用裸名称引用（如 `SELECT total_flow AS 指标 ... GROUP BY 维度`）
