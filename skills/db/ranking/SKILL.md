---
id: ranking
label: 排名分析
description: 按指标排序找出 TopN / BottomN，基于绝对值排名
keywords: [排名, TopN, 最高, 最低, 最大, 最小, 前几, 前十, 第一, 排行, 排列, 倒数, 垫底]
priority_boost: 0
sql_hints:
  summary: ORDER BY + LIMIT → 适合"找 TopN / 最大最小"（基于绝对值排名，不涉及变化量计算）
  order_by_direction: DESC
  limit: 10
meta:
  requires_dimension: true
  intent_hint_template: "按{metric}排序，找出 Top{limit}，需要 GROUP BY 维度 ORDER BY 指标 DESC LIMIT N"
  stats_fields: [top_items, top_value, bottom_value]
  issue_fields: [insufficient_data, single_item]
---

# When to Activate

找出 TopN / BottomN，按某个指标排序取极值，包括但不限于：
- 按某指标排序，找出前 N 名或后 N 名
- 找出"最大""最小""最高""最低"的项
- 生成排行榜、Top 列表

**典型表述**：
- "销售额最高的5个产品"
- "排名前十的部门"
- "找出利润最低的3个地区"
- "哪个产品收入最多"

**关键特征**：基于**绝对值**排序，不涉及两期变化量的计算。结果需要 LIMIT 截断。

# Not For

- 需要计算变化幅度再排序 → 用 `comparison`（如"下降最多的产品"需要先算差值）
- 需要看时间趋势 → 用 `trend`
- 需要拆解构成但不限制数量 → 用 `source`

# Disambiguation

- "销售额最高的5个产品" → ranking（基于绝对值排序 + LIMIT）
- "销售额下降最多的产品" → 不是 ranking → comparison（需要计算两期变化量）
- 基于绝对值排序 → ranking；基于变化量排序 → comparison
- "各产品销售额" → 不是 ranking → source（拆解所有，不取 TopN）
- "前5个产品的销售额" → ranking（有 TopN 限制）
- "产品销售额排行" → ranking（暗含排序+可能的截断）

# Planner Prompt

## 排名分析 (ranking)

**适用场景**：找出 TopN / BottomN，按某个指标排序取极值

**不适用**：
- 需要计算变化幅度再排序 → `comparison`
- 需要看时间趋势 → `trend`
- 查看全部分组数值不截断 → `source`

**SQL 模式**：`GROUP BY 维度 + ORDER BY 指标 DESC/ASC + LIMIT N`

**核心概念**：
排名分析的本质是**在绝对值维度上排序并截断**。关键决策点：
1. **排序指标**：用户想按什么指标排名
2. **排序方向**：Top（DESC）还是 Bottom（ASC）
3. **截断数量**：LIMIT N，用户没明确说时默认 10
4. **分组维度**：按什么维度分组后再排序

**与 comparison 的关键区分**：
- ranking 是对**一个时间点/条件**的数据排序（基于绝对值）
- comparison 是对**两个时间点/条件**的数据计算差值后排序（基于变化量）
- "销售额最高" → ranking；"销售额增长最多" → comparison

**注意事项**：
1. 如果用户说"Top N"但没给具体 N，默认用 10
2. 排序方向要与用户语义匹配（"最高"→ DESC，"最低"→ ASC）
3. ranking 的结果常被后续 drilldown 任务使用

**任务示例**：
```json
{
  "id": "ranking_1",
  "type": "ranking",
  "description": "找出2025年销售额最高的5个产品",
  "depends_on": [],
  "notes": ["排序方向: DESC", "限制数量: 5", "分组维度: product"],
  "virtual_fields": []
}
```

# SQL Instruction

### 排名分析 SQL 生成规则

**基本结构**：
```
SELECT 维度列, 聚合函数(指标列) AS 指标别名
FROM 表
WHERE 时间/业务筛选条件
GROUP BY 维度列
ORDER BY 指标别名 DESC/ASC
LIMIT N
```

**详细规则**：
1. **GROUP BY**：按用户指定的维度分组
2. **ORDER BY 方向**：根据任务描述决定
   - "最高/最大/Top" → `DESC`
   - "最低/最小/Bottom/倒数" → `ASC`
3. **LIMIT**：任务描述中的 N，未指定时默认 10
4. **单 SQL 原则**：只生成 1 个 SQL

**常见错误**：
- ❌ 忘记 LIMIT 导致返回所有行（ranking 的关键是截断）
- ❌ 排序方向与用户语义相反（"最低"用了 DESC）
- ❌ 在 ranking 中加入 LAG 计算变化量（这是 comparison 的职责）
- ❌ 缺少 GROUP BY 导致返回明细行而非聚合排名
