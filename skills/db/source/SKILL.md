---
id: source
label: 来源分析
description: 按维度拆解单期数据，查看各分组的数值分布/贡献来源
keywords: [来源, 构成, 拆解, 贡献, 组成, 哪些, 分布, 各个, 按XX分, 明细, 分别, 各, 每个]
priority_boost: 0
sql_hints:
  summary: 按维度 GROUP BY + DESC → 适合"按XX拆解构成/贡献/分别查看"（仅单期，不涉及变化量计算）
  order_by_direction: DESC
meta:
  requires_dimension: true
  intent_hint_template: "按维度「{dimension}」拆解来源构成，GROUP BY 该维度并按数值降序排序"
  stats_fields: [top_contributor, top_ratio, category_count]
  issue_fields: [single_category, low_coverage]
---

# When to Activate

按维度拆解**单期**的数据分布，查看各分组的数值，包括但不限于：
- 按某个维度（产品/地区/部门/渠道等）分组聚合，查看各组的数值
- 需要同时查看多个分组的数值（"A和B分别是多少"）
- 找出"主要贡献来源""各项分布""组成结构"
- 对某个指标进行维度拆解，形成分布视图

**典型表述**：
- "各部门的销售额分别是多少"
- "按地区看收入分布"
- "哪些产品贡献了主要收入"
- "A和B各自是多少"
- "2023年、2024年的销售额分别是多少"（按年份分组）

**关键特征**：只涉及**单个时间段/条件**的拆解，不涉及两期对比或变化量计算。只要出现"分别""各""每个""按XX"等分组语义，就适用 source。

# Not For

- 涉及"变化/增减/跌幅/涨幅"等需要计算两期差值的分析 → 用 `comparison`
- 需要时间趋势/走势 → 用 `trend`
- 需要计算百分比/占比 → 用 `ratio`（如果只看各项的绝对值分布，用 source）
- 不需要分组、只要一个总数 → 用 `basic`

# Disambiguation

- "各部门销售额" → source（按维度拆解各项数值）
- "A和B分别是多少" → source（按维度分组查看多个值）
- "找出哪些产品销售额下降最多" → 不是 source → comparison（需要两期数据计算变化）
- 涉及"变化/增减/跌幅/涨幅/下降最多/增长最快" → comparison（即使同时有维度拆解）
- 仅按维度拆解构成，不涉及变化量 → source
- "各产品销售额" → source（看绝对值分布）
- "各产品销售额占比" → ratio（需要计算百分比）
- "各产品销售额变化" → comparison（涉及变化）
- "总销售额是多少" → basic（单一总数，无分组）

# Planner Prompt

## 来源分析 (source)

**适用场景**：按维度拆解**单期**的数据分布，查看各分组数值

**不适用**：
- 涉及"变化/增减/跌幅/涨幅"等需要计算两期差值的分析 → `comparison`
- 需要时间趋势 → `trend`
- 不需要分组 → `basic`

**SQL 模式**：`GROUP BY 维度列 + ORDER BY 指标 DESC`

**核心概念**：
来源分析的本质是**在某个时间截面上，按维度拆解指标的分布**。关键决策点：
1. **维度选择**：根据用户意图选择最合适的 GROUP BY 维度
2. **排序方向**：默认按指标降序（看 Top 贡献者），但以任务描述为准
3. **单期原则**：只处理一个时间段的数据，如果需要对比两期变化用 comparison

**注意事项**：
1. 维度选择要与用户意图匹配，不要随意选择分析维度
2. 如果用户没有明确指定排序方向，默认按数值降序
3. 通常不需要 LIMIT，除非用户明确说"前几名"
4. 如果用户同时想看绝对值和占比，考虑拆为 source + ratio 两步
5. **★ 分组维度完整性**：notes 中指定的所有分组维度（如 `分组维度: 年, 国内/海外`）必须**全部**出现在 SQL 的 GROUP BY 中。缺少任何一个维度都会导致数据被错误聚合（如多年数据混在一起求和）

**任务示例**：
```json
{
  "id": "source_1",
  "type": "source",
  "description": "按部门拆解2024年销售额，查看各部门的数值",
  "depends_on": [],
  "notes": ["分组维度: department", "排序: 按销售额降序"],
  "virtual_fields": []
}
```

# SQL Instruction

### 来源/构成分析 SQL 生成规则

**基本结构**：
```
SELECT 维度列, 聚合函数(指标列) AS 指标别名
FROM 表
WHERE 时间/业务筛选条件
GROUP BY 维度列
ORDER BY 指标别名 DESC
```

**详细规则**：
1. **维度选择**：根据任务描述选择最合适的维度进行 GROUP BY
2. **★ 分组维度完整性**：任务 notes 中指定的**所有**分组维度必须全部出现在 GROUP BY 和 SELECT 中。例如 notes 写了 `分组维度: 年, 国内/海外`，则 GROUP BY 必须同时包含 `"年"` 和 `"国内/海外"`
3. **排序方向**：通常按数值降序排序（看 Top 贡献），具体以任务描述为准
4. **单 SQL 原则**：只生成 1 个 SQL
5. **不要计算占比**：来源分析只返回各维度的聚合值，占比计算由 ratio 类型负责
6. **不要计算差值**：不要在 SQL 中加入 LAG/LEAD 或两期对比逻辑

**常见错误**：
- ❌ 在 source SQL 中计算占比百分比（这是 ratio 的职责）
- ❌ 在 source SQL 中加入两期对比逻辑（这是 comparison 的职责）
- ❌ 选择了用户没有提到的维度进行 GROUP BY
- ❌ 遗漏必要的 WHERE 筛选条件（如时间范围）
- ❌ notes 中写了多个分组维度但 GROUP BY 只写了部分（如 notes 要求按"年, 国内/海外"分组，SQL 却只 GROUP BY "国内/海外"，导致多年数据被混合聚合）
