---
id: comparison
label: 对比分析
description: 两个时间段/条件的对比，计算差值与增长率（同比/环比）
keywords: [对比, 同比, 环比, 涨幅, 跌幅, 变化, 增减, 下降最多, 增长最快, 增长率, 变化量, 增幅, 降幅, 升幅]
priority_boost: 0
sql_hints:
  summary: LAG/窗口函数计算差值/增长率 → 适合"对比两期差异、找增减幅度最大的项"
  comparison: yoy
meta:
  requires_time: true
  intent_hint_template: "对比两个时间段或条件下的指标，计算差值或增长率"
  stats_fields: [delta, growth_rate, comparison_base]
  issue_fields: [insufficient_comparison_data]
---

# When to Activate

需要**计算两期差值或增长率**的分析，包括但不限于：
- 同比/环比对比，计算增长率或变化幅度
- 找出"下降最多""增长最快""跌幅最大"的项
- 按维度分组后计算各自的变化量并排序
- 两个条件/场景之间的数值对比

**典型表述**：
- "今年比去年增长了多少"
- "哪些产品销售额下降最多"
- "同比增长率是多少"
- "环比变化最大的部门"
- "找出跌幅最大的产品"

**核心判断**：只要涉及"变化/增减/跌幅/涨幅/同比/环比"，无论是否同时有维度拆解，都必须用 comparison。

# Not For

- 单纯看走势、不需要计算差值 → 用 `trend`（如"看看近几年销售额"）
- 单纯排名、不涉及变化量 → 用 `ranking`（如"销售额最高的5个产品"）
- 单纯拆解构成、不涉及变化量 → 用 `source`（如"按部门拆解销售额"）

# Disambiguation

- 涉及"变化/增减/跌幅/涨幅/同比/环比/下降最多/增长最快" → comparison（即使同时有维度拆解）
- "找出跌幅最大的产品" → comparison（按产品 GROUP BY 两期数据，计算变化量后排序）
- "哪些产品销售额下降最多" → comparison（需要对比两期数据）
- "按部门拆解销售额" → 不是 comparison → source（不涉及变化量）
- "销售额最高的产品" → 不是 comparison → ranking（基于绝对值排序）
- "看看近几年销售额" → 不是 comparison → trend（只看走势不算差值）

# Planner Prompt

## 对比分析 (comparison)

**适用场景**：需要**计算两期差值或增长率**的分析，包括：
- 同比/环比对比，计算增长率/变化幅度
- 找出"下降最多""增长最快""跌幅最大"的项
- 按维度分组后计算各自的变化量并排序

**不适用**：
- 单纯看走势 → `trend`
- 单纯排名不涉及变化量 → `ranking`
- 单纯拆解构成 → `source`

**SQL 模式**：LAG/LEAD 窗口函数或子查询计算差值、增长率

**核心概念**：
对比分析的本质是**在两个基准之间计算数值差异**。关键决策点：
1. **对比基准**：时间维度（同比/环比）还是条件维度（A vs B）
2. **输出指标**：差值、增长率、还是两者都要
3. **排序语义**：必须严格匹配用户意图的方向（"下降最多"≠"变化最大"）

**★ 核心判断**：只要涉及"变化/增减/跌幅/涨幅/同比/环比"，就必须用 comparison

**注意事项**：
1. 对比任务必须明确两个时间点或两组条件
2. 如果用户没有明确说对比哪两期，需要从上下文推断（如"同比"→ 当期 vs 上年同期）
3. 输出结果应同时包含基期值、当期值、变化值（和/或增长率），便于理解
4. **★ 对比完整性**：如果用户提到 N 个时间点（如 2023、2024、2025 年），notes 中必须列出**所有需要对比的年份对**（如 `"对比年份: 2023→2024, 2024→2025"`），不要遗漏任何一对
5. **★ 上游数据依赖检查**：如果 comparison 依赖上游 source 任务的结果做年度对比，必须确保上游 source 的 GROUP BY 包含年份维度。在 notes 中写明：`"要求上游按年分组"`

**任务示例**：
```json
{
  "id": "comparison_1",
  "type": "comparison",
  "description": "对比2023和2024年各产品销售额，找出下降幅度最大的产品",
  "notes": ["对比方式: 同比", "分组维度: product", "排序: 按变化值ASC"],
  "virtual_fields": []
}
```

# SQL Instruction

### 对比分析 SQL 生成规则

**基本结构**：
```
-- 方式一：子查询对比（两期数据可能不对称时，用 FULL OUTER JOIN）
SELECT COALESCE(a.维度, b.维度) AS 维度,
  COALESCE(b.当期值, 0) AS 当期值,
  COALESCE(a.基期值, 0) AS 基期值,
  (COALESCE(b.当期值, 0) - COALESCE(a.基期值, 0)) AS 变化值,
  ROUND((COALESCE(b.当期值, 0) - COALESCE(a.基期值, 0)) * 100.0 / NULLIF(a.基期值, 0), 2) AS 增长率
FROM (基期子查询) a
FULL OUTER JOIN (当期子查询) b ON a.维度 IS NOT DISTINCT FROM b.维度
ORDER BY 变化值 ASC/DESC

-- 方式二：窗口函数（适合多期连续对比，维度两期都有时）
SELECT 维度, 时间, 值,
  LAG(值) OVER (PARTITION BY 维度 ORDER BY 时间) AS 上期值,
  值 - LAG(值) OVER (PARTITION BY 维度 ORDER BY 时间) AS 变化值
FROM ...
```

**详细规则**：
1. **差值计算**：`当期值 - 基期值`
2. **增长率计算**：`(当期 - 基期) / NULLIF(基期, 0) * 100`，用 `NULLIF` 防止除零
3. **结果必须包含**：基期值、当期值、变化值（和/或增长率），不要只返回差值
4. **★ 排序方向必须严格匹配任务描述的语义**：
   - "下降最多/跌幅最大" → `ORDER BY 变化值 ASC`（取最负的值）
   - "增长最多/涨幅最大" → `ORDER BY 变化值 DESC`（取最正的值）
   - **禁止使用 ABS() 排序**，因为 ABS 会混淆增长和下降的方向
5. **NULLIF 防御**：分母始终用 `NULLIF(基期值, 0)` 包裹
6. 只生成 1 个 SQL

**★ JOIN 类型选择（重要）**：
- **引用上游临时表时**：两个临时表的维度值可能不对称，**必须使用 `FULL OUTER JOIN`**，用 `COALESCE(a.维度, b.维度)` 合并维度列，用 `COALESCE(值, 0)` 补零
- **NULL 维度值的 JOIN**：`NULL = NULL` 在 SQL 中为 FALSE，维度可能含 NULL。**必须使用 `IS NOT DISTINCT FROM` 代替 `=`** 作为 JOIN 条件
- **从原始表子查询时**：如果能确保两期维度值完全对称，可以用 INNER JOIN

**常见错误**：
- ❌ 使用 `ORDER BY ABS(变化值) DESC` 找"下降最多"（ABS 会把增长和下降混在一起）
- ❌ 只返回增长率不返回原始值（用户无法验证计算是否正确）
- ❌ 忘记 NULLIF 导致基期为零时除零报错
- ❌ 排序方向与用户语义相反（"下降最多"用了 DESC）
- ❌ 使用 INNER JOIN 导致只在一期出现的维度被丢弃
- ❌ 使用 `ON a.维度 = b.维度` 导致 NULL 值的行无法 JOIN（应用 `IS NOT DISTINCT FROM`）

**★ 与虚拟字段（指标型）配合的规则**：
- 指标型虚拟字段（如 `total_flow`）本身已含 `SUM(CASE WHEN ...)` 聚合表达式
- **禁止**在任何位置（包括子查询中）对指标虚拟字段再套聚合函数：
  - ❌ `SUM(total_flow)` → 展开后变为 `SUM(SUM(CASE WHEN ...))` → **报错：嵌套聚合**
  - ❌ `SELECT SUM(total_flow) FROM ...` 子查询 → 展开后同样嵌套聚合 → **报错**
- **禁止**在 CASE WHEN 的 THEN 分支中引用指标型虚拟字段：
  - ❌ `SUM(CASE WHEN "年"=2023 THEN total_flow ELSE 0 END)` → 展开后嵌套聚合，报错
- **正确做法**：使用子查询对比时，每个子查询中直接用**裸名称**引用指标虚拟字段（不套 SUM）
  ```sql
  -- ✅ 正确：子查询中直接引用 total_flow（裸名称），不套 SUM
  SELECT COALESCE(a."维度", b."维度") AS "维度",
         a.基期值, b.当期值, (b.当期值 - a.基期值) AS 变化值
  FROM (SELECT "维度", total_flow AS 基期值 FROM "表" WHERE "base_valid_data" AND "年"=2023 GROUP BY "维度") a
  FULL OUTER JOIN
       (SELECT "维度", total_flow AS 当期值 FROM "表" WHERE "base_valid_data" AND "年"=2024 GROUP BY "维度") b
       ON a."维度" IS NOT DISTINCT FROM b."维度"
  ```
- **或者**直接使用 GROUP BY 加上年份维度，让 total_flow 自然按年聚合，然后用窗口函数计算差值
