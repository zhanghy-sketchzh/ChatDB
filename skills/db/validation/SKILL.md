---
id: validation
label: 数据验证
description: 验证 SQL 正确性、检查口径一致性、诊断空结果或报错原因
keywords: [验证, 诊断, 检查, 确认, 排查, 为什么没有数据, 数据对不上]
priority_boost: 0
sql_hints:
  summary: 诊断查询 → 检查数据存在性、字段值分布、条件匹配
meta:
  intent_hint_template: "执行诊断查询，检查数据是否存在、字段值分布、条件匹配情况"
  stats_fields: [row_count, distinct_values, sample_values]
  issue_fields: [no_data, field_not_found, value_mismatch]
---

# When to Activate

SQL 报错或返回空结果时，需要诊断原因，包括但不限于：
- 上游 SQL 执行失败，需要检查表结构或字段名
- 查询返回空结果，需要验证数据是否存在
- 怀疑筛选条件有误，需要检查字段值的实际分布
- 结果数值异常，需要验证口径和数据范围

**典型场景**（非用户直接提问，通常由 Planner 在重试路径中插入）：
- SQL 报错"column not found" → 检查表中的实际列名
- 查询返回 0 行 → 检查 WHERE 条件中的值是否存在
- 数据看起来不对 → 检查时间范围、筛选条件是否正确

**关键特征**：validation 不是用户主动选择的分析类型，而是 Planner 在检测到异常时自动插入的诊断步骤。

# Not For

- 正常分析流程中不应主动使用
- 用户的首次查询（在遇到问题之前不需要 validation）
- 能直接修复的 SQL 语法错误（直接修复即可，不需要诊断查询）

# Disambiguation

- Planner 检测到 SQL 失败并选择重试 → 可插入 validation 诊断
- 用户说"数据对不上" → 可能需要 validation
- 用户说"帮我查XX数据" → 不是 validation，用对应的分析类型

# Planner Prompt

## 数据验证 (validation)

**适用场景**：SQL 报错或空结果时的诊断

**不适用**：
- 正常分析流程不应主动使用
- 仅在 Planner 选择重试路径时插入

**SQL 模式**：`SELECT DISTINCT / COUNT(*) / 采样查询`

**核心概念**：
数据验证的本质是**在执行失败后，通过轻量级探查定位问题原因**。常见诊断策略：
1. **存在性检查**：确认表/列是否存在，条件值是否有数据
2. **值分布检查**：查看字段的 DISTINCT 值，确认筛选条件是否匹配
3. **采样检查**：取少量数据行，确认数据格式和内容

**注意事项**：
1. 诊断 SQL 应该尽量轻量（COUNT/DISTINCT/LIMIT 少量行），避免全表扫描
2. 诊断结果应能直接指导下一步的 SQL 修复
3. 一个 validation 任务通常只做一个方向的诊断，不要混合多种检查

**任务示例**：
```json
{
  "id": "validation_1",
  "type": "validation",
  "description": "检查某列中是否存在指定的筛选值",
  "depends_on": [],
  "notes": ["诊断类型: 值存在性检查"],
  "virtual_fields": []
}
```

# SQL Instruction

### 数据验证 SQL 生成规则

**诊断策略与对应 SQL**：

**策略1 - 值存在性检查**（确认某个值是否存在）：
```sql
SELECT DISTINCT "列名" FROM 表名
WHERE "列名" LIKE '%{search_value}%'
LIMIT 20
```

**策略2 - 行数检查**（确认筛选条件是否有数据）：
```sql
SELECT COUNT(*) AS row_count FROM 表名
WHERE <原查询的筛选条件>
```

**策略3 - 字段值分布**（查看字段实际有哪些值）：
```sql
SELECT DISTINCT "列名", COUNT(*) AS cnt
FROM 表名
GROUP BY "列名"
ORDER BY cnt DESC
LIMIT 30
```

**策略4 - 数据采样**（查看原始数据样例）：
```sql
SELECT * FROM 表名
WHERE <筛选条件>
LIMIT 5
```

**详细规则**：
1. **轻量优先**：诊断 SQL 必须有 LIMIT，避免全表扫描
2. **一次一策略**：每个 validation 任务只用一种诊断策略
3. **目标明确**：SQL 的目的是回答"为什么上游 SQL 失败了"
4. 只生成 1 个 SQL

**常见错误**：
- ❌ 诊断 SQL 没有 LIMIT，对大表全表扫描
- ❌ 混合多种诊断策略在一个 SQL 中
- ❌ 诊断 SQL 比原 SQL 还复杂
- ❌ 诊断结果不能指导问题修复
