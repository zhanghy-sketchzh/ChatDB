"""
DuckDB SQL 语法规则

定义 DuckDB 特定的 SQL 语法规则和约束，供 SQL 生成智能体使用。
"""

# 引号与名称（合并：基础引号 + 精确匹配）
DUCKDB_QUOTE_AND_NAMING = """
### 引号与名称
- 列名/表名：含中文、数字开头、特殊字符或空格时用双引号；字符串值用单引号。例：WHERE "部门" = '销售部'
- 表名用 Schema 完整表名；列名逐字符匹配表结构；字符串条件匹配实际值
"""

# WHERE / GROUP BY
DUCKDB_WHERE_GROUPBY = """
### WHERE 与 GROUP BY
- AND/OR 混用须用括号明确优先级；WHERE 不能引用 SELECT 别名
- 非聚合列必须在 GROUP BY 中；ORDER BY 列须在 SELECT/GROUP BY 中；CTE 各层列引用一致
"""

# CTE / 子查询 / 数值 / LIMIT
DUCKDB_CTE_NUMERIC_LIMIT = """
### CTE、子查询与数值
- CTE/表别名用英文或拼音；同一 SELECT 不能引用本层别名；禁止 SELECT 中多行子查询，用 JOIN 或窗口函数
- 数值：ROUND(column, 2)；除零用 NULLIF；比例：ROUND(SUM(CASE WHEN cond THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)
- 仅当用户要求「前N条」等时加 LIMIT

### 日期函数类型限制（重要！）
- QUARTER()/MONTH()/YEAR() 等日期函数**只接受 DATE/TIMESTAMP 类型**，不接受 BIGINT/INTEGER
- 若月份列是整数格式（如 202501、202502），**禁止**使用 QUARTER("月份")，或根据列元信息中的具体值范围（如 202501~202512）使用 CASE WHEN 映射
"""

# 多表查询策略
DUCKDB_MULTI_TABLE_STRATEGIES = """
### 多表查询策略

**策略1: UNION ALL（合并相似结构的表）**
- 适用场景：多个表结构相似，需要合并查询（如：国内表+海外表）
- 每个 SELECT 必须有相同数量的列，类型兼容
- 不存在的字段用 NULL 填充
- 示例：
```sql
SELECT "员工姓名", "奖金金额", '国内' AS "来源" FROM "国内表"
UNION ALL
SELECT "员工姓名", "奖金金额", '海外' AS "来源" FROM "海外表"
ORDER BY "奖金金额" DESC;
```

**策略2: JOIN（关联不同表）**
- 适用场景：需要关联不同表的数据
- 禁止使用 ON 1=1，必须有明确的关联条件
- 示例：
```sql
SELECT a."员工姓名", b."部门名称"
FROM "员工表" a
JOIN "部门表" b ON a."部门ID" = b."部门ID";
```

**策略3: 子查询**
- 适用场景：需要先聚合再筛选
- 示例：
```sql
SELECT * FROM (
    SELECT "员工姓名", SUM("奖金金额") AS "奖金合计"
    FROM "奖金表"
    GROUP BY "员工姓名"
) subquery
ORDER BY "奖金合计" DESC LIMIT 1;
```
"""

# 常见错误
DUCKDB_COMMON_ERRORS = """
### 常见错误避免
禁止 FULL OUTER JOIN ON 1=1；禁止 GREATEST/LEAST 比较不同表（用 UNION ALL）；禁止 SELECT 引用同层别名、中文 CTE/别名。
"""


def get_duckdb_syntax_rules(include_multi_table: bool = False) -> str:
    """
    获取 DuckDB SQL 语法规则（紧凑版，供生成 prompt 使用）。
    """
    rules = [
        DUCKDB_QUOTE_AND_NAMING,
        DUCKDB_WHERE_GROUPBY,
        DUCKDB_CTE_NUMERIC_LIMIT,
        DUCKDB_COMMON_ERRORS,
    ]
    if include_multi_table:
        rules.insert(-1, DUCKDB_MULTI_TABLE_STRATEGIES)
    return "\n".join(rules)


def get_analysis_constraints(
    table_name: str | None = None,
    table_names: list[str] | None = None,
    is_multi_table: bool = False,
) -> str:
    """
    获取分析约束条件

    Args:
        table_name: 单表名称
        table_names: 多表名称列表
        is_multi_table: 是否为多表模式

    Returns:
        约束条件字符串
    """
    if is_multi_table and table_names:
        return f"""
**可用的数据表**：{', '.join(table_names)}

**多表查询策略选择**：
1. UNION ALL：结构相似的表合并查询（如国内+海外）
2. JOIN：不同类型表的关联查询
3. 单表：用户明确指定某个表时

**表名/字段名必须与 Schema 完全匹配**
"""
    elif table_name:
        return f"""
**数据表**：{table_name}
**表名/字段名必须与 Schema 完全匹配**
"""
    return ""
