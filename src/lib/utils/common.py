"""
lib.utils.common — 公共工具函数

仅包含所有场景包共用的通用函数。
DB 特有函数（select_best_table, build_schema_text 等）保留在 chatdb.utils.common。
"""

from typing import Any
import json
import re


def format_rows(rows: list[dict[str, Any]], max_rows: int = 10) -> str:
    """格式化结果行为表格"""
    if not rows:
        return "无数据"

    sample = rows[:max_rows]
    columns = list(sample[0].keys())

    lines = [" | ".join(columns)]
    lines.append(" | ".join(["---"] * len(columns)))

    for row in sample:
        values = [str(row.get(col, ""))[:50] for col in columns]
        lines.append(" | ".join(values))

    if len(rows) > max_rows:
        lines.append(f"... 共 {len(rows)} 行")

    return "\n".join(lines)


def parse_json(response: str) -> dict[str, Any]:
    """解析 JSON 响应（支持从文本中提取）"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def parse_json_array(response: str) -> list[Any]:
    """解析 JSON 数组响应"""
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[[\s\S]*\]', response)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def clean_sql(sql: str) -> str:
    """清理 SQL 文本"""
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```\s*", "", sql)
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql
