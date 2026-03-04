"""
chatdb.utils.common — ChatDB 工具函数

通用函数已迁移到 lib.utils.common，此处：
1. re-export 所有通用函数
2. 保留 ChatDB 特有的 DB 函数
"""

from typing import Any

# re-export 公共函数
from lib.utils.common import (  # noqa: F401
    format_rows,
    parse_json,
    parse_json_array,
    clean_sql,
)


# ==================== ChatDB 特有函数 ====================


def select_best_table(query: str, tables: list[dict[str, Any]]) -> str:
    """选择最佳表"""
    if not tables:
        return ""

    if len(tables) == 1:
        return tables[0].get("table_name", "")

    best_score = 0
    best_table = tables[0].get("table_name", "")
    query_words = set(query)

    for table in tables:
        desc = table.get("table_description", "")
        score = len(query_words & set(desc))
        if score > best_score:
            best_score = score
            best_table = table.get("table_name", "")

    return best_table


def build_schema_text(tables_meta: list[dict[str, Any]]) -> str:
    """构建 schema 文本"""
    lines = []
    for table in tables_meta:
        table_name = table.get("table_name", "")
        lines.append(f"表: {table_name}")

        columns = table.get("columns_info") or table.get("columns", [])
        if columns:
            lines.append("列:")
            for col in columns:
                col_name = col.get("name", col.get("column_name", ""))
                col_type = col.get("type", col.get("column_type", ""))
                lines.append(f"  - {col_name}: {col_type}")

        lines.append("")
    return "\n".join(lines)


def get_tables_info(schema_info) -> list[dict[str, Any]]:
    """从 SchemaInfo 获取表信息（增强版）"""
    tables_info = []
    for table in schema_info.tables:
        columns_info = []
        for col in table.columns:
            col_info = {
                "name": col.name,
                "type": col.type,
            }
            if hasattr(col, "sample_values") and col.sample_values:
                col_info["sample_values"] = col.sample_values
            columns_info.append(col_info)

        table_info = {
            "table_name": table.name,
            "column_count": len(table.columns),
            "table_description": table.comment or "",
            "columns_info": columns_info,
        }

        if hasattr(table, "row_count"):
            table_info["row_count"] = table.row_count

        if hasattr(table, "column_profiles"):
            table_info["column_profiles"] = table.column_profiles

        tables_info.append(table_info)
    return tables_info
