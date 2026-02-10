"""
公共工具函数
"""

from typing import Any
import json
import re


def select_best_table(query: str, tables: list[dict[str, Any]]) -> str:
    """
    选择最佳表
    
    Args:
        query: 用户查询
        tables: 可用表列表
    
    Returns:
        最佳表名
    """
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
    """
    构建 schema 文本
    
    Args:
        tables_meta: 表元数据列表
    
    Returns:
        格式化的 schema 文本
    """
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
    """
    从 SchemaInfo 获取表信息（增强版）
    
    Args:
        schema_info: SchemaInfo 对象
    
    Returns:
        表信息列表，包含：
        - table_name: 表名
        - column_count: 列数
        - row_count: 行数（如果有）
        - table_description: 表描述
        - columns_info: 列信息列表
        - column_profiles: 列元信息（如果有）
    """
    tables_info = []
    for table in schema_info.tables:
        # 构建列信息
        columns_info = []
        for col in table.columns:
            col_info = {
                "name": col.name,
                "type": col.type,
            }
            # 如果有示例值，添加进来
            if hasattr(col, "sample_values") and col.sample_values:
                col_info["sample_values"] = col.sample_values
            columns_info.append(col_info)
        
        table_info = {
            "table_name": table.name,
            "column_count": len(table.columns),
            "table_description": table.comment or "",
            "columns_info": columns_info,
        }
        
        # 如果有行数信息
        if hasattr(table, "row_count"):
            table_info["row_count"] = table.row_count
        
        # 如果有 column_profiles
        if hasattr(table, "column_profiles"):
            table_info["column_profiles"] = table.column_profiles
        
        tables_info.append(table_info)
    return tables_info


def format_rows(rows: list[dict[str, Any]], max_rows: int = 10) -> str:
    """
    格式化结果行为表格
    
    Args:
        rows: 结果行列表
        max_rows: 最大显示行数
    
    Returns:
        格式化的表格文本
    """
    if not rows:
        return "无数据"
    
    sample = rows[:max_rows]
    columns = list(sample[0].keys())
    
    lines = [" | ".join(columns)]
    lines.append(" | ".join(["---"] * len(columns)))
    
    for row in sample:
        values = [str(row.get(col, ""))[:50] for col in columns]  # 截断过长值
        lines.append(" | ".join(values))
    
    if len(rows) > max_rows:
        lines.append(f"... 共 {len(rows)} 行")
    
    return "\n".join(lines)


def parse_json(response: str) -> dict[str, Any]:
    """
    解析 JSON 响应（支持从文本中提取）
    
    Args:
        response: LLM 响应文本
    
    Returns:
        解析的 JSON 对象
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {}


def parse_json_array(response: str) -> list[Any]:
    """
    解析 JSON 数组响应
    
    Args:
        response: LLM 响应文本
    
    Returns:
        解析的数组
    """
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
    """
    清理 SQL 文本
    
    Args:
        sql: 原始 SQL
    
    Returns:
        清理后的 SQL
    """
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```\s*", "", sql)
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql
