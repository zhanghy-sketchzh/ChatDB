"""
双格式 Schema 构建器

生成两种 Schema 格式：
1. DDL Schema：标准 SQL DDL，每个字段带摘要注释
2. Light Schema：Markdown 轻量格式，自然语言友好
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from chatdb.preprocessing.column_profiler import ColumnProfile, ColumnProfiler


@dataclass
class DDLSchema:
    """DDL Schema"""
    table_name: str
    create_sql: str
    columns: list[dict[str, Any]]
    
    def to_prompt(self) -> str:
        """生成带注释的 DDL"""
        return self.create_sql


@dataclass
class LightSchema:
    """轻量 Schema（Markdown 格式）"""
    table_name: str
    description: str
    row_count: int
    columns: list[dict[str, str]]
    
    def to_markdown(self) -> str:
        """生成 Markdown 格式"""
        lines = [
            f"## {self.table_name}",
            f"> {self.description}",
            f"> 数据量: {self.row_count:,} 行",
            "",
            "| 字段 | 类型 | 说明 |",
            "|------|------|------|",
        ]
        for col in self.columns:
            lines.append(f"| {col['name']} | {col['type']} | {col['summary']} |")
        return "\n".join(lines)
    
    def to_prompt(self) -> str:
        """生成 Prompt 友好的文本"""
        lines = [
            f"表名: {self.table_name}",
            f"描述: {self.description}",
            f"行数: {self.row_count:,}",
            "",
            "字段列表:",
        ]
        for col in self.columns:
            lines.append(f"  - {col['name']} ({col['type']}): {col['summary']}")
        return "\n".join(lines)


class SchemaBuilder:
    """双格式 Schema 构建器"""
    
    def __init__(self, profiler: ColumnProfiler | None = None):
        self.profiler = profiler or ColumnProfiler()
    
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_description: str = "",
        id_columns: list[str] | None = None,
    ) -> tuple[DDLSchema, LightSchema]:
        """
        从 DataFrame 构建双格式 Schema
        
        Args:
            df: 数据
            table_name: 表名
            table_description: 表描述
            id_columns: ID 列列表
            
        Returns:
            (DDLSchema, LightSchema)
        """
        # 生成列摘要
        profiles = self.profiler.profile_dataframe(df, id_columns)
        
        # 构建 DDL Schema
        ddl_schema = self._build_ddl(df, table_name, profiles)
        
        # 构建 Light Schema
        light_schema = self._build_light(
            table_name, table_description or f"数据表 {table_name}",
            len(df), profiles
        )
        
        return ddl_schema, light_schema
    
    def build_from_duckdb(
        self,
        db_path: str,
        table_name: str,
        table_description: str = "",
        id_columns: list[str] | None = None,
    ) -> tuple[DDLSchema, LightSchema]:
        """从 DuckDB 数据库构建双格式 Schema"""
        import duckdb
        
        conn = duckdb.connect(db_path, read_only=True)
        try:
            df = conn.execute(f'SELECT * FROM "{table_name}"').fetchdf()
            return self.build_from_dataframe(df, table_name, table_description, id_columns)
        finally:
            conn.close()
    
    def _build_ddl(
        self,
        df: pd.DataFrame,
        table_name: str,
        profiles: list[ColumnProfile],
    ) -> DDLSchema:
        """构建 DDL Schema"""
        lines = [f'CREATE TABLE "{table_name}" (']
        
        columns_info = []
        for profile in profiles:
            # 获取 SQL 类型
            sql_type = self._pandas_to_sql_type(df[profile.name].dtype)
            
            # 生成注释
            comment = profile.to_text_summary()
            
            # DDL 行
            col_line = f'    "{profile.name}" {sql_type}'
            if comment:
                col_line += f"  -- {comment}"
            
            lines.append(col_line + ",")
            
            columns_info.append({
                "name": profile.name,
                "type": sql_type,
                "summary": comment,
                "profile": profile.to_dict(),
            })
        
        # 移除最后一个逗号
        lines[-1] = lines[-1].rstrip(",")
        lines.append(");")
        
        return DDLSchema(
            table_name=table_name,
            create_sql="\n".join(lines),
            columns=columns_info,
        )
    
    def _build_light(
        self,
        table_name: str,
        description: str,
        row_count: int,
        profiles: list[ColumnProfile],
    ) -> LightSchema:
        """构建 Light Schema"""
        columns = []
        for profile in profiles:
            columns.append({
                "name": profile.name,
                "type": profile.dtype,
                "summary": profile.to_text_summary(),
            })
        
        return LightSchema(
            table_name=table_name,
            description=description,
            row_count=row_count,
            columns=columns,
        )
    
    @staticmethod
    def _pandas_to_sql_type(dtype) -> str:
        """Pandas 类型转 SQL 类型"""
        dtype_str = str(dtype)
        if dtype_str in ("int64", "int32", "Int64"):
            return "BIGINT"
        elif dtype_str in ("float64", "float32"):
            return "DOUBLE"
        elif "datetime" in dtype_str:
            return "TIMESTAMP"
        elif dtype_str == "bool":
            return "BOOLEAN"
        else:
            return "VARCHAR"
