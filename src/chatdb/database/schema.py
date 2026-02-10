"""
数据库 Schema 检查器和生成器

提供数据库 Schema 的提取、生成、格式化等功能，供 LLM Agent 使用。
支持从数据库连接器提取 Schema，也支持从 DataFrame 生成 Schema。
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

from chatdb.utils.exceptions import SchemaError
from chatdb.utils.logger import logger
from chatdb.database.base import BaseDatabaseConnector


@dataclass
class ColumnInfo:
    """列信息"""

    name: str
    type: str
    nullable: bool
    primary_key: bool
    default: str | None = None
    comment: str | None = None


@dataclass
class TableInfo:
    """表信息"""

    name: str
    columns: list[ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[dict[str, Any]]
    comment: str | None = None


@dataclass
class SchemaInfo:
    """数据库 Schema 信息"""

    database_name: str
    tables: list[TableInfo]

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 友好的文本格式"""
        lines = [f"数据库: {self.database_name}", "=" * 50, ""]

        for table in self.tables:
            lines.append(f"表名: {table.name}")
            if table.comment:
                lines.append(f"描述: {table.comment}")
            lines.append("-" * 30)

            for col in table.columns:
                pk_marker = " [PK]" if col.primary_key else ""
                null_marker = " NULL" if col.nullable else " NOT NULL"
                comment = f" -- {col.comment}" if col.comment else ""
                lines.append(f"  {col.name}: {col.type}{pk_marker}{null_marker}{comment}")

            if table.foreign_keys:
                lines.append("  外键关系:")
                for fk in table.foreign_keys:
                    lines.append(
                        f"    {fk['column']} -> {fk['referred_table']}.{fk['referred_column']}"
                    )

            lines.append("")

        return "\n".join(lines)


class SchemaManager:
    """Schema 管理器 - 统一管理 Schema 的检查和生成"""

    def __init__(
        self,
        connector: BaseDatabaseConnector | None = None,
        llm_client=None,
        model_name: str | None = None,
    ):
        """
        初始化 Schema 管理器

        Args:
            connector: 数据库连接器实例（用于检查数据库 Schema）
            llm_client: LLM 客户端（BaseLLM 实例，用于生成 Schema）
            model_name: LLM 模型名称（可选，如果提供会覆盖llm_client的model）
        """
        from chatdb.llm.base import BaseLLM
        
        self.connector = connector
        self.llm_client = llm_client
        self.model_name = model_name or (llm_client.model if isinstance(llm_client, BaseLLM) else None)

    # ==================== 数据库 Schema 检查方法 ====================

    async def get_schema_info(self) -> SchemaInfo:
        """
        获取完整的数据库 Schema 信息

        Returns:
            SchemaInfo 对象

        Raises:
            SchemaError: 如果连接器未提供或获取失败
        """
        if not self.connector:
            raise SchemaError("需要提供数据库连接器才能获取 Schema 信息")

        try:
            engine = self.connector.engine
            tables = await self._get_tables(engine)

            # 从连接 URL 提取数据库名
            db_name = self._extract_db_name(engine.url)

            return SchemaInfo(database_name=db_name, tables=tables)

        except Exception as e:
            logger.error(f"获取 Schema 信息失败: {e}")
            raise SchemaError(f"获取 Schema 信息失败: {e}")

    async def _get_tables(self, engine) -> list[TableInfo]:
        """获取所有表信息（支持同步和异步引擎）"""
        from sqlalchemy import inspect as sqlalchemy_inspect
        from sqlalchemy.ext.asyncio import AsyncEngine
        
        def sync_inspect(connection):
            inspector = sqlalchemy_inspect(connection)
            table_names = inspector.get_table_names()
            result = []

            for table_name in table_names:
                columns = []
                pk_columns = inspector.get_pk_constraint(table_name).get(
                    "constrained_columns", []
                )

                for col in inspector.get_columns(table_name):
                    columns.append(
                        ColumnInfo(
                            name=col["name"],
                            type=str(col["type"]),
                            nullable=col.get("nullable", True),
                            primary_key=col["name"] in pk_columns,
                            default=str(col.get("default")) if col.get("default") else None,
                            comment=col.get("comment"),
                        )
                    )

                # 获取外键信息
                foreign_keys = []
                for fk in inspector.get_foreign_keys(table_name):
                    for i, col in enumerate(fk["constrained_columns"]):
                        foreign_keys.append(
                            {
                                "column": col,
                                "referred_table": fk["referred_table"],
                                "referred_column": fk["referred_columns"][i],
                            }
                        )

                result.append(
                    TableInfo(
                        name=table_name,
                        columns=columns,
                        primary_keys=pk_columns,
                        foreign_keys=foreign_keys,
                        comment=None,
                    )
                )

            return result

        # 判断是否为异步引擎
        if isinstance(engine, AsyncEngine):
            async with engine.connect() as conn:
                tables = await conn.run_sync(sync_inspect)
        else:
            # 同步引擎（如 DuckDB）
            with engine.connect() as conn:
                tables = sync_inspect(conn)

        return tables

    async def get_table_sample(self, table_name: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        获取表的示例数据

        Args:
            table_name: 表名
            limit: 返回行数

        Returns:
            示例数据列表
        """
        if not self.connector:
            raise SchemaError("需要提供数据库连接器才能获取表样本")
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        return await self.connector.execute_query(sql)

    @staticmethod
    def _extract_db_name(url) -> str:
        """从 SQLAlchemy URL 提取数据库名"""
        return url.database or "unknown"

    # ==================== DataFrame Schema 生成方法 ====================

    def detect_id_columns_with_llm(
        self, df: pd.DataFrame, table_name: str
    ) -> List[str]:
        """使用 LLM 识别 ID 列"""
        if not self.llm_client:
            raise ValueError("LLM客户端未配置，无法识别ID列")

        import asyncio
        from chatdb.llm.base import BaseLLM, Message

        if not isinstance(self.llm_client, BaseLLM):
            raise ValueError("llm_client 必须是 BaseLLM 实例")

        columns_info = []
        for col in df.columns:
            sample_values = df[col].dropna().head(5).tolist()
            sample_str = ", ".join([str(v) for v in sample_values])[:100]
            columns_info.append(
                f"  - {col} (唯一值: {len(df[col].dropna().unique())}, 示例: {sample_str})"
            )

        prompt = f"""分析数据表字段，识别ID列（标识符列，如员工ID、订单号、编码等）。

表名: {table_name}
字段:
{chr(10).join(columns_info)}

返回JSON: {{"id_columns": ["列名1", "列名2"]}}
无ID列返回: {{"id_columns": []}}"""

        # 调用LLM
        from chatdb.utils.logger import log_llm_interaction
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，使用 run_coroutine_threadsafe
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.llm_client.generate([Message(role="user", content=prompt)], temperature=0, max_tokens=500)
                    )
                    response = future.result()
            else:
                response = loop.run_until_complete(
                    self.llm_client.generate([Message(role="user", content=prompt)], temperature=0, max_tokens=500)
                )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            response = asyncio.run(
                self.llm_client.generate([Message(role="user", content=prompt)], temperature=0, max_tokens=500)
            )

        # 记录 LLM 交互（输入输出封装在一起）
        log_llm_interaction(logger, "ID列识别", prompt, response.content, max_prompt_chars=300, max_response_chars=500)

        if not response.content:
            return []

        # 提取JSON
        from chatdb.llm.base import _extract_json_from_text

        json_str = _extract_json_from_text(response.content)
        if not json_str:
            return []

        try:
            result = json.loads(json_str)
            return [
                col for col in result.get("id_columns", []) if col in df.columns
            ]
        except Exception as e:
            logger.warning(f"解析ID列JSON失败: {e}")
        return []


    def generate_basic_schema_json(
        self, df: pd.DataFrame, table_name: str, id_columns: List[str] = None
    ) -> str:
        """生成基础的Schema JSON（不使用LLM，仅基于代码分析）
        
        包含字段统计信息：
        - 分类字段：唯一值 top10（或全部，如果 <= 20）
        - 数值字段：范围、均值、中位数
        """
        if id_columns is None:
            id_columns = []

        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0

            is_id_column = col in id_columns

            if dtype in ["int64", "int32", "Int64"]:
                data_type = "整数"
            elif dtype in ["float64", "float32"]:
                data_type = "小数"
            elif "datetime" in dtype:
                data_type = "日期时间"
            else:
                data_type = "文本"

            col_info = {
                "column_name": col,
                "data_type": data_type,
                "description": f"字段 {col}",
                "null_percentage": round(null_pct, 1),
                "is_id_column": is_id_column,
            }

            # 分类字段：提取唯一值（top10 或全部 <= 20）
            if dtype == "object":
                unique_vals = df[col].dropna().unique()
                unique_count = len(unique_vals)
                if unique_count <= 20:
                    col_info["unique_values_top20"] = [
                        str(v) for v in unique_vals[:20]
                    ]
                elif unique_count > 20:
                    # 取出现次数最多的 top10
                    value_counts = df[col].value_counts()
                    col_info["unique_values_top10"] = [
                        str(v) for v in value_counts.head(10).index.tolist()
                    ]
                    col_info["unique_count"] = unique_count

            # 数值字段：统计范围、均值、中位数
            if dtype in [
                "int64",
                "int32",
                "Int64",
                "float64",
                "float32",
            ] and not is_id_column:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    min_val = col_data.min()
                    max_val = col_data.max()
                    mean_val = col_data.mean()
                    median_val = col_data.median()
                    col_info["statistics_summary"] = (
                        f"范围: {min_val:.2f} ~ {max_val:.2f}, "
                        f"均值: {mean_val:.2f}, 中位数: {median_val:.2f}"
                    )

            columns.append(col_info)

        schema = {
            "table_name": table_name,
            "table_description": f"数据表 {table_name}",
            "id_columns": id_columns,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
        }

        return json.dumps(schema, ensure_ascii=False, indent=2)

    def prepare_er_info(self, df: pd.DataFrame, table_name: str) -> str:
        """准备ER信息（表结构）"""
        er_lines = [f"表名: {table_name}"]
        er_lines.append(f"行数: {len(df)}")
        er_lines.append(f"列数: {len(df.columns)}")
        er_lines.append("\n字段列表:")

        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            er_lines.append(f"  - {col} ({dtype}, 缺失率: {null_pct:.1f}%)")

        return "\n".join(er_lines)

    def prepare_numeric_stats(self, df: pd.DataFrame) -> str:
        """准备数值列的描述统计"""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            return "无数值列"

        stats_lines = ["数值列描述统计:"]
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_lines.append(f"\n  {col}:")
                stats_lines.append(f"    最小值: {col_data.min():.2f}")
                stats_lines.append(f"    最大值: {col_data.max():.2f}")
                stats_lines.append(f"    平均值: {col_data.mean():.2f}")
                stats_lines.append(f"    中位数: {col_data.median():.2f}")
                stats_lines.append(f"    标准差: {col_data.std():.2f}")

        return "\n".join(stats_lines)

    def prepare_categorical_distribution(self, df: pd.DataFrame) -> str:
        """准备分类列的唯一值分布"""
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical_cols:
            return "无分类列"

        dist_lines = ["分类列唯一值分布:"]
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            unique_count = len(unique_vals)

            dist_lines.append(f"\n  {col} (唯一值数量: {unique_count}):")

            if unique_count <= 20:
                value_counts = df[col].value_counts()
                for val, count in value_counts.head(20).items():
                    dist_lines.append(
                        f"    - '{val}': {count}条 ({count / len(df) * 100:.1f}%)"
                    )
            else:
                value_counts = df[col].value_counts()
                dist_lines.append("    前10个最常见值:")
                for val, count in value_counts.head(10).items():
                    dist_lines.append(
                        f"    - '{val}': {count}条 ({count / len(df) * 100:.1f}%)"
                    )

        return "\n".join(dist_lines)

    def format_schema_as_prompt(
        self, schema_json: str, df: pd.DataFrame, table_name: str
    ) -> str:
        """
        将Schema JSON格式化为文本prompt
        用于后续的query改写和SQL生成
        """
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError:
            return f"数据表: {table_name}\n数据规模: {len(df)}行 × {len(df.columns)}列"

        lines = []
        lines.append("=== 数据表Schema理解 ===")
        lines.append(f"表名: {schema.get('table_name', table_name)}")
        lines.append(f"表描述: {schema.get('table_description', '')}")
        lines.append("")

        lines.append("=== 字段详细信息 ===")
        for col in schema.get("columns", []):
            lines.append(f"\n字段: {col.get('column_name')}")
            lines.append(f"  类型: {col.get('data_type')}")
            lines.append(f"  描述: {col.get('description')}")

            if "unique_values_top20" in col:
                unique_vals = col["unique_values_top20"]
                lines.append(
                    f"  出现次数前20的值: {', '.join([str(v) for v in unique_vals])}"
                )

            if "statistics_summary" in col:
                lines.append(f"  统计: {col['statistics_summary']}")

        return "\n".join(lines)

    def enrich_schema_json(
        self,
        schema_json: str,
        df: pd.DataFrame,
        table_name: str,
        id_columns: List[str],
        data_processor=None,
    ) -> str:
        """
        丰富 Schema JSON，添加技术性字段描述和统计信息

        Args:
            schema_json: 简化的 Schema JSON 字符串
            df: DataFrame 对象
            table_name: 表名
            id_columns: ID 列列表
            data_processor: 数据处理器（可选）

        Returns:
            丰富后的 Schema JSON 字符串
        """
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError:
            return schema_json

        # 为每个列添加技术性描述和统计信息
        for col_info in schema.get("columns", []):
            col_name = col_info.get("column_name")
            if col_name not in df.columns:
                continue

            # 生成列描述
            description = self._generate_column_description(
                df, col_name, col_info, id_columns, data_processor
            )
            if description:
                col_info["description"] = description
            
            # 添加统计信息
            self._add_column_statistics(df, col_name, col_info, id_columns)

        return json.dumps(schema, ensure_ascii=False, indent=2)

    def _add_column_statistics(
        self, 
        df: pd.DataFrame, 
        col_name: str, 
        col_info: dict, 
        id_columns: List[str]
    ) -> None:
        """
        为列添加统计信息
        
        - 分类字段：唯一值 top10（或全部，如果 <= 20）
        - 数值字段：范围、均值、中位数
        """
        if col_name not in df.columns:
            return
        
        dtype = str(df[col_name].dtype)
        is_id_column = col_name in id_columns
        
        # 添加缺失率
        null_count = df[col_name].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        if null_pct > 0:
            col_info["null_percentage"] = round(null_pct, 1)
        
        # 添加是否为ID列
        col_info["is_id_column"] = is_id_column
        
        # 分类字段：提取唯一值（top10 或全部 <= 20）
        if dtype == "object" and not is_id_column:
            unique_vals = df[col_name].dropna().unique()
            unique_count = len(unique_vals)
            if unique_count <= 20:
                col_info["unique_values_top20"] = [str(v) for v in unique_vals[:20]]
            elif unique_count > 20:
                # 取出现次数最多的 top10
                value_counts = df[col_name].value_counts()
                col_info["unique_values_top10"] = [
                    str(v) for v in value_counts.head(10).index.tolist()
                ]
                col_info["unique_count"] = unique_count

        # 数值字段：统计范围、均值、中位数
        if dtype in ["int64", "int32", "Int64", "float64", "float32"] and not is_id_column:
            col_data = df[col_name].dropna()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                median_val = col_data.median()
                col_info["statistics_summary"] = (
                    f"范围: {min_val:.2f} ~ {max_val:.2f}, "
                    f"均值: {mean_val:.2f}, 中位数: {median_val:.2f}"
                )

    def _generate_column_description(
        self,
        df: pd.DataFrame,
        col_name: str,
        col_info: dict,
        id_columns: List[str],
        data_processor=None,
    ) -> str:
        """
        生成列的详细描述

        Args:
            df: DataFrame 对象
            col_name: 列名
            col_info: 列信息字典
            id_columns: ID 列列表
            data_processor: 数据处理器（可选）

        Returns:
            列描述字符串
        """
        dtype = str(df[col_name].dtype)
        null_count = df[col_name].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0

        parts = []

        # 数据类型描述
        if dtype in ["int64", "int32", "Int64"]:
            parts.append("整数类型")
        elif dtype in ["float64", "float32"]:
            parts.append("小数类型")
        elif "datetime" in dtype:
            parts.append("日期时间类型")
        else:
            parts.append("文本类型")

        # 缺失率
        if null_pct > 0:
            parts.append(f"缺失率 {null_pct:.1f}%")

        # ID 列标记
        if col_name in id_columns:
            parts.append("标识符列")

        # 唯一值信息
        unique_count = len(df[col_name].dropna().unique())
        if unique_count <= 20:
            parts.append(f"唯一值 {unique_count} 个")
        else:
            parts.append(f"唯一值 {unique_count} 个（较多）")

        return "，".join(parts) if parts else col_info.get("description", f"字段 {col_name}")

    async def generate_multi_table_schemas_with_llm(
        self, tables_basic_info: List[Dict], filename: str, data_processor=None
    ) -> Dict[str, str]:
        """
        统一生成多个表的Schema理解JSON（一次LLM调用生成所有表的描述）
        
        Args:
            tables_basic_info: 所有表的基础信息列表，每个元素包含：
                - table_name: 表名
                - df: DataFrame对象
                - id_columns: ID列列表
            filename: 文件名
            data_processor: 数据处理器（可选，用于 enrich_schema_json）
            
        Returns:
            字典，key为table_name，value为schema JSON字符串
        """
        if not self.llm_client:
            raise ValueError("LLM客户端未配置，无法生成多表Schema")
        
        # 构建所有表的基础信息
        tables_info_for_prompt = []
        for table_info in tables_basic_info:
            table_name = table_info["table_name"]
            df = table_info["df"]
            
            er_info = self.prepare_er_info(df, table_name)
            numeric_stats = self.prepare_numeric_stats(df)
            categorical_distribution = self.prepare_categorical_distribution(df)
            sample_data = df.head(3).to_dict("records")
            
            tables_info_for_prompt.append({
                "table_name": table_name,
                "er_info": er_info,
                "numeric_stats": numeric_stats,
                "categorical_distribution": categorical_distribution,
                "sample_data": sample_data,
            })
        
        # 构建统一的prompt
        prompt = self._build_multi_table_schema_prompt(tables_info_for_prompt, filename)

        # 调用LLM生成所有表的schema
        from chatdb.llm.base import call_llm_for_schema
        # LLM 交互日志会在 call_llm_for_schema 中统一处理

        llm_result = await call_llm_for_schema(self.llm_client, prompt, self.model_name)
        
        # 解析LLM返回的结果
        try:
            all_schemas_simplified = json.loads(llm_result)
        except json.JSONDecodeError as e:
            logger.error(f"解析多表schema JSON失败: {e}")
            raise
        
        # 为每个表补充技术性字段
        result = {}
        for table_info in tables_basic_info:
            table_name = table_info["table_name"]
            df = table_info["df"]
            id_columns = table_info["id_columns"]
            
            # 获取该表的简化schema
            simplified_schema = all_schemas_simplified.get(table_name, {})
            simplified_json = json.dumps(simplified_schema, ensure_ascii=False)
            
            # 补充技术性字段
            enriched_json = self.enrich_schema_json(
                simplified_json, df, table_name, id_columns, data_processor
            )
            result[table_name] = enriched_json
        
        return result

    def _build_multi_table_schema_prompt(
        self, tables_info: List[Dict], filename: str
    ) -> str:
        """
        构建多表统一生成schema的prompt
        
        Args:
            tables_info: 所有表的信息列表
            filename: 文件名
        """
        # 转换sample_data中的特殊类型为可JSON序列化的格式
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            elif hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        # 构建每个表的信息文本
        tables_text = []
        for idx, table_info in enumerate(tables_info, 1):
            table_name = table_info["table_name"]
            er_info = table_info["er_info"]
            numeric_stats = table_info["numeric_stats"]
            categorical_distribution = table_info["categorical_distribution"]
            sample_data = convert_to_serializable(table_info["sample_data"])
            sample_data_str = json.dumps(sample_data, ensure_ascii=False, indent=2)
            
            table_text = f"""
=== 表{idx}: {table_name} ===

{er_info}

数值列描述统计:
{numeric_stats}

分类列唯一值分布:
{categorical_distribution}

样本数据（前3行）:
{sample_data_str}
"""
            tables_text.append(table_text)
        
        all_tables_text = "\n".join(tables_text)
        all_table_names = [t["table_name"] for t in tables_info]
        all_table_names_str = "、".join(all_table_names)
        
        prompt = f"""你是一个数据分析专家。请分析Excel文件"{filename}"中的{len(tables_info)}个数据表，为每个表生成Schema理解的JSON。

{all_tables_text}

**任务要求**：
为每个表生成 `table_description`（表的整体描述，说明这是什么数据，适合做什么分析）

请严格按照以下JSON格式输出：

```json
{{
  "{all_table_names[0]}": {{
    "table_description": "表的整体描述...",
  }},
  "{all_table_names[1] if len(all_table_names) > 1 else 'table2'}": {{
    "table_description": "表的整体描述...",
  }},
  ...
}}
```

请直接输出JSON，不要有其他文字："""
        
        return prompt


# ==================== 向后兼容别名 ====================

SchemaInspector = SchemaManager
SchemaGenerator = SchemaManager

