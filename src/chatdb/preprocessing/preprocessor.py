"""
数据预处理器

统一入口，执行完整的离线预处理流程：
1. 生成双格式 Schema (DDL + Light)
2. 生成列摘要
3. 构建 BM25 索引
4. 存储元数据到 meta_data.db
5. （可选）构建向量索引
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from chatdb.preprocessing.column_profiler import ColumnProfile, ColumnProfiler
from chatdb.preprocessing.schema_builder import DDLSchema, LightSchema, SchemaBuilder
from chatdb.preprocessing.text_index import IndexDocument, TextIndex
from chatdb.storage import MetaDataStore


@dataclass
class PreprocessResult:
    """预处理结果"""
    table_name: str
    row_count: int
    column_count: int
    
    # 双格式 Schema
    ddl_schema: DDLSchema
    light_schema: LightSchema
    
    # 列摘要
    column_profiles: list[ColumnProfile]
    
    # 索引统计
    index_doc_count: int = 0
    
    def get_ddl_prompt(self) -> str:
        """获取 DDL 格式的 Prompt"""
        return self.ddl_schema.to_prompt()
    
    def get_light_prompt(self) -> str:
        """获取轻量格式的 Prompt"""
        return self.light_schema.to_prompt()
    
    def get_markdown(self) -> str:
        """获取 Markdown 格式"""
        return self.light_schema.to_markdown()
    
    def to_dict(self) -> dict[str, Any]:
        """转为字典"""
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "ddl_schema": self.ddl_schema.create_sql,
            "light_schema": self.light_schema.to_prompt(),
            "column_profiles": [p.to_dict() for p in self.column_profiles],
            "index_doc_count": self.index_doc_count,
        }


class DataPreprocessor:
    """
    数据预处理器
    
    将数据库/DataFrame 转换为 LLM 友好的检索与提示资源
    """
    
    def __init__(
        self,
        text_index: TextIndex | None = None,
        meta_store: MetaDataStore | None = None,
        enable_value_index: bool = True,
        value_sample_limit: int = 1000,
    ):
        """
        Args:
            text_index: 文本索引实例，None 则使用默认路径
            meta_store: 元数据存储实例，None 则使用默认路径
            enable_value_index: 是否索引单元格值
            value_sample_limit: 单列值索引的采样数量限制
        """
        self.profiler = ColumnProfiler()
        self.schema_builder = SchemaBuilder(self.profiler)
        self.text_index = text_index or TextIndex()
        self.meta_store = meta_store or MetaDataStore()
        self.enable_value_index = enable_value_index
        self.value_sample_limit = value_sample_limit
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_description: str = "",
        id_columns: list[str] | None = None,
        file_name: str = "",
        db_name: str = "",
        db_path: str = "",
        source_type: str = "excel",
        save_to_meta: bool = True,
    ) -> PreprocessResult:
        """
        预处理 DataFrame
        
        Args:
            df: 数据
            table_name: 表名
            table_description: 表描述
            id_columns: ID 列列表
            file_name: 来源文件名
            db_name: 数据库名
            db_path: 数据库路径
            source_type: 数据源类型
            save_to_meta: 是否保存到元数据库
            
        Returns:
            PreprocessResult
        """
        # 1. 生成双格式 Schema
        ddl_schema, light_schema = self.schema_builder.build_from_dataframe(
            df, table_name, table_description, id_columns
        )
        
        # 2. 生成列摘要
        column_profiles = self.profiler.profile_dataframe(df, id_columns)
        
        # 3. 构建 BM25 索引
        index_doc_count = self._build_text_index(df, table_name, table_description, column_profiles)
        
        result = PreprocessResult(
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            ddl_schema=ddl_schema,
            light_schema=light_schema,
            column_profiles=column_profiles,
            index_doc_count=index_doc_count,
        )
        
        # 4. 保存到元数据库
        if save_to_meta:
            self._save_to_meta(
                result, 
                file_name=file_name or table_name,
                db_name=db_name or "default",
                db_path=db_path,
                source_type=source_type,
                id_columns=id_columns,
                table_description=table_description,
            )
        
        return result
    
    def _save_to_meta(
        self,
        result: PreprocessResult,
        file_name: str,
        db_name: str,
        db_path: str,
        source_type: str,
        id_columns: list[str] | None,
        table_description: str,
    ) -> None:
        """保存预处理结果到元数据库"""
        # 计算内容哈希
        content_hash = hashlib.sha256(
            f"{result.table_name}{result.row_count}{result.column_count}{result.get_ddl_prompt()}".encode()
        ).hexdigest()[:16]
        
        self.meta_store.save(
            file_hash=content_hash,
            table_hash=content_hash,
            table_name=result.table_name,
            file_name=file_name,
            db_name=db_name,
            db_path=db_path,
            row_count=result.row_count,
            column_count=result.column_count,
            schema_info=result.get_ddl_prompt(),
            source_type=source_type,
            columns_info=[{"name": p.name, "dtype": p.dtype} for p in result.column_profiles],
            table_description=table_description,
            summary_prompt=result.get_light_prompt(),
            id_columns=id_columns,
            create_table_sql=result.ddl_schema.create_sql,
            column_profiles=[p.to_dict() for p in result.column_profiles],
            ddl_schema=result.get_ddl_prompt(),
            light_schema=result.get_light_prompt(),
        )
    
    def preprocess_duckdb_table(
        self,
        db_path: str,
        table_name: str,
        table_description: str = "",
        id_columns: list[str] | None = None,
    ) -> PreprocessResult:
        """
        预处理 DuckDB 表
        
        Args:
            db_path: 数据库路径
            table_name: 表名
            table_description: 表描述
            id_columns: ID 列列表
        """
        import duckdb
        
        conn = duckdb.connect(db_path, read_only=True)
        try:
            df = conn.execute(f'SELECT * FROM "{table_name}"').fetchdf()
            return self.preprocess_dataframe(df, table_name, table_description, id_columns)
        finally:
            conn.close()
    
    def _build_text_index(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_description: str,
        column_profiles: list[ColumnProfile],
    ) -> int:
        """构建文本索引"""
        docs: list[IndexDocument] = []
        
        # 1. 表级文档
        table_content = f"{table_name} {table_description} " + " ".join(df.columns.tolist())
        docs.append(IndexDocument(
            doc_id=f"table:{table_name}",
            doc_type="table",
            table_name=table_name,
            content=table_content,
            metadata={"description": table_description, "row_count": len(df)},
        ))
        
        # 2. 列级文档
        for profile in column_profiles:
            col_content = f"{profile.name} {profile.dtype} {profile.to_text_summary()}"
            docs.append(IndexDocument(
                doc_id=f"column:{table_name}.{profile.name}",
                doc_type="column",
                table_name=table_name,
                column_name=profile.name,
                content=col_content,
                metadata={"name": profile.name, "dtype": profile.dtype, "summary": profile.to_text_summary()},
            ))
        
        # 3. 值级文档：从 unique_values 中索引
        if self.enable_value_index:
            for profile in column_profiles:
                if profile.unique_values:
                    # 取前 value_sample_limit 个值索引
                    for val, count in profile.unique_values[:self.value_sample_limit]:
                        if len(val) > 1:  # 过滤太短的值
                            docs.append(IndexDocument(
                                doc_id=f"value:{table_name}.{profile.name}:{hash(val)}",
                                doc_type="value",
                                table_name=table_name,
                                column_name=profile.name,
                                content=val,
                                metadata={"value": val, "count": count},
                            ))
        
        # 添加到索引
        self.text_index.add_documents(docs)
        return len(docs)
    
    def search(self, query: str, top_k: int = 10):
        """搜索相关内容"""
        return self.text_index.search(query, top_k)
    
    def search_tables(self, query: str, top_k: int = 5):
        """搜索相关表"""
        return self.text_index.search_tables(query, top_k)
    
    def search_columns(self, query: str, table_name: str | None = None, top_k: int = 10):
        """搜索相关列"""
        return self.text_index.search_columns(query, table_name, top_k)
    
    def search_values(self, query: str, top_k: int = 20):
        """搜索相关值"""
        return self.text_index.search_values(query, top_k)
    
    def delete_table(self, table_name: str) -> int:
        """删除表的所有索引"""
        return self.text_index.delete_by_table(table_name)
