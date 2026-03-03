"""
数据预处理器

统一入口，执行完整的离线预处理流程：
1. 生成双格式 Schema (DDL + Light)
2. 生成列摘要
3. 构建 BM25 索引
4. 存储元数据到 meta_data.db
5. （可选）构建向量索引
6. （可选）LLM 生成表理解文本
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from chatdb.preprocessing.column_profiler import ColumnProfile, ColumnProfiler
from chatdb.preprocessing.schema_builder import DDLSchema, LightSchema, SchemaBuilder
from chatdb.preprocessing.text_index import IndexDocument, TextIndex
from chatdb.storage import MetaDataStore

logger = logging.getLogger(__name__)


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
        meta_info: dict[str, Any] | None = None,
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
            meta_info: meta_db / YML meta 中的额外信息，用于增强 BM25 索引
                       支持的 key: display_name, grain, table_understanding,
                       virtual_field_synonyms (list[str])
            
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
        index_doc_count = self._build_text_index(
            df, table_name, table_description, column_profiles, meta_info
        )
        
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
        meta_info: dict[str, Any] | None = None,
    ) -> PreprocessResult:
        """
        预处理 DuckDB 表
        
        Args:
            db_path: 数据库路径
            table_name: 表名
            table_description: 表描述
            id_columns: ID 列列表
            meta_info: meta_db / YML meta 中的额外信息，用于增强 BM25 索引
        """
        import duckdb
        
        conn = duckdb.connect(db_path, read_only=True)
        try:
            df = conn.execute(f'SELECT * FROM "{table_name}"').fetchdf()
            return self.preprocess_dataframe(
                df, table_name, table_description, id_columns, meta_info=meta_info,
            )
        finally:
            conn.close()
    
    def _build_text_index(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_description: str,
        column_profiles: list[ColumnProfile],
        meta_info: dict[str, Any] | None = None,
    ) -> int:
        """构建文本索引
        
        表级文档使用丰富的内容构建，以提高 BM25 召回率：
        - table_name + table_description（基础信息）
        - meta_info 中的 display_name、grain、table_understanding（meta_db 信息）
        - meta_info 中的 virtual_field_synonyms（YML 虚拟字段同义词）
        - 所有列名
        - 列统计摘要（分类字段高频值、数值字段范围）
        """
        docs: list[IndexDocument] = []
        meta_info = meta_info or {}
        
        # 1. 表级文档：构建丰富的可检索文本
        content_parts: list[str] = [table_name]
        
        # 加入 table_description
        if table_description:
            content_parts.append(table_description)
        
        # 加入 meta_db / YML meta 中的额外信息
        display_name = meta_info.get("display_name", "")
        if display_name and display_name != table_name:
            content_parts.append(display_name)
        
        grain = meta_info.get("grain", "")
        if grain:
            content_parts.append(grain)
        
        table_understanding = meta_info.get("table_understanding", "")
        if table_understanding:
            # 截取前 500 字符避免索引文档过长
            content_parts.append(table_understanding[:500])
        
        # 加入 YML 虚拟字段的同义词（业务术语）
        synonyms = meta_info.get("virtual_field_synonyms", [])
        if synonyms:
            content_parts.extend(synonyms)
        
        # 加入所有列名
        content_parts.extend(df.columns.tolist())
        
        # 加入列统计摘要（让表级文档也能匹配到列值相关的查询）
        for profile in column_profiles:
            if profile.is_id:
                continue
            summary = profile.to_text_summary(top_k=10)
            if summary:
                content_parts.append(f"{profile.name} {summary}")
        
        table_content = " ".join(content_parts)
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

    async def generate_table_understanding(
        self,
        llm: Any,
        table_name: str,
        table_description: str = "",
        row_count: int = 0,
        column_count: int = 0,
        column_profiles: list[dict[str, Any]] | None = None,
        content_hash: str | None = None,
        source_type: str = "excel",
        force: bool = False,
        yml_config: dict[str, Any] | None = None,
    ) -> str:
        """
        为指定表生成 LLM 理解文本并缓存到 meta_data.db。

        如果 meta_data.db 中已有缓存且 force=False，直接返回缓存。

        Args:
            llm: BaseLLM 实例
            table_name: 表名
            table_description: 表的简要描述
            row_count: 行数
            column_count: 列数
            column_profiles: 列元信息（dict 格式）
            content_hash: 内容哈希（用于更新缓存），None 时按 table_name 查找
            source_type: 数据源类型
            force: 是否强制重新生成
            yml_config: YML 业务配置（可选），提供报表层级、组织范围等额外信息

        Returns:
            表理解文本
        """
        # 尝试从缓存读取
        if not force:
            cached = self.meta_store.get_by_table_name(table_name, source_type)
            if cached and cached.get("table_understanding"):
                logger.info(f"表 '{table_name}' 理解文本命中缓存")
                return cached["table_understanding"]

        from chatdb.preprocessing.table_understanding import generate_table_understanding

        understanding = await generate_table_understanding(
            llm=llm,
            table_name=table_name,
            table_description=table_description,
            row_count=row_count,
            column_count=column_count,
            column_profiles=column_profiles,
            yml_config=yml_config,
        )

        # 写入缓存
        if content_hash:
            self.meta_store.update_table_understanding(content_hash, understanding, source_type)
        else:
            # 按 table_name 查找后更新
            meta = self.meta_store.get_by_table_name(table_name, source_type)
            if meta:
                h = meta.get("content_hash") or meta.get("table_hash") or meta.get("file_hash")
                if h:
                    self.meta_store.update_table_understanding(h, understanding, source_type)

        return understanding


def extract_meta_info_from_yml(yml_config: dict[str, Any]) -> dict[str, Any]:
    """
    从 YML 配置中提取 meta_info，用于增强 BM25 索引。
    
    自动提取：
    - display_name: 表的业务展示名
    - grain: 数据粒度描述
    - virtual_field_synonyms: 虚拟字段的所有同义词（业务术语）
    
    Args:
        yml_config: 完整的 YML 配置字典（metrics_config.yml 解析后的内容）
    
    Returns:
        meta_info 字典，可直接传给 preprocess_dataframe(..., meta_info=...)
    
    Example:
        import yaml
        with open("data/yml/metrics_config.yml") as f:
            yml = yaml.safe_load(f)
        meta_info = extract_meta_info_from_yml(yml)
        preprocessor.preprocess_dataframe(df, "脚本测试数据", meta_info=meta_info)
    """
    meta = yml_config.get("meta", {})
    info: dict[str, Any] = {}
    
    if meta.get("display_name"):
        info["display_name"] = meta["display_name"]
    if meta.get("description"):
        info["description"] = meta["description"]
    if meta.get("grain"):
        info["grain"] = meta["grain"]
    
    # 从 virtual_fields 中提取所有同义词
    synonyms: list[str] = []
    virtual_fields = yml_config.get("virtual_fields", {})
    for field_id, field_def in virtual_fields.items():
        if isinstance(field_def, dict):
            # 加入字段描述
            desc = field_def.get("description", "")
            if desc:
                synonyms.append(desc)
            # 加入同义词列表
            syns = field_def.get("synonyms", [])
            if syns:
                synonyms.extend(syns)
    
    if synonyms:
        info["virtual_field_synonyms"] = synonyms
    
    return info
