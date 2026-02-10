"""
离线预处理模块

将数据库转换为 LLM 友好的检索与提示资源：
1. 双格式 Schema：DDL Schema + Light Schema
2. 列摘要：数值统计 / 分类高频值
3. BM25 关键字索引
4. 向量库（预留接口）
"""

from chatdb.preprocessing.column_profiler import ColumnProfile, ColumnProfiler
from chatdb.preprocessing.preprocessor import DataPreprocessor, PreprocessResult
from chatdb.preprocessing.schema_builder import DDLSchema, LightSchema, SchemaBuilder
from chatdb.preprocessing.text_index import BM25Index, IndexDocument, KeywordMatch, SearchResult, TextIndex
from chatdb.preprocessing.vector_store import (
    BaseVectorStore,
    CellVectorStore,
    ExampleVectorStore,
    VectorDocument,
    VectorSearchResult,
)

__all__ = [
    # Schema
    "SchemaBuilder",
    "DDLSchema",
    "LightSchema",
    # Column Profile
    "ColumnProfiler",
    "ColumnProfile",
    # Text Index (BM25)
    "TextIndex",
    "BM25Index",
    "IndexDocument",
    "SearchResult",
    "KeywordMatch",
    # Vector Store (预留)
    "BaseVectorStore",
    "CellVectorStore",
    "ExampleVectorStore",
    "VectorDocument",
    "VectorSearchResult",
    # Main Preprocessor
    "DataPreprocessor",
    "PreprocessResult",
]
