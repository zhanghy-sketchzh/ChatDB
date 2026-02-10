"""
向量索引模块（预留接口）

提供向量化检索能力：
- VD_cell：数据库单元格向量库，用于检索"内容相关"的值
- VD_example：训练示例向量库，用于检索相似问题做 few-shot

当前为接口预留，实际实现需要依赖 embedding 模型
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorDocument:
    """向量文档"""
    doc_id: str
    doc_type: str  # cell, example
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """向量存储基类"""
    
    @abstractmethod
    def add(self, docs: list[VectorDocument]) -> None:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        """搜索"""
        pass
    
    @abstractmethod
    def delete(self, doc_ids: list[str]) -> None:
        """删除文档"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空"""
        pass


class CellVectorStore(BaseVectorStore):
    """
    单元格向量库 (VD_cell)
    
    存储所有 text 类型单元格的向量，用于检索"内容相关"的值
    如地名、公司名、产品名等
    
    TODO: 实际实现需要:
    - Embedding 模型（如 text2vec, sentence-transformers）
    - 向量数据库（如 faiss, chromadb, milvus）
    """
    
    def __init__(self, embedding_model=None, vector_db=None):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self._documents: dict[str, VectorDocument] = {}
    
    def add(self, docs: list[VectorDocument]) -> None:
        """添加单元格向量"""
        for doc in docs:
            self._documents[doc.doc_id] = doc
        # TODO: 实际向量化和存储
    
    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        """搜索相似单元格"""
        # TODO: 实际向量检索
        return []
    
    def delete(self, doc_ids: list[str]) -> None:
        """删除"""
        for doc_id in doc_ids:
            self._documents.pop(doc_id, None)
    
    def clear(self) -> None:
        """清空"""
        self._documents.clear()


class ExampleVectorStore(BaseVectorStore):
    """
    示例向量库 (VD_example)
    
    存储「自然语言问题 + SQL」的向量，用于检索相似问题做 few-shot
    
    TODO: 实际实现需要:
    - Embedding 模型
    - 向量数据库
    """
    
    def __init__(self, embedding_model=None, vector_db=None):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self._documents: dict[str, VectorDocument] = {}
    
    def add(self, docs: list[VectorDocument]) -> None:
        """添加示例"""
        for doc in docs:
            self._documents[doc.doc_id] = doc
        # TODO: 实际向量化和存储
    
    def add_example(self, question: str, sql: str, metadata: dict | None = None) -> str:
        """
        添加问题-SQL 示例
        
        Args:
            question: 自然语言问题
            sql: 对应的 SQL
            metadata: 额外元数据（如表名、难度等）
        """
        import uuid
        doc_id = str(uuid.uuid4())
        doc = VectorDocument(
            doc_id=doc_id,
            doc_type="example",
            content=f"问题: {question}\nSQL: {sql}",
            metadata={"question": question, "sql": sql, **(metadata or {})},
        )
        self.add([doc])
        return doc_id
    
    def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        """搜索相似问题"""
        # TODO: 实际向量检索
        return []
    
    def get_few_shot_examples(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """
        获取 few-shot 示例
        
        Returns:
            [{"question": "...", "sql": "..."}, ...]
        """
        results = self.search(query, top_k)
        return [r.metadata for r in results if "question" in r.metadata]
    
    def delete(self, doc_ids: list[str]) -> None:
        """删除"""
        for doc_id in doc_ids:
            self._documents.pop(doc_id, None)
    
    def clear(self) -> None:
        """清空"""
        self._documents.clear()
