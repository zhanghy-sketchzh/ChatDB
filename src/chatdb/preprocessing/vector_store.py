"""
向量索引模块

提供向量化检索能力：
- VD_cell：数据库单元格向量库，用于检索"内容相关"的值
- VD_example：训练示例向量库，用于检索相似问题做 few-shot

ExampleVectorStore 已实现 BM25 关键词检索兜底，无需 embedding 模型即可使用。
未来接入 embedding 模型后，search() 可升级为语义向量检索。
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
    
    存储「自然语言问题 + SQL」，用于检索相似问题做 few-shot。
    
    实现策略：
    - BM25 关键词检索（已实现，开箱即用）
    - 向量语义检索（预留，接入 embedding 模型后自动升级）
    
    持久化：SQLite 存储示例，BM25 索引懒加载到内存。
    """
    
    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_model: Any = None,
    ):
        """
        Args:
            db_path: SQLite 持久化路径，None 则纯内存模式
            embedding_model: 预留，未来接入 embedding 模型
        """
        self.embedding_model = embedding_model
        self._documents: dict[str, VectorDocument] = {}
        
        # BM25 索引（懒加载）
        self._bm25: Any = None  # BM25Index
        self._bm25_dirty: bool = True

        # SQLite 持久化
        self._db_path: Path | None = Path(db_path) if db_path else None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        if not self._db_path:
            return
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS examples (
                    doc_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    explanation TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)

    def _load_from_db(self) -> None:
        if not self._db_path or not self._db_path.exists():
            return
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT doc_id, question, sql, explanation, metadata FROM examples").fetchall()
        for row in rows:
            doc_id, question, sql_text, explanation, meta_json = row
            meta = json.loads(meta_json) if meta_json else {}
            meta.update({"question": question, "sql": sql_text, "explanation": explanation})
            doc = VectorDocument(
                doc_id=doc_id,
                doc_type="example",
                content=f"{question}",
                metadata=meta,
            )
            self._documents[doc_id] = doc
        self._bm25_dirty = True

    @property
    def bm25(self) -> Any:
        """懒加载 BM25 索引"""
        if self._bm25 is None or self._bm25_dirty:
            from chatdb.preprocessing.text_index import BM25Index, IndexDocument
            self._bm25 = BM25Index()
            for doc in self._documents.values():
                self._bm25.add_document(IndexDocument(
                    doc_id=doc.doc_id,
                    doc_type="example",
                    table_name="",
                    content=doc.content,
                    metadata=doc.metadata,
                ))
            self._bm25_dirty = False
        return self._bm25
    
    def add(self, docs: list[VectorDocument]) -> None:
        """添加示例文档"""
        for doc in docs:
            self._documents[doc.doc_id] = doc
        self._bm25_dirty = True

    def add_example(
        self,
        question: str,
        sql: str,
        explanation: str = "",
        metadata: dict | None = None,
    ) -> str:
        """
        添加问题-SQL 示例
        
        Args:
            question: 自然语言问题
            sql: 对应的 SQL
            explanation: 可选说明
            metadata: 额外元数据（如表名、难度等）
        """
        import uuid
        doc_id = str(uuid.uuid4())[:12]
        meta = {"question": question, "sql": sql, "explanation": explanation, **(metadata or {})}
        doc = VectorDocument(
            doc_id=doc_id,
            doc_type="example",
            content=question,
            metadata=meta,
        )
        self.add([doc])

        # 持久化
        if self._db_path:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO examples (doc_id, question, sql, explanation, metadata) VALUES (?,?,?,?,?)",
                    (doc_id, question, sql, explanation, json.dumps(meta, ensure_ascii=False)),
                )

        return doc_id
    
    def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        """
        搜索相似问题
        
        策略：
        1. 如果有 embedding_model → 向量语义检索（TODO）
        2. 否则 → BM25 关键词检索（已实现）
        """
        if not self._documents:
            return []

        # BM25 兜底
        results = self.bm25.search(query, top_k)
        return [
            VectorSearchResult(
                doc_id=r.doc_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
    
    def get_few_shot_examples(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """
        获取 few-shot 示例
        
        Returns:
            [{"question": "...", "sql": "...", "explanation": "..."}, ...]
        """
        results = self.search(query, top_k)
        return [
            {
                "question": r.metadata.get("question", ""),
                "sql": r.metadata.get("sql", ""),
                "explanation": r.metadata.get("explanation", ""),
            }
            for r in results
            if r.metadata.get("question")
        ]
    
    def delete(self, doc_ids: list[str]) -> None:
        """删除"""
        for doc_id in doc_ids:
            self._documents.pop(doc_id, None)
        self._bm25_dirty = True
        if self._db_path:
            with sqlite3.connect(str(self._db_path)) as conn:
                placeholders = ",".join("?" * len(doc_ids))
                conn.execute(f"DELETE FROM examples WHERE doc_id IN ({placeholders})", doc_ids)
    
    def clear(self) -> None:
        """清空"""
        self._documents.clear()
        self._bm25 = None
        self._bm25_dirty = True
        if self._db_path:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("DELETE FROM examples")

    @property
    def count(self) -> int:
        """示例数量"""
        return len(self._documents)
