"""
文本索引模块

提供 BM25 等关键字索引，用于快速锁定相关表/列/值。
支持「稠密 + 稀疏」双检索通道。

关键词提取流程:
1. 用户查询 → jieba 分词 → 关键词列表
2. 关键词 → BM25 搜索 → 匹配 value 类型文档
3. value 文档来自分类字段的 unique_values
4. 通过匹配的 value 定位相关字段和表
"""

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jieba
import jieba.analyse


@dataclass
class IndexDocument:
    """索引文档"""
    doc_id: str
    doc_type: str  # table, column, value
    table_name: str
    column_name: str | None = None
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果"""
    doc_id: str
    doc_type: str
    table_name: str
    column_name: str | None
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass
class KeywordMatch:
    """关键词匹配结果"""
    keyword: str
    matched_value: str
    table_name: str
    column_name: str
    score: float


class BM25Index:
    """
    BM25 关键字索引
    
    支持中英文分词，用于快速检索相关表/列/值
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        
        # 文档存储
        self.documents: dict[str, IndexDocument] = {}
        
        # 倒排索引: term -> {doc_id: term_freq}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        
        # 文档长度
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length: float = 0
        
        # 文档频率
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
    
    def add_document(self, doc: IndexDocument) -> None:
        """添加文档到索引"""
        self.documents[doc.doc_id] = doc
        
        # 分词
        tokens = self._tokenize(doc.content)
        self.doc_lengths[doc.doc_id] = len(tokens)
        
        # 更新倒排索引
        term_freq: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        
        for term, freq in term_freq.items():
            self.inverted_index[term][doc.doc_id] = freq
            self.doc_freq[term] += 1
        
        self.total_docs += 1
        self._update_avg_length()
    
    def add_documents(self, docs: list[IndexDocument]) -> None:
        """批量添加文档"""
        for doc in docs:
            self.add_document(doc)
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            搜索结果列表
        """
        query_tokens = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            # 计算 IDF
            df = self.doc_freq[token]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            
            # 计算每个文档的 BM25 分数
            for doc_id, tf in self.inverted_index[token].items():
                doc_len = self.doc_lengths[doc_id]
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_id] += idf * (numerator / denominator)
        
        # 排序并返回
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                doc_type=doc.doc_type,
                table_name=doc.table_name,
                column_name=doc.column_name,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
            ))
        
        return results
    
    def search_tables(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """只搜索表级文档"""
        results = self.search(query, top_k * 3)
        table_results = [r for r in results if r.doc_type == "table"]
        return table_results[:top_k]
    
    def search_columns(self, query: str, table_name: str | None = None, top_k: int = 10) -> list[SearchResult]:
        """搜索列级文档"""
        results = self.search(query, top_k * 3)
        column_results = [r for r in results if r.doc_type == "column"]
        if table_name:
            column_results = [r for r in column_results if r.table_name == table_name]
        return column_results[:top_k]
    
    def search_values(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """搜索值级文档"""
        results = self.search(query, top_k * 3)
        value_results = [r for r in results if r.doc_type == "value"]
        return value_results[:top_k]
    
    def extract_keywords(self, query: str, top_k: int = 10) -> list[str]:
        """
        从查询中提取关键词
        
        使用 jieba 的 TF-IDF 提取关键词
        """
        keywords = jieba.analyse.extract_tags(query, topK=top_k)
        return keywords
    
    def match_keywords_to_values(
        self, 
        query: str, 
        top_k_keywords: int = 5,
        top_k_values: int = 10
    ) -> list[KeywordMatch]:
        """
        关键词匹配分类字段值
        
        流程:
        1. 提取查询关键词
        2. 每个关键词用 BM25 搜索 value 类型文档
        3. 返回匹配结果（关键词 → 字段值 → 表.列）
        
        Args:
            query: 用户查询
            top_k_keywords: 提取的关键词数量
            top_k_values: 每个关键词匹配的值数量
            
        Returns:
            KeywordMatch 列表
        """
        keywords = self.extract_keywords(query, top_k_keywords)
        matches: list[KeywordMatch] = []
        
        for kw in keywords:
            results = self.search_values(kw, top_k_values)
            for r in results:
                matches.append(KeywordMatch(
                    keyword=kw,
                    matched_value=r.content,
                    table_name=r.table_name,
                    column_name=r.column_name or "",
                    score=r.score,
                ))
        
        # 按分数排序
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches
    
    def find_relevant_columns(self, query: str, top_k: int = 10) -> dict[str, list[str]]:
        """
        根据查询找到相关的表和列
        
        Returns:
            {table_name: [column_name, ...]}
        """
        matches = self.match_keywords_to_values(query)
        
        # 按 table.column 去重
        result: dict[str, set[str]] = defaultdict(set)
        for m in matches[:top_k]:
            if m.column_name:
                result[m.table_name].add(m.column_name)
        
        return {t: list(cols) for t, cols in result.items()}
    
    def _tokenize(self, text: str) -> list[str]:
        """中英文分词"""
        # 转小写
        text = text.lower()
        # jieba 分词
        tokens = list(jieba.cut(text))
        # 过滤停用词和空白
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
        return tokens
    
    def _update_avg_length(self) -> None:
        """更新平均文档长度"""
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def clear(self) -> None:
        """清空索引"""
        self.documents.clear()
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self.doc_freq.clear()
        self.total_docs = 0
        self.avg_doc_length = 0


class TextIndex:
    """
    持久化文本索引
    
    使用 SQLite 存储，支持增量更新
    """
    
    def __init__(self, db_path: str | Path | None = None):
        """
        Args:
            db_path: 索引数据库路径，默认为 data/pilot/text_index.db
        """
        if db_path is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            db_path = project_root / "data" / "pilot" / "text_index.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # 内存索引（懒加载）
        self._bm25: BM25Index | None = None
    
    def _init_db(self) -> None:
        """初始化数据库"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    doc_type TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    column_name TEXT,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_table ON documents(table_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type)")
    
    @property
    def bm25(self) -> BM25Index:
        """懒加载 BM25 索引"""
        if self._bm25 is None:
            self._bm25 = BM25Index()
            self._load_to_memory()
        return self._bm25
    
    def _load_to_memory(self) -> None:
        """加载到内存索引"""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
            for row in rows:
                doc = IndexDocument(
                    doc_id=row[0],
                    doc_type=row[1],
                    table_name=row[2],
                    column_name=row[3],
                    content=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                )
                self._bm25.add_document(doc)
    
    def add_document(self, doc: IndexDocument) -> None:
        """添加文档"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (doc.doc_id, doc.doc_type, doc.table_name, doc.column_name,
                 doc.content, json.dumps(doc.metadata, ensure_ascii=False))
            )
        
        # 更新内存索引
        if self._bm25 is not None:
            self._bm25.add_document(doc)
    
    def add_documents(self, docs: list[IndexDocument]) -> None:
        """批量添加文档"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                [(d.doc_id, d.doc_type, d.table_name, d.column_name,
                  d.content, json.dumps(d.metadata, ensure_ascii=False)) for d in docs]
            )
        
        # 重建内存索引
        self._bm25 = None
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """搜索"""
        return self.bm25.search(query, top_k)
    
    def search_tables(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """搜索表"""
        return self.bm25.search_tables(query, top_k)
    
    def search_columns(self, query: str, table_name: str | None = None, top_k: int = 10) -> list[SearchResult]:
        """搜索列"""
        return self.bm25.search_columns(query, table_name, top_k)
    
    def search_values(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """搜索值"""
        return self.bm25.search_values(query, top_k)
    
    def extract_keywords(self, query: str, top_k: int = 10) -> list[str]:
        """提取关键词"""
        return self.bm25.extract_keywords(query, top_k)
    
    def match_keywords_to_values(
        self, 
        query: str, 
        top_k_keywords: int = 5,
        top_k_values: int = 10
    ) -> list[KeywordMatch]:
        """关键词匹配分类字段值"""
        return self.bm25.match_keywords_to_values(query, top_k_keywords, top_k_values)
    
    def find_relevant_columns(self, query: str, top_k: int = 10) -> dict[str, list[str]]:
        """根据查询找到相关的表和列"""
        return self.bm25.find_relevant_columns(query, top_k)
    
    def delete_by_table(self, table_name: str) -> int:
        """删除表相关的所有文档"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM documents WHERE table_name = ?", (table_name,))
            deleted = cursor.rowcount
        
        # 重建内存索引
        self._bm25 = None
        return deleted
    
    def clear(self) -> None:
        """清空索引"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM documents")
        self._bm25 = None
