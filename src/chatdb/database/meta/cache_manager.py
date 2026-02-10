"""
通用数据源缓存管理器

支持多种数据源（Excel、CSV等）的元数据缓存管理。
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCacheManager:
    """通用数据源缓存管理器"""

    def __init__(self, excel_meta_db_path: str = None):
        """
        初始化缓存管理器

        Args:
            excel_meta_db_path: Excel 元数据数据库路径，默认为 data/pilot/excel_meta_data.db
        """
        # Excel 元数据数据库路径（data/pilot/excel_meta_data.db）
        if excel_meta_db_path is None:
            # 从项目根目录计算 data/pilot 路径
            current_file = Path(__file__)
            # 假设项目结构：src/chatdb/database/meta/cache_manager.py
            # 需要回到项目根目录，然后到 data/pilot
            project_root = current_file.parent.parent.parent.parent.parent
            excel_meta_dir = project_root / "data" / "pilot"
            excel_meta_dir.mkdir(parents=True, exist_ok=True)
            excel_meta_db_path = excel_meta_dir / "excel_meta_data.db"
        
        self.excel_meta_db_path = Path(excel_meta_db_path)
        self.excel_meta_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_meta_db()

    def _init_meta_db(self):
        """初始化元数据数据库"""
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()
        
        try:
            # 统一的表元数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_meta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash VARCHAR(64) NOT NULL,
                    table_hash VARCHAR(64),
                    source_type VARCHAR(50) NOT NULL DEFAULT 'excel',
                    table_name VARCHAR(255) NOT NULL,
                    sheet_name VARCHAR(255),
                    file_name VARCHAR(255) NOT NULL,
                    db_name VARCHAR(255) NOT NULL,
                    db_path TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    columns_info TEXT,
                    schema_info TEXT NOT NULL,
                    table_description TEXT,
                    summary_prompt TEXT,
                    id_columns TEXT,
                    create_table_sql TEXT,
                    column_profiles TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    UNIQUE(file_hash, sheet_name)
                )
            """)
            
            # 向后兼容：添加可能缺失的列
            cursor.execute("PRAGMA table_info(table_meta)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            new_columns = [
                ("table_hash", "VARCHAR(64)"),
                ("columns_info", "TEXT"),
                ("summary_prompt", "TEXT"),
                ("source_type", "VARCHAR(50) DEFAULT 'excel'"),
                ("column_profiles", "TEXT"),
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE table_meta ADD COLUMN {col_name} {col_type}")
            
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def calculate_content_hash(df: pd.DataFrame, filename: str) -> str:
        """
        计算内容哈希值（基于DataFrame，用于向后兼容）

        Args:
            df: DataFrame 对象
            filename: 文件名（包含在哈希计算中）

        Returns:
            SHA256 哈希值
        """
        content_parts = [
            filename,
            ",".join(df.columns.tolist()),
            df.to_csv(index=False),
        ]
        content = "\n".join(content_parts)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_file_hash(file_path: str, extra_info: List[str] = None) -> str:
        """
        计算文件级别的哈希值（基于文件内容）

        Args:
            file_path: 文件路径
            extra_info: 额外信息列表（如sheet名称列表），用于区分不同的处理方式

        Returns:
            SHA256 哈希值
        """
        sha256_hash = hashlib.sha256()

        # 读取文件内容并计算哈希
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        # 如果指定了额外信息，将其也纳入哈希计算
        if extra_info:
            info_str = ",".join(sorted(extra_info))
            sha256_hash.update(info_str.encode("utf-8"))

        return sha256_hash.hexdigest()

    def get_cached_info(self, content_hash: str, source_type: str = "excel") -> Optional[Dict]:
        """根据内容哈希获取缓存信息（向后兼容方法）"""
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()
        
        # 使用 table_hash 字段查找
        cursor.execute(
            """
            SELECT 
                file_hash, table_hash, source_type, file_name, table_name, sheet_name,
                db_name, db_path, row_count, column_count, columns_info, 
                schema_info, summary_prompt, id_columns, created_at, last_accessed, access_count
            FROM table_meta
            WHERE table_hash = ? AND source_type = ?
            LIMIT 1
        """,
            (content_hash, source_type),
        )
        row = cursor.fetchone()

        if row:
            cursor.execute(
                """
                UPDATE table_meta
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE table_hash = ? AND source_type = ?
            """,
                (content_hash, source_type),
            )
            conn.commit()
            conn.close()
            return {
                "content_hash": row[1] or row[0],  # table_hash 或 file_hash
                "source_type": row[2],
                "original_filename": row[3],
                "table_name": row[4],
                "sheet_name": row[5],
                "db_name": row[6],
                "db_path": row[7],
                "row_count": row[8],
                "column_count": row[9],
                "columns_info": json.loads(row[10]) if row[10] else [],
                "data_schema_json": row[11],
                "summary_prompt": row[12],
                "id_columns": json.loads(row[13]) if row[13] else [],
                "created_at": row[14],
                "last_accessed": row[15],
                "access_count": row[16],
            }

        conn.close()
        return None

    def save_cache_info(
        self,
        content_hash: str,
        original_filename: str,
        table_name: str,
        db_name: str,
        db_path: str,
        df: pd.DataFrame,
        source_type: str = "excel",
        summary_prompt: str = None,
        data_schema_json: str = None,
        id_columns: List[str] = None,
    ):
        """
        保存缓存信息（向后兼容方法，单表模式）

        Args:
            content_hash: 内容哈希值（作为 table_hash）
            original_filename: 原始文件名
            table_name: 表名
            db_name: 数据库名
            db_path: 数据库路径
            df: DataFrame 对象
            source_type: 数据源类型（excel, csv等）
            summary_prompt: 数据理解提示词
            data_schema_json: 数据schema JSON
            id_columns: ID列名列表
        """
        columns_info = [
            {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
        ]
        
        # 使用 file_hash = content_hash（单表模式）
        self._save_to_table_meta(
            file_hash=content_hash,
            table_hash=content_hash,
            table_name=table_name,
            sheet_name=None,
            file_name=original_filename,
            db_name=db_name,
            db_path=db_path,
            row_count=len(df),
            column_count=len(df.columns),
            columns_info=columns_info,
            schema_info=data_schema_json or json.dumps({}, ensure_ascii=False),
            table_description=self._extract_table_description(data_schema_json),
            summary_prompt=summary_prompt,
            id_columns=id_columns or [],
            create_table_sql=None,
            source_type=source_type,
        )

    def update_summary_prompt(self, content_hash: str, summary_prompt: str, source_type: str = "excel"):
        """更新数据理解提示词"""
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE table_meta
            SET summary_prompt = ?, last_accessed = CURRENT_TIMESTAMP
            WHERE table_hash = ? AND source_type = ?
        """,
            (summary_prompt, content_hash, source_type),
        )

        conn.commit()
        conn.close()

    def delete_cache_by_hash(self, content_hash: str, source_type: str = "excel"):
        """根据内容哈希删除缓存"""
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM table_meta WHERE table_hash = ? AND source_type = ?",
            (content_hash, source_type),
        )
        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted > 0

    def delete_cache_by_filename(self, filename: str, source_type: str = "excel"):
        """根据文件名删除缓存"""
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM table_meta WHERE file_name = ? AND source_type = ?",
            (filename, source_type),
        )
        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted > 0

    def list_all_cache(self, source_type: str = None) -> List[Dict]:
        """
        列出所有缓存记录

        Args:
            source_type: 数据源类型过滤，None表示所有类型
        """
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()

        if source_type:
            cursor.execute("""
                SELECT 
                    file_hash, table_hash, source_type, file_name, table_name,
                    row_count, column_count, created_at, last_accessed, access_count
                FROM table_meta
                WHERE source_type = ?
                ORDER BY last_accessed DESC
            """, (source_type,))
        else:
            cursor.execute("""
                SELECT 
                    file_hash, table_hash, source_type, file_name, table_name,
                    row_count, column_count, created_at, last_accessed, access_count
                FROM table_meta
                ORDER BY last_accessed DESC
            """)

        records = cursor.fetchall()
        conn.close()

        result = []
        for row in records:
            result.append(
                {
                    "content_hash": row[1] or row[0],  # table_hash 或 file_hash
                    "source_type": row[2],
                    "original_filename": row[3],
                    "table_name": row[4],
                    "row_count": row[5],
                    "column_count": row[6],
                    "created_at": row[7],
                    "last_accessed": row[8],
                    "access_count": row[9],
                }
            )

        return result

    def get_tables_by_file_hash(self, file_hash: str, source_type: str = "excel") -> List[Dict]:
        """
        根据文件哈希获取所有表的缓存信息（多表模式）
        
        Args:
            file_hash: 文件哈希值
            source_type: 数据源类型
            
        Returns:
            表信息列表，包含 column_profiles（列元信息：唯一值数量、高频值、统计信息）
        """
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT 
                file_hash, table_hash, source_type, sheet_name, file_name, table_name,
                db_name, db_path, row_count, column_count, columns_info,
                schema_info, summary_prompt, id_columns, create_table_sql,
                created_at, last_accessed, access_count,
                column_profiles, table_description
            FROM table_meta
            WHERE file_hash = ? AND source_type = ?
            ORDER BY id
        """,
            (file_hash, source_type),
        )
        rows = cursor.fetchall()
        
        if rows:
            cursor.execute(
                """
                UPDATE table_meta
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE file_hash = ? AND source_type = ?
            """,
                (file_hash, source_type),
            )
            conn.commit()
        
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                "file_hash": row[0],
                "table_hash": row[1],
                "source_type": row[2],
                "sheet_name": row[3],
                "original_filename": row[4],
                "table_name": row[5],
                "db_name": row[6],
                "db_path": row[7],
                "row_count": row[8],
                "column_count": row[9],
                "columns_info": json.loads(row[10]) if row[10] else [],
                "data_schema_json": row[11],
                "summary_prompt": row[12],
                "id_columns": json.loads(row[13]) if row[13] else [],
                "create_table_sql": row[14],
                "created_at": row[15],
                "last_accessed": row[16],
                "access_count": row[17],
                # 新增：column_profiles 包含丰富的列元信息
                "column_profiles": json.loads(row[18]) if row[18] else [],
                "table_description": row[19],
            })
        
        return result

    def save_table_cache_info(
        self,
        file_hash: str,
        sheet_name: str,
        table_hash: str,
        original_filename: str,
        table_name: str,
        db_name: str,
        db_path: str,
        df: pd.DataFrame,
        source_type: str = "excel",
        summary_prompt: str = None,
        data_schema_json: str = None,
        id_columns: List[str] = None,
        create_table_sql: str = None,
    ):
        """
        保存单个表的缓存信息（多表模式）
        
        Args:
            file_hash: 文件哈希值
            sheet_name: sheet名称（对于非Excel可能是None）
            table_hash: 表哈希值
            original_filename: 原始文件名
            table_name: 表名
            db_name: 数据库名
            db_path: 数据库路径
            df: DataFrame 对象
            source_type: 数据源类型
            summary_prompt: 数据理解提示词
            data_schema_json: 数据schema JSON
            id_columns: ID列名列表
            create_table_sql: 建表SQL语句
        """
        columns_info = [
            {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
        ]
        
        self._save_to_table_meta(
            file_hash=file_hash,
            table_hash=table_hash,
            table_name=table_name,
            sheet_name=sheet_name,
            file_name=original_filename,
            db_name=db_name,
            db_path=db_path,
            row_count=len(df),
            column_count=len(df.columns),
            columns_info=columns_info,
            schema_info=data_schema_json or json.dumps({}, ensure_ascii=False),
            table_description=self._extract_table_description(data_schema_json),
            summary_prompt=summary_prompt,
            id_columns=id_columns or [],
            create_table_sql=create_table_sql,
            source_type=source_type,
        )
    
    def _extract_table_description(self, schema_json: str) -> Optional[str]:
        """从 schema JSON 中提取表描述"""
        if not schema_json:
            return None
        try:
            schema = json.loads(schema_json)
            return schema.get("table_description")
        except Exception:
            return None
    
    def _save_to_table_meta(
        self,
        file_hash: str,
        table_hash: Optional[str],
        table_name: str,
        sheet_name: Optional[str],
        file_name: str,
        db_name: str,
        db_path: str,
        row_count: int,
        column_count: int,
        columns_info: List[Dict],
        schema_info: str,
        table_description: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        id_columns: List[str] = None,
        create_table_sql: Optional[str] = None,
        source_type: str = "excel",
    ):
        """
        保存到统一的表元数据表（table_meta）
        
        Args:
            file_hash: 文件哈希值
            table_hash: 表哈希值
            table_name: 表名
            sheet_name: sheet名称
            file_name: 文件名
            db_name: 数据库名
            db_path: 数据库路径
            row_count: 行数
            column_count: 列数
            columns_info: 列信息列表
            schema_info: Schema JSON 字符串
            table_description: 表描述
            summary_prompt: 提示词
            id_columns: ID列列表
            create_table_sql: 建表SQL
            source_type: 数据源类型
        """
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()
        
        id_columns_json = json.dumps(id_columns if id_columns else [], ensure_ascii=False)
        columns_info_json = json.dumps(columns_info, ensure_ascii=False)
        
        # 检查记录是否已存在
        cursor.execute("""
            SELECT access_count FROM table_meta
            WHERE file_hash = ? AND sheet_name = ?
        """, (file_hash, sheet_name))
        
        existing = cursor.fetchone()
        if existing:
            # 更新现有记录
            cursor.execute("""
                UPDATE table_meta
                SET table_hash = ?, source_type = ?, table_name = ?, file_name = ?, 
                    db_name = ?, db_path = ?, row_count = ?, column_count = ?,
                    columns_info = ?, schema_info = ?, table_description = ?,
                    summary_prompt = ?, id_columns = ?, create_table_sql = ?,
                    last_accessed = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE file_hash = ? AND sheet_name = ?
            """, (
                table_hash, source_type, table_name, file_name, db_name, db_path,
                row_count, column_count, columns_info_json, schema_info,
                table_description, summary_prompt, id_columns_json, create_table_sql,
                file_hash, sheet_name
            ))
        else:
            # 插入新记录
            cursor.execute("""
                INSERT INTO table_meta
                (file_hash, table_hash, source_type, table_name, sheet_name, file_name, 
                 db_name, db_path, row_count, column_count, columns_info, schema_info,
                 table_description, summary_prompt, id_columns, create_table_sql,
                 created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            """, (
                file_hash, table_hash, source_type, table_name, sheet_name, file_name,
                db_name, db_path, row_count, column_count, columns_info_json, schema_info,
                table_description, summary_prompt, id_columns_json, create_table_sql
            ))
        
        conn.commit()
        conn.close()

    def delete_tables_by_file_hash(self, file_hash: str, source_type: str = "excel") -> int:
        """
        根据文件哈希删除所有相关表的缓存
        
        Args:
            file_hash: 文件哈希值
            source_type: 数据源类型
            
        Returns:
            删除的记录数
        """
        conn = sqlite3.connect(str(self.excel_meta_db_path))
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM table_meta WHERE file_hash = ? AND source_type = ?",
            (file_hash, source_type),
        )
        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted
