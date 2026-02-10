"""
表元数据存储管理

管理所有数据源（Excel、CSV等）的表结构元数据。
数据存储在 data/pilot/meta_data.db
"""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)


def _get_default_db_path() -> Path:
    """获取默认的元数据数据库路径"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    db_dir = project_root / "data" / "pilot"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "meta_data.db"


class MetaDataStore:
    """表元数据存储"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _get_default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """数据库连接上下文"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """初始化数据库表"""
        with self._conn() as conn:
            conn.execute("""
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
                    ddl_schema TEXT,
                    light_schema TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    UNIQUE(file_hash, sheet_name)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_table_meta_hash ON table_meta(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_table_meta_table ON table_meta(table_hash)")
            
            # 检查并添加新列（兼容已存在的表）
            cursor = conn.execute("PRAGMA table_info(table_meta)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            for col, col_type in [("column_profiles", "TEXT"), ("ddl_schema", "TEXT"), ("light_schema", "TEXT")]:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE table_meta ADD COLUMN {col} {col_type}")

    # ============ 哈希计算 ============

    @staticmethod
    def calc_file_hash(file_path: str | Path, extra_info: list[str] | None = None) -> str:
        """计算文件哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        if extra_info:
            sha256.update(",".join(sorted(extra_info)).encode())
        return sha256.hexdigest()

    @staticmethod
    def calc_content_hash(df: pd.DataFrame, filename: str) -> str:
        """计算 DataFrame 内容哈希（向后兼容）"""
        content = "\n".join([filename, ",".join(df.columns.tolist()), df.to_csv(index=False)])
        return hashlib.sha256(content.encode()).hexdigest()

    # ============ 查询方法 ============

    def get_by_hash(self, content_hash: str, source_type: str = "excel") -> dict[str, Any] | None:
        """根据哈希获取表信息"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM table_meta WHERE (table_hash = ? OR file_hash = ?) AND source_type = ? LIMIT 1",
                (content_hash, content_hash, source_type)
            ).fetchone()
            
            if row:
                conn.execute(
                    "UPDATE table_meta SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                    (row["id"],)
                )
                return self._row_to_dict(row)
        return None

    def get_by_file_hash(self, file_hash: str, source_type: str = "excel") -> list[dict[str, Any]]:
        """根据文件哈希获取所有表"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM table_meta WHERE file_hash = ? AND source_type = ? ORDER BY id",
                (file_hash, source_type)
            ).fetchall()
            
            if rows:
                conn.execute(
                    "UPDATE table_meta SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE file_hash = ? AND source_type = ?",
                    (file_hash, source_type)
                )
            return [self._row_to_dict(r) for r in rows]

    def get_by_table_name(self, table_name: str, source_type: str | None = None) -> dict[str, Any] | None:
        """根据表名获取表信息，优先返回 column_profiles 最丰富的记录"""
        with self._conn() as conn:
            if source_type:
                row = conn.execute(
                    """SELECT * FROM table_meta 
                       WHERE table_name = ? AND source_type = ? 
                       ORDER BY length(column_profiles) DESC, last_accessed DESC 
                       LIMIT 1""",
                    (table_name, source_type)
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT * FROM table_meta 
                       WHERE table_name = ? 
                       ORDER BY length(column_profiles) DESC, last_accessed DESC 
                       LIMIT 1""",
                    (table_name,)
                ).fetchone()
            
            if row:
                conn.execute(
                    "UPDATE table_meta SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                    (row["id"],)
                )
                return self._row_to_dict(row)
        return None

    def list_all(self, source_type: str | None = None) -> list[dict[str, Any]]:
        """列出所有表"""
        with self._conn() as conn:
            if source_type:
                rows = conn.execute(
                    "SELECT * FROM table_meta WHERE source_type = ? ORDER BY last_accessed DESC",
                    (source_type,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM table_meta ORDER BY last_accessed DESC").fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ============ 保存方法 ============

    def save(
        self,
        file_hash: str,
        table_name: str,
        file_name: str,
        db_name: str,
        db_path: str,
        row_count: int,
        column_count: int,
        schema_info: str,
        source_type: str = "excel",
        table_hash: str | None = None,
        sheet_name: str | None = None,
        columns_info: list[dict] | None = None,
        table_description: str | None = None,
        summary_prompt: str | None = None,
        id_columns: list[str] | None = None,
        create_table_sql: str | None = None,
        column_profiles: list[dict] | None = None,
        ddl_schema: str | None = None,
        light_schema: str | None = None,
    ) -> None:
        """保存表元数据"""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM table_meta WHERE file_hash = ? AND sheet_name IS ?",
                (file_hash, sheet_name)
            ).fetchone()

            columns_json = json.dumps(columns_info or [], ensure_ascii=False)
            id_columns_json = json.dumps(id_columns or [], ensure_ascii=False)
            profiles_json = json.dumps(column_profiles or [], ensure_ascii=False)

            if existing:
                conn.execute("""
                    UPDATE table_meta SET
                        table_hash = ?, source_type = ?, table_name = ?, file_name = ?,
                        db_name = ?, db_path = ?, row_count = ?, column_count = ?,
                        columns_info = ?, schema_info = ?, table_description = ?,
                        summary_prompt = ?, id_columns = ?, create_table_sql = ?,
                        column_profiles = ?, ddl_schema = ?, light_schema = ?,
                        last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE id = ?
                """, (
                    table_hash, source_type, table_name, file_name, db_name, db_path,
                    row_count, column_count, columns_json, schema_info, table_description,
                    summary_prompt, id_columns_json, create_table_sql, profiles_json,
                    ddl_schema, light_schema, existing["id"]
                ))
            else:
                conn.execute("""
                    INSERT INTO table_meta (
                        file_hash, table_hash, source_type, table_name, sheet_name, file_name,
                        db_name, db_path, row_count, column_count, columns_info, schema_info,
                        table_description, summary_prompt, id_columns, create_table_sql,
                        column_profiles, ddl_schema, light_schema
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_hash, table_hash, source_type, table_name, sheet_name, file_name,
                    db_name, db_path, row_count, column_count, columns_json, schema_info,
                    table_description, summary_prompt, id_columns_json, create_table_sql,
                    profiles_json, ddl_schema, light_schema
                ))

    def save_from_df(
        self,
        content_hash: str,
        original_filename: str,
        table_name: str,
        db_name: str,
        db_path: str,
        df: pd.DataFrame,
        source_type: str = "excel",
        **kwargs
    ) -> None:
        """从 DataFrame 保存（向后兼容方法）"""
        columns_info = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
        self.save(
            file_hash=content_hash,
            table_hash=content_hash,
            table_name=table_name,
            file_name=original_filename,
            db_name=db_name,
            db_path=db_path,
            row_count=len(df),
            column_count=len(df.columns),
            schema_info=kwargs.get("data_schema_json", "{}"),
            source_type=source_type,
            columns_info=columns_info,
            **{k: v for k, v in kwargs.items() if k != "data_schema_json"}
        )

    # ============ 更新/删除 ============

    def update_summary(self, content_hash: str, summary_prompt: str, source_type: str = "excel") -> None:
        """更新数据理解提示词"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE table_meta SET summary_prompt = ?, last_accessed = CURRENT_TIMESTAMP WHERE (table_hash = ? OR file_hash = ?) AND source_type = ?",
                (summary_prompt, content_hash, content_hash, source_type)
            )

    def delete_by_hash(self, content_hash: str, source_type: str = "excel") -> bool:
        """根据哈希删除"""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM table_meta WHERE (table_hash = ? OR file_hash = ?) AND source_type = ?",
                (content_hash, content_hash, source_type)
            )
            return cursor.rowcount > 0

    def delete_by_file_hash(self, file_hash: str, source_type: str = "excel") -> int:
        """根据文件哈希删除所有相关表"""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM table_meta WHERE file_hash = ? AND source_type = ?",
                (file_hash, source_type)
            )
            return cursor.rowcount

    def delete_by_filename(self, filename: str, source_type: str = "excel") -> bool:
        """根据文件名删除"""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM table_meta WHERE file_name = ? AND source_type = ?",
                (filename, source_type)
            )
            return cursor.rowcount > 0

    # ============ 辅助方法 ============

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """行转字典"""
        return {
            "file_hash": row["file_hash"],
            "table_hash": row["table_hash"],
            "content_hash": row["table_hash"] or row["file_hash"],  # 向后兼容
            "source_type": row["source_type"],
            "table_name": row["table_name"],
            "sheet_name": row["sheet_name"],
            "original_filename": row["file_name"],
            "db_name": row["db_name"],
            "db_path": row["db_path"],
            "row_count": row["row_count"],
            "column_count": row["column_count"],
            "columns_info": json.loads(row["columns_info"]) if row["columns_info"] else [],
            "data_schema_json": row["schema_info"],
            "table_description": row["table_description"],
            "summary_prompt": row["summary_prompt"],
            "id_columns": json.loads(row["id_columns"]) if row["id_columns"] else [],
            "create_table_sql": row["create_table_sql"],
            "column_profiles": json.loads(row["column_profiles"]) if row["column_profiles"] else [],
            "ddl_schema": row["ddl_schema"],
            "light_schema": row["light_schema"],
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
        }


# 向后兼容别名
DataCacheManager = MetaDataStore
