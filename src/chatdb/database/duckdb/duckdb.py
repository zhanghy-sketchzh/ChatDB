"""
DuckDB 数据库连接器

DuckDB 不支持异步驱动，使用同步引擎包装为异步接口。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from chatdb.utils.config import settings
from chatdb.utils.exceptions import ConnectionError, QueryExecutionError
from chatdb.utils.logger import logger


class DuckDBConnector:
    """DuckDB 数据库连接器（同步引擎，异步接口包装）"""

    def __init__(
        self,
        database: str | None = None,
        connection_url: str | None = None,
    ):
        """
        初始化 DuckDB 连接器

        Args:
            database: 数据库文件路径（如果为 None 则使用内存数据库）
            connection_url: 完整的连接 URL（如果提供则忽略其他参数）
        """
        self._database = database
        if connection_url:
            self._connection_url = connection_url
        elif database:
            self._connection_url = f"duckdb:///{database}"
        else:
            db_path = getattr(settings.database, 'duckdb_database', None)
            self._connection_url = f"duckdb:///{db_path}" if db_path else "duckdb:///:memory:"
        
        self._engine: Engine | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def engine(self) -> Engine:
        """获取数据库引擎"""
        if self._engine is None:
            raise ConnectionError("数据库未连接，请先调用 connect() 方法")
        return self._engine

    @property
    def connection_url(self) -> str:
        """获取连接 URL"""
        return self._connection_url

    async def connect(self) -> None:
        """建立数据库连接"""
        try:
            self._engine = create_engine(
                self._connection_url,
                echo=settings.api.debug,
            )
            # 测试连接
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"[DuckDBConnector] 数据库连接成功: {self._connection_url}")
        except Exception as e:
            logger.error(f"[DuckDBConnector] 数据库连接失败: {e}")
            raise ConnectionError(f"数据库连接失败: {e}")

    async def disconnect(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("[DuckDBConnector] 数据库连接已关闭")
        self._executor.shutdown(wait=False)

    def _execute_sync(self, sql: str, params: dict | None = None) -> list[dict[str, Any]]:
        """同步执行查询"""
        # DuckDB 通过 SQLAlchemy 执行时不需要末尾分号，会导致解析错误
        sql = sql.rstrip().rstrip(';')
        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]

    async def execute_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        fetch: bool = True,
    ) -> list[dict[str, Any]]:
        """异步执行 SQL 查询"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, self._execute_sync, sql, params
            )
        except Exception as e:
            logger.error(f"[DuckDBConnector] SQL 执行失败: {sql[:100]}... | 错误: {e}")
            raise QueryExecutionError(f"SQL 执行失败: {e}", {"sql": sql})

    def get_tables_meta(self) -> list[dict[str, Any]]:
        """
        获取数据库中所有表的元数据
        
        Returns:
            表元数据列表，每个元素包含：
            - table_name: 表名
            - row_count: 行数
            - column_count: 列数
            - columns_info: 列信息列表
            - create_table_sql: 建表 SQL
            - table_description: 表描述
            - column_profiles: 列元信息（唯一值数量、高频值、统计信息）
        """
        if self._engine is None:
            raise ConnectionError("数据库未连接，请先调用 connect() 方法")
        
        # 尝试从 MetaDataStore 获取丰富的元数据
        meta_store = self._get_meta_store()
        
        tables_meta = []
        
        with self._engine.connect() as conn:
            # 获取所有表名
            tables_result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in tables_result.fetchall()]
            
            for table_name in tables:
                # 获取表结构
                columns_result = conn.execute(text(f'DESCRIBE "{table_name}"'))
                columns_data = columns_result.fetchall()
                columns_info = [
                    {"name": col[0], "type": col[1]}
                    for col in columns_data
                ]
                
                # 获取行数
                count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                count_row = count_result.fetchone()
                row_count = count_row[0] if count_row else 0
                
                # 生成建表 SQL
                columns_sql = [f'    "{col[0]}" {col[1]}' for col in columns_data]
                create_table_sql = f'CREATE TABLE "{table_name}" (\n' + ",\n".join(columns_sql) + "\n);"
                
                table_meta = {
                    "table_name": table_name,
                    "row_count": row_count,
                    "column_count": len(columns_info),
                    "columns_info": columns_info,
                    "create_table_sql": create_table_sql,
                    "table_description": f"表 {table_name}，包含 {row_count} 行，{len(columns_info)} 列",
                }
                
                # 从 MetaDataStore 获取 column_profiles
                if meta_store:
                    stored_meta = meta_store.get_by_table_name(table_name)
                    if stored_meta:
                        column_profiles = stored_meta.get("column_profiles", [])
                        if column_profiles:
                            table_meta["column_profiles"] = column_profiles
                        # 使用存储的描述（如果有）
                        stored_desc = stored_meta.get("table_description")
                        if stored_desc:
                            table_meta["table_description"] = stored_desc
                
                tables_meta.append(table_meta)
        
        return tables_meta
    
    def _get_meta_store(self):
        """获取 MetaDataStore 实例（懒加载）"""
        try:
            from chatdb.storage.meta_data import MetaDataStore
            # 使用默认路径 data/pilot/meta_data.db
            return MetaDataStore()
        except Exception:
            return None

    async def get_tables_meta_async(self) -> list[dict[str, Any]]:
        """异步获取数据库中所有表的元数据"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_tables_meta)

    async def __aenter__(self) -> "DuckDBConnector":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

