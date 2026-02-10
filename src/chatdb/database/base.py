"""
数据库连接器基类和工厂函数

定义通用接口和共同方法，所有具体连接器都继承此类。
同时提供工厂函数用于创建连接器实例。
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from chatdb.utils.config import settings
from chatdb.utils.exceptions import ConnectionError, QueryExecutionError
from chatdb.utils.logger import logger


class BaseDatabaseConnector(ABC):
    """数据库连接器基类 - 定义通用接口和共同方法"""

    def __init__(self, connection_url: str | None = None):
        """
        初始化数据库连接器

        Args:
            connection_url: 数据库连接 URL
        """
        self._connection_url = connection_url or self._get_default_connection_url()
        self._engine: AsyncEngine | None = None
        self._session_factory: sessionmaker | None = None

    @abstractmethod
    def _get_default_connection_url(self) -> str:
        """获取默认连接 URL（子类必须实现）"""
        pass

    @property
    def engine(self) -> AsyncEngine:
        """获取数据库引擎"""
        if self._engine is None:
            raise ConnectionError("数据库未连接，请先调用 connect() 方法")
        return self._engine

    @property
    def connection_url(self) -> str:
        """获取连接 URL（隐藏敏感信息）"""
        return self._mask_url(self._connection_url)

    async def connect(self) -> None:
        """建立数据库连接"""
        try:
            self._engine = create_async_engine(
                self._connection_url,
                echo=settings.api.debug,
                pool_pre_ping=True,
                pool_size=self._get_pool_size(),
                max_overflow=self._get_max_overflow(),
            )

            self._session_factory = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # 测试连接
            await self._test_connection()

            logger.info(f"[{self.__class__.__name__}] 数据库连接成功: {self.connection_url}")

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] 数据库连接失败: {e}")
            raise ConnectionError(f"数据库连接失败: {e}")

    async def _test_connection(self) -> None:
        """测试数据库连接（子类可重写）"""
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    def _get_pool_size(self) -> int:
        """获取连接池大小（子类可重写）"""
        return 5

    def _get_max_overflow(self) -> int:
        """获取最大溢出连接数（子类可重写）"""
        return 10

    async def disconnect(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info(f"[{self.__class__.__name__}] 数据库连接已关闭")

    async def execute_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        fetch: bool = True,
    ) -> list[dict[str, Any]]:
        """
        执行 SQL 查询

        Args:
            sql: SQL 查询语句
            params: 查询参数
            fetch: 是否获取结果

        Returns:
            查询结果列表
        """
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text(sql), params or {})

                if fetch:
                    rows = result.fetchall()
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]

                await conn.commit()
                return []

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] SQL 执行失败: {sql[:100]}... | 错误: {e}")
            raise QueryExecutionError(f"SQL 执行失败: {e}", {"sql": sql})

    def get_session(self) -> AsyncSession:
        """获取数据库会话"""
        if self._session_factory is None:
            raise ConnectionError("数据库未连接，请先调用 connect() 方法")
        return self._session_factory()

    @staticmethod
    def _mask_url(url: str) -> str:
        """隐藏连接 URL 中的敏感信息"""
        import re

        return re.sub(r"://[^@]+@", "://***:***@", url)

    @staticmethod
    def read_excel_file(
        excel_file_path: str, sheet_name=None, header=None
    ) -> Any:
        """
        智能读取 Excel 文件，支持 .xls 和 .xlsx 格式（公共方法）

        Args:
            excel_file_path: Excel 文件路径
            sheet_name: sheet 名称
            header: 表头行索引

        Returns:
            DataFrame
        """
        import pandas as pd
        from pathlib import Path

        file_ext = Path(excel_file_path).suffix.lower()

        try:
            if file_ext == ".xls":
                # 使用 xlrd 引擎读取旧版 .xls 文件
                logger.info(f"检测到 .xls 格式，使用 xlrd 引擎读取")
                return pd.read_excel(
                    excel_file_path, sheet_name=sheet_name, header=header, engine="xlrd"
                )
            else:
                # 使用默认的 openpyxl 引擎读取 .xlsx 文件
                return pd.read_excel(
                    excel_file_path, sheet_name=sheet_name, header=header
                )
        except Exception as e:
            logger.error(f"读取 Excel 文件失败: {e}")
            # 如果默认方式失败，尝试另一种引擎
            try:
                if file_ext == ".xls":
                    logger.warning(f"xlrd 读取失败，尝试使用 openpyxl")
                    return pd.read_excel(
                        excel_file_path,
                        sheet_name=sheet_name,
                        header=header,
                        engine="openpyxl",
                    )
                else:
                    logger.warning(f"openpyxl 读取失败，尝试使用 xlrd")
                    return pd.read_excel(
                        excel_file_path,
                        sheet_name=sheet_name,
                        header=header,
                        engine="xlrd",
                    )
            except Exception as e2:
                logger.error(f"所有引擎都无法读取文件: {e2}")
                raise Exception(f"无法读取 Excel 文件 {excel_file_path}: {e2}")

    async def __aenter__(self) -> "BaseDatabaseConnector":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    @classmethod
    def create(
        cls,
        db_type: Literal["postgresql", "mysql", "sqlite", "duckdb", "excel"] | None = None,
        connection_url: str | None = None,
        **kwargs,
    ) -> "BaseDatabaseConnector":
        """
        创建数据库连接器实例（类方法）

        Args:
            db_type: 数据库类型
            connection_url: 连接 URL（如果提供则直接使用）
            **kwargs: 其他连接参数（host, port, user, password, database 等）

        Returns:
            数据库连接器实例

        Examples:
            >>> # 使用默认配置
            >>> connector = BaseDatabaseConnector.create()
            >>> 
            >>> # 指定数据库类型
            >>> connector = BaseDatabaseConnector.create(db_type="mysql")
            >>> 
            >>> # 使用连接 URL
            >>> connector = BaseDatabaseConnector.create(connection_url="postgresql+asyncpg://...")
            >>> 
            >>> # 指定连接参数
            >>> connector = BaseDatabaseConnector.create(
            ...     db_type="postgresql",
            ...     host="localhost",
            ...     port=5432,
            ...     user="postgres",
            ...     password="password",
            ...     database="mydb"
            ... )
            >>> 
            >>> # Excel 文件
            >>> connector = BaseDatabaseConnector.create(
            ...     db_type="excel",
            ...     file_path="/path/to/file.xlsx",
            ...     sheet_name="Sheet1"
            ... )
        """
        # 延迟导入避免循环依赖
        from chatdb.database.duckdb import DuckDBConnector
        from chatdb.database.excel.register import ExcelConnector
        from chatdb.database.mysql import MySQLConnector
        from chatdb.database.postgresql import PostgreSQLConnector
        from chatdb.database.sqlite import SQLiteConnector

        # 如果提供了连接 URL，尝试从 URL 推断类型
        if connection_url:
            if connection_url.startswith("postgresql"):
                return PostgreSQLConnector(connection_url=connection_url)
            elif connection_url.startswith("mysql"):
                return MySQLConnector(connection_url=connection_url)
            elif connection_url.startswith("sqlite"):
                return SQLiteConnector(connection_url=connection_url)
            elif connection_url.startswith("duckdb"):
                return DuckDBConnector(connection_url=connection_url)
            else:
                raise ValueError(f"无法从连接 URL 推断数据库类型: {connection_url}")

        # 使用指定的数据库类型或默认类型
        db_type = db_type or settings.database.default_db_type

        if db_type == "postgresql":
            return PostgreSQLConnector(**kwargs)
        elif db_type == "mysql":
            return MySQLConnector(**kwargs)
        elif db_type == "sqlite":
            return SQLiteConnector(**kwargs)
        elif db_type == "duckdb":
            return DuckDBConnector(**kwargs)
        elif db_type == "excel":
            if "file_path" not in kwargs:
                raise ValueError("Excel 连接器需要 file_path 参数")
            return ExcelConnector(**kwargs)
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")


# ==================== 向后兼容函数 ====================

def create_connector(
    db_type: Literal["postgresql", "mysql", "sqlite", "duckdb", "excel"] | None = None,
    connection_url: str | None = None,
    **kwargs,
) -> BaseDatabaseConnector:
    """创建数据库连接器实例（向后兼容函数）"""
    return BaseDatabaseConnector.create(db_type=db_type, connection_url=connection_url, **kwargs)


def DatabaseConnector(
    connection_url: str | None = None,
    db_type: str | None = None,
) -> BaseDatabaseConnector:
    """数据库连接器工厂函数（向后兼容）"""
    return BaseDatabaseConnector.create(db_type=db_type, connection_url=connection_url)

