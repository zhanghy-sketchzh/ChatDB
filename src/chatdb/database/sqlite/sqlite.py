"""
SQLite 数据库连接器
"""

from chatdb.utils.config import settings
from chatdb.database.base import BaseDatabaseConnector


class SQLiteConnector(BaseDatabaseConnector):
    """SQLite 数据库连接器"""

    def __init__(
        self,
        database: str | None = None,
        connection_url: str | None = None,
    ):
        """
        初始化 SQLite 连接器

        Args:
            database: 数据库文件路径
            connection_url: 完整的连接 URL（如果提供则忽略其他参数）
        """
        if connection_url:
            super().__init__(connection_url)
        else:
            self._database = database or settings.database.sqlite_database
            super().__init__()

    def _get_default_connection_url(self) -> str:
        """获取 SQLite 连接 URL"""
        return f"sqlite+aiosqlite:///{self._database}"

    def _get_pool_size(self) -> int:
        """SQLite 不需要连接池"""
        return 1

    def _get_max_overflow(self) -> int:
        """SQLite 不需要溢出连接"""
        return 0

