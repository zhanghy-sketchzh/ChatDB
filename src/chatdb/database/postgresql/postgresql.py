"""
PostgreSQL 数据库连接器
"""

from chatdb.utils.config import settings
from chatdb.database.base import BaseDatabaseConnector


class PostgreSQLConnector(BaseDatabaseConnector):
    """PostgreSQL 数据库连接器"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        connection_url: str | None = None,
    ):
        """
        初始化 PostgreSQL 连接器

        Args:
            host: 主机地址
            port: 端口
            user: 用户名
            password: 密码
            database: 数据库名
            connection_url: 完整的连接 URL（如果提供则忽略其他参数）
        """
        if connection_url:
            super().__init__(connection_url)
        else:
            self._host = host or settings.database.postgres_host
            self._port = port or settings.database.postgres_port
            self._user = user or settings.database.postgres_user
            self._password = password or settings.database.postgres_password
            self._database = database or settings.database.postgres_database
            super().__init__()

    def _get_default_connection_url(self) -> str:
        """获取 PostgreSQL 连接 URL"""
        return (
            f"postgresql+asyncpg://{self._user}:{self._password}"
            f"@{self._host}:{self._port}/{self._database}"
        )

