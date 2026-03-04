"""
chatdb.utils.config — ChatDB 配置管理

通用配置类已迁移到 lib.utils.config，此处：
1. re-export 所有通用符号（保持旧导入路径兼容）
2. 定义 ChatDB 特有的 DatabaseSettings
3. 继承 BaseAppSettings 构建含 database 段的 Settings
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# re-export 公共配置
from lib.utils.config import (  # noqa: F401
    LLMSettings,
    APISettings,
    LogSettings,
    BaseAppSettings,
    _load_toml_config,
    _apply_toml_to_settings,
)


class DatabaseSettings(BaseSettings):
    """数据库配置（ChatDB 特有）"""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")
    postgres_database: str = Field(default="chatdb")

    # MySQL
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="root")
    mysql_password: str = Field(default="root")
    mysql_database: str = Field(default="chatdb")

    # SQLite
    sqlite_database: str = Field(default="./data/chatdb.db")

    # DuckDB
    duckdb_database: str = Field(default="", description="DuckDB 数据库路径（空则使用内存数据库）")

    # 默认数据库类型
    default_db_type: Literal["postgresql", "mysql", "sqlite", "duckdb", "excel"] = Field(
        default="postgresql", description="默认数据库类型"
    )

    def get_connection_url(self, db_type: str | None = None) -> str:
        """获取数据库连接 URL"""
        db_type = db_type or self.default_db_type

        if db_type == "postgresql":
            return (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            )
        elif db_type == "mysql":
            return (
                f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            )
        elif db_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_database}"
        elif db_type == "duckdb":
            if self.duckdb_database:
                return f"duckdb:///{self.duckdb_database}"
            else:
                return "duckdb:///:memory:"
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")


class Settings(BaseAppSettings):
    """ChatDB 应用主配置（含 database 段）"""

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


@lru_cache
def get_settings() -> Settings:
    """获取 ChatDB 应用配置（单例模式）"""
    toml_config = _load_toml_config()
    _settings = Settings()
    _apply_toml_to_settings(_settings, toml_config)

    # 数据库配置（ChatDB 特有）
    if "database" in toml_config:
        db_config = toml_config["database"]
        for key in ["postgres_host", "postgres_port", "postgres_user", "postgres_password",
                     "postgres_database", "mysql_host", "mysql_port", "mysql_user",
                     "mysql_password", "mysql_database", "sqlite_database", "duckdb_database"]:
            if key in db_config:
                setattr(_settings.database, key, db_config[key])
        if db_config.get("default_db_type"):
            _settings.database.default_db_type = db_config["default_db_type"]

    return _settings


# 全局配置实例
settings = get_settings()
