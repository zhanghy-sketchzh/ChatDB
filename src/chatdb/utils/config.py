"""
应用配置管理

使用 pydantic-settings 管理配置，支持从环境变量、.env 文件和 config.toml 文件加载配置。
配置优先级：环境变量 > config.toml > 默认值
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 兼容 Python 3.11+ 的 tomllib
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # 需要安装: pip install tomli
    except ImportError:
        tomllib = None


class LLMSettings(BaseSettings):
    """LLM 配置"""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API Base URL"
    )
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI 模型名称")

    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")
    anthropic_model: str = Field(default="claude-3-opus-20240229", description="Claude 模型名称")

    # 混元 (Hunyuan)
    hunyuan_api_key: str = Field(default="", description="混元 API Key (Bearer Token)")
    hunyuan_api_base: str = Field(
        default="http://hunyuanapi.woa.com/openapi/v1/chat/completions",
        description="混元 API Base URL",
    )
    hunyuan_model: str = Field(default="hunyuan-t1-latest", description="混元模型名称")

    # 默认提供商
    default_llm_provider: Literal["openai", "anthropic", "hunyuan"] = Field(
        default="openai", description="默认 LLM 提供商"
    )

    def get_hunyuan_params(self) -> dict:
        """获取混元 LLM 的参数字典"""
        return {
            "model": self.hunyuan_model,
            "api_key": self.hunyuan_api_key,
            "api_base": self.hunyuan_api_base,
            "enable_enhancement": False,
            "sensitive_business": True,
        }


class DatabaseSettings(BaseSettings):
    """数据库配置"""

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


class APISettings(BaseSettings):
    """API 服务配置"""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)


class LogSettings(BaseSettings):
    """日志配置"""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: str = Field(default="INFO")
    file: str = Field(default="./log/chatdb.log")


class Settings(BaseSettings):
    """应用主配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 支持从 pyproject.toml 读取配置（通过环境变量）
        extra="ignore",
    )

    # 子配置
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    log: LogSettings = Field(default_factory=LogSettings)


def _load_toml_config() -> dict:
    """从 config.toml 文件加载配置"""
    if tomllib is None:
        return {}
    
    config_path = Path("config.toml")
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        # 如果 TOML 解析失败，返回空字典
        return {}


@lru_cache
def get_settings() -> Settings:
    """获取应用配置（单例模式）"""
    # 先加载 TOML 配置
    toml_config = _load_toml_config()
    
    # 创建配置实例（会自动从环境变量和 .env 文件读取）
    settings = Settings()
    
    # 如果 TOML 中有配置，且环境变量中没有设置，则使用 TOML 中的值
    if toml_config:
        # LLM 配置
        if "llm" in toml_config:
            llm_config = toml_config["llm"]
            if not settings.llm.openai_api_key and llm_config.get("openai_api_key"):
                settings.llm.openai_api_key = llm_config["openai_api_key"]
            if not settings.llm.openai_api_base and llm_config.get("openai_api_base"):
                settings.llm.openai_api_base = llm_config["openai_api_base"]
            if not settings.llm.openai_model and llm_config.get("openai_model"):
                settings.llm.openai_model = llm_config["openai_model"]
            
            if not settings.llm.anthropic_api_key and llm_config.get("anthropic_api_key"):
                settings.llm.anthropic_api_key = llm_config["anthropic_api_key"]
            if not settings.llm.anthropic_model and llm_config.get("anthropic_model"):
                settings.llm.anthropic_model = llm_config["anthropic_model"]
            
            if llm_config.get("hunyuan_api_key"):
                settings.llm.hunyuan_api_key = llm_config["hunyuan_api_key"]
            if llm_config.get("hunyuan_api_base"):
                settings.llm.hunyuan_api_base = llm_config["hunyuan_api_base"]
            if llm_config.get("hunyuan_model"):
                settings.llm.hunyuan_model = llm_config["hunyuan_model"]
            
            if llm_config.get("default_llm_provider"):
                settings.llm.default_llm_provider = llm_config["default_llm_provider"]
        
        # 数据库配置
        if "database" in toml_config:
            db_config = toml_config["database"]
            for key in ["postgres_host", "postgres_port", "postgres_user", "postgres_password", 
                       "postgres_database", "mysql_host", "mysql_port", "mysql_user", 
                       "mysql_password", "mysql_database", "sqlite_database", "duckdb_database"]:
                if key in db_config:
                    setattr(settings.database, key, db_config[key])
            if db_config.get("default_db_type"):
                settings.database.default_db_type = db_config["default_db_type"]
        
        # API 配置
        if "api" in toml_config:
            api_config = toml_config["api"]
            for key in ["host", "port", "debug"]:
                if key in api_config:
                    setattr(settings.api, key, api_config[key])
        
        # 日志配置
        if "log" in toml_config:
            log_config = toml_config["log"]
            for key in ["level", "file"]:
                if key in log_config:
                    setattr(settings.log, key, log_config[key])
    
    return settings


# 全局配置实例
settings = get_settings()

