"""
lib.utils.config — 公共配置管理

提供所有场景包共用的配置类：LLMSettings、APISettings、LogSettings、Settings。
DB 配置（DatabaseSettings）保留在 chatdb.utils.config 中。

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

    # Venus 代理平台（支持 GLM、Qwen 等模型，OpenAI 兼容协议）
    venus_api_key: str = Field(default="", description="Venus 代理 Token")
    venus_api_base: str = Field(
        default="http://v2.open.venus.oa.com/llmproxy",
        description="Venus API Base URL",
    )
    venus_model: str = Field(default="glm-5", description="Venus 模型名称")

    # 默认提供商
    default_llm_provider: Literal["openai", "anthropic", "hunyuan", "venus"] = Field(
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

    def get_venus_params(self) -> dict:
        """获取 Venus 代理平台的参数字典"""
        return {
            "model": self.venus_model,
            "api_key": self.venus_api_key,
            "api_base": self.venus_api_base,
        }


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


class BaseAppSettings(BaseSettings):
    """
    应用主配置基类

    子场景包继承此类并添加自己的配置段。
    例如 chatdb.utils.config.Settings 增加 database: DatabaseSettings。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 公共子配置
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)
    log: LogSettings = Field(default_factory=LogSettings)


# ============================================================
# TOML 配置加载
# ============================================================

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
        return {}


def _apply_toml_to_settings(settings: BaseAppSettings, toml_config: dict) -> None:
    """将 TOML 配置覆盖到 settings 对象（通用部分）"""
    if not toml_config:
        return

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

        if llm_config.get("venus_api_key"):
            settings.llm.venus_api_key = llm_config["venus_api_key"]
        if llm_config.get("venus_api_base"):
            settings.llm.venus_api_base = llm_config["venus_api_base"]
        if llm_config.get("venus_model"):
            settings.llm.venus_model = llm_config["venus_model"]

        if llm_config.get("default_llm_provider"):
            settings.llm.default_llm_provider = llm_config["default_llm_provider"]

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


# 向后兼容：Settings 别名（chatdb.utils.config 会覆盖为带 database 段的子类）
Settings = BaseAppSettings


@lru_cache
def get_settings() -> BaseAppSettings:
    """获取应用配置（单例模式）— 基础版，不含 DatabaseSettings"""
    toml_config = _load_toml_config()
    _settings = BaseAppSettings()
    _apply_toml_to_settings(_settings, toml_config)
    return _settings


# 全局配置实例
settings = get_settings()
