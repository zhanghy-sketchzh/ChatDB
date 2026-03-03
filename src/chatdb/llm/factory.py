"""
LLM 工厂类

提供统一的 LLM 实例创建接口。
"""

from typing import Literal

from chatdb.utils.config import settings

from chatdb.llm.anthropic_llm import AnthropicLLM
from chatdb.llm.base import BaseLLM
from chatdb.llm.hunyuan_llm import HunyuanLLM
from chatdb.llm.openai_llm import OpenAILLM
from chatdb.llm.venus_llm import VenusLLM


class LLMFactory:
    """LLM 工厂类"""

    _providers = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "hunyuan": HunyuanLLM,
        "venus": VenusLLM,
    }

    @classmethod
    def create(
        cls,
        provider: Literal["openai", "anthropic", "hunyuan", "venus"] | None = None,
        use_config: bool = True,
        **kwargs,
    ) -> BaseLLM:
        """
        创建 LLM 实例

        Args:
            provider: LLM 提供商
            use_config: 是否使用配置文件中的参数（如果 kwargs 中没有提供）
            **kwargs: LLM 参数（会覆盖配置中的值）

        Returns:
            LLM 实例
        """
        provider = provider or settings.llm.default_llm_provider

        if provider not in cls._providers:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

        # 从配置中获取基础参数，再用 kwargs 覆盖
        if use_config:
            config_params = {}
            if provider == "hunyuan":
                config_params = settings.llm.get_hunyuan_params()
            elif provider == "openai":
                config_params = {
                    "model": settings.llm.openai_model,
                    "api_key": settings.llm.openai_api_key,
                    "api_base": settings.llm.openai_api_base,
                }
            elif provider == "venus":
                config_params = settings.llm.get_venus_params()
            elif provider == "anthropic":
                config_params = {
                    "model": settings.llm.anthropic_model,
                    "api_key": settings.llm.anthropic_api_key,
                }
            # kwargs 覆盖配置值
            config_params.update(kwargs)
            kwargs = config_params

        return cls._providers[provider](**kwargs)

    @classmethod
    def register(cls, name: str, llm_class: type[BaseLLM]) -> None:
        """
        注册新的 LLM 提供商

        Args:
            name: 提供商名称
            llm_class: LLM 类
        """
        cls._providers[name] = llm_class

