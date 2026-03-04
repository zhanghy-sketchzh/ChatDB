"""
lib.llm.factory — LLM 工厂类
"""

from typing import Literal

from lib.utils.config import settings
from lib.llm.base import BaseLLM
from lib.llm.anthropic_llm import AnthropicLLM
from lib.llm.hunyuan_llm import HunyuanLLM
from lib.llm.openai_llm import OpenAILLM
from lib.llm.venus_llm import VenusLLM


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
        """创建 LLM 实例"""
        provider = provider or settings.llm.default_llm_provider

        if provider not in cls._providers:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

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
            config_params.update(kwargs)
            kwargs = config_params

        return cls._providers[provider](**kwargs)

    @classmethod
    def register(cls, name: str, llm_class: type[BaseLLM]) -> None:
        """注册新的 LLM 提供商"""
        cls._providers[name] = llm_class
