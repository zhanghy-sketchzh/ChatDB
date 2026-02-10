"""
OpenAI LLM 实现

支持 OpenAI GPT 系列模型的调用。
"""

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from chatdb.utils.config import settings
from chatdb.utils.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from chatdb.utils.logger import logger
from chatdb.llm.base import BaseLLM, LLMResponse, Message


class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """
        初始化 OpenAI LLM

        Args:
            model: 模型名称
            api_key: API Key
            api_base: API Base URL
            temperature: 温度参数
            max_tokens: 最大输出 token 数
        """
        super().__init__(
            model=model or settings.llm.openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.client = AsyncOpenAI(
            api_key=api_key or settings.llm.openai_api_key,
            base_url=api_base or settings.llm.openai_api_base,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        调用 OpenAI API 生成响应

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            LLM 响应对象
        """
        try:
            # 转换消息格式
            api_messages = [{"role": m.role, "content": m.content} for m in messages]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **self.extra_params,
            )

            choice = response.choices[0]

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw_response=response,
            )

        except Exception as e:
            error_msg = str(e).lower()

            if "rate limit" in error_msg:
                logger.warning(f"OpenAI API 限流: {e}")
                raise LLMRateLimitError(f"API 限流: {e}")
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"OpenAI API 连接失败: {e}")
                raise LLMConnectionError(f"API 连接失败: {e}")
            else:
                logger.error(f"OpenAI API 调用失败: {e}")
                raise LLMResponseError(f"API 调用失败: {e}")

