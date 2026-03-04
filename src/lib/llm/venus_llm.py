"""
lib.llm.venus_llm — Venus 代理平台 LLM 实现
"""

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from lib.utils.config import settings
from lib.utils.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from lib.utils.logger import logger
from lib.llm.base import BaseLLM, LLMResponse, Message


class VenusLLM(BaseLLM):
    """Venus 代理平台 LLM 实现（OpenAI 兼容协议）"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 32768,
        **kwargs,
    ):
        super().__init__(
            model=model or settings.llm.venus_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.top_p = top_p
        self.client = AsyncOpenAI(
            api_key=api_key or settings.llm.venus_api_key,
            base_url=api_base or settings.llm.venus_api_base,
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
        try:
            api_messages = [{"role": m.role, "content": m.content} for m in messages]

            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=api_messages,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
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

            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"Venus API 限流: {e}")
                raise LLMRateLimitError(f"API 限流: {e}")
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"Venus API 连接失败: {e}")
                raise LLMConnectionError(f"API 连接失败: {e}")
            else:
                logger.error(f"Venus API 调用失败: {e}")
                raise LLMResponseError(f"API 调用失败: {e}")
