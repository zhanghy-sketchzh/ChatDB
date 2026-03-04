"""
lib.llm.anthropic_llm — Anthropic Claude LLM 实现
"""

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from lib.utils.config import settings
from lib.utils.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from lib.utils.logger import logger
from lib.llm.base import BaseLLM, LLMResponse, Message


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM 实现"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(
            model=model or settings.llm.anthropic_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.client = AsyncAnthropic(
            api_key=api_key or settings.llm.anthropic_api_key,
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
            system_content = ""
            api_messages = []

            for m in messages:
                if m.role == "system":
                    system_content = m.content
                else:
                    api_messages.append({"role": m.role, "content": m.content})

            response = await self.client.messages.create(
                model=self.model,
                system=system_content if system_content else None,
                messages=api_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                raw_response=response,
            )

        except Exception as e:
            error_msg = str(e).lower()

            if "rate limit" in error_msg:
                logger.warning(f"Anthropic API 限流: {e}")
                raise LLMRateLimitError(f"API 限流: {e}")
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"Anthropic API 连接失败: {e}")
                raise LLMConnectionError(f"API 连接失败: {e}")
            else:
                logger.error(f"Anthropic API 调用失败: {e}")
                raise LLMResponseError(f"API 调用失败: {e}")
