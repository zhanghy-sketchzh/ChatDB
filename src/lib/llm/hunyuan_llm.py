"""
lib.llm.hunyuan_llm — 混元（Hunyuan）LLM 实现
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from lib.utils.config import settings
from lib.utils.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from lib.utils.logger import logger
from lib.llm.base import BaseLLM, LLMResponse, Message


class HunyuanLLM(BaseLLM):
    """混元 LLM 实现"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
        enable_enhancement: bool = False,
        sensitive_business: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model or settings.llm.hunyuan_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.api_key = api_key or settings.llm.hunyuan_api_key
        self.api_base = api_base or settings.llm.hunyuan_api_base
        self.enable_enhancement = enable_enhancement
        self.sensitive_business = sensitive_business

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

            json_data = {
                "model": kwargs.get("model", self.model),
                "messages": api_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "enable_enhancement": kwargs.get("enable_enhancement", self.enable_enhancement),
                "sensitive_business": kwargs.get("sensitive_business", self.sensitive_business),
            }

            json_data.update(self.extra_params)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=180.0, verify=False) as client:
                response = await client.post(
                    self.api_base,
                    headers=headers,
                    json=json_data,
                )

                response.raise_for_status()
                result = response.json()

                choices = result.get("choices", [])
                if not choices:
                    raise LLMResponseError("API 响应中没有 choices 字段")

                choice = choices[0]
                message = choice.get("message", {})
                content = message.get("content", "")

                usage = result.get("usage", {})
                usage_info = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

                return LLMResponse(
                    content=content,
                    model=result.get("model", self.model),
                    usage=usage_info,
                    raw_response=result,
                )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 429:
                logger.warning(f"混元 API 限流: {e}")
                raise LLMRateLimitError(f"API 限流: {e}")
            elif status_code == 403:
                error_detail = ""
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get("error", {}).get("message", str(e))
                except Exception:
                    error_detail = str(e)
                logger.error(f"混元 API 认证失败 (403): {error_detail}")
                raise LLMResponseError(f"API 认证失败 (403): {error_detail}")
            else:
                error_detail = ""
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get("error", {}).get("message", str(e))
                except Exception:
                    error_detail = str(e)
                logger.error(f"混元 API HTTP 错误 ({status_code}): {error_detail}")
                raise LLMResponseError(f"API HTTP 错误 ({status_code}): {error_detail}")

        except httpx.RequestError as e:
            logger.error(f"混元 API 连接失败: {type(e).__name__}: {e!r}")
            raise LLMConnectionError(f"API 连接失败: {type(e).__name__}: {e!r}")

        except Exception as e:
            error_msg = str(e).lower()

            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"混元 API 限流: {e}")
                raise LLMRateLimitError(f"API 限流: {e}")
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"混元 API 连接失败: {type(e).__name__}: {e!r}")
                raise LLMConnectionError(f"API 连接失败: {type(e).__name__}: {e!r}")
            else:
                logger.error(f"混元 API 调用失败: {e}")
                raise LLMResponseError(f"API 调用失败: {e}")
