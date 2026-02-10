"""
混元（Hunyuan）LLM 实现

支持腾讯混元大模型的调用。
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from chatdb.utils.config import settings
from chatdb.utils.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from chatdb.utils.logger import logger
from chatdb.llm.base import BaseLLM, LLMResponse, Message


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
        """
        初始化混元 LLM

        Args:
            model: 模型名称
            api_key: API Key (Bearer Token)
            api_base: API Base URL
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            enable_enhancement: 是否启用增强
            sensitive_business: 是否敏感业务
        """
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
        """
        调用混元 API 生成响应

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            LLM 响应对象
        """
        try:
            # 转换消息格式
            api_messages = [{"role": m.role, "content": m.content} for m in messages]

            # 构建请求数据
            json_data = {
                "model": kwargs.get("model", self.model),
                "messages": api_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "enable_enhancement": kwargs.get("enable_enhancement", self.enable_enhancement),
                "sensitive_business": kwargs.get("sensitive_business", self.sensitive_business),
            }

            # 添加额外参数
            json_data.update(self.extra_params)

            # 构建请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # 发送异步请求
            # 对于内网 API，禁用 SSL 验证避免证书权限问题
            async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
                response = await client.post(
                    self.api_base,
                    headers=headers,
                    json=json_data,
                )

                # 检查响应状态
                response.raise_for_status()

                # 解析响应
                result = response.json()

                # 提取响应内容
                choices = result.get("choices", [])
                if not choices:
                    raise LLMResponseError("API 响应中没有 choices 字段")

                choice = choices[0]
                message = choice.get("message", {})
                content = message.get("content", "")

                # 提取使用量信息
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
                # 403 通常是认证问题
                error_detail = ""
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get("error", {}).get("message", str(e))
                except Exception:
                    error_detail = str(e)
                logger.error(f"混元 API 认证失败 (403): {error_detail}")
                logger.error(f"请检查: 1) API Key 是否正确 2) API Key 是否有权限访问模型 '{self.model}' 3) API URL 是否正确")
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
            logger.error(f"混元 API 连接失败: {e}")
            raise LLMConnectionError(f"API 连接失败: {e}")

        except Exception as e:
            error_msg = str(e).lower()

            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"混元 API 限流: {e}")
                raise LLMRateLimitError(f"API 限流: {e}")
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"混元 API 连接失败: {e}")
                raise LLMConnectionError(f"API 连接失败: {e}")
            else:
                logger.error(f"混元 API 调用失败: {e}")
                raise LLMResponseError(f"API 调用失败: {e}")

