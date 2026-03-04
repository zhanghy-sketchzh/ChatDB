"""
lib.llm.base — LLM 基础类

定义 LLM 调用的抽象接口，支持多种 LLM 提供商。
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from lib.utils.logger import logger, log_llm_debug, is_llm_debug_enabled


@dataclass
class Message:
    """消息对象"""

    role: str  # system / user / assistant
    content: str


@dataclass
class LLMResponse:
    """LLM 响应对象"""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None


class BaseLLM(ABC):
    """LLM 抽象基类"""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
        self._on_llm_call: Any = None

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        pass

    async def chat(self, prompt: str, system_prompt: str | None = None, caller_name: str = "LLM", **kwargs) -> str:
        """简单对话接口"""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        import time as _time
        _t0 = _time.time()
        response = await self.generate(messages, **kwargs)
        _duration_ms = int((_time.time() - _t0) * 1000)

        if is_llm_debug_enabled():
            log_llm_debug(
                caller_name=caller_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                response=response.content,
                model=self.model,
            )

        if self._on_llm_call is not None:
            try:
                self._on_llm_call(
                    caller_name=caller_name,
                    prompt_preview=prompt[:500],
                    response_preview=response.content[:500],
                    model=response.model or self.model,
                    input_tokens=response.usage.get("input_tokens", 0) or response.usage.get("prompt_tokens", 0),
                    output_tokens=response.usage.get("output_tokens", 0) or response.usage.get("completion_tokens", 0),
                    duration_ms=_duration_ms,
                )
            except Exception:
                pass

        return response.content


async def call_llm_for_schema(
    llm_client: BaseLLM, prompt: str, model_name: Optional[str] = None
) -> str:
    """调用LLM生成Schema JSON（非流式）"""
    if not isinstance(llm_client, BaseLLM):
        raise ValueError("llm_client 必须是 BaseLLM 实例")

    try:
        messages = [Message(role="user", content=prompt)]

        response = await llm_client.generate(
            messages,
            temperature=0,
            max_tokens=20480,
            model=model_name or llm_client.model,
        )

        from lib.utils.logger import log_llm_interaction
        log_llm_interaction(logger, "生成表描述信息", prompt, response.content, max_prompt_chars=300, max_response_chars=500)

        if not response.content:
            raise Exception("LLM返回空结果")

        text = response.content.strip()
        json_str = extract_json_from_text(text)

        if not json_str:
            raise Exception("无法提取JSON内容")

        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {e}")
            raise

    except Exception as e:
        logger.error(f"调用LLM失败: {e}")
        raise


def extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取JSON内容"""
    if "```json" in text.lower():
        start_idx = text.lower().find("```json")
        if start_idx >= 0:
            content_start = text.find("\n", start_idx) + 1
            if content_start > 0:
                end_idx = text.find("```", content_start)
                if end_idx > content_start:
                    return text[content_start:end_idx].strip()

    if "```" in text:
        start_idx = text.find("```")
        content_start = text.find("\n", start_idx) + 1
        if content_start > 0:
            end_idx = text.find("```", content_start)
            if end_idx > content_start:
                return text[content_start:end_idx].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end].strip()

    return None


# 向后兼容别名
_extract_json_from_text = extract_json_from_text
