"""
LLM 基础类

定义 LLM 调用的抽象接口，支持多种 LLM 提供商。
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from chatdb.utils.logger import logger, log_llm_debug, is_llm_debug_enabled


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
        """
        初始化 LLM

        Args:
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            **kwargs: 其他参数
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        生成响应

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            LLM 响应对象
        """
        pass

    async def chat(self, prompt: str, system_prompt: str | None = None, caller_name: str = "LLM") -> str:
        """
        简单对话接口

        Args:
            prompt: 用户输入
            system_prompt: 系统提示
            caller_name: 调用者名称（用于 debug 日志）

        Returns:
            模型响应文本
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        response = await self.generate(messages)
        
        # LLM Debug 模式：输出完整的输入输出
        if is_llm_debug_enabled():
            log_llm_debug(
                caller_name=caller_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                response=response.content,
                model=self.model,
            )
        
        return response.content


async def call_llm_for_schema(
    llm_client: BaseLLM, prompt: str, model_name: Optional[str] = None
) -> str:
    """
    调用LLM生成Schema JSON（非流式）

    Args:
        llm_client: LLM客户端实例（BaseLLM）
        prompt: 提示词
        model_name: 模型名称（可选，如果提供会覆盖llm_client的model）

    Returns:
        Schema JSON字符串

    Raises:
        Exception: 如果LLM调用失败或返回空结果
    """
    if not isinstance(llm_client, BaseLLM):
        raise ValueError("llm_client 必须是 BaseLLM 实例")

    try:
        # 构建消息
        messages = [Message(role="user", content=prompt)]

        # 调用LLM
        response = await llm_client.generate(
            messages,
            temperature=0,
            max_tokens=20480,
            model=model_name or llm_client.model,
        )

        # 记录 LLM 交互（输入输出封装在一起）
        from chatdb.utils.logger import log_llm_interaction
        log_llm_interaction(logger, "生成表描述信息", prompt, response.content, max_prompt_chars=300, max_response_chars=500)

        if not response.content:
            raise Exception("LLM返回空结果")

        # 提取并验证JSON
        text = response.content.strip()
        json_str = _extract_json_from_text(text)

        if not json_str:
            raise Exception("无法提取JSON内容")

        try:
            # 验证JSON格式
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {e}")
            raise

    except Exception as e:
        logger.error(f"调用LLM失败: {e}")
        raise


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    从文本中提取JSON内容

    Args:
        text: 包含JSON的文本

    Returns:
        提取的JSON字符串，如果未找到则返回None
    """
    # 尝试提取 ```json ... ``` 格式
    if "```json" in text.lower():
        start_idx = text.lower().find("```json")
        if start_idx >= 0:
            content_start = text.find("\n", start_idx) + 1
            if content_start > 0:
                end_idx = text.find("```", content_start)
                if end_idx > content_start:
                    return text[content_start:end_idx].strip()

    # 尝试提取 ``` ... ``` 格式
    if "```" in text:
        start_idx = text.find("```")
        content_start = text.find("\n", start_idx) + 1
        if content_start > 0:
            end_idx = text.find("```", content_start)
            if end_idx > content_start:
                return text[content_start:end_idx].strip()

    # 尝试提取 {...} 格式
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end].strip()

    return None

