"""
lib.utils.exceptions — 公共异常类

定义所有场景包共用的基础异常。
DB/SQL 特有异常保留在 chatdb.utils.exceptions 中。
"""

from typing import Any


class ChatDBError(Exception):
    """基础异常类"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


# ==================== LLM 相关异常 ====================


class LLMError(ChatDBError):
    """LLM 调用异常"""

    pass


class LLMConnectionError(LLMError):
    """LLM API 连接异常"""

    pass


class LLMResponseError(LLMError):
    """LLM 响应解析异常"""

    pass


class LLMRateLimitError(LLMError):
    """LLM API 限流异常"""

    pass


# ==================== Agent 相关异常 ====================


class AgentError(ChatDBError):
    """Agent 执行异常"""

    pass


class AgentTimeoutError(AgentError):
    """Agent 执行超时异常"""

    pass


class AgentValidationError(AgentError):
    """Agent 输入验证异常"""

    pass
