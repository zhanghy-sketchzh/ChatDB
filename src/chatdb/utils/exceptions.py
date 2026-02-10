"""
自定义异常类

定义项目中使用的各类异常，便于统一异常处理和错误追踪。
"""

from typing import Any


class ChatDBError(Exception):
    """ChatDB 基础异常类"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


# ==================== 数据库相关异常 ====================


class DatabaseError(ChatDBError):
    """数据库操作异常"""

    pass


class ConnectionError(DatabaseError):
    """数据库连接异常"""

    pass


class QueryExecutionError(DatabaseError):
    """SQL 查询执行异常"""

    pass


class SchemaError(DatabaseError):
    """数据库 Schema 获取异常"""

    pass


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


# ==================== SQL 相关异常 ====================


class SQLError(ChatDBError):
    """SQL 相关异常"""

    pass


class SQLGenerationError(SQLError):
    """SQL 生成异常"""

    pass


class SQLValidationError(SQLError):
    """SQL 验证异常"""

    pass


class UnsafeSQLError(SQLError):
    """不安全的 SQL 异常"""

    pass

