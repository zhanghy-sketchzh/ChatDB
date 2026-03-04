"""
chatdb.utils.exceptions — ChatDB 异常类

通用异常已迁移到 lib.utils.exceptions，此处：
1. re-export 所有通用异常
2. 定义 ChatDB 特有的 DB/SQL 异常
"""

# re-export 公共异常
from lib.utils.exceptions import (  # noqa: F401
    ChatDBError,
    LLMError,
    LLMConnectionError,
    LLMResponseError,
    LLMRateLimitError,
    AgentError,
    AgentTimeoutError,
    AgentValidationError,
)


# ==================== 数据库相关异常（ChatDB 特有）====================


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


# ==================== SQL 相关异常（ChatDB 特有）====================


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
