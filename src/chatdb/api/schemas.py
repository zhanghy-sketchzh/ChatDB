"""
API 数据模型

定义 API 请求和响应的数据结构。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ==================== 查询相关 ====================


class QueryRequest(BaseModel):
    """查询请求"""

    query: str = Field(..., description="自然语言查询", min_length=1, max_length=2000)
    db_type: Literal["postgresql", "mysql", "sqlite"] | None = Field(
        default=None, description="数据库类型"
    )
    skip_validation: bool = Field(default=False, description="是否跳过 SQL 验证")
    skip_summary: bool = Field(default=False, description="是否跳过结果总结")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "查询销售额最高的前10个产品",
                    "db_type": "postgresql",
                    "skip_validation": False,
                    "skip_summary": False,
                }
            ]
        }
    }


class ValidationInfo(BaseModel):
    """SQL 验证信息"""

    is_valid: bool
    message: str


class AgentResultInfo(BaseModel):
    """智能体执行结果"""

    status: str
    message: str
    error: str | None = None


class QueryResponse(BaseModel):
    """查询响应"""

    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="原始查询")
    sql: str = Field(default="", description="生成的 SQL")
    result: list[dict[str, Any]] = Field(default_factory=list, description="查询结果")
    row_count: int = Field(default=0, description="结果行数")
    summary: str = Field(default="", description="结果总结")
    error: str | None = Field(default=None, description="错误信息")
    validation: ValidationInfo | None = None
    agent_results: dict[str, AgentResultInfo] | None = None


# ==================== 数据库连接相关 ====================


class DatabaseConnectionRequest(BaseModel):
    """数据库连接请求"""

    db_type: Literal["postgresql", "mysql", "sqlite"] = Field(..., description="数据库类型")
    host: str | None = Field(default=None, description="主机地址")
    port: int | None = Field(default=None, description="端口")
    user: str | None = Field(default=None, description="用户名")
    password: str | None = Field(default=None, description="密码")
    database: str = Field(..., description="数据库名")


class DatabaseConnectionResponse(BaseModel):
    """数据库连接响应"""

    success: bool
    message: str
    connection_id: str | None = None


# ==================== Schema 相关 ====================


class ColumnInfoResponse(BaseModel):
    """列信息响应"""

    name: str
    type: str
    nullable: bool
    primary_key: bool
    default: str | None = None
    comment: str | None = None


class TableInfoResponse(BaseModel):
    """表信息响应"""

    name: str
    columns: list[ColumnInfoResponse]
    primary_keys: list[str]
    foreign_keys: list[dict[str, Any]]
    comment: str | None = None


class SchemaResponse(BaseModel):
    """Schema 响应"""

    database_name: str
    tables: list[TableInfoResponse]


# ==================== 健康检查 ====================


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = "healthy"
    version: str
    database_connected: bool = False


