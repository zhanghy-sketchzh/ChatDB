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
    session_id: str | None = Field(default=None, description="会话 ID（传入启用多轮对话记忆）")
    skip_validation: bool = Field(default=False, description="是否跳过 SQL 验证")
    skip_summary: bool = Field(default=False, description="是否跳过结果总结")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "查询销售额最高的前10个产品",
                    "db_type": "postgresql",
                    "session_id": None,
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


class ClarificationResponse(BaseModel):
    """人类介入回复（用于 /query/continue 端点）"""

    session_id: str = Field(..., description="run_context 中的 session_id")
    step_id: int = Field(0, description="run_context 中的 step_id")
    chosen_option: str = Field(..., description="用户选择的 option id")
    extra_input: str | None = Field(None, description="用户自由输入的补充信息")
    original_query: str = Field("", description="原始用户查询（用于恢复上下文）")


# ==================== 报告相关 ====================


class ReportRequest(BaseModel):
    """报告生成请求"""

    query: str = Field(..., description="报告主题描述", min_length=1, max_length=5000)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")
    max_sections: int = Field(default=7, description="最大章节数", ge=2, le=15)
    include_evidence: bool = Field(default=True, description="是否包含证据链")
    verify: bool = Field(default=True, description="是否启用质量审计")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "帮我分析今年的销售趋势，生成一份分析报告",
                    "db_path": "data/duckdb/sales.duckdb",
                    "max_sections": 5,
                }
            ]
        }
    }


class ReportResponse(BaseModel):
    """报告生成响应"""

    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="原始查询")
    title: str = Field(default="", description="报告标题")
    markdown: str = Field(default="", description="完整报告 Markdown")
    outline: dict[str, Any] | None = Field(default=None, description="大纲 JSON")
    section_count: int = Field(default=0, description="章节数")
    evidence_count: int = Field(default=0, description="证据条数")
    verification: dict[str, Any] | None = Field(default=None, description="质量审计结果")
    error: str | None = Field(default=None, description="错误信息")
    elapsed_seconds: float = Field(default=0.0, description="耗时（秒）")


class ReportAGUIRequest(BaseModel):
    """报告 AG-UI SSE 请求"""

    query: str = Field(..., description="报告主题描述", min_length=1)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")
    thread_id: str | None = Field(default=None, description="AG-UI 线程 ID")
    run_id: str | None = Field(default=None, description="AG-UI 运行 ID")
    max_sections: int = Field(default=7, description="最大章节数", ge=2, le=15)
    verify: bool = Field(default=True, description="是否启用质量审计")


# ==================== 网关相关 ====================


class GatewayRequest(BaseModel):
    """统一网关请求 — 自动分流到 chatdb 或 chatreport"""

    query: str = Field(..., description="自然语言输入", min_length=1)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "今年销售额最高的产品是什么", "db_path": "data/duckdb/sales.duckdb"},
                {"query": "帮我写一份销售分析报告", "db_path": "data/duckdb/sales.duckdb"},
            ]
        }
    }


class GatewayResponse(BaseModel):
    """统一网关响应"""

    routed_to: Literal["chatdb", "chatreport"] = Field(..., description="路由去向")
    query_type: str = Field(..., description="分类结果")
    # chatdb 结果字段
    sql: str | None = Field(default=None, description="生成的 SQL")
    result: list[dict[str, Any]] | None = Field(default=None, description="查询结果")
    row_count: int = Field(default=0, description="结果行数")
    summary: str = Field(default="", description="结果总结")
    # chatreport 结果字段
    report_markdown: str | None = Field(default=None, description="报告 Markdown")
    report_outline: dict[str, Any] | None = Field(default=None, description="报告大纲")
    # 通用字段
    success: bool = Field(default=True, description="是否成功")
    error: str | None = Field(default=None, description="错误信息")


class GatewayAGUIRequest(BaseModel):
    """统一网关 AG-UI SSE 请求"""

    query: str = Field(..., description="自然语言输入", min_length=1)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")
    thread_id: str | None = Field(default=None, description="AG-UI 线程 ID")
    run_id: str | None = Field(default=None, description="AG-UI 运行 ID")


