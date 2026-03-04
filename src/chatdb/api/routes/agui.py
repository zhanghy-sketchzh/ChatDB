"""
AG-UI 协议端点

提供符合 AG-UI 标准的 SSE 流式端点，将 ChatDB 多 Agent 执行流程
实时推送给前端。支持两种输入模式：
  1. 标准 AG-UI RunAgentInput（兼容 CopilotKit 等客户端）
  2. 简化版 ChatDB 请求（仅需 query + db_path）
  3. 人类介入恢复（/continue 端点，同样以 SSE 流返回）
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder

from chatdb.core.agui_adapter import AGUIAdapter

router = APIRouter(prefix="/agui", tags=["AG-UI"])


class ChatDBRunRequest(BaseModel):
    """简化版请求（兼容 ChatDB 原有 /chat 接口参数）"""
    query: str = Field(..., description="自然语言查询", min_length=1)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")
    thread_id: str | None = Field(default=None, description="AG-UI 线程 ID")
    run_id: str | None = Field(default=None, description="AG-UI 运行 ID")


@router.post("/", summary="AG-UI SSE 流式端点（简化版）")
async def agui_chat_endpoint(request_body: ChatDBRunRequest, request: Request):
    """
    接收 ChatDB 查询请求，以 AG-UI SSE 事件流返回执行过程。

    前端可实时接收：
      - STEP_STARTED / STEP_FINISHED（语义解析、Planner、SQL 执行、总结）
      - TOOL_CALL_*（各 Agent 的输入输出）
      - TEXT_MESSAGE_*（最终回答）
      - STATE_SNAPSHOT（结果数据快照）
      - RUN_STARTED / RUN_FINISHED / RUN_ERROR（生命周期）
    """
    import uuid

    thread_id = request_body.thread_id or uuid.uuid4().hex
    run_id = request_body.run_id or uuid.uuid4().hex

    orchestrator = await _create_orchestrator(request_body.db_path)

    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)
    adapter = AGUIAdapter(encoder=encoder)

    return StreamingResponse(
        adapter.stream(
            orchestrator=orchestrator,
            query=request_body.query,
            thread_id=thread_id,
            run_id=run_id,
            session_id=request_body.session_id,
        ),
        media_type=encoder.get_content_type(),
    )


@router.post("/run", summary="AG-UI 标准端点（RunAgentInput）")
async def agui_standard_endpoint(input_data: RunAgentInput, request: Request):
    """
    兼容标准 AG-UI RunAgentInput 的端点。

    从 messages 中提取最后一条用户消息作为 query，
    从 state 中读取 db_path。适配 CopilotKit 等标准客户端。
    """
    import uuid

    query = _extract_query(input_data)
    db_path = _extract_db_path(input_data)

    orchestrator = await _create_orchestrator(db_path)

    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)
    adapter = AGUIAdapter(encoder=encoder)

    return StreamingResponse(
        adapter.stream(
            orchestrator=orchestrator,
            query=query,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
            session_id=input_data.thread_id,
        ),
        media_type=encoder.get_content_type(),
    )


class AGUIContinueRequest(BaseModel):
    """人类介入恢复请求（AG-UI SSE 流式版本）"""
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str = Field(..., description="原始查询的 session_id")
    chosen_option: str = Field(..., description="用户选择的 option id")
    extra_input: str | None = Field(default=None, description="用户自由输入补充")
    original_query: str = Field("", description="原始用户查询")
    thread_id: str | None = Field(default=None, description="AG-UI 线程 ID")
    run_id: str | None = Field(default=None, description="AG-UI 运行 ID")


@router.post("/continue", summary="AG-UI 人类介入恢复端点（SSE 流式）")
async def agui_continue_endpoint(request_body: AGUIContinueRequest, request: Request):
    """
    当查询返回 need_clarification 状态后，用户提交选择，
    此端点以 AG-UI SSE 事件流返回恢复执行的全过程。

    与 /query/continue（返回 JSON）不同，此端点提供实时进度事件。
    """
    import uuid

    thread_id = request_body.thread_id or uuid.uuid4().hex
    run_id = request_body.run_id or uuid.uuid4().hex

    # 拼接用户澄清到原始查询中，让 LLM 自主决定如何调整
    clarification = f"（用户澄清：选择了「{request_body.chosen_option}」"
    if request_body.extra_input:
        clarification += f"，补充说明：{request_body.extra_input}"
    clarification += "）"
    resumed_query = (
        f"{request_body.original_query} {clarification}"
        if request_body.original_query
        else clarification
    )

    orchestrator = await _create_orchestrator(request_body.db_path)

    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)
    adapter = AGUIAdapter(encoder=encoder)

    return StreamingResponse(
        adapter.stream(
            orchestrator=orchestrator,
            query=resumed_query,
            thread_id=thread_id,
            run_id=run_id,
            session_id=request_body.session_id,
        ),
        media_type=encoder.get_content_type(),
    )


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

async def _create_orchestrator(db_path: str) -> Any:
    """根据文件路径创建 Orchestrator 实例。"""
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.core import AgentOrchestrator
    from lib.llm import LLMFactory

    file_path = Path(db_path)

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        from chatdb.api.routes.chat import _import_csv_to_duckdb
        actual_path = _import_csv_to_duckdb(db_path)
    elif suffix == ".duckdb":
        actual_path = db_path
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")

    db = DuckDBConnector(database=actual_path)
    await db.connect()
    tables_meta = db.get_tables_meta()
    llm = LLMFactory.create(provider="hunyuan")

    return AgentOrchestrator(llm, db, tables_meta=tables_meta)


def _extract_query(input_data: RunAgentInput) -> str:
    """从 RunAgentInput.messages 中提取最后一条用户消息文本。"""
    for msg in reversed(input_data.messages):
        if getattr(msg, "role", None) == "user":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if hasattr(part, "text"):
                        return part.text.strip()
    raise ValueError("未找到用户消息")


def _extract_db_path(input_data: RunAgentInput) -> str:
    """从 RunAgentInput.state 或 forwarded_props 中提取 db_path。"""
    state = input_data.state or {}
    if isinstance(state, dict) and state.get("db_path"):
        return state["db_path"]
    props = input_data.forwarded_props or {}
    if isinstance(props, dict) and props.get("db_path"):
        return props["db_path"]
    raise ValueError("未提供 db_path（请通过 state.db_path 或 forwarded_props.db_path 传入）")
