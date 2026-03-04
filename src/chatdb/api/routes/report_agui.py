"""
报告 AG-UI SSE 端点

提供报告生成的 AG-UI 流式端点，将 ReportOrchestrator 的执行过程
以 AG-UI SSE 事件流实时推送给前端。

事件映射：
  ReportOrchestrator 阶段     →  AG-UI 事件
  ─────────────────────────────────────────
  generate_report 开始         →  RUN_STARTED
  大纲生成                     →  STEP(outline) + TOOL_CALL
  章节规划                     →  STEP(section_plan) + TOOL_CALL
  数据查询（调 ChatDB）        →  STEP(data_query) + TOOL_CALL
  章节撰写                     →  STEP(section_write) + TEXT_MESSAGE(章节内容流式)
  质量审计                     →  STEP(verify) + CUSTOM(verification_result)
  承上启下                     →  CUSTOM(transition)
  全局拼装                     →  TEXT_MESSAGE(最终报告流式)
  报告完成                     →  STATE_SNAPSHOT + RUN_FINISHED
"""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ag_ui.encoder import EventEncoder

from chatdb.api.schemas import ReportAGUIRequest
from lib.utils.logger import logger

router = APIRouter(prefix="/report/agui", tags=["Report AG-UI"])


@router.post("/", summary="报告生成 AG-UI SSE 流式端点")
async def report_agui_endpoint(request_body: ReportAGUIRequest, request: Request):
    """
    报告生成的 AG-UI 流式端点。

    接收报告生成请求，以 AG-UI SSE 事件流返回生成过程。
    前端可实时展示大纲、章节进度、数据查询结果等。
    """
    thread_id = request_body.thread_id or uuid.uuid4().hex
    run_id = request_body.run_id or uuid.uuid4().hex

    logger.info(
        f"收到报告 AG-UI 请求: {request_body.query[:80]}... "
        f"thread={thread_id}"
    )

    # 创建 ChatDB orchestrator（作为数据源）
    from chatdb.api.routes.report import _create_chatdb_orchestrator

    chatdb_orch = await _create_chatdb_orchestrator(request_body.db_path)

    # 创建 ReportOrchestrator
    from chatreport.core.orchestrator import ReportOrchestrator
    from chatreport.core.agui_adapter import ReportAGUIAdapter
    from chatreport.tools.data_service import DataService

    report_orch = ReportOrchestrator(
        llm=chatdb_orch.llm,
        data_service=DataService(chatdb_orch),
    )

    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)
    adapter = ReportAGUIAdapter(encoder=encoder)

    return StreamingResponse(
        adapter.stream(
            report_orchestrator=report_orch,
            query=request_body.query,
            thread_id=thread_id,
            run_id=run_id,
            session_id=request_body.session_id,
            max_sections=request_body.max_sections,
            verify=request_body.verify,
        ),
        media_type=encoder.get_content_type(),
    )
