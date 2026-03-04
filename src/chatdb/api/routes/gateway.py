"""
统一网关路由

自动分流请求到 ChatDB（数据查询）或 ChatReport（报告生成）。
前端只需对接一个端点，无需关心走的是查询还是报告。

端点：
  POST /gateway      — REST 同步，返回 GatewayResponse
  POST /gateway/agui — AG-UI SSE 流式，根据分类结果选用不同 adapter
"""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from chatdb.api.schemas import GatewayRequest, GatewayResponse, GatewayAGUIRequest
from lib.utils.logger import logger

router = APIRouter(prefix="/gateway", tags=["Gateway"])


@router.post("/", response_model=GatewayResponse, summary="统一入口 — 自动分流")
async def gateway_endpoint(request: GatewayRequest) -> GatewayResponse:
    """
    自动判断用户意图：
    - 数据查询类 → ChatDB.process_query()
    - 报告生成类 → ChatReport.generate_report()
    """
    from lib.classify.router import QueryRouter
    from chatdb.api.routes.report import _create_chatdb_orchestrator

    try:
        chatdb_orch = await _create_chatdb_orchestrator(request.db_path)

        # 分类（有 LLM 就用 LLM，否则纯规则）
        query_router = QueryRouter()
        decision = await query_router.classify(
            request.query, llm=chatdb_orch.llm
        )

        logger.info(
            f"网关分流: query={request.query[:50]}... → "
            f"{decision.target} ({decision.query_type}, conf={decision.confidence:.2f})"
        )

        if decision.target == "chatreport":
            # 走 ChatReport 报告生成
            from chatreport.core.orchestrator import ReportOrchestrator
            from chatreport.tools.data_service import DataService

            report_orch = ReportOrchestrator(
                llm=chatdb_orch.llm,
                data_service=DataService(chatdb_orch),
            )
            result = await report_orch.generate_report(
                query=request.query,
                session_id=request.session_id,
            )
            return GatewayResponse(
                routed_to="chatreport",
                query_type=decision.query_type,
                report_markdown=result.get("markdown"),
                report_outline=result.get("outline"),
                summary=result.get("markdown", "")[:500],
                success=result.get("success", True),
                error=result.get("error"),
            )
        else:
            # 走 ChatDB 数据查询
            result = await chatdb_orch.process_query(
                request.query, session_id=request.session_id
            )
            return GatewayResponse(
                routed_to="chatdb",
                query_type=decision.query_type,
                sql=result.get("sql"),
                result=result.get("result"),
                row_count=result.get("row_count", 0),
                summary=result.get("summary", ""),
                success=result.get("success", True),
                error=result.get("error"),
            )
    except FileNotFoundError as e:
        return GatewayResponse(
            routed_to="chatdb",
            query_type="error",
            success=False,
            error=str(e),
        )
    except Exception as e:
        logger.exception(f"网关错误: {e}")
        return GatewayResponse(
            routed_to="chatdb",
            query_type="error",
            success=False,
            error=str(e),
        )


@router.post("/agui", summary="统一 AG-UI 入口 — 自动分流 + SSE 流式")
async def gateway_agui_endpoint(request_body: GatewayAGUIRequest, request: Request):
    """
    统一 AG-UI 流式入口。

    先 classify，再根据结果选用 ChatDB 或 ChatReport 的 AGUIAdapter。
    首个 CUSTOM 事件携带 route_decision 告知前端当前模式。
    """
    from lib.classify.router import QueryRouter
    from ag_ui.encoder import EventEncoder

    thread_id = request_body.thread_id or uuid.uuid4().hex
    run_id = request_body.run_id or uuid.uuid4().hex

    # 创建 ChatDB orchestrator
    from chatdb.api.routes.report import _create_chatdb_orchestrator

    chatdb_orch = await _create_chatdb_orchestrator(request_body.db_path)

    # 分类
    query_router = QueryRouter()
    decision = await query_router.classify(
        request_body.query, llm=chatdb_orch.llm
    )

    logger.info(
        f"网关 AG-UI 分流: query={request_body.query[:50]}... → "
        f"{decision.target} ({decision.query_type})"
    )

    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)

    if decision.target == "chatreport":
        # 报告模式 → ReportAGUIAdapter
        from chatreport.core.orchestrator import ReportOrchestrator
        from chatreport.core.agui_adapter import ReportAGUIAdapter
        from chatreport.tools.data_service import DataService

        report_orch = ReportOrchestrator(
            llm=chatdb_orch.llm,
            data_service=DataService(chatdb_orch),
        )
        adapter = ReportAGUIAdapter(encoder=encoder)

        async def _report_stream():
            # 先发 route_decision
            yield _encode_route_decision(encoder, decision, thread_id, run_id)
            async for chunk in adapter.stream(
                report_orchestrator=report_orch,
                query=request_body.query,
                thread_id=thread_id,
                run_id=run_id,
                session_id=request_body.session_id,
            ):
                yield chunk

        return StreamingResponse(
            _report_stream(),
            media_type=encoder.get_content_type(),
        )
    else:
        # 查询模式 → ChatDB AGUIAdapter
        from chatdb.core.agui_adapter import AGUIAdapter

        adapter = AGUIAdapter(encoder=encoder)

        async def _query_stream():
            # 先发 route_decision
            yield _encode_route_decision(encoder, decision, thread_id, run_id)
            async for chunk in adapter.stream(
                orchestrator=chatdb_orch,
                query=request_body.query,
                thread_id=thread_id,
                run_id=run_id,
                session_id=request_body.session_id,
            ):
                yield chunk

        return StreamingResponse(
            _query_stream(),
            media_type=encoder.get_content_type(),
        )


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _encode_route_decision(encoder, decision, thread_id: str, run_id: str) -> str:
    """编码 route_decision 为 AG-UI CUSTOM 事件字符串。"""
    from ag_ui.core import CustomEvent, EventType

    event = CustomEvent(
        type=EventType.CUSTOM,
        name="route_decision",
        value={
            "target": decision.target,
            "query_type": decision.query_type,
            "confidence": decision.confidence,
        },
    )
    return encoder.encode(event)
