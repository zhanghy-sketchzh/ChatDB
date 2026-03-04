"""
报告生成路由

提供报告生成的 REST 端点，直接调用 ChatReport 的 ReportOrchestrator。
不经过 ChatDB 的 process_query。
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from chatdb.api.schemas import ReportRequest, ReportResponse
from lib.utils.logger import logger

router = APIRouter(prefix="/report", tags=["Report"])


@router.post("/", response_model=ReportResponse, summary="生成数据分析报告")
async def generate_report(request: ReportRequest) -> ReportResponse:
    """
    独立的报告生成端点。

    直接调用 ReportOrchestrator，不经过 ChatDB 的 process_query。
    流程：大纲规划 → 章节 DAG → 子问题 → 数据查询 → 撰写 → 审计 → 拼装。
    """
    try:
        logger.info(f"收到报告生成请求: {request.query[:80]}...")

        chatdb_orch = await _create_chatdb_orchestrator(request.db_path)

        from chatreport.core.orchestrator import ReportOrchestrator
        from chatreport.tools.data_service import DataService

        report_orch = ReportOrchestrator(
            llm=chatdb_orch.llm,
            data_service=DataService(chatdb_orch),
        )

        result = await report_orch.generate_report(
            query=request.query,
            session_id=request.session_id,
            max_sections=request.max_sections,
            verify=request.verify,
        )

        return ReportResponse(
            success=result.get("success", True),
            query=request.query,
            title=result.get("title", ""),
            markdown=result.get("markdown", ""),
            outline=result.get("outline"),
            section_count=result.get("section_count", 0),
            evidence_count=result.get("evidence_count", 0),
            verification=result.get("global_verification"),
            error=result.get("error"),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"报告生成错误: {e}")
        return ReportResponse(
            success=False,
            query=request.query,
            error=str(e),
        )


# ------------------------------------------------------------------
# 辅助函数（复用 chat/agui 的 orchestrator 创建逻辑）
# ------------------------------------------------------------------

async def _create_chatdb_orchestrator(db_path: str):
    """根据文件路径创建 ChatDB Orchestrator 实例。"""
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.core import AgentOrchestrator
    from lib.llm import LLMFactory

    file_path = Path(db_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {db_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        from chatdb.api.routes.chat import _import_csv_to_duckdb

        actual_path = _import_csv_to_duckdb(db_path)
    elif suffix == ".duckdb":
        actual_path = db_path
    else:
        raise ValueError(f"不支持的文件类型: {suffix}，支持 .duckdb 和 .csv")

    db = DuckDBConnector(database=actual_path)
    await db.connect()
    tables_meta = db.get_tables_meta()
    llm = LLMFactory.create(provider="hunyuan")
    return AgentOrchestrator(llm, db, tables_meta=tables_meta)
