"""
查询路由

处理自然语言查询相关的 API 请求。
"""

from fastapi import APIRouter, Depends, HTTPException

from chatdb.core import AgentOrchestrator
from chatdb.api.dependencies import get_orchestrator
from chatdb.api.schemas import QueryRequest, QueryResponse, ClarificationResponse
from chatdb.utils.exceptions import AgentError, ChatDBError
from chatdb.utils.logger import logger

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    summary="执行自然语言查询",
    description="将自然语言查询转换为 SQL，执行并返回结果和总结",
)
async def execute_query(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """
    执行自然语言数据库查询

    - **query**: 自然语言查询语句
    - **db_type**: 数据库类型（可选）
    - **skip_validation**: 是否跳过 SQL 验证
    - **skip_summary**: 是否跳过结果总结
    """
    try:
        logger.info(f"收到查询请求: {request.query[:50]}...")

        result = await orchestrator.process_query(
            query=request.query,
            session_id=request.session_id,
        )

        return QueryResponse(**result)

    except AgentError as e:
        logger.error(f"智能体执行错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except ChatDBError as e:
        logger.error(f"应用错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"未知错误: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post(
    "/sql-only",
    response_model=QueryResponse,
    summary="仅生成 SQL",
    description="将自然语言转换为 SQL，不执行查询",
)
async def generate_sql_only(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """
    仅生成 SQL，不执行查询

    适用于只需要获取 SQL 语句的场景。
    """
    try:
        # 生成 SQL
        from chatdb.agents.base import AgentContext
        from chatdb.database.schema import SchemaInspector

        inspector = SchemaInspector(orchestrator.db_connector)
        schema_info = await inspector.get_schema_info()

        context = AgentContext(
            user_query=request.query,
            schema_text=schema_info.to_prompt_text(),
        )

        result = await orchestrator.agents["generator"].execute(context)

        return QueryResponse(
            success=result.status.value == "success",
            query=request.query,
            sql=context.generated_sql,
            result=[],
            row_count=0,
            summary="",
            error=result.error,
        )

    except Exception as e:
        logger.exception(f"SQL 生成错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/continue",
    summary="继续被暂停的分析",
    description="当查询返回 need_clarification 状态时，用户提交选择后调用此端点继续执行",
)
async def continue_query(
    request: ClarificationResponse,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> dict:
    """
    继续被暂停的分析

    - **session_id**: 上次查询返回的 run_context.session_id
    - **chosen_option**: 用户选择的选项 ID
    - **extra_input**: 用户自由输入（可选）
    - **original_query**: 原始查询（用于恢复上下文）
    """
    try:
        logger.info(f"继续分析: session={request.session_id}, option={request.chosen_option}")

        # 将用户选择作为补充上下文，拼接到原始查询中重新执行
        # 由 LLM 决定如何调整后续计划（不做硬编码映射）
        clarification = f"（用户澄清：选择了「{request.chosen_option}」"
        if request.extra_input:
            clarification += f"，补充说明：{request.extra_input}"
        clarification += "）"

        resumed_query = f"{request.original_query} {clarification}" if request.original_query else clarification

        result = await orchestrator.process_query(
            query=resumed_query,
            session_id=request.session_id,
        )

        return result

    except Exception as e:
        logger.exception(f"继续分析错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

