"""
查询路由

处理自然语言查询相关的 API 请求。
"""

from fastapi import APIRouter, Depends, HTTPException

from chatdb.core import AgentOrchestrator
from chatdb.api.dependencies import get_orchestrator
from chatdb.api.schemas import QueryRequest, QueryResponse
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
            user_query=request.query,
            skip_validation=request.skip_validation,
            skip_summary=request.skip_summary,
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

