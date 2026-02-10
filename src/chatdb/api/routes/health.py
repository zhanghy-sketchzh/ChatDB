"""
健康检查路由
"""

from fastapi import APIRouter

from chatdb import __version__
from chatdb.api.dependencies import app_state
from chatdb.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务健康状态",
)
async def health_check() -> HealthResponse:
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        database_connected=app_state.is_initialized,
    )


@router.get(
    "/",
    summary="根路径",
    description="API 根路径信息",
)
async def root():
    """根路径"""
    return {
        "name": "ChatDB API",
        "version": __version__,
        "description": "基于 LLM 多智能体的自然语言数据库查询系统",
        "docs": "/docs",
    }


