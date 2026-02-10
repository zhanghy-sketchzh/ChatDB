"""
FastAPI 应用工厂

创建和配置 FastAPI 应用实例。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatdb import __version__
from chatdb.api.dependencies import app_state
from chatdb.api.routes import database_router, health_router, query_router, chat_router
from chatdb.utils.config import settings
from chatdb.utils.logger import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    setup_logging()
    logger.info("ChatDB API 正在启动...")

    try:
        await app_state.initialize()
        logger.info("应用初始化完成")
    except Exception as e:
        logger.warning(f"初始化时发生错误（服务仍将启动）: {e}")

    yield

    # 关闭时
    logger.info("ChatDB API 正在关闭...")
    await app_state.shutdown()
    logger.info("应用已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="ChatDB API",
        description="""
# ChatDB - 基于 LLM 多智能体的自然语言数据库查询系统

## 功能特性

- 🗣️ **自然语言查询**: 使用自然语言描述查询需求
- 🔄 **多数据库支持**: PostgreSQL、MySQL、SQLite
- 🤖 **多智能体协作**: SQL 生成、验证、结果总结
- 🔒 **安全查询**: 仅支持 SELECT 查询，防止数据篡改

## 使用方式

1. 配置数据库连接
2. 发送自然语言查询
3. 获取 SQL、查询结果和智能总结
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(query_router)
    app.include_router(database_router)

    return app


# 创建应用实例
app = create_app()

