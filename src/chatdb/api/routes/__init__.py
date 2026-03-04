"""API 路由模块"""

from chatdb.api.routes.query import router as query_router
from chatdb.api.routes.database import router as database_router
from chatdb.api.routes.health import router as health_router
from chatdb.api.routes.chat import router as chat_router
from chatdb.api.routes.agui import router as agui_router
from chatdb.api.routes.report import router as report_router
from chatdb.api.routes.report_agui import router as report_agui_router
from chatdb.api.routes.gateway import router as gateway_router

__all__ = [
    "query_router",
    "database_router",
    "health_router",
    "chat_router",
    "agui_router",
    "report_router",
    "report_agui_router",
    "gateway_router",
]



