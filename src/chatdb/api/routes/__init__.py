"""API 路由模块"""

from chatdb.api.routes.query import router as query_router
from chatdb.api.routes.database import router as database_router
from chatdb.api.routes.health import router as health_router
from chatdb.api.routes.chat import router as chat_router

__all__ = ["query_router", "database_router", "health_router", "chat_router"]



