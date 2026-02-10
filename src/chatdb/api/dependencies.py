"""
API 依赖注入

管理共享资源和依赖项。
"""

from typing import AsyncGenerator

from chatdb.core import AgentOrchestrator
from chatdb.utils.config import settings
from chatdb.database.base import BaseDatabaseConnector, create_connector
from chatdb.database.schema import SchemaInspector
from chatdb.llm.factory import LLMFactory


class AppState:
    """应用状态 - 管理共享资源"""

    def __init__(self):
        self.db_connector: BaseDatabaseConnector | None = None
        self.orchestrator: AgentOrchestrator | None = None
        self._initialized: bool = False

    async def initialize(self) -> None:
        """初始化应用资源（延迟初始化，不预连接数据库）"""
        if self._initialized:
            return
        # 不再预连接数据库，改为按需连接
        self._initialized = True

    async def shutdown(self) -> None:
        """关闭应用资源"""
        if self.db_connector:
            await self.db_connector.disconnect()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# 全局应用状态
app_state = AppState()


async def get_db_connector() -> BaseDatabaseConnector:
    """获取数据库连接器"""
    if not app_state.db_connector:
        raise RuntimeError("数据库未初始化")
    return app_state.db_connector


async def get_orchestrator() -> AgentOrchestrator:
    """获取智能体协调器"""
    if not app_state.orchestrator:
        raise RuntimeError("应用未初始化")
    return app_state.orchestrator


async def get_schema_inspector() -> SchemaInspector:
    """获取 Schema 检查器"""
    connector = await get_db_connector()
    return SchemaInspector(connector)

