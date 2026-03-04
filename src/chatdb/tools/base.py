"""
chatdb.tools.base — 兼容层（真正代码已迁移到 lib.tools.base）

保留 chatdb 特有的 AgentBackedTool.execute 实现（绑定 chatdb.agents.base）。
"""
from lib.tools.base import (  # noqa: F401
    AgentBackedTool as _LibAgentBackedTool,
    BaseTool,
    SubToolDef,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext


class AgentBackedTool(_LibAgentBackedTool):
    """
    chatdb 专用的 AgentBackedTool — 绑定 chatdb.agents.base 的类型检查。

    execute 默认实现会校验 _build_context 返回 AgentContext 类型。
    """

    async def execute(self, **kwargs: Any) -> ToolResult:
        """通过封装的 Agent 执行，并映射结果为 ToolResult。"""
        from chatdb.agents.base import AgentContext, AgentResult
        context = self._build_context(**kwargs)
        if not isinstance(context, AgentContext):
            return ToolResult.fail("_build_context 必须返回 AgentContext")
        result = await self._get_agent().execute(context)
        return self._map_result(result)


__all__ = [
    "ToolParameter",
    "SubToolDef",
    "ToolMetadata",
    "ToolResult",
    "BaseTool",
    "AgentBackedTool",
]
