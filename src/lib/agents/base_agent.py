"""
lib/agents/base_agent.py — Agent 抽象基类

从 chatdb/agents/base.py 迁移并泛化，去掉 ChatDB 特有逻辑
（history 管理、ToolRegistry 集成、workflow history 等）。

分层设计：
- lib/agents/base_agent.py — 纯粹的 Agent 抽象（LLM + 工具 + 日志）
- chatdb/agents/base.py — 继承此基类，增加 ChatDB 特有功能

核心接口：
- register_tool / get_tool / use_tool: 工具管理
- run(state, context): ReAct 模式执行
- execute(context): 简单模式执行（兼容旧版）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from lib.llm import BaseLLM
from lib.utils.logger import get_component_logger

if TYPE_CHECKING:
    from lib.tools.base import BaseTool, ToolResult


class AgentStatus(str, Enum):
    """智能体状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AgentContext:
    """Agent 通用上下文 — 子类可扩展"""
    user_query: str
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """智能体执行结果"""
    status: AgentStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseAgent(ABC):
    """
    Agent 抽象基类 — 纯粹的 Agent 抽象

    提供：
    - LLM 交互
    - 工具注册和调用
    - 组件日志

    子类实现：
    - execute(context) -> AgentResult: 核心执行逻辑
    - get_system_prompt() -> str: 系统提示词（可选）
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        description: str = "",
        tools: list["BaseTool"] | None = None,
    ):
        self.name = name
        self.llm = llm
        self.description = description
        self._log = get_component_logger(name)

        # 工具注册（name -> tool）
        self._tools: dict[str, "BaseTool"] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)

    # ============ 工具管理 ============

    def register_tool(self, tool: "BaseTool") -> None:
        """注册工具到本 Agent"""
        self._tools[tool.name] = tool

    @property
    def tools(self) -> dict[str, "BaseTool"]:
        """获取所有已注册工具"""
        return self._tools

    @property
    def has_tools(self) -> bool:
        return len(self._tools) > 0

    def get_tool(self, name: str) -> "BaseTool | None":
        """获取已注册的工具"""
        return self._tools.get(name)

    def add_tool(self, tool: "BaseTool") -> None:
        """register_tool 的别名"""
        self.register_tool(tool)

    async def use_tool(self, name: str, **kwargs: Any) -> "ToolResult":
        """使用工具（简单模式）"""
        from lib.tools.base import ToolResult

        tool = self._tools.get(name)
        if not tool:
            available = list(self._tools.keys())
            return ToolResult.fail(
                f"工具 '{name}' 不可用。可用工具: {available}"
            )

        is_valid, error = tool.validate_params(**kwargs)
        if not is_valid:
            return ToolResult.fail(error)

        return await tool.execute(**kwargs)

    async def call_tool(
        self,
        name: str,
        state: Any,
        context: Any,
        **kwargs: Any,
    ) -> None:
        """调用工具（ReAct 模式，直接修改 state）"""
        tool = self._tools.get(name)
        if not tool:
            state.set_error(f"工具 '{name}' 不可用")
            return
        await tool(state, context, **kwargs)

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """获取所有工具的 Schema（用于 LLM Function Calling）"""
        return [tool.to_function_schema() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """获取所有工具的描述文本"""
        if not self._tools:
            return "无可用工具"
        lines = ["可用工具："]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description.split(chr(10))[0]}")
        return "\n".join(lines)

    def get_tools_prompt(self, include_subtools: bool = False) -> str:
        """获取工具描述（供 LLM prompt 使用）"""
        if not self._tools:
            return "无可用工具"
        lines = ["## 可用工具\n"]
        for tool in self._tools.values():
            lines.append(tool.get_prompt_description(include_subtools))
            lines.append("")
        return "\n".join(lines)

    # ============ ReAct 模式核心接口 ============

    async def run(self, state: Any, context: Any) -> None:
        """
        ReAct 模式执行入口。

        子类应重写此方法根据 state 选择合适的 Tool 并调用。
        默认实现调用 execute() 方法（兼容旧版）。
        """
        result = await self.execute(context)
        if result.status == AgentStatus.FAILED:
            state.set_error(result.error or result.message)

    # ============ 抽象方法 ============

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """执行智能体任务"""
        ...

    def get_system_prompt(self) -> str:
        """获取系统提示词（可覆写）"""
        return ""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, tools={list(self._tools.keys())})>"
