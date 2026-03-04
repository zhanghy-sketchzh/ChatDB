"""
智能体基础类（Agent + Tool 架构）

继承自 lib.agents.BaseAgent，增加 ChatDB 特有功能：
- 历史管理（ChatHistoryManager）
- ReAct 模式的 ErrorType 集成
- ToolRegistry 集成

设计理念（参考 Agno）：
- Agent 只是"使用者"，Tool 才是"能力"
- Agent 只能调用自己注册的 Tool
- 通过 ToolRegistry 管理 Agent→Tool 映射
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from lib.agents.base_agent import (
    BaseAgent as _LibBaseAgent,
    AgentStatus,
    AgentResult,
)
from lib.agents.base_agent import AgentContext as _LibAgentContext
from lib.llm import BaseLLM
from lib.storage.chat_history import (
    ChatHistoryManager,
    HistoryConfig,
)
from lib.storage.task_history import TaskHistoryDB

if TYPE_CHECKING:
    from chatdb.tools.base import BaseTool, ToolResult
    from chatdb.tools.registry import ToolRegistry
    from chatdb.core.react_state import ReActState


# Re-export AgentStatus 和 AgentResult 保持兼容
__all__ = [
    "AgentStatus",
    "AgentContext",
    "AgentResult",
    "BaseAgent",
]


@dataclass
class AgentContext(_LibAgentContext):
    """智能体上下文 — ChatDB 扩展版，增加 schema/history/sql 字段"""

    # Schema 信息（API 兼容层需要）
    schema_text: str = ""

    # 历史对话上下文（滑动窗口管理）
    chat_history: list[dict[str, str]] = field(default_factory=list)

    # 生成的 SQL（API 兼容层读取）
    generated_sql: str = ""


class BaseAgent(_LibBaseAgent):
    """
    ChatDB 智能体抽象基类

    在 lib.agents.BaseAgent 基础上增加：
    - 历史管理器（ChatHistoryManager）
    - ReAct 模式 ErrorType 集成
    - ToolRegistry 集成方法
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        description: str = "",
        # 工具配置
        tools: list["BaseTool"] | None = None,
        # 历史存储配置
        db_path: str | Path | None = None,
        history_config: HistoryConfig | None = None,
        # 快捷配置项
        add_history_to_context: bool = True,
        num_history_runs: int = 3,
        num_history_messages: int | None = None,
        enable_history_tool: bool = False,
        search_across_sessions: bool = False,
    ):
        super().__init__(name=name, llm=llm, description=description, tools=tools)

        # 历史管理器
        self._history_manager: ChatHistoryManager | None = None

        if db_path:
            config = history_config or HistoryConfig(
                add_history_to_context=add_history_to_context,
                num_history_runs=num_history_runs,
                num_history_messages=num_history_messages,
                enable_history_tool=enable_history_tool,
                search_across_sessions=search_across_sessions,
            )
            db = TaskHistoryDB(db_path)
            self._history_manager = ChatHistoryManager(db, config)
            self._history_manager.set_agent(name)

    # ============ ToolRegistry 集成 ============

    def register_tool_to_registry(
        self,
        tool: "BaseTool",
        registry: "ToolRegistry",
    ) -> None:
        """注册工具到本 Agent，同时注册到全局 Registry"""
        self._tools[tool.name] = tool
        registry.register_for_agent(tool, self.name)

    # ============ ReAct 模式（ChatDB 特有 ErrorType 集成）============

    async def call_tool(
        self,
        name: str,
        state: "ReActState",
        context: AgentContext,
        **kwargs: Any,
    ) -> None:
        """调用工具（ReAct 模式，带 ErrorType）"""
        tool = self._tools.get(name)
        if not tool:
            from chatdb.core.react_state import ErrorType
            state.set_error(f"工具 '{name}' 不可用", ErrorType.OTHER)
            return
        await tool(state, context, **kwargs)

    async def run(
        self,
        state: "ReActState",
        context: AgentContext,
    ) -> None:
        """ReAct 模式执行入口"""
        result = await self.execute(context)
        if result.status == AgentStatus.FAILED:
            from chatdb.core.react_state import ErrorType
            state.set_error(result.error or result.message, ErrorType.OTHER)

    # ============ 历史管理 ============

    @property
    def history(self) -> ChatHistoryManager | None:
        """获取历史管理器"""
        return self._history_manager

    @property
    def has_history(self) -> bool:
        return self._history_manager is not None

    def start_session(self, session_id: str | None = None, metadata: dict | None = None) -> str | None:
        if self._history_manager:
            return self._history_manager.start_session(session_id, metadata)
        return None

    def set_session(self, session_id: str) -> None:
        if self._history_manager:
            self._history_manager.set_session(session_id)

    def add_to_history(
        self,
        user_input: str,
        assistant_output: str,
        tool_calls: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        if self._history_manager:
            return self._history_manager.add_interaction(
                user_input=user_input,
                assistant_output=assistant_output,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        return None

    def get_history_context(self, num_runs: int | None = None) -> str:
        if self._history_manager:
            return self._history_manager.get_history_context(num_runs)
        return ""

    def get_history_as_chat_format(self, num_runs: int | None = None) -> list[dict[str, str]]:
        if self._history_manager:
            return self._history_manager.get_history_as_chat_format(num_runs)
        return []

    def get_workflow_history(self, num_runs: int | None = None) -> list[tuple[str, str]]:
        if self._history_manager:
            return self._history_manager.get_workflow_history(num_runs)
        return []

    def get_workflow_history_context(self, num_runs: int | None = None) -> str:
        if self._history_manager:
            return self._history_manager.get_workflow_history_context(num_runs)
        return ""

    def search_history(self, keyword: str, limit: int = 10) -> list:
        if self._history_manager:
            return self._history_manager.search_history(keyword, limit)
        return []

    def get_tool_call_history(self, num_runs: int = 5) -> list[dict[str, Any]]:
        if self._history_manager:
            return self._history_manager.get_tool_call_history(num_runs)
        return []

    def prepare_context_with_history(self, context: AgentContext) -> AgentContext:
        if not self._history_manager:
            return context
        if context.session_id:
            self._history_manager.set_session(context.session_id)
        if self._history_manager.config.add_history_to_context and not context.chat_history:
            context.chat_history = self.get_history_as_chat_format()
        return context

    def save_execution_to_history(
        self,
        context: AgentContext,
        result: AgentResult,
        include_metadata: bool = True,
    ) -> str | None:
        if not self._history_manager:
            return None
        output = result.data.get("summary", result.message)
        if result.data.get("generated_sql"):
            output = f"SQL: {result.data['generated_sql']}\n\n{output}"
        metadata = None
        if include_metadata:
            metadata = {
                "agent": self.name,
                "status": result.status.value,
            }
            if result.error:
                metadata["error"] = result.error
        return self.add_to_history(
            user_input=context.user_query,
            assistant_output=output,
            metadata=metadata,
        )

    # ============ 抽象方法 ============

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """执行智能体任务"""
        ...
