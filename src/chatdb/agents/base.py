"""
智能体基础类（Agent + Tool 架构）

设计理念（参考 Agno）：
- Agent 只是"使用者"，Tool 才是"能力"
- Agent 只能调用自己注册的 Tool
- 通过 ToolRegistry 管理 Agent→Tool 映射

定义智能体的抽象接口和通用功能：
- register_tool: 注册工具到本 Agent
- get_tool: 获取已注册的工具
- use_tool: 使用工具
- run: 根据状态选择并调用工具
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from chatdb.llm.base import BaseLLM
from chatdb.storage.chat_history import (
    ChatHistoryDB,
    ChatHistoryManager,
    HistoryConfig,
)

if TYPE_CHECKING:
    from chatdb.tools.base import BaseTool, ToolResult
    from chatdb.tools.registry import ToolRegistry
    from chatdb.core.react_state import ReActState


class AgentStatus(str, Enum):
    """智能体状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AgentContext:
    """智能体上下文 - 在智能体之间传递的共享数据"""

    # 用户原始查询
    user_query: str

    # 数据库 Schema 信息
    schema_text: str = ""

    # 历史对话上下文（滑动窗口管理）- 兼容旧版
    chat_history: list[dict[str, str]] = field(default_factory=list)

    # 会话 ID（用于历史管理）
    session_id: str | None = None

    # 表选择结果
    available_tables: list[dict[str, Any]] = field(default_factory=list)
    selected_tables: list[str] = field(default_factory=list)
    selection_reason: str = ""

    # 结构化意图（SemanticParser 输出）
    query_intent: Any = None  # StructuredIntent
    
    # YAML 配置（业务口径）
    yml_config: dict[str, Any] = field(default_factory=dict)
    
    # 当前正在执行的任务（Planner → SQLAgent）
    current_task: dict[str, Any] | None = None

    # 生成的 SQL
    generated_sql: str = ""

    # SQL 验证结果
    is_sql_valid: bool = False
    sql_validation_message: str = ""

    # 查询结果
    query_result: list[dict[str, Any]] = field(default_factory=list)
    query_error: str = ""

    # 结果总结
    summary: str = ""

    # 额外上下文（用于总结等）
    extra_context: str = ""

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """智能体执行结果"""
    status: AgentStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseAgent(ABC):
    """
    智能体抽象基类（Agent + Tool 架构）
    
    设计理念（参考 Agno）：
    - Agent 只是"使用者"，不直接实现具体能力
    - Agent 只能调用自己注册的 Tool
    - 通过 register_tool 绑定工具，通过 get_tool 获取
    
    核心接口：
    - register_tool(tool_cls): 注册工具类到本 Agent
    - get_tool(name): 获取已注册的工具
    - use_tool(name, **kwargs): 调用工具
    - run(state, context): 根据状态选择并调用工具（ReAct 模式）
    
    使用示例：
    ```python
    class PlannerAgent(BaseAgent):
        def __init__(self, llm):
            super().__init__("planner", llm)
            self.register_tool(SemanticParseTool(llm))
        
        async def run(self, state, context):
            if state.need_intent:
                await self.get_tool("semantic_parse")(state, context)
    ```
    （SQL 流程由 Orchestrator 通过 SQLTool.run_workflow / run_generate / run_execute_and_evaluate 调用）
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
        """
        初始化智能体

        Args:
            name: 智能体名称（用于 ToolRegistry 的 Agent-Tool 绑定）
            llm: LLM 实例
            description: 智能体描述
            tools: 初始工具列表
            db_path: 历史数据库路径
            history_config: 历史配置
        """
        self.name = name
        self.llm = llm
        self.description = description
        
        # 工具注册（name -> tool）
        self._tools: dict[str, "BaseTool"] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
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
            db = ChatHistoryDB(db_path)
            self._history_manager = ChatHistoryManager(db, config)
            self._history_manager.set_agent(name)

    # ============ 工具管理（核心 API）============

    def register_tool(self, tool: "BaseTool") -> None:
        """
        注册工具到本 Agent
        
        绑定后，该工具只能被本 Agent 使用（通过 ToolRegistry 控制）
        
        Args:
            tool: 工具实例
        """
        self._tools[tool.name] = tool
    
    def register_tool_to_registry(
        self,
        tool: "BaseTool",
        registry: "ToolRegistry",
    ) -> None:
        """
        注册工具到本 Agent，同时注册到全局 Registry
        
        这是推荐的注册方式，确保 Agent-Tool 关系被全局管理
        """
        self._tools[tool.name] = tool
        registry.register_for_agent(tool, self.name)

    @property
    def tools(self) -> dict[str, "BaseTool"]:
        """获取所有已注册工具"""
        return self._tools
    
    @property
    def has_tools(self) -> bool:
        """是否有已注册工具"""
        return len(self._tools) > 0
    
    def get_tool(self, name: str) -> "BaseTool | None":
        """
        获取已注册的工具
        
        Args:
            name: 工具名称
        
        Returns:
            工具实例，不存在则返回 None
        """
        return self._tools.get(name)
    
    def add_tool(self, tool: "BaseTool") -> None:
        """添加工具（register_tool 的别名，兼容旧版）"""
        self.register_tool(tool)
    
    async def use_tool(self, name: str, **kwargs: Any) -> "ToolResult":
        """
        使用工具（简单模式）
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
        
        Returns:
            ToolResult: 工具执行结果
        
        Raises:
            ValueError: 工具不存在或未注册
        """
        from chatdb.tools.base import ToolResult
        
        tool = self._tools.get(name)
        if not tool:
            available = list(self._tools.keys())
            return ToolResult.fail(
                f"工具 '{name}' 不可用。可用工具: {available}"
            )
        
        # 验证参数
        is_valid, error = tool.validate_params(**kwargs)
        if not is_valid:
            return ToolResult.fail(error)
        
        return await tool.execute(**kwargs)
    
    async def call_tool(
        self,
        name: str,
        state: "ReActState",
        context: AgentContext,
        **kwargs: Any,
    ) -> None:
        """
        调用工具（ReAct 模式）
        
        工具直接修改 state/context，不返回数据
        
        Args:
            name: 工具名称
            state: ReAct 状态
            context: Agent 上下文
            **kwargs: 额外参数
        """
        tool = self._tools.get(name)
        if not tool:
            from chatdb.core.react_state import ErrorType
            state.set_error(f"工具 '{name}' 不可用", ErrorType.OTHER)
            return
        
        await tool(state, context, **kwargs)
    
    def get_tools_schema(self) -> list[dict[str, Any]]:
        """获取所有工具的 Schema（用于 LLM Function Calling）"""
        return [tool.to_function_schema() for tool in self._tools.values()]
    
    def get_tools_description(self) -> str:
        """获取所有工具的描述文本（用于 Prompt）"""
        if not self._tools:
            return "无可用工具"
        
        lines = ["可用工具："]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description.split(chr(10))[0]}")
        return "\n".join(lines)
    
    def get_tools_prompt(self, include_subtools: bool = False) -> str:
        """
        获取工具描述（供 LLM prompt 使用）
        
        Args:
            include_subtools: 是否包含子工具详情（渐进式披露）
        """
        if not self._tools:
            return "无可用工具"
        
        lines = ["## 可用工具\n"]
        for tool in self._tools.values():
            lines.append(tool.get_prompt_description(include_subtools))
            lines.append("")
        return "\n".join(lines)

    # ============ ReAct 模式核心接口 ============

    async def run(
        self,
        state: "ReActState",
        context: AgentContext,
    ) -> None:
        """
        ReAct 模式执行入口
        
        根据 state 选择合适的 Tool 并调用。
        子类应重写此方法实现具体逻辑。
        
        Args:
            state: ReAct 状态（包含 need_* 标记）
            context: Agent 上下文
        """
        # 默认实现：调用 execute 方法（兼容旧版）
        result = await self.execute(context)
        
        # 将结果同步到 state
        if result.status == AgentStatus.FAILED:
            from chatdb.core.react_state import ErrorType
            state.set_error(result.error or result.message, ErrorType.OTHER)

    # ============ 兼容旧版接口 ============

    @property
    def history(self) -> ChatHistoryManager | None:
        """获取历史管理器"""
        return self._history_manager

    @property
    def has_history(self) -> bool:
        """是否启用了历史功能"""
        return self._history_manager is not None

    def start_session(self, session_id: str | None = None, metadata: dict | None = None) -> str | None:
        """开始新会话或恢复已有会话"""
        if self._history_manager:
            return self._history_manager.start_session(session_id, metadata)
        return None

    def set_session(self, session_id: str) -> None:
        """设置当前会话"""
        if self._history_manager:
            self._history_manager.set_session(session_id)

    def add_to_history(
        self,
        user_input: str,
        assistant_output: str,
        tool_calls: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """添加交互到历史"""
        if self._history_manager:
            return self._history_manager.add_interaction(
                user_input=user_input,
                assistant_output=assistant_output,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        return None

    def get_history_context(self, num_runs: int | None = None) -> str:
        """获取格式化的历史上下文"""
        if self._history_manager:
            return self._history_manager.get_history_context(num_runs)
        return ""

    def get_history_as_chat_format(self, num_runs: int | None = None) -> list[dict[str, str]]:
        """获取 chat 格式的历史"""
        if self._history_manager:
            return self._history_manager.get_history_as_chat_format(num_runs)
        return []

    def get_workflow_history(self, num_runs: int | None = None) -> list[tuple[str, str]]:
        """获取工作流历史"""
        if self._history_manager:
            return self._history_manager.get_workflow_history(num_runs)
        return []

    def get_workflow_history_context(self, num_runs: int | None = None) -> str:
        """获取工作流历史上下文"""
        if self._history_manager:
            return self._history_manager.get_workflow_history_context(num_runs)
        return ""

    def search_history(self, keyword: str, limit: int = 10) -> list:
        """搜索历史消息"""
        if self._history_manager:
            return self._history_manager.search_history(keyword, limit)
        return []

    def get_tool_call_history(self, num_runs: int = 5) -> list[dict[str, Any]]:
        """获取工具调用历史"""
        if self._history_manager:
            return self._history_manager.get_tool_call_history(num_runs)
        return []

    def prepare_context_with_history(self, context: AgentContext) -> AgentContext:
        """准备带历史的上下文"""
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
        """保存执行结果到历史"""
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
        """执行智能体任务（兼容旧版）"""
        pass

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return ""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, tools={list(self._tools.keys())})>"
