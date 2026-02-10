"""
Tool 基类 - 结构化工具定义（Agent + Tool 架构）

设计理念（参考 Agno）：
- Tool 是"能力"的抽象，Agent 只是"使用者"
- 每个 Tool 有清晰的元数据（ToolMetadata）
- 支持渐进式披露：顶层 Tool + subtools
- 可文档化：每个 Tool 可关联 TOOL.md

每个 Tool 包含：
- metadata: 工具元数据（名称、描述、分类、输入输出定义等）
- execute: 执行方法（直接操作 state/context）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext


# ============================================================
# 工具元数据定义
# ============================================================

@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, number, boolean, object, array
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None  # 枚举值
    
    def to_schema(self) -> dict[str, Any]:
        """转换为 JSON Schema 格式"""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolMetadata:
    """
    工具元数据 - 完整的工具"说明书"
    
    设计理念：
    - 既供 LLM 理解（description、inputs、outputs）
    - 也供系统管理（category、version、dependencies）
    - 支持渐进式披露（subtools）
    """
    name: str                                   # 工具名，如 "semantic_parse"
    description: str                            # 工具干什么（一句话）
    category: str = "general"                   # 分类：analysis/execution/diagnosis/planning
    
    # 输入输出定义（用于 LLM 理解和文档生成）
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)   # {"param": {"type": "str", "description": "..."}}
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)  # {"result": {"type": "...", "description": "..."}}
    
    # 版本和文档
    version: str = "1.0"
    doc_path: str | None = None                 # 对应 TOOL.md 路径
    
    # 子工具（渐进式披露）
    subtools: list[str] = field(default_factory=list)  # 子工具名称列表
    
    # 依赖和标记
    is_core: bool = False                       # 是否核心工具
    dependencies: list[str] = field(default_factory=list)  # 依赖的其他工具
    
    def to_prompt_text(self, include_subtools: bool = False) -> str:
        """
        生成供 LLM 使用的描述文本
        
        Args:
            include_subtools: 是否包含子工具详情（渐进式披露）
        """
        lines = [f"### {self.name}"]
        lines.append(self.description)
        
        if self.inputs:
            lines.append("\n**输入参数：**")
            for param, info in self.inputs.items():
                type_str = info.get("type", "any")
                desc = info.get("description", "")
                lines.append(f"- `{param}` ({type_str}): {desc}")
        
        if self.outputs:
            lines.append("\n**输出结果：**")
            for result, info in self.outputs.items():
                type_str = info.get("type", "any")
                desc = info.get("description", "")
                lines.append(f"- `{result}` ({type_str}): {desc}")
        
        if include_subtools and self.subtools:
            lines.append("\n**子能力：**")
            for subtool in self.subtools:
                lines.append(f"- {subtool}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "version": self.version,
            "doc_path": self.doc_path,
            "subtools": self.subtools,
            "is_core": self.is_core,
            "dependencies": self.dependencies,
        }


# ============================================================
# 工具执行结果
# ============================================================

@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    message: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {
            "success": self.success,
            "data": self.data,
            "message": self.message,
        }
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def ok(cls, data: dict[str, Any] | None = None, message: str = "") -> "ToolResult":
        """创建成功结果"""
        return cls(success=True, data=data or {}, message=message)
    
    @classmethod
    def fail(cls, error: str, data: dict[str, Any] | None = None) -> "ToolResult":
        """创建失败结果"""
        return cls(success=False, error=error, data=data or {})


# ============================================================
# 工具基类
# ============================================================

class BaseTool(ABC):
    """
    工具基类（新架构）
    
    设计理念（参考 Agno）：
    - Tool 是独立的"能力单元"
    - 通过 metadata 提供完整的"说明书"
    - execute 直接操作 state/context（而非返回数据让 Agent 处理）
    - 支持两种模式：
      1. 简单模式：execute(**kwargs) -> ToolResult
      2. ReAct 模式：__call__(state, context, **kwargs) -> None（直接修改 state）
    
    子类需要实现：
    - metadata: 工具元数据
    - execute 或 __call__: 执行方法
    """
    
    def __init__(self, metadata: ToolMetadata | None = None):
        """
        初始化工具
        
        Args:
            metadata: 工具元数据，如果不提供则使用类属性
        """
        self._metadata = metadata
    
    @property
    def metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        if self._metadata:
            return self._metadata
        # 子类可以通过覆盖此属性提供默认元数据
        return ToolMetadata(
            name=self.name,
            description=self.description,
        )
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（唯一标识）"""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具描述
        
        供 LLM 理解工具用途，应包含：
        - 功能说明
        - 适用场景
        - 输入输出说明
        """
        ...
    
    @property
    def category(self) -> str:
        """工具分类"""
        return self.metadata.category
    
    @property
    def parameters(self) -> list[ToolParameter]:
        """参数定义列表（兼容旧版）"""
        return []
    
    @property
    def subtools(self) -> list[str]:
        """子工具列表"""
        return self.metadata.subtools
    
    def get_doc(self) -> str | None:
        """获取工具文档内容（从 TOOL.md）"""
        doc_path = self.metadata.doc_path
        if doc_path and Path(doc_path).exists():
            return Path(doc_path).read_text(encoding="utf-8")
        return None
    
    # ============================================================
    # 执行接口
    # ============================================================
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        简单模式执行（兼容旧版）
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            ToolResult: 执行结果
        """
        raise NotImplementedError("子类需要实现 execute 方法")
    
    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        """
        ReAct 模式执行（新架构核心）
        
        直接读写 state/context，不返回数据
        
        Args:
            state: ReAct 状态
            context: Agent 上下文
            **kwargs: 额外参数
        """
        # 默认实现：调用 execute 并将结果写入 state
        result = await self.execute(**kwargs)
        if not result.success:
            state.set_error(result.error or "工具执行失败")
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def validate_params(self, **kwargs: Any) -> tuple[bool, str]:
        """
        验证参数
        
        Returns:
            (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False, f"缺少必需参数: {param.name}"
        return True, ""
    
    def to_function_schema(self) -> dict[str, Any]:
        """
        转换为 OpenAI Function Calling 格式
        
        便于与 LLM 集成
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def get_prompt_description(self, include_subtools: bool = False) -> str:
        """获取供 LLM 使用的描述文本"""
        return self.metadata.to_prompt_text(include_subtools)
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


# ============================================================
# Agent 能力封装工具基类
# ============================================================

class AgentBackedTool(BaseTool):
    """
    封装 Agent 能力的工具基类

    子类实现：_get_agent()、_build_context(**kwargs)、_map_result(AgentResult)。
    execute 默认：构建 context → 调用 agent.execute(context) → 将 AgentResult 转为 ToolResult。
    """

    def _get_agent(self):
        """返回本工具封装的 Agent 实例（延迟初始化由子类实现）。"""
        raise NotImplementedError

    def _build_context(self, **kwargs: Any) -> Any:
        """根据 execute 参数构建 AgentContext。"""
        raise NotImplementedError

    def _map_result(self, agent_result: Any) -> ToolResult:
        """将 AgentResult 转为 ToolResult。"""
        raise NotImplementedError

    async def execute(self, **kwargs: Any) -> ToolResult:
        """通过封装的 Agent 执行，并映射结果为 ToolResult。"""
        from chatdb.agents.base import AgentContext, AgentResult
        context = self._build_context(**kwargs)
        if not isinstance(context, AgentContext):
            return ToolResult.fail("_build_context 必须返回 AgentContext")
        result = await self._get_agent().execute(context)
        return self._map_result(result)
