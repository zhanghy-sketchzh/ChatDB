"""
lib.tools — 通用工具基类

提供场景无关的工具抽象：
- ToolParameter: 参数定义
- SubToolDef: 子工具定义（LLM 可调用的原子指令）
- ToolMetadata: 工具元数据
- ToolResult: 工具执行结果
- BaseTool: 工具基类
- AgentBackedTool: Agent 能力封装工具基类
"""

from lib.tools.base import (
    AgentBackedTool,
    BaseTool,
    SubToolDef,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)

__all__ = [
    "ToolParameter",
    "SubToolDef",
    "ToolMetadata",
    "ToolResult",
    "BaseTool",
    "AgentBackedTool",
]
