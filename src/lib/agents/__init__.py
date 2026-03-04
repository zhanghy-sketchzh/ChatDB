"""
lib.agents — Agent 框架基类

提供场景无关的 Agent 抽象：
- BaseAgent: Agent 抽象基类（LLM + 工具 + 日志）
- AgentStatus: 智能体状态枚举
- AgentContext: 通用上下文
- AgentResult: 执行结果
- BasePlanner: 规划 Agent 基类
- BaseWriter: 内容生成 Agent 基类
"""

from lib.agents.base_agent import (
    BaseAgent,
    AgentStatus,
    AgentContext,
    AgentResult,
)
from lib.agents.base_planner import BasePlanner
from lib.agents.base_writer import BaseWriter

__all__ = [
    "BaseAgent",
    "AgentStatus",
    "AgentContext",
    "AgentResult",
    "BasePlanner",
    "BaseWriter",
]
