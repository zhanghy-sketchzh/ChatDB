"""核心模块 - 多 Agent 协调器

注意：为避免循环导入，部分导入需要延迟加载
"""

from chatdb.core.react_state import ReActState, ReActPhase, ErrorType
from chatdb.core.messages import TaskRequest, TaskResponse, TaskResultEntry, PlanDecision
from chatdb.core.scratch_pad import ScratchPadManager
from chatdb.core.context_retriever import ContextRetriever, RetrievalResult
from chatdb.core.semantic_parse import SemanticParseTool
from chatdb.core.summarize import SummarizeAnswerTool


def __getattr__(name):
    """延迟导入 orchestrator 相关内容，避免循环导入"""
    if name in ("AgentOrchestrator", "run_query"):
        from chatdb.core.orchestrator import AgentOrchestrator, run_query
        return {
            "AgentOrchestrator": AgentOrchestrator,
            "run_query": run_query,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 调度器（延迟加载）
    "AgentOrchestrator",
    "run_query",
    # 状态
    "ReActState",
    "ReActPhase",
    "ErrorType",
    # Agent 通信消息
    "TaskRequest",
    "TaskResponse",
    "TaskResultEntry",
    "PlanDecision",
    # Scratch Pad（文件暂存）
    "ScratchPadManager",
    # Context Retriever（检索增强）
    "ContextRetriever",
    "RetrievalResult",
    # 流程步骤
    "SemanticParseTool",
    "SummarizeAnswerTool",
]
