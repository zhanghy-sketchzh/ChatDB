"""
lib.core — 核心框架模块

提供场景无关的编排框架基类：
- BaseState: 通用状态机基类
- BaseOrchestrator: 编排器抽象基类（Template Method Pattern）
- BaseAGUIAdapter: AG-UI SSE 适配器基类
- BaseDAG / DAGNode / NodeStatus: DAG 拓扑调度框架
- ResultCache: 查询结果 LRU+TTL 缓存
- BaseScratchPadManager: 文件暂存管理器基类
"""

from lib.core.result_cache import ResultCache
from lib.core.scratch_pad import BaseScratchPadManager
from lib.core.base_state import BaseState, BasePhase
from lib.core.base_orchestrator import BaseOrchestrator
from lib.core.base_agui_adapter import BaseAGUIAdapter
from lib.core.base_dag import BaseDAG, DAGNode, NodeStatus

__all__ = [
    # State
    "BaseState",
    "BasePhase",
    # Orchestrator
    "BaseOrchestrator",
    # AGUI Adapter
    "BaseAGUIAdapter",
    # DAG
    "BaseDAG",
    "DAGNode",
    "NodeStatus",
    # Cache & Scratch
    "ResultCache",
    "BaseScratchPadManager",
]
