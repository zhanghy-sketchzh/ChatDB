"""
lib/core/base_orchestrator.py — 编排器抽象基类

从 chatdb.core.orchestrator.AgentOrchestrator 和
chatreport.core.orchestrator.ReportOrchestrator 提取的公共模式。

使用 Template Method Pattern 定义标准流程：
1. init_state — 初始化状态
2. pre_process — 前置处理（分类、语义解析等）
3. plan — 计划生成
4. execute — 计划执行（串行/并行）
5. post_process — 后置处理（总结、持久化等）

子类只需实现各阶段的业务逻辑。

继承关系：
- chatdb.core.orchestrator.AgentOrchestrator(BaseOrchestrator)
- chatreport.core.orchestrator.ReportOrchestrator(BaseOrchestrator)
"""

from abc import ABC, abstractmethod
import asyncio
import time
from pathlib import Path
from typing import Any

from lib.llm import BaseLLM
from lib.utils.logger import get_component_logger


class BaseOrchestrator(ABC):
    """
    多 Agent 编排器抽象基类

    定义标准流程骨架（Template Method Pattern）：
    1. init_state → 初始化状态对象
    2. pre_process → 前置处理（分类、语义解析等）
    3. plan → 生成执行计划
    4. execute → 执行计划
    5. post_process → 后置处理（总结、持久化等）

    event_sink 机制：
    - chatdb 和 chatreport 的事件推送完全相同
    - 通过 asyncio.Queue 与 AGUIAdapter 解耦
    """

    def __init__(
        self,
        llm: BaseLLM,
        component_name: str = "Orchestrator",
        scratch_base: str = "data/scratch",
    ):
        self.llm = llm
        self._log = get_component_logger(component_name)
        self._event_sink: asyncio.Queue | None = None
        self._scratch_base = scratch_base

    # ================================================================
    # event_sink 机制（chatdb 和 chatreport 完全相同）
    # ================================================================

    async def _emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """向 AG-UI 事件队列推送一条内部事件（无队列时静默忽略）。"""
        if self._event_sink is not None:
            await self._event_sink.put((event_type, data or {}))

    # ================================================================
    # 主入口模板方法（Template Method Pattern）
    # ================================================================

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        event_sink: asyncio.Queue | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        主执行入口 — 模板方法

        定义固定的编排骨架，子类通过覆写各阶段钩子实现业务逻辑。
        """
        self._event_sink = event_sink
        start_time = time.time()

        state = await self.init_state(query, session_id, **kwargs)

        try:
            state = await self.pre_process(state, **kwargs)
            state = await self.plan(state, **kwargs)
            state = await self.execute(state, **kwargs)
            state = await self.post_process(state, **kwargs)

            return self.build_result(state, query, start_time)

        except Exception as e:
            self._log.error(f"执行失败: {e}")
            await self._emit("error", {"error": str(e)})
            return self.build_error_result(state, query, start_time, e)

        finally:
            if self._event_sink is not None:
                await self._event_sink.put(None)  # 结束信号
                self._event_sink = None

    # ================================================================
    # 子类必须实现的抽象方法
    # ================================================================

    @abstractmethod
    async def init_state(self, query: str, session_id: str | None, **kwargs: Any) -> Any:
        """初始化状态对象"""
        ...

    @abstractmethod
    async def pre_process(self, state: Any, **kwargs: Any) -> Any:
        """前置处理（分类、语义解析、上下文加载等）"""
        ...

    @abstractmethod
    async def plan(self, state: Any, **kwargs: Any) -> Any:
        """生成执行计划"""
        ...

    @abstractmethod
    async def execute(self, state: Any, **kwargs: Any) -> Any:
        """执行计划"""
        ...

    @abstractmethod
    async def post_process(self, state: Any, **kwargs: Any) -> Any:
        """后置处理（总结、持久化等）"""
        ...

    @abstractmethod
    def build_result(self, state: Any, query: str, start_time: float) -> dict[str, Any]:
        """构建返回结果"""
        ...

    # ================================================================
    # 可覆写的默认实现
    # ================================================================

    def build_error_result(
        self,
        state: Any,
        query: str,
        start_time: float,
        error: Exception,
    ) -> dict[str, Any]:
        """构建错误结果（可覆写）"""
        return {
            "success": False,
            "query": query,
            "error": str(error),
            "elapsed_ms": (time.time() - start_time) * 1000,
        }

    # ================================================================
    # Scratch Pad 公共方法
    # ================================================================

    def save_scratch(self, session_id: str, filename: str, content: str) -> None:
        """保存中间结果到 scratch 目录"""
        try:
            path = Path(self._scratch_base) / session_id / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            self._log.warn(f"Scratch 写入失败 ({filename}): {e}")
