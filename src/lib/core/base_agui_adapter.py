"""
lib/core/base_agui_adapter.py — AG-UI SSE 适配器基类

从 chatdb.core.agui_adapter.AGUIAdapter 和
chatreport.core.agui_adapter.ReportAGUIAdapter 提取的 90%+ 相同代码。

提供完全通用的：
- stream() 主循环
- _consume() 队列消费
- _map_event() 事件映射（含所有基础映射方法）

子类只需覆写：
- _drive_orchestrator(): 启动后台任务
- _build_final_output(): 处理最终结果（summary/markdown/快照）

收益：chatdb 和 chatreport 的 agui_adapter 各自减少约 200 行重复代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator

from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    StateSnapshotEvent,
    CustomEvent,
)
from ag_ui.encoder import EventEncoder

from lib.utils.logger import get_component_logger


class BaseAGUIAdapter(ABC):
    """
    AG-UI SSE 适配器基类

    提供完全通用的事件流骨架：
    - stream(): 主循环（RUN_STARTED → consume → final_output → RUN_FINISHED）
    - _consume(): 队列消费（超时处理 + 事件映射）
    - _map_event(): 事件映射（step_start/end, tool_*, text_chunk, custom）

    子类只需实现：
    - _drive_orchestrator(): 启动后台任务并返回结果 dict
    - _build_final_output(): 处理最终结果（yield SSE 帧）
    - 可覆写 timeout / _extra_handlers() 等

    事件映射方法均为 @staticmethod，子类可直接复用或覆写。
    """

    # 子类可覆写的默认超时（秒）
    timeout: int = 600

    def __init__(
        self,
        component_name: str = "AGUIAdapter",
        encoder: EventEncoder | None = None,
    ):
        self._encoder = encoder or EventEncoder()
        self._event_count = 0
        self._start_time = 0.0
        self._log = get_component_logger(component_name)

    # ================================================================
    # 主循环
    # ================================================================

    async def stream(
        self,
        orchestrator: Any,
        query: str,
        thread_id: str,
        run_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """通用 AG-UI 事件流主循环"""
        self._event_count = 0
        self._start_time = time.time()
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        enc = self._encoder.encode

        self._log.info(
            f"开始事件流: thread={thread_id[:8]}… run={run_id[:8]}… "
            f"query={query[:60]}…"
        )

        # --- RUN_STARTED ---
        yield enc(RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id,
        ))
        self._event_count += 1

        # 子类启动后台任务
        task = asyncio.create_task(
            self._drive_orchestrator(orchestrator, query, queue, **kwargs)
        )
        self._log.debug("后台任务已启动")

        final_result: dict[str, Any] = {}

        try:
            async for agui_event_str in self._consume(queue, thread_id, run_id):
                yield agui_event_str

            final_result = await task

        except Exception as exc:
            self._log.error(f"执行异常: {exc}")
            yield enc(RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(exc),
            ))
            self._event_count += 1
            if not task.done():
                task.cancel()
            return

        # 子类处理最终输出
        async for event_str in self._build_final_output(final_result, thread_id, run_id):
            yield event_str

        # --- RUN_FINISHED ---
        yield enc(RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=thread_id,
            run_id=run_id,
        ))
        self._event_count += 1

        elapsed = time.time() - self._start_time
        self._log.info(f"事件流完成: 共 {self._event_count} 个事件, 耗时 {elapsed:.1f}s")

    # ================================================================
    # 子类必须实现
    # ================================================================

    @abstractmethod
    async def _drive_orchestrator(
        self,
        orchestrator: Any,
        query: str,
        queue: asyncio.Queue,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """启动 Orchestrator 后台任务，返回最终结果 dict"""
        ...

    @abstractmethod
    async def _build_final_output(
        self,
        result: dict[str, Any],
        thread_id: str,
        run_id: str,
    ) -> AsyncGenerator[str, None]:
        """处理最终结果，yield SSE 帧（如 TEXT_MESSAGE、STATE_SNAPSHOT）"""
        ...

    # ================================================================
    # 队列消费（完全通用）
    # ================================================================

    async def _consume(
        self,
        queue: asyncio.Queue,
        thread_id: str,
        run_id: str,
    ) -> AsyncGenerator[str, None]:
        """从队列读取内部事件，转换为 AG-UI SSE 帧。"""
        enc = self._encoder.encode
        queue_count = 0

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=self.timeout)
            except asyncio.TimeoutError:
                self._log.error(
                    f"执行超时 ({self.timeout}s), 已消费 {queue_count} 个内部事件"
                )
                yield enc(RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=f"执行超时 ({self.timeout}s)",
                ))
                self._event_count += 1
                return

            if item is None:
                self._log.debug(f"队列结束信号: 共消费 {queue_count} 个内部事件")
                return

            event_type, data = item
            queue_count += 1

            agui_events = self._map_event(event_type, data, thread_id, run_id)
            for evt in agui_events:
                self._event_count += 1
                yield enc(evt)

    # ================================================================
    # 事件映射（通用 + 可扩展）
    # ================================================================

    def _map_event(
        self,
        event_type: str,
        data: dict[str, Any],
        thread_id: str,
        run_id: str,
    ) -> list:
        """将单个内部事件映射为一组 AG-UI 事件对象。"""
        handlers = self._get_event_handlers()
        handler = handlers.get(event_type)

        if handler:
            result = handler(data)
            self._log_event(event_type, data, result)
            return result

        # 未知事件 → CUSTOM
        self._log.debug(f"  未知内部事件 '{event_type}' → CUSTOM")
        return [CustomEvent(type=EventType.CUSTOM, name=event_type, value=data)]

    def _get_event_handlers(self) -> dict[str, Any]:
        """
        返回事件类型 → 处理函数的映射。

        子类可覆写此方法来添加/替换处理函数。
        """
        handlers: dict[str, Any] = {
            "step_start": self._map_step_start,
            "step_end": self._map_step_end,
            "tool_start": self._map_tool_start,
            "tool_args": self._map_tool_args,
            "tool_end": self._map_tool_end,
            "tool_result": self._map_tool_result,
            "text_chunk": self._map_text_chunk,
            "custom": self._map_custom,
            "error": self._map_error,
        }
        return handlers

    def _log_event(self, event_type: str, data: dict[str, Any], result: list) -> None:
        """事件日志（可覆写调整日志级别）"""
        if event_type == "step_start":
            self._log.info(f"→ STEP_STARTED: {data.get('name', '?')}")
        elif event_type == "step_end":
            self._log.info(f"← STEP_FINISHED: {data.get('name', '?')}")
        elif event_type == "tool_start":
            self._log.info(f"  🔧 TOOL_CALL_START: {data.get('tool_name', '?')}")
        elif event_type == "tool_end":
            self._log.debug(f"  🔧 TOOL_CALL_END: {data.get('call_id', '?')[:12]}…")
        elif event_type == "tool_result":
            content = data.get("result", {})
            preview = json.dumps(content, ensure_ascii=False)[:80] if content else "(empty)"
            self._log.debug(f"  📋 TOOL_CALL_RESULT: {preview}…")
        else:
            self._log.debug(f"  映射 {event_type} → {len(result)} 个 AG-UI 事件")

    # ================================================================
    # 映射方法（static，子类可直接复用或覆写）
    # ================================================================

    @staticmethod
    def _map_step_start(data: dict[str, Any]) -> list:
        return [StepStartedEvent(
            type=EventType.STEP_STARTED,
            step_name=data.get("name", "unknown"),
        )]

    @staticmethod
    def _map_step_end(data: dict[str, Any]) -> list:
        return [StepFinishedEvent(
            type=EventType.STEP_FINISHED,
            step_name=data.get("name", "unknown"),
        )]

    @staticmethod
    def _map_tool_start(data: dict[str, Any]) -> list:
        return [ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=data.get("call_id", uuid.uuid4().hex),
            tool_call_name=data.get("tool_name", ""),
        )]

    @staticmethod
    def _map_tool_args(data: dict[str, Any]) -> list:
        return [ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=data.get("call_id", ""),
            delta=json.dumps(data.get("args", {}), ensure_ascii=False),
        )]

    @staticmethod
    def _map_tool_end(data: dict[str, Any]) -> list:
        return [ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=data.get("call_id", ""),
        )]

    @staticmethod
    def _map_tool_result(data: dict[str, Any]) -> list:
        msg_id = data.get("message_id", uuid.uuid4().hex)
        return [ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=msg_id,
            tool_call_id=data.get("call_id", ""),
            content=json.dumps(data.get("result", {}), ensure_ascii=False),
        )]

    @staticmethod
    def _map_text_chunk(data: dict[str, Any]) -> list:
        msg_id = data.get("message_id", uuid.uuid4().hex)
        events: list = []
        if data.get("start"):
            events.append(TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=msg_id,
                role="assistant",
            ))
        if data.get("delta"):
            events.append(TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=msg_id,
                delta=data["delta"],
            ))
        if data.get("end"):
            events.append(TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=msg_id,
            ))
        return events

    @staticmethod
    def _map_custom(data: dict[str, Any]) -> list:
        return [CustomEvent(
            type=EventType.CUSTOM,
            name=data.get("name", "event"),
            value=data.get("value", {}),
        )]

    @staticmethod
    def _map_error(data: dict[str, Any]) -> list:
        return [RunErrorEvent(
            type=EventType.RUN_ERROR,
            message=data.get("error", "未知错误"),
        )]
