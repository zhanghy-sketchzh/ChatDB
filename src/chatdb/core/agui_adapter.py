"""
AG-UI 协议适配器

将 AgentOrchestrator 的多 Agent 执行流程转换为 AG-UI 标准事件流。
通过 asyncio.Queue 与 Orchestrator 解耦：Orchestrator 写入内部事件，
Adapter 消费并转换为 AG-UI SSE 事件推送给前端。

事件映射：
  Orchestrator 阶段          →  AG-UI 事件
  ─────────────────────────────────────────
  process_query 开始          →  RUN_STARTED
  语义解析                    →  STEP + TOOL_CALL (semantic_parse)
  Planner 生成计划            →  STEP + TOOL_CALL (planner)
  SQLAgent 执行任务           →  STEP + TOOL_CALL (sql_task_*)
  Planner 决策                →  CUSTOM (planner_decision)
  生成总结 / 最终回答         →  TEXT_MESSAGE
  完成 / 异常                 →  RUN_FINISHED / RUN_ERROR
"""

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

from chatdb.utils.logger import get_component_logger

_log = get_component_logger("AGUIAdapter")


class AGUIAdapter:
    """将 Orchestrator 内部事件流转换为 AG-UI SSE 字符串。"""

    def __init__(self, encoder: EventEncoder | None = None):
        self._encoder = encoder or EventEncoder()
        self._event_count = 0
        self._start_time = 0.0

    async def stream(
        self,
        orchestrator: Any,
        query: str,
        thread_id: str,
        run_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        驱动 Orchestrator 执行并逐步 yield AG-UI SSE 帧。

        Args:
            orchestrator: AgentOrchestrator 实例
            query: 用户自然语言查询
            thread_id: AG-UI 会话线程 ID
            run_id: AG-UI 运行 ID
            session_id: ChatDB 会话 ID（多轮对话）
        """
        self._event_count = 0
        self._start_time = time.time()

        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        enc = self._encoder.encode

        _log.info(f"开始事件流: thread={thread_id[:8]}… run={run_id[:8]}… query={query[:60]}…")

        # --- RUN_STARTED ---
        yield enc(RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id,
        ))
        self._event_count += 1

        # 后台启动 Orchestrator
        task = asyncio.create_task(
            orchestrator.process_query(query, session_id=session_id, event_sink=queue)
        )
        _log.debug("Orchestrator 后台任务已启动")

        final_result: dict[str, Any] = {}
        has_error = False

        try:
            async for agui_event_str in self._consume(queue, thread_id, run_id):
                yield agui_event_str

            final_result = await task

        except Exception as exc:
            has_error = True
            _log.error(f"执行异常: {exc}")
            yield enc(RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(exc),
            ))
            self._event_count += 1
            if not task.done():
                task.cancel()
            return

        # --- 最终文本消息（summary）---
        summary = final_result.get("summary", "")
        if summary:
            msg_id = uuid.uuid4().hex
            _log.debug(f"发送 TEXT_MESSAGE: {summary[:80]}…")
            yield enc(TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=msg_id,
                role="assistant",
            ))
            yield enc(TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=msg_id,
                delta=summary,
            ))
            yield enc(TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=msg_id,
            ))
            self._event_count += 3

        # --- 结果状态快照 ---
        snapshot = _build_state_snapshot(final_result)
        if snapshot:
            row_count = snapshot.get("row_count", 0)
            _log.debug(f"发送 STATE_SNAPSHOT: success={snapshot.get('success')}, rows={row_count}")
            yield enc(StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=snapshot,
            ))
            self._event_count += 1

        # --- RUN_FINISHED ---
        yield enc(RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=thread_id,
            run_id=run_id,
        ))
        self._event_count += 1

        elapsed = time.time() - self._start_time
        _log.info(f"事件流完成: 共 {self._event_count} 个事件, 耗时 {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 内部：消费 Orchestrator 事件队列
    # ------------------------------------------------------------------

    async def _consume(
        self,
        queue: asyncio.Queue,
        thread_id: str,
        run_id: str,
    ) -> AsyncGenerator[str, None]:
        """从队列读取内部事件，转换为 AG-UI SSE 帧。"""
        enc = self._encoder.encode
        timeout = 600  # 单次等待上限
        queue_count = 0

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                _log.error(f"Orchestrator 执行超时 ({timeout}s), 已消费 {queue_count} 个内部事件")
                yield enc(RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message="Orchestrator 执行超时",
                ))
                self._event_count += 1
                return

            if item is None:
                _log.debug(f"队列结束信号: 共消费 {queue_count} 个内部事件")
                return

            event_type, data = item
            queue_count += 1

            agui_events = self._map_event(event_type, data, thread_id, run_id)
            for evt in agui_events:
                self._event_count += 1
                yield enc(evt)

    def _map_event(
        self,
        event_type: str,
        data: dict[str, Any],
        thread_id: str,
        run_id: str,
    ) -> list:
        """将单个内部事件映射为一组 AG-UI 事件对象。"""
        _HANDLERS = {
            "step_start": self._map_step_start,
            "step_end": self._map_step_end,
            "tool_start": self._map_tool_start,
            "tool_args": self._map_tool_args,
            "tool_end": self._map_tool_end,
            "tool_result": self._map_tool_result,
            "text_chunk": self._map_text_chunk,
            "custom": self._map_custom,
        }
        handler = _HANDLERS.get(event_type)
        if handler:
            result = handler(data)
            # 对关键事件输出 INFO 日志，其余用 DEBUG
            if event_type == "step_start":
                _log.info(f"→ STEP_STARTED: {data.get('name', '?')}")
            elif event_type == "step_end":
                _log.info(f"← STEP_FINISHED: {data.get('name', '?')}")
            elif event_type == "tool_start":
                _log.info(f"  🔧 TOOL_CALL_START: {data.get('tool_name', '?')}")
            elif event_type == "tool_end":
                _log.debug(f"  🔧 TOOL_CALL_END: {data.get('call_id', '?')[:12]}…")
            elif event_type == "tool_result":
                content = data.get("result", {})
                preview = json.dumps(content, ensure_ascii=False)[:80] if content else "(empty)"
                _log.debug(f"  📋 TOOL_CALL_RESULT: {preview}…")
            else:
                _log.debug(f"  映射 {event_type} → {len(result)} 个 AG-UI 事件")
            return result

        _log.debug(f"  未知内部事件 '{event_type}' → CUSTOM")
        return [CustomEvent(type=EventType.CUSTOM, name=event_type, value=data)]

    # ---------- 映射方法 ----------

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
            name=data.get("name", "chatdb_event"),
            value=data.get("value", {}),
        )]


def _build_state_snapshot(result: dict[str, Any]) -> dict[str, Any] | None:
    """从查询结果构建前端可消费的状态快照。"""
    if not result:
        return None
    return {
        "success": result.get("success", False),
        "query": result.get("query", ""),
        "rewritten_query": result.get("rewritten_query"),
        "sql": result.get("sql", ""),
        "row_count": result.get("row_count", 0),
        "table_name": result.get("table_name", ""),
        "result": result.get("result", [])[:50],
        "intent": result.get("intent"),
    }
