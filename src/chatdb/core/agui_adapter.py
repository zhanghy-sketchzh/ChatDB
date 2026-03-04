"""
AG-UI 协议适配器

将 AgentOrchestrator 的多 Agent 执行流程转换为 AG-UI 标准事件流。
通过 asyncio.Queue 与 Orchestrator 解耦：Orchestrator 写入内部事件，
Adapter 消费并转换为 AG-UI SSE 事件推送给前端。

继承自 lib.core.BaseAGUIAdapter，仅实现 chatdb 特有的：
- _drive_orchestrator(): 启动 AgentOrchestrator.process_query
- _build_final_output(): 输出 summary + state_snapshot
"""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator

from ag_ui.core import (
    EventType,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    StateSnapshotEvent,
)

from lib.core.base_agui_adapter import BaseAGUIAdapter


class AGUIAdapter(BaseAGUIAdapter):
    """将 AgentOrchestrator 内部事件流转换为 AG-UI SSE 字符串。"""

    timeout = 600  # 10 分钟超时

    def __init__(self, encoder=None):
        super().__init__(component_name="AGUIAdapter", encoder=encoder)

    async def stream(
        self,
        orchestrator: Any,
        query: str,
        thread_id: str,
        run_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """驱动 Orchestrator 执行并逐步 yield AG-UI SSE 帧。"""
        async for event_str in super().stream(
            orchestrator,
            query,
            thread_id,
            run_id,
            session_id=session_id,
        ):
            yield event_str

    async def _drive_orchestrator(
        self,
        orchestrator: Any,
        query: str,
        queue: asyncio.Queue,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """启动 AgentOrchestrator.process_query 后台任务"""
        session_id = kwargs.get("session_id")
        return await orchestrator.process_query(
            query, session_id=session_id, event_sink=queue
        )

    async def _build_final_output(
        self,
        result: dict[str, Any],
        thread_id: str,
        run_id: str,
    ) -> AsyncGenerator[str, None]:
        """输出最终文本消息（summary）和状态快照"""
        enc = self._encoder.encode

        # --- 最终文本消息（summary）---
        summary = result.get("summary", "")
        if summary:
            msg_id = uuid.uuid4().hex
            self._log.debug(f"发送 TEXT_MESSAGE: {summary[:80]}…")
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
        snapshot = _build_state_snapshot(result)
        if snapshot:
            row_count = snapshot.get("row_count", 0)
            self._log.debug(f"发送 STATE_SNAPSHOT: success={snapshot.get('success')}, rows={row_count}")
            yield enc(StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=snapshot,
            ))
            self._event_count += 1


def _build_state_snapshot(result: dict[str, Any]) -> dict[str, Any] | None:
    """从查询结果构建前端可消费的状态快照。"""
    if not result:
        return None
    snapshot: dict[str, Any] = {
        "success": result.get("success", False),
        "status": result.get("status", "completed"),
        "query": result.get("query", ""),
        "rewritten_query": result.get("rewritten_query"),
        "sql": result.get("sql", ""),
        "row_count": result.get("row_count", 0),
        "table_name": result.get("table_name", ""),
        "result": result.get("result", [])[:50],
        "intent": result.get("intent"),
    }

    # human intervention 相关字段透传
    if result.get("clarification_request"):
        snapshot["clarification_request"] = result["clarification_request"]
    if result.get("run_context"):
        snapshot["run_context"] = result["run_context"]

    # research_mode 结构化结论透传
    for key in ("confidence", "key_findings", "limitations", "suggested_follow_ups"):
        if result.get(key) is not None:
            snapshot[key] = result[key]

    return snapshot
