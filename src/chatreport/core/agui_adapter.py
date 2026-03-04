"""
ReportAGUIAdapter — 报告 AG-UI 协议适配器

继承自 lib.core.BaseAGUIAdapter，仅实现 chatreport 特有的：
- _drive_orchestrator(): 启动 ReportOrchestrator.generate_report
- _build_final_output(): 输出 state_snapshot（报告通过 text_chunk 流式输出）
- _get_event_handlers(): 覆写移除 tool_* 映射，添加 error 映射

事件映射：
  ReportOrchestrator 阶段         →  AG-UI 事件
  ─────────────────────────────────────────────
  generate_report 开始             →  RUN_STARTED
  大纲生成                         →  STEP + CUSTOM (outline_end)
  章节开始/结束                    →  CUSTOM (section_start/section_end)
  子问题规划                       →  CUSTOM (sub_question_plan)
  数据查询                         →  CUSTOM (data_query_start/data_query_end)
  章节撰写完成                     →  CUSTOM (section_write_end)
  章节审计                         →  CUSTOM (verification_result)
  全局审计                         →  STEP + CUSTOM (global_verification)
  报告拼装                         →  STEP + TEXT_MESSAGE (报告流式输出)
  完成 / 异常                      →  RUN_FINISHED / RUN_ERROR
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from ag_ui.core import (
    EventType,
    StateSnapshotEvent,
)

from lib.core.base_agui_adapter import BaseAGUIAdapter


class ReportAGUIAdapter(BaseAGUIAdapter):
    """将 ReportOrchestrator 内部事件流转换为 AG-UI SSE 字符串。"""

    timeout = 1200  # 报告生成较慢，超时 20 分钟

    def __init__(self, encoder=None):
        super().__init__(component_name="ReportAGUIAdapter", encoder=encoder)

    async def stream(
        self,
        orchestrator: Any,
        query: str,
        thread_id: str,
        run_id: str,
        session_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """驱动 ReportOrchestrator 执行并逐步 yield AG-UI SSE 帧。"""
        async for event_str in super().stream(
            orchestrator,
            query,
            thread_id,
            run_id,
            session_id=session_id,
            chat_history=chat_history,
        ):
            yield event_str

    async def _drive_orchestrator(
        self,
        orchestrator: Any,
        query: str,
        queue: asyncio.Queue,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """启动 ReportOrchestrator.generate_report 后台任务"""
        session_id = kwargs.get("session_id")
        chat_history = kwargs.get("chat_history")
        return await orchestrator.generate_report(
            query=query,
            session_id=session_id,
            chat_history=chat_history,
            event_sink=queue,
        )

    async def _build_final_output(
        self,
        result: dict[str, Any],
        thread_id: str,
        run_id: str,
    ) -> AsyncGenerator[str, None]:
        """输出报告状态快照（报告内容通过 text_chunk 流式输出）"""
        enc = self._encoder.encode

        snapshot = _build_report_snapshot(result)
        if snapshot:
            self._log.debug(
                f"发送 STATE_SNAPSHOT: success={snapshot.get('success')}, "
                f"sections={snapshot.get('section_count')}"
            )
            yield enc(StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=snapshot,
            ))
            self._event_count += 1

    def _get_event_handlers(self) -> dict[str, Any]:
        """
        覆写事件处理器：报告场景不需要 tool_* 映射。
        保留 step_start, step_end, text_chunk, custom, error。
        """
        return {
            "step_start": self._map_step_start,
            "step_end": self._map_step_end,
            "text_chunk": self._map_text_chunk,
            "custom": self._map_custom,
            "error": self._map_error,
        }


def _build_report_snapshot(result: dict[str, Any]) -> dict[str, Any] | None:
    """从报告结果构建前端可消费的状态快照。"""
    if not result:
        return None
    return {
        "success": result.get("success", False),
        "title": result.get("title", ""),
        "section_count": result.get("section_count", 0),
        "evidence_count": result.get("evidence_count", 0),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
        "session_id": result.get("session_id", ""),
        "has_global_verification": result.get("global_verification") is not None,
        "markdown_length": len(result.get("markdown", "")),
    }
