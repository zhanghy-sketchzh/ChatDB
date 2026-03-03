"""
会话记忆辅助模块

从 orchestrator 拆出的会话历史加载/保存逻辑。
"""

from typing import Any

from chatdb.core.react_state import ReActState
from chatdb.storage.chat_history import ChatHistoryManager
from chatdb.storage.task_history import TaskTracker


class HistoryHelper:
    """会话历史的加载、保存和格式化"""

    def __init__(
        self,
        history_manager: ChatHistoryManager | None,
        task_tracker: TaskTracker,
    ):
        self._history_manager = history_manager
        self._task_tracker = task_tracker

    def load_chat_history(
        self,
        session_id: str | None,
        log,
    ) -> list[dict[str, str]]:
        """加载会话历史"""
        if not self._history_manager or not session_id:
            return []
        self._history_manager.start_session(session_id)
        history = self._history_manager.get_history_as_chat_format()
        if history:
            log.info(f"已加载 {len(history) // 2} 轮历史对话 (session={session_id[:8]}...)")
        return history

    def save_to_history(
        self,
        session_id: str | None,
        query: str,
        summary: str,
        state: ReActState,
    ) -> None:
        """将本轮结果保存到会话历史"""
        if not session_id:
            return
        output_parts = []
        if summary:
            output_parts.append(summary)

        result_data_brief = self._extract_result_data_brief(state)
        if result_data_brief:
            output_parts.append(result_data_brief)

        if state.executed_sqls:
            sql_parts = ["[执行SQL]"]
            for tid, sql in state.executed_sqls.items():
                sql_parts.append(f"  [{tid}] {sql}")
            output_parts.append("\n".join(sql_parts))
        else:
            sql = state.final_sql or state.current_sql
            if sql:
                output_parts.append(f"[SQL] {sql}")

        assistant_output = "\n".join(output_parts) if output_parts else "(无结果)"
        self._task_tracker.set_assistant_output(assistant_output)

    @staticmethod
    def _extract_result_data_brief(state: ReActState) -> str:
        """从查询结果中提取关键数据摘要"""
        rows = state.execute_result.get("rows", []) if state.execute_result else []
        if not rows:
            return ""
        brief_rows = rows[:10]
        parts = ["[查询结果数据]"]
        for i, row in enumerate(brief_rows, 1):
            items = list(row.items())[:5]
            row_str = ", ".join(f"{k}={v}" for k, v in items)
            parts.append(f"  {i}. {row_str}")
        if len(rows) > 10:
            parts.append(f"  ...共 {len(rows)} 行")
        return "\n".join(parts)

    @staticmethod
    def format_history_for_prompt(chat_history: list[dict[str, str]]) -> str:
        """将 chat_history 格式化为 prompt 注入文本"""
        if not chat_history:
            return ""
        lines = ["## 历史对话\n"]
        for msg in chat_history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
            lines.append("")
        return "\n".join(lines) + "\n"
