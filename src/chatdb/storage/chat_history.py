"""
聊天历史管理器

基于 TaskHistoryDB 的对话历史管理，不再使用独立的 sessions / runs / messages 表。
所有数据统一存储在 tasks 表中。

ChatHistoryManager 提供高级接口：
- 多轮上下文注入
- 按需查询历史
- 格式化历史输出
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from chatdb.storage.task_history import TaskHistoryDB


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """单条消息"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RunRecord:
    """单次运行记录（向后兼容）"""
    run_id: str
    session_id: str
    agent_name: str
    user_input: str
    assistant_output: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "messages": [m.to_dict() for m in self.messages],
            "tool_calls": self.tool_calls,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HistoryConfig:
    """历史配置"""
    # 是否自动注入历史到上下文
    add_history_to_context: bool = True
    # 注入最近多少轮对话
    num_history_runs: int = 3
    # 更细粒度：注入多少条消息（优先级高于 num_history_runs）
    num_history_messages: int | None = None
    # 是否启用历史查询工具
    enable_history_tool: bool = False
    # 是否允许跨会话搜索
    search_across_sessions: bool = False
    # 跨多少个历史会话搜索
    num_history_sessions: int = 2


class ChatHistoryManager:
    """
    聊天历史管理器
    
    基于 TaskHistoryDB 实现，所有数据存储在 tasks 表中。
    提供高级接口用于：
    - 自动上下文注入
    - 按需查询历史
    - 格式化历史输出
    """

    def __init__(self, db: TaskHistoryDB, config: HistoryConfig | None = None):
        self.db = db
        self.config = config or HistoryConfig()
        self._current_session_id: str | None = None
        self._agent_name: str = "default"

    def set_agent(self, agent_name: str) -> None:
        """设置当前 Agent 名称"""
        self._agent_name = agent_name

    def set_session(self, session_id: str) -> None:
        """设置当前会话 ID"""
        self._current_session_id = session_id

    @property
    def session_id(self) -> str:
        """获取当前会话 ID，不存在则生成"""
        if not self._current_session_id:
            self._current_session_id = str(uuid.uuid4())
        return self._current_session_id

    def start_session(self, session_id: str | None = None, metadata: dict | None = None) -> str:
        """开始新会话或恢复已有会话"""
        self._current_session_id = session_id or str(uuid.uuid4())
        return self._current_session_id

    def add_interaction(
        self,
        user_input: str,
        assistant_output: str,
        tool_calls: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        添加一次交互记录。
        
        注意：在新架构中，交互数据直接存储在 tasks 表的 assistant_output 字段。
        这个方法现在由 orchestrator 通过 tracker.set_assistant_output() 调用，
        不再独立写入 runs/messages 表。
        
        为向后兼容保留此方法，返回一个虚拟 run_id。
        """
        # 在新架构中，assistant_output 已由 tracker.set_assistant_output() 写入 tasks 表
        # 此方法仅做兼容处理
        return str(uuid.uuid4())

    # ============ 历史获取 ============

    def get_history_runs(self, num_runs: int | None = None) -> list[RunRecord]:
        """获取历史运行记录（从 tasks 表读取）"""
        if not self._current_session_id:
            return []
        num = num_runs or self.config.num_history_runs
        tasks = self.db.get_tasks_by_session(self._current_session_id, limit=num + 10)
        # 过滤出成功且有 assistant_output 的任务
        runs = []
        for t in tasks:
            if t.status == "success" and t.assistant_output:
                runs.append(RunRecord(
                    run_id=t.task_id,
                    session_id=t.session_id,
                    agent_name=self._agent_name,
                    user_input=t.user_query,
                    assistant_output=t.assistant_output,
                    created_at=t.created_at,
                    metadata=t.metadata,
                ))
        # 返回最近 num 条
        return runs[-num:] if num else runs

    def get_history_messages(self, num_messages: int | None = None) -> list[Message]:
        """获取历史消息（从 tasks 表推导）"""
        runs = self.get_history_runs()
        messages = []
        for run in runs:
            messages.append(Message(role=MessageRole.USER, content=run.user_input))
            messages.append(Message(role=MessageRole.ASSISTANT, content=run.assistant_output))
        if num_messages:
            return messages[-num_messages:]
        return messages

    def get_history_as_chat_format(self, num_runs: int | None = None) -> list[dict[str, str]]:
        """
        获取历史记录，格式化为 chat 格式
        
        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        if not self._current_session_id:
            return []
        num = num_runs or self.config.num_history_runs
        return self.db.get_chat_history(self._current_session_id, num)

    # ============ 上下文注入 ============

    def get_history_context(self, num_runs: int | None = None) -> str:
        """获取格式化的历史上下文（用于注入到 prompt）"""
        if not self.config.add_history_to_context or not self._current_session_id:
            return ""
        num = num_runs or self.config.num_history_runs
        return self.db.get_chat_history_context(self._current_session_id, num)

    def get_workflow_history_context(self, num_runs: int | None = None) -> str:
        """获取工作流历史上下文"""
        runs = self.get_history_runs(num_runs)
        if not runs:
            return ""

        lines = ["<workflow_history_context>"]
        for i, run in enumerate(runs, 1):
            lines.append(f"[run-{i}]")
            lines.append(f"input: {run.user_input}")
            lines.append(f"response: {run.assistant_output}")
            lines.append("")
        lines.append("</workflow_history_context>")
        return "\n".join(lines)

    def get_workflow_history(self, num_runs: int | None = None) -> list[tuple[str, str]]:
        """获取工作流历史（结构化数据）"""
        runs = self.get_history_runs(num_runs)
        return [(run.user_input, run.assistant_output) for run in runs]

    # ============ 搜索 ============

    def search_history(self, keyword: str, limit: int = 10) -> list[Message]:
        """搜索历史消息"""
        tasks = self.db.search_history(keyword, session_id=self._current_session_id, limit=limit)
        messages = []
        for t in tasks:
            messages.append(Message(
                role=MessageRole.USER,
                content=t.user_query,
                timestamp=t.created_at,
            ))
            if t.assistant_output:
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=t.assistant_output,
                    timestamp=t.created_at,
                ))
        return messages

    def get_tool_call_history(self, num_runs: int = 5) -> list[dict[str, Any]]:
        """获取工具调用历史（从 plan_nodes 推导）"""
        if not self._current_session_id:
            return []
        tasks = self.db.get_tasks_by_session(self._current_session_id, limit=num_runs)
        all_calls = []
        for t in tasks:
            for node in t.nodes:
                if node.sql:
                    all_calls.append({
                        "task_id": t.task_id,
                        "plan_task_id": node.plan_task_id,
                        "sql": node.sql,
                        "timestamp": node.started_at.isoformat(),
                    })
        return all_calls

    # ============ 跨会话 ============

    def get_cross_session_history(self, num_sessions: int | None = None) -> list[RunRecord]:
        """获取跨会话历史"""
        if not self.config.search_across_sessions:
            return self.get_history_runs()
        # 跨会话搜索：获取最近的任务
        tasks = self.db.get_recent_tasks(limit=self.config.num_history_runs * (num_sessions or self.config.num_history_sessions))
        runs = []
        for t in tasks:
            if t.assistant_output:
                runs.append(RunRecord(
                    run_id=t.task_id,
                    session_id=t.session_id,
                    agent_name=self._agent_name,
                    user_input=t.user_query,
                    assistant_output=t.assistant_output,
                    created_at=t.created_at,
                    metadata=t.metadata,
                ))
        return runs
