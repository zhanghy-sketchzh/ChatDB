"""
聊天历史存储与管理

参考 Agno 设计，实现分层记忆 + 可控注入 + 工具化访问：
- Agent 层：单智能体会话历史
- Workflow 层：工作流多次运行历史
- 支持自动上下文注入和按需查询两种模式
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


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
    """单次运行记录（包含输入输出）"""
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


def _get_default_history_db_path() -> Path:
    """获取默认的历史数据库路径"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    db_dir = project_root / "data" / "pilot"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "history.db"


class ChatHistoryDB:
    """
    聊天历史数据库存储层
    
    使用 SQLite 存储会话和消息数据
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _get_default_history_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # 运行记录表（每次对话为一个 run）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    assistant_output TEXT NOT NULL,
                    tool_calls TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # 消息表（更细粒度的消息存储）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (run_id) REFERENCES runs(run_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")

    # ============ 会话管理 ============
    
    def create_session(self, agent_name: str, session_id: str | None = None, metadata: dict | None = None) -> str:
        """创建新会话"""
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, agent_name, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, agent_name, now, now, json.dumps(metadata or {}))
            )
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话信息"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def get_recent_sessions(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """获取最近的会话列表"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE agent_name = ? ORDER BY updated_at DESC LIMIT ?",
                (agent_name, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            return row is not None

    # ============ 运行记录管理 ============
    
    def add_run(
        self,
        session_id: str,
        agent_name: str,
        user_input: str,
        assistant_output: str,
        tool_calls: list[dict] | None = None,
        messages: list[Message] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """添加一次运行记录"""
        run_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            # 插入运行记录
            conn.execute(
                """INSERT INTO runs (run_id, session_id, agent_name, user_input, assistant_output, tool_calls, created_at, metadata) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, session_id, agent_name, user_input, assistant_output, 
                 json.dumps(tool_calls or []), now, json.dumps(metadata or {}))
            )
            
            # 插入消息记录
            if messages:
                for msg in messages:
                    conn.execute(
                        """INSERT INTO messages (message_id, run_id, session_id, role, content, timestamp, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (msg.message_id, run_id, session_id, msg.role.value, 
                         msg.content, msg.timestamp.isoformat(), json.dumps(msg.metadata))
                    )
            
            # 更新会话时间
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        
        return run_id

    def get_runs(self, session_id: str, limit: int | None = None, offset: int = 0) -> list[RunRecord]:
        """获取会话的运行记录"""
        with self._get_connection() as conn:
            query = "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at DESC"
            params: list[Any] = [session_id]
            
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            
            records = []
            for row in rows:
                row_dict = dict(row)
                # 获取该 run 的所有消息
                msg_rows = conn.execute(
                    "SELECT * FROM messages WHERE run_id = ? ORDER BY timestamp",
                    (row_dict["run_id"],)
                ).fetchall()
                
                messages = [
                    Message(
                        message_id=m["message_id"],
                        role=MessageRole(m["role"]),
                        content=m["content"],
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        metadata=json.loads(m["metadata"]),
                    )
                    for m in msg_rows
                ]
                
                records.append(RunRecord(
                    run_id=row_dict["run_id"],
                    session_id=row_dict["session_id"],
                    agent_name=row_dict["agent_name"],
                    user_input=row_dict["user_input"],
                    assistant_output=row_dict["assistant_output"],
                    messages=messages,
                    tool_calls=json.loads(row_dict["tool_calls"]),
                    created_at=datetime.fromisoformat(row_dict["created_at"]),
                    metadata=json.loads(row_dict["metadata"]),
                ))
            
            return records

    def get_recent_runs(self, agent_name: str, num_runs: int = 3, session_id: str | None = None) -> list[RunRecord]:
        """获取最近的运行记录"""
        with self._get_connection() as conn:
            if session_id:
                query = "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?"
                params: list[Any] = [session_id, num_runs]
            else:
                query = "SELECT * FROM runs WHERE agent_name = ? ORDER BY created_at DESC LIMIT ?"
                params = [agent_name, num_runs]
            
            rows = conn.execute(query, params).fetchall()
            
            records = []
            for row in rows:
                row_dict = dict(row)
                records.append(RunRecord(
                    run_id=row_dict["run_id"],
                    session_id=row_dict["session_id"],
                    agent_name=row_dict["agent_name"],
                    user_input=row_dict["user_input"],
                    assistant_output=row_dict["assistant_output"],
                    tool_calls=json.loads(row_dict["tool_calls"]),
                    created_at=datetime.fromisoformat(row_dict["created_at"]),
                    metadata=json.loads(row_dict["metadata"]),
                ))
            
            # 按时间正序返回（最早的在前）
            return list(reversed(records))

    # ============ 消息管理 ============
    
    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        """获取会话的所有消息"""
        with self._get_connection() as conn:
            query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp"
            params: list[Any] = [session_id]
            
            if limit:
                query += " DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(query, params).fetchall()
                rows = list(reversed(rows))  # 恢复正序
            else:
                rows = conn.execute(query, params).fetchall()
            
            return [
                Message(
                    message_id=row["message_id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]

    def search_messages(self, keyword: str, agent_name: str | None = None, limit: int = 20) -> list[Message]:
        """搜索消息内容"""
        with self._get_connection() as conn:
            if agent_name:
                query = """
                    SELECT m.* FROM messages m 
                    JOIN runs r ON m.run_id = r.run_id 
                    WHERE m.content LIKE ? AND r.agent_name = ?
                    ORDER BY m.timestamp DESC LIMIT ?
                """
                params: list[Any] = [f"%{keyword}%", agent_name, limit]
            else:
                query = "SELECT * FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?"
                params = [f"%{keyword}%", limit]
            
            rows = conn.execute(query, params).fetchall()
            return [
                Message(
                    message_id=row["message_id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]

    # ============ 清理 ============
    
    def clear_session(self, session_id: str) -> None:
        """清空会话的所有记录"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM runs WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def clear_all(self) -> None:
        """清空所有数据"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM sessions")


class ChatHistoryManager:
    """
    聊天历史管理器
    
    提供高级接口用于：
    - 自动上下文注入
    - 按需查询历史
    - 格式化历史输出
    """

    def __init__(self, db: ChatHistoryDB, config: HistoryConfig | None = None):
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
        """获取当前会话 ID，不存在则创建"""
        if not self._current_session_id:
            self._current_session_id = self.db.create_session(self._agent_name)
        return self._current_session_id

    def start_session(self, session_id: str | None = None, metadata: dict | None = None) -> str:
        """开始新会话或恢复已有会话"""
        if session_id and self.db.session_exists(session_id):
            self._current_session_id = session_id
        else:
            self._current_session_id = self.db.create_session(
                self._agent_name, session_id=session_id, metadata=metadata
            )
        return self._current_session_id

    def add_interaction(
        self,
        user_input: str,
        assistant_output: str,
        tool_calls: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """添加一次交互（user input + assistant output）"""
        messages = [
            Message(role=MessageRole.USER, content=user_input),
            Message(role=MessageRole.ASSISTANT, content=assistant_output),
        ]
        
        return self.db.add_run(
            session_id=self.session_id,
            agent_name=self._agent_name,
            user_input=user_input,
            assistant_output=assistant_output,
            tool_calls=tool_calls,
            messages=messages,
            metadata=metadata,
        )

    # ============ 历史获取 ============
    
    def get_history_runs(self, num_runs: int | None = None) -> list[RunRecord]:
        """获取历史运行记录"""
        num = num_runs or self.config.num_history_runs
        return self.db.get_recent_runs(
            agent_name=self._agent_name,
            num_runs=num,
            session_id=self._current_session_id,
        )

    def get_history_messages(self, num_messages: int | None = None) -> list[Message]:
        """获取历史消息"""
        if not self._current_session_id:
            return []
        num = num_messages or self.config.num_history_messages
        return self.db.get_messages(self._current_session_id, limit=num)

    def get_history_as_chat_format(self, num_runs: int | None = None) -> list[dict[str, str]]:
        """
        获取历史记录，格式化为 chat 格式
        
        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        runs = self.get_history_runs(num_runs)
        history = []
        for run in runs:
            history.append({"role": "user", "content": run.user_input})
            history.append({"role": "assistant", "content": run.assistant_output})
        return history

    # ============ 上下文注入 ============
    
    def get_history_context(self, num_runs: int | None = None) -> str:
        """
        获取格式化的历史上下文（用于注入到 prompt）
        
        Returns:
            格式化的历史上下文字符串
        """
        if not self.config.add_history_to_context:
            return ""
        
        runs = self.get_history_runs(num_runs)
        if not runs:
            return ""
        
        lines = ["<chat_history_context>"]
        for i, run in enumerate(runs, 1):
            lines.append(f"[对话 {i}]")
            lines.append(f"用户: {run.user_input}")
            lines.append(f"助手: {run.assistant_output}")
            lines.append("")
        lines.append("</chat_history_context>")
        
        return "\n".join(lines)

    def get_workflow_history_context(self, num_runs: int | None = None) -> str:
        """
        获取工作流历史上下文（用于 Workflow 层）
        
        Returns:
            格式化的工作流历史上下文
        """
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
        """
        获取工作流历史（结构化数据）
        
        Returns:
            [(user_input, assistant_output), ...]
        """
        runs = self.get_history_runs(num_runs)
        return [(run.user_input, run.assistant_output) for run in runs]

    # ============ 搜索 ============
    
    def search_history(self, keyword: str, limit: int = 10) -> list[Message]:
        """搜索历史消息"""
        return self.db.search_messages(keyword, agent_name=self._agent_name, limit=limit)

    def get_tool_call_history(self, num_runs: int = 5) -> list[dict[str, Any]]:
        """获取工具调用历史"""
        runs = self.get_history_runs(num_runs)
        all_calls = []
        for run in runs:
            for call in run.tool_calls:
                call["run_id"] = run.run_id
                call["timestamp"] = run.created_at.isoformat()
                all_calls.append(call)
        return all_calls

    # ============ 跨会话 ============
    
    def get_cross_session_history(self, num_sessions: int | None = None) -> list[RunRecord]:
        """获取跨会话历史"""
        if not self.config.search_across_sessions:
            return self.get_history_runs()
        
        num = num_sessions or self.config.num_history_sessions
        sessions = self.db.get_recent_sessions(self._agent_name, limit=num)
        
        all_runs = []
        for session in sessions:
            runs = self.db.get_runs(session["session_id"], limit=self.config.num_history_runs)
            all_runs.extend(runs)
        
        # 按时间排序
        all_runs.sort(key=lambda r: r.created_at)
        return all_runs[-self.config.num_history_runs * num:]
