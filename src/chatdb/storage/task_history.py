"""
任务执行历史存储

记录每个 Agent 环节的输入输出、SQL 查询结果等信息。
用于调试、审计和分析。
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


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentStep:
    """单个 Agent 执行步骤"""
    step_id: str
    task_id: str
    agent_name: str
    status: TaskStatus
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TaskRecord:
    """完整任务记录"""
    task_id: str
    session_id: str
    user_query: str
    status: TaskStatus
    steps: list[AgentStep] = field(default_factory=list)
    final_sql: str | None = None
    final_result: Any = None
    summary: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "final_sql": self.final_sql,
            "final_result": self.final_result,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": self.metadata,
        }


def _get_default_db_path() -> Path:
    """获取默认的历史数据库路径"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    db_dir = project_root / "data" / "pilot"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "history.db"


class TaskHistoryDB:
    """任务历史数据库"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _get_default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """数据库连接上下文"""
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
        with self._conn() as conn:
            # 任务表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    final_sql TEXT,
                    final_result TEXT,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # Agent 步骤表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_steps (
                    step_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    input_data TEXT NOT NULL,
                    output_data TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            # 索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_task ON agent_steps(task_id)")

    # ============ 任务管理 ============

    def create_task(self, session_id: str, user_query: str, metadata: dict | None = None) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, session_id, user_query, status, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, session_id, user_query, TaskStatus.PENDING.value, now, json.dumps(metadata or {}))
            )
        return task_id

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """更新任务状态"""
        with self._conn() as conn:
            finished_at = datetime.now().isoformat() if status in (TaskStatus.SUCCESS, TaskStatus.FAILED) else None
            conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ? WHERE task_id = ?",
                (status.value, finished_at, task_id)
            )

    def complete_task(
        self,
        task_id: str,
        final_sql: str | None = None,
        final_result: Any = None,
        summary: str | None = None,
        status: TaskStatus = TaskStatus.SUCCESS,
    ) -> None:
        """完成任务"""
        with self._conn() as conn:
            result_json = json.dumps(final_result, ensure_ascii=False, default=str) if final_result else None
            conn.execute(
                "UPDATE tasks SET status = ?, final_sql = ?, final_result = ?, summary = ?, finished_at = ? WHERE task_id = ?",
                (status.value, final_sql, result_json, summary, datetime.now().isoformat(), task_id)
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        """获取任务记录"""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None

            steps = self._get_steps(conn, task_id)
            return self._row_to_task(row, steps)

    def get_tasks_by_session(self, session_id: str, limit: int = 20) -> list[TaskRecord]:
        """获取会话的任务列表"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
            return [self._row_to_task(r, self._get_steps(conn, r["task_id"])) for r in rows]

    def get_recent_tasks(self, limit: int = 20) -> list[TaskRecord]:
        """获取最近的任务"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._row_to_task(r, self._get_steps(conn, r["task_id"])) for r in rows]

    # ============ Agent 步骤管理 ============

    def add_step(
        self,
        task_id: str,
        agent_name: str,
        input_data: dict[str, Any],
        status: TaskStatus = TaskStatus.RUNNING,
    ) -> str:
        """添加 Agent 步骤"""
        step_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_steps (step_id, task_id, agent_name, status, input_data, started_at) VALUES (?, ?, ?, ?, ?, ?)",
                (step_id, task_id, agent_name, status.value, json.dumps(input_data, ensure_ascii=False, default=str), now)
            )
        return step_id

    def complete_step(
        self,
        step_id: str,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        status: TaskStatus | None = None,
    ) -> None:
        """完成 Agent 步骤"""
        with self._conn() as conn:
            # 获取开始时间计算耗时
            row = conn.execute("SELECT started_at FROM agent_steps WHERE step_id = ?", (step_id,)).fetchone()
            duration_ms = None
            if row:
                started = datetime.fromisoformat(row["started_at"])
                duration_ms = int((datetime.now() - started).total_seconds() * 1000)

            final_status = status or (TaskStatus.FAILED if error else TaskStatus.SUCCESS)
            output_json = json.dumps(output_data, ensure_ascii=False, default=str) if output_data else None

            conn.execute(
                "UPDATE agent_steps SET status = ?, output_data = ?, error = ?, finished_at = ?, duration_ms = ? WHERE step_id = ?",
                (final_status.value, output_json, error, datetime.now().isoformat(), duration_ms, step_id)
            )

    def get_steps(self, task_id: str) -> list[AgentStep]:
        """获取任务的所有步骤"""
        with self._conn() as conn:
            return self._get_steps(conn, task_id)

    # ============ 辅助方法 ============

    def _get_steps(self, conn: sqlite3.Connection, task_id: str) -> list[AgentStep]:
        """获取任务的所有步骤"""
        rows = conn.execute(
            "SELECT * FROM agent_steps WHERE task_id = ? ORDER BY started_at",
            (task_id,)
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def _row_to_task(self, row: sqlite3.Row, steps: list[AgentStep]) -> TaskRecord:
        """行转任务记录"""
        return TaskRecord(
            task_id=row["task_id"],
            session_id=row["session_id"],
            user_query=row["user_query"],
            status=TaskStatus(row["status"]),
            steps=steps,
            final_sql=row["final_sql"],
            final_result=json.loads(row["final_result"]) if row["final_result"] else None,
            summary=row["summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_step(self, row: sqlite3.Row) -> AgentStep:
        """行转步骤记录"""
        return AgentStep(
            step_id=row["step_id"],
            task_id=row["task_id"],
            agent_name=row["agent_name"],
            status=TaskStatus(row["status"]),
            input_data=json.loads(row["input_data"]) if row["input_data"] else {},
            output_data=json.loads(row["output_data"]) if row["output_data"] else None,
            error=row["error"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            duration_ms=row["duration_ms"],
        )

    # ============ 清理 ============

    def clear_session(self, session_id: str) -> None:
        """清空会话的任务"""
        with self._conn() as conn:
            task_ids = [r["task_id"] for r in conn.execute(
                "SELECT task_id FROM tasks WHERE session_id = ?", (session_id,)
            ).fetchall()]

            for task_id in task_ids:
                conn.execute("DELETE FROM agent_steps WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE session_id = ?", (session_id,))

    def clear_all(self) -> None:
        """清空所有数据"""
        with self._conn() as conn:
            conn.execute("DELETE FROM agent_steps")
            conn.execute("DELETE FROM tasks")


class TaskTracker:
    """
    任务追踪器
    
    用于在 Agent 执行过程中记录各环节信息
    """

    def __init__(self, db: TaskHistoryDB | None = None):
        self.db = db or TaskHistoryDB()
        self._current_task_id: str | None = None
        self._current_session_id: str | None = None

    def start_task(self, session_id: str, user_query: str, metadata: dict | None = None) -> str:
        """开始新任务"""
        self._current_session_id = session_id
        self._current_task_id = self.db.create_task(session_id, user_query, metadata)
        self.db.update_task_status(self._current_task_id, TaskStatus.RUNNING)
        return self._current_task_id

    def add_agent_step(self, agent_name: str, input_data: dict[str, Any]) -> str:
        """添加 Agent 步骤"""
        if not self._current_task_id:
            raise ValueError("No active task. Call start_task first.")
        return self.db.add_step(self._current_task_id, agent_name, input_data)

    def complete_agent_step(
        self,
        step_id: str,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """完成 Agent 步骤"""
        self.db.complete_step(step_id, output_data, error)

    def complete_task(
        self,
        final_sql: str | None = None,
        final_result: Any = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """完成任务"""
        if not self._current_task_id:
            return

        status = TaskStatus.FAILED if error else TaskStatus.SUCCESS
        self.db.complete_task(self._current_task_id, final_sql, final_result, summary, status)
        self._current_task_id = None

    @property
    def task_id(self) -> str | None:
        return self._current_task_id

    def get_current_task(self) -> TaskRecord | None:
        """获取当前任务"""
        if not self._current_task_id:
            return None
        return self.db.get_task(self._current_task_id)
