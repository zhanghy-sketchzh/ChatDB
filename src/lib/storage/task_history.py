"""
lib.storage.task_history — 任务执行历史存储

从 chatdb.storage.task_history 真正迁移。
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from lib.utils.logger import get_component_logger


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# 数据模型
# ============================================================

@dataclass
class AgentStep:
    """Agent 环节记录"""
    step_id: str
    task_id: str
    agent_name: str
    status: TaskStatus
    input_data: dict[str, Any] = field(default_factory=dict)
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
class PlanNode:
    """DAG 计划节点记录"""
    node_id: str
    task_id: str
    step_id: str
    plan_task_id: str
    task_type: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.RUNNING
    sql: str = ""
    row_count: int = 0
    result_sample: list[dict[str, Any]] = field(default_factory=list)
    is_parallel: bool = False
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "plan_task_id": self.plan_task_id,
            "task_type": self.task_type,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "sql": self.sql,
            "row_count": self.row_count,
            "result_sample": self.result_sample,
            "is_parallel": self.is_parallel,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
        }


@dataclass
class LLMCallRecord:
    """LLM 调用记录"""
    call_id: str
    task_id: str
    step_id: str | None
    caller_name: str
    prompt_preview: str
    response_preview: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "caller_name": self.caller_name,
            "prompt_preview": self.prompt_preview,
            "response_preview": self.response_preview,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TaskRecord:
    """完整任务记录"""
    task_id: str
    session_id: str
    user_query: str
    rewritten_query: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    final_sql: str | None = None
    final_result: Any = None
    summary: str | None = None
    plan_json: str | None = None
    assistant_output: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    nodes: list[PlanNode] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    total_duration_ms: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_steps(self) -> list[AgentStep]:
        return self.steps

    @property
    def plan_nodes(self) -> list[PlanNode]:
        return self.nodes

    def get_nodes_for_step(self, step_id: str) -> list[PlanNode]:
        return [n for n in self.nodes if n.step_id == step_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "rewritten_query": self.rewritten_query,
            "status": self.status.value,
            "final_sql": self.final_sql,
            "final_result": self.final_result,
            "summary": self.summary,
            "plan_json": self.plan_json,
            "assistant_output": self.assistant_output,
            "steps": [s.to_dict() for s in self.steps],
            "nodes": [n.to_dict() for n in self.nodes],
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "total_duration_ms": self.total_duration_ms,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": self.metadata,
        }

    def display(self) -> str:
        lines = [
            f"═══ 任务 [{self.status.value}] ═══",
            f"问题: {self.user_query}",
        ]
        if self.rewritten_query and self.rewritten_query != self.user_query:
            lines.append(f"改写: {self.rewritten_query}")
        if self.total_duration_ms:
            lines.append(f"耗时: {self.total_duration_ms}ms")
        lines.append("")

        for step in self.steps:
            status_icon = {"success": "✓", "failed": "✗", "running": "…", "skipped": "⊘"}.get(
                step.status.value, "?"
            )
            dur = f" ({step.duration_ms}ms)" if step.duration_ms else ""
            lines.append(f"  {status_icon} [{step.agent_name}]{dur}")

            for node in self.get_nodes_for_step(step.step_id):
                n_icon = {"success": "✓", "failed": "✗", "skipped": "⊘"}.get(
                    node.status.value, "?"
                )
                par = " [并行]" if node.is_parallel else ""
                lines.append(f"      {n_icon} {node.plan_task_id}: {node.description[:60]}{par}")
                if node.sql:
                    sql_preview = node.sql.replace("\n", " ")[:80]
                    lines.append(f"        SQL: {sql_preview}...")
                if node.row_count:
                    lines.append(f"        结果: {node.row_count} 行")
                if node.error:
                    lines.append(f"        错误: {node.error[:100]}")

        if self.summary:
            lines.append(f"\n总结: {self.summary[:200]}...")
        return "\n".join(lines)


# ============================================================
# 数据库层
# ============================================================

def _get_default_db_path() -> Path:
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    db_dir = project_root / "data" / "pilot"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "history.db"


class TaskHistoryDB:
    """任务历史数据库（4 张表）"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _get_default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
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
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    rewritten_query TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    final_sql TEXT,
                    final_result TEXT,
                    summary TEXT,
                    plan_json TEXT,
                    assistant_output TEXT,
                    total_duration_ms INTEGER,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_steps (
                    step_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    input_data TEXT DEFAULT '{}',
                    output_data TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_nodes (
                    node_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    plan_task_id TEXT NOT NULL,
                    task_type TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    depends_on TEXT DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'pending',
                    sql TEXT,
                    row_count INTEGER DEFAULT 0,
                    result_sample TEXT DEFAULT '[]',
                    is_parallel INTEGER DEFAULT 0,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id),
                    FOREIGN KEY (step_id) REFERENCES agent_steps(step_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    call_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    step_id TEXT,
                    node_id TEXT,
                    caller_name TEXT NOT NULL,
                    prompt_preview TEXT,
                    response_preview TEXT,
                    model TEXT DEFAULT '',
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_steps_task ON agent_steps(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_nodes_task ON plan_nodes(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_nodes_step ON plan_nodes(step_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_task ON llm_calls(task_id)")

            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        migrations = [
            ("rewritten_query", "TEXT"),
            ("plan_json", "TEXT"),
            ("total_duration_ms", "INTEGER"),
            ("assistant_output", "TEXT"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_type}")

        llm_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_calls)").fetchall()}
        if "node_id" not in llm_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN node_id TEXT")

        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "execution_steps" in all_tables:
            conn.execute("""
                INSERT OR IGNORE INTO agent_steps 
                    (step_id, task_id, agent_name, status, input_data, output_data, 
                     error, started_at, finished_at, duration_ms)
                SELECT step_id, task_id, agent_name, status, input_data, output_data,
                       error, started_at, finished_at, duration_ms
                FROM execution_steps WHERE step_type = 'agent'
            """)
            conn.execute("""
                INSERT OR IGNORE INTO plan_nodes
                    (node_id, task_id, step_id, plan_task_id, task_type, description,
                     depends_on, status, sql, row_count, result_sample, is_parallel,
                     error, started_at, finished_at, duration_ms)
                SELECT step_id, task_id, parent_step_id, agent_name, plan_task_type, description,
                       depends_on, status, sql, row_count, result_sample, is_parallel,
                       error, started_at, finished_at, duration_ms
                FROM execution_steps WHERE step_type = 'node'
            """)
            conn.execute("DROP TABLE execution_steps")

        legacy_tables = ["sessions", "runs", "messages"]
        for table in legacy_tables:
            if table in all_tables:
                conn.execute(f"DROP TABLE {table}")  # noqa: S608

    # ============ 任务管理 ============

    def create_task(self, session_id: str, user_query: str, metadata: dict | None = None) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO tasks (task_id, session_id, user_query, status, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, session_id, user_query, TaskStatus.PENDING.value, now, json.dumps(metadata or {})),
            )
        return task_id

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        with self._conn() as conn:
            finished_at = datetime.now().isoformat() if status in (TaskStatus.SUCCESS, TaskStatus.FAILED) else None
            conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ? WHERE task_id = ?",
                (status.value, finished_at, task_id),
            )

    def update_task_rewritten_query(self, task_id: str, rewritten_query: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET rewritten_query = ? WHERE task_id = ?", (rewritten_query, task_id))

    def update_task_plan(self, task_id: str, plan_json: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET plan_json = ? WHERE task_id = ?", (plan_json, task_id))

    def update_task_assistant_output(self, task_id: str, assistant_output: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET assistant_output = ? WHERE task_id = ?", (assistant_output, task_id))

    def complete_task(self, task_id: str, final_sql: str | None = None, final_result: Any = None,
                      summary: str | None = None, status: TaskStatus = TaskStatus.SUCCESS,
                      total_duration_ms: int | None = None) -> None:
        with self._conn() as conn:
            result_json = json.dumps(final_result, ensure_ascii=False, default=str) if final_result else None
            conn.execute(
                """UPDATE tasks SET status = ?, final_sql = ?, final_result = ?, summary = ?,
                   total_duration_ms = ?, finished_at = ? WHERE task_id = ?""",
                (status.value, final_sql, result_json, summary, total_duration_ms,
                 datetime.now().isoformat(), task_id),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            steps = self._get_steps(conn, task_id)
            nodes = self._get_nodes(conn, task_id)
            llm_calls = self._get_llm_calls(conn, task_id)
            return self._row_to_task(row, steps, nodes, llm_calls)

    def get_tasks_by_session(self, session_id: str, limit: int = 20) -> list[TaskRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [
                self._row_to_task(r, self._get_steps(conn, r["task_id"]),
                                  self._get_nodes(conn, r["task_id"]),
                                  self._get_llm_calls(conn, r["task_id"]))
                for r in rows
            ]

    def get_recent_tasks(self, limit: int = 20) -> list[TaskRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                self._row_to_task(r, self._get_steps(conn, r["task_id"]),
                                  self._get_nodes(conn, r["task_id"]),
                                  self._get_llm_calls(conn, r["task_id"]))
                for r in rows
            ]

    def get_chat_history(self, session_id: str, num_runs: int = 3) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT user_query, assistant_output FROM tasks 
                   WHERE session_id = ? AND status = 'success' AND assistant_output IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, num_runs),
            ).fetchall()

        history = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["user_query"]})
            history.append({"role": "assistant", "content": row["assistant_output"]})
        return history

    def get_chat_history_context(self, session_id: str, num_runs: int = 3) -> str:
        history = self.get_chat_history(session_id, num_runs)
        if not history:
            return ""

        lines = ["<chat_history_context>"]
        for i in range(0, len(history), 2):
            turn = i // 2 + 1
            lines.append(f"[对话 {turn}]")
            lines.append(f"用户: {history[i]['content']}")
            if i + 1 < len(history):
                lines.append(f"助手: {history[i + 1]['content']}")
            lines.append("")
        lines.append("</chat_history_context>")
        return "\n".join(lines)

    def search_history(self, keyword: str, session_id: str | None = None, limit: int = 20) -> list[TaskRecord]:
        with self._conn() as conn:
            if session_id:
                rows = conn.execute(
                    """SELECT * FROM tasks WHERE session_id = ? AND 
                       (user_query LIKE ? OR summary LIKE ? OR assistant_output LIKE ?)
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM tasks WHERE 
                       user_query LIKE ? OR summary LIKE ? OR assistant_output LIKE ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit),
                ).fetchall()
            return [self._row_to_task(r, [], [], []) for r in rows]

    # ============ Agent 步骤管理 ============

    def add_step(self, task_id: str, agent_name: str, input_data: dict[str, Any] | None = None,
                 status: TaskStatus = TaskStatus.RUNNING) -> str:
        step_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        input_json = json.dumps(input_data or {}, ensure_ascii=False, default=str)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agent_steps 
                   (step_id, task_id, agent_name, status, input_data, started_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (step_id, task_id, agent_name, status.value, input_json, now),
            )
        return step_id

    def complete_step(self, step_id: str, output_data: dict[str, Any] | None = None,
                      error: str | None = None, status: TaskStatus | None = None) -> None:
        with self._conn() as conn:
            row = conn.execute("SELECT started_at FROM agent_steps WHERE step_id = ?", (step_id,)).fetchone()
            duration_ms = None
            if row:
                started = datetime.fromisoformat(row["started_at"])
                duration_ms = int((datetime.now() - started).total_seconds() * 1000)

            final_status = status or (TaskStatus.FAILED if error else TaskStatus.SUCCESS)
            output_json = json.dumps(output_data, ensure_ascii=False, default=str) if output_data else None
            conn.execute(
                """UPDATE agent_steps SET status = ?, output_data = ?, error = ?, 
                   finished_at = ?, duration_ms = ? WHERE step_id = ?""",
                (final_status.value, output_json, error, datetime.now().isoformat(), duration_ms, step_id),
            )

    # ============ Plan Node 管理 ============

    def add_plan_node(self, step_id: str, task_id: str, plan_task_id: str, task_type: str,
                      description: str, depends_on: list[str] | None = None,
                      is_parallel: bool = False) -> str:
        node_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO plan_nodes
                   (node_id, task_id, step_id, plan_task_id, task_type,
                    description, depends_on, status, started_at, is_parallel)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node_id, task_id, step_id, plan_task_id, task_type,
                 description, json.dumps(depends_on or []), TaskStatus.RUNNING.value,
                 now, 1 if is_parallel else 0),
            )
        return node_id

    def complete_plan_node(self, node_id: str, sql: str = "", row_count: int = 0,
                           result_sample: list[dict[str, Any]] | None = None,
                           error: str | None = None, status: TaskStatus | None = None) -> None:
        with self._conn() as conn:
            row = conn.execute("SELECT started_at FROM plan_nodes WHERE node_id = ?", (node_id,)).fetchone()
            duration_ms = None
            if row:
                started = datetime.fromisoformat(row["started_at"])
                duration_ms = int((datetime.now() - started).total_seconds() * 1000)

            final_status = status or (TaskStatus.FAILED if error else TaskStatus.SUCCESS)
            sample_json = json.dumps(result_sample or [], ensure_ascii=False, default=str)
            conn.execute(
                """UPDATE plan_nodes SET status = ?, sql = ?, row_count = ?, result_sample = ?,
                   error = ?, finished_at = ?, duration_ms = ? WHERE node_id = ?""",
                (final_status.value, sql, row_count, sample_json, error,
                 datetime.now().isoformat(), duration_ms, node_id),
            )

    # ============ LLM 调用日志 ============

    def add_llm_call(self, task_id: str, caller_name: str, prompt_preview: str,
                     response_preview: str, step_id: str | None = None, node_id: str | None = None,
                     model: str = "", input_tokens: int = 0, output_tokens: int = 0,
                     duration_ms: int = 0) -> str:
        call_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO llm_calls
                   (call_id, task_id, step_id, node_id, caller_name,
                    prompt_preview, response_preview, model,
                    input_tokens, output_tokens, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (call_id, task_id, step_id, node_id, caller_name,
                 prompt_preview[:500], response_preview[:500], model,
                 input_tokens, output_tokens, duration_ms, now),
            )
        return call_id

    # ============ 查询辅助 ============

    def _get_steps(self, conn: sqlite3.Connection, task_id: str) -> list[AgentStep]:
        rows = conn.execute(
            "SELECT * FROM agent_steps WHERE task_id = ? ORDER BY started_at", (task_id,)
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def _get_nodes(self, conn: sqlite3.Connection, task_id: str) -> list[PlanNode]:
        rows = conn.execute(
            "SELECT * FROM plan_nodes WHERE task_id = ? ORDER BY started_at", (task_id,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def _get_llm_calls(self, conn: sqlite3.Connection, task_id: str) -> list[LLMCallRecord]:
        rows = conn.execute(
            "SELECT * FROM llm_calls WHERE task_id = ? ORDER BY created_at", (task_id,)
        ).fetchall()
        return [self._row_to_llm_call(r) for r in rows]

    def get_steps(self, task_id: str) -> list[AgentStep]:
        with self._conn() as conn:
            return self._get_steps(conn, task_id)

    def get_nodes(self, task_id: str) -> list[PlanNode]:
        with self._conn() as conn:
            return self._get_nodes(conn, task_id)

    def _row_to_task(self, row: sqlite3.Row, steps: list[AgentStep],
                     nodes: list[PlanNode] | None = None,
                     llm_calls: list[LLMCallRecord] | None = None) -> TaskRecord:
        keys = row.keys()
        return TaskRecord(
            task_id=row["task_id"],
            session_id=row["session_id"],
            user_query=row["user_query"],
            rewritten_query=row["rewritten_query"] if "rewritten_query" in keys else None,
            status=TaskStatus(row["status"]),
            final_sql=row["final_sql"],
            final_result=json.loads(row["final_result"]) if row["final_result"] else None,
            summary=row["summary"],
            plan_json=row["plan_json"] if "plan_json" in keys else None,
            assistant_output=row["assistant_output"] if "assistant_output" in keys else None,
            steps=steps,
            nodes=nodes or [],
            llm_calls=llm_calls or [],
            total_duration_ms=row["total_duration_ms"] if "total_duration_ms" in keys else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_step(self, row: sqlite3.Row) -> AgentStep:
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

    def _row_to_node(self, row: sqlite3.Row) -> PlanNode:
        return PlanNode(
            node_id=row["node_id"],
            task_id=row["task_id"],
            step_id=row["step_id"],
            plan_task_id=row["plan_task_id"],
            task_type=row["task_type"] or "",
            description=row["description"] or "",
            depends_on=json.loads(row["depends_on"]) if row["depends_on"] else [],
            status=TaskStatus(row["status"]),
            sql=row["sql"] or "",
            row_count=row["row_count"] or 0,
            result_sample=json.loads(row["result_sample"]) if row["result_sample"] else [],
            is_parallel=bool(row["is_parallel"]),
            error=row["error"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            duration_ms=row["duration_ms"],
        )

    def _row_to_llm_call(self, row: sqlite3.Row) -> LLMCallRecord:
        return LLMCallRecord(
            call_id=row["call_id"],
            task_id=row["task_id"],
            step_id=row["step_id"],
            caller_name=row["caller_name"],
            prompt_preview=row["prompt_preview"] or "",
            response_preview=row["response_preview"] or "",
            model=row["model"] or "",
            input_tokens=row["input_tokens"] or 0,
            output_tokens=row["output_tokens"] or 0,
            duration_ms=row["duration_ms"] or 0,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ============ 清理 ============

    def clear_session(self, session_id: str) -> None:
        with self._conn() as conn:
            task_ids = [r["task_id"] for r in conn.execute(
                "SELECT task_id FROM tasks WHERE session_id = ?", (session_id,)
            ).fetchall()]

            for tid in task_ids:
                conn.execute("DELETE FROM llm_calls WHERE task_id = ?", (tid,))
                conn.execute("DELETE FROM plan_nodes WHERE task_id = ?", (tid,))
                conn.execute("DELETE FROM agent_steps WHERE task_id = ?", (tid,))
            conn.execute("DELETE FROM tasks WHERE session_id = ?", (session_id,))

    def clear_all(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM llm_calls")
            conn.execute("DELETE FROM plan_nodes")
            conn.execute("DELETE FROM agent_steps")
            conn.execute("DELETE FROM tasks")


class TaskTracker:
    """任务追踪器"""

    def __init__(self, db: TaskHistoryDB | None = None):
        self.db = db or TaskHistoryDB()
        self._log = get_component_logger("TaskTracker")
        self._current_task_id: str | None = None
        self._current_session_id: str | None = None
        self._task_start_time: float = 0.0
        self._active_step_id: str | None = None
        self._active_node_id: str | None = None

    @property
    def task_id(self) -> str | None:
        return self._current_task_id

    @property
    def active_step_id(self) -> str | None:
        return self._active_step_id

    @property
    def active_node_id(self) -> str | None:
        return self._active_node_id

    def start_task(self, session_id: str, user_query: str, metadata: dict | None = None) -> str:
        self._current_session_id = session_id
        self._task_start_time = time.time()
        self._current_task_id = self.db.create_task(session_id, user_query, metadata)
        self.db.update_task_status(self._current_task_id, TaskStatus.RUNNING)
        self._log.debug(f"任务开始: {self._current_task_id[:8]}... query={user_query[:40]}")
        return self._current_task_id

    def set_rewritten_query(self, rewritten_query: str) -> None:
        if self._current_task_id:
            self.db.update_task_rewritten_query(self._current_task_id, rewritten_query)

    def set_plan(self, plan_display: str) -> None:
        if self._current_task_id:
            self.db.update_task_plan(self._current_task_id, plan_display)

    def set_assistant_output(self, assistant_output: str) -> None:
        if self._current_task_id:
            self.db.update_task_assistant_output(self._current_task_id, assistant_output)

    def end_task(self, final_sql: str | None = None, final_result: Any = None,
                 summary: str | None = None, error: str | None = None) -> None:
        if not self._current_task_id:
            return
        duration_ms = int((time.time() - self._task_start_time) * 1000) if self._task_start_time else None
        status = TaskStatus.FAILED if error else TaskStatus.SUCCESS
        self.db.complete_task(self._current_task_id, final_sql, final_result, summary, status, duration_ms)
        self._log.debug(f"任务完成: {self._current_task_id[:8]}... status={status.value} duration={duration_ms}ms")
        self._current_task_id = None
        self._active_step_id = None
        self._active_node_id = None

    def start_step(self, agent_name: str, input_data: dict[str, Any] | None = None) -> str:
        if not self._current_task_id:
            return ""
        step_id = self.db.add_step(self._current_task_id, agent_name, input_data)
        self._active_step_id = step_id
        self._active_node_id = None
        return step_id

    def end_step(self, step_id: str, output_data: dict[str, Any] | None = None,
                 error: str | None = None) -> None:
        if not step_id:
            return
        self.db.complete_step(step_id, output_data, error)
        if self._active_step_id == step_id:
            self._active_step_id = None
            self._active_node_id = None

    def start_node(self, step_id: str, plan_task_id: str, task_type: str, description: str,
                   depends_on: list[str] | None = None, is_parallel: bool = False) -> str:
        if not self._current_task_id or not step_id:
            return ""
        node_id = self.db.add_plan_node(
            step_id, self._current_task_id, plan_task_id,
            task_type, description, depends_on, is_parallel,
        )
        self._active_node_id = node_id
        return node_id

    def end_node(self, node_id: str, sql: str = "", row_count: int = 0,
                 result_sample: list[dict[str, Any]] | None = None,
                 error: str | None = None) -> None:
        if not node_id:
            return
        self.db.complete_plan_node(node_id, sql, row_count, result_sample, error)
        if self._active_node_id == node_id:
            self._active_node_id = None

    def log_llm_call(self, caller_name: str, prompt_preview: str, response_preview: str,
                     model: str = "", input_tokens: int = 0, output_tokens: int = 0,
                     duration_ms: int = 0) -> str:
        if not self._current_task_id:
            return ""
        return self.db.add_llm_call(
            task_id=self._current_task_id,
            caller_name=caller_name,
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            step_id=self._active_step_id,
            node_id=self._active_node_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

    def get_current_task(self) -> TaskRecord | None:
        if not self._current_task_id:
            return None
        return self.db.get_task(self._current_task_id)

    def get_session_tasks(self, session_id: str | None = None) -> list[TaskRecord]:
        sid = session_id or self._current_session_id
        if not sid:
            return []
        return self.db.get_tasks_by_session(sid)

    def get_task_token_stats(self, task_id: str | None = None) -> dict[str, Any]:
        tid = task_id or self._current_task_id
        if not tid:
            return {}

        with self.db._conn() as conn:
            calls = conn.execute(
                """SELECT caller_name, model, input_tokens, output_tokens, duration_ms 
                   FROM llm_calls WHERE task_id = ? ORDER BY created_at""",
                (tid,)
            ).fetchall()

            if not calls:
                return {}

            by_caller = {}
            total_input = 0
            total_output = 0
            total_duration = 0

            for call in calls:
                caller = call["caller_name"]
                if caller not in by_caller:
                    by_caller[caller] = {
                        "calls": 0, "input_tokens": 0, "output_tokens": 0,
                        "duration_ms": 0, "models": set()
                    }
                by_caller[caller]["calls"] += 1
                by_caller[caller]["input_tokens"] += call["input_tokens"]
                by_caller[caller]["output_tokens"] += call["output_tokens"]
                by_caller[caller]["duration_ms"] += call["duration_ms"]
                by_caller[caller]["models"].add(call["model"])
                total_input += call["input_tokens"]
                total_output += call["output_tokens"]
                total_duration += call["duration_ms"]

            for caller_stats in by_caller.values():
                caller_stats["models"] = list(caller_stats["models"])

            return {
                "task_id": tid,
                "total_calls": len(calls),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "total_duration_ms": total_duration,
                "by_caller": by_caller,
                "avg_tokens_per_call": (total_input + total_output) / len(calls) if calls else 0,
            }

    def get_session_token_stats(self, session_id: str | None = None) -> dict[str, Any]:
        sid = session_id or self._current_session_id
        if not sid:
            return {}

        with self.db._conn() as conn:
            task_ids = [r["task_id"] for r in conn.execute(
                "SELECT task_id FROM tasks WHERE session_id = ?", (sid,)
            ).fetchall()]

            if not task_ids:
                return {}

            total_stats = {
                "session_id": sid, "task_count": len(task_ids),
                "total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
                "total_tokens": 0, "total_duration_ms": 0,
                "by_caller": {}, "by_task": {}
            }

            for task_id in task_ids:
                task_stats = self.get_task_token_stats(task_id)
                if not task_stats:
                    continue
                total_stats["by_task"][task_id] = task_stats
                total_stats["total_calls"] += task_stats["total_calls"]
                total_stats["total_input_tokens"] += task_stats["total_input_tokens"]
                total_stats["total_output_tokens"] += task_stats["total_output_tokens"]
                total_stats["total_tokens"] += task_stats["total_tokens"]
                total_stats["total_duration_ms"] += task_stats["total_duration_ms"]

                for caller, stats in task_stats["by_caller"].items():
                    if caller not in total_stats["by_caller"]:
                        total_stats["by_caller"][caller] = {
                            "calls": 0, "input_tokens": 0, "output_tokens": 0,
                            "duration_ms": 0, "models": set()
                        }
                    caller_data = total_stats["by_caller"][caller]
                    caller_data["calls"] += stats["calls"]
                    caller_data["input_tokens"] += stats["input_tokens"]
                    caller_data["output_tokens"] += stats["output_tokens"]
                    caller_data["duration_ms"] += stats["duration_ms"]
                    caller_data["models"].update(stats["models"])

            for caller_stats in total_stats["by_caller"].values():
                caller_stats["models"] = list(caller_stats["models"])

            return total_stats

    def format_token_report(self, task_id: str | None = None) -> str:
        stats = self.get_task_token_stats(task_id)
        if not stats:
            return "无token使用数据"

        lines = [
            f"## Token使用报告 - 任务 {stats['task_id'][:8]}...",
            "",
            f"**总计**: {stats['total_calls']} 次调用, {stats['total_tokens']:,} tokens",
            f"- 输入: {stats['total_input_tokens']:,} tokens",
            f"- 输出: {stats['total_output_tokens']:,} tokens",
            f"- 总耗时: {stats['total_duration_ms']:,} ms",
            f"- 平均每次调用: {stats['avg_tokens_per_call']:.1f} tokens",
            "",
            "**按调用者分组**:",
        ]

        for caller, caller_stats in stats["by_caller"].items():
            total_caller_tokens = caller_stats["input_tokens"] + caller_stats["output_tokens"]
            percentage = (total_caller_tokens / stats["total_tokens"]) * 100 if stats["total_tokens"] > 0 else 0
            lines.append(
                f"- {caller}: {caller_stats['calls']} 次, "
                f"{total_caller_tokens:,} tokens ({percentage:.1f}%), "
                f"{caller_stats['duration_ms']:,} ms"
            )

        return "\n".join(lines)
