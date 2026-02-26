"""
任务执行历史存储

四张表，职责清晰：

1. tasks（大宽表）
   - 一行 = 一次用户查询的完整记录
   - 包含：session 信息 + 用户问题 + 改写 + 计划 + 执行结果 + 对话历史 + 时间/耗时

2. agent_steps（Agent 环节表）
   - 一行 = 一个 Agent 环节（semantic_parse / planner / sql_execute / summarize）
   - 记录各环节的输入/输出/耗时

3. plan_nodes（DAG 计划节点表）
   - 一行 = 一个 DAG 执行节点（ranking_1 / drilldown_2 等）
   - 通过 step_id 关联到所属的 agent_step（通常是 sql_execute）
   - 记录 SQL / 行数 / 结果样本 / 依赖关系

4. llm_calls（LLM 调用日志）
   - 一行 = 一次 LLM 请求
   - 记录 caller / prompt / response / model / tokens / 耗时
   - 通过 task_id / step_id / node_id 关联到具体环节
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

from chatdb.utils.logger import get_component_logger


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
    """
    Agent 环节记录
    
    记录 Orchestrator 中每个 Agent 环节的执行信息：
    - semantic_parse: 语义解析
    - planner: 计划生成
    - sql_execute: SQL 执行（下属 plan_nodes）
    - summarize: 结果总结
    """
    step_id: str
    task_id: str
    agent_name: str              # semantic_parse / planner / sql_execute / summarize
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
    """
    DAG 计划节点记录
    
    记录 sql_execute 阶段中每个 DAG 任务节点的执行信息。
    通过 step_id 关联到所属的 AgentStep。
    """
    node_id: str
    task_id: str
    step_id: str                 # 所属 agent_step 的 step_id
    plan_task_id: str            # DAG 节点 ID（如 ranking_1）
    task_type: str               # ranking / validation / summary 等
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
    caller_name: str           # 调用者标识（semantic_parse / sql_generate / planner 等）
    prompt_preview: str        # prompt 前 500 字
    response_preview: str      # response 前 500 字
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
    """
    完整任务记录（大宽表模型）
    
    一行 = 一次用户查询的全部信息：
    - session 归属
    - 原始/改写问题
    - 分析计划
    - 最终 SQL / 结果 / 总结
    - 对话记录（assistant_output 用于多轮上下文注入）
    - Agent 步骤列表 + Plan 节点列表
    - LLM 调用列表
    """
    task_id: str
    session_id: str
    user_query: str
    rewritten_query: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    # --- 执行结果 ---
    final_sql: str | None = None
    final_result: Any = None
    summary: str | None = None
    plan_json: str | None = None
    # --- 对话记录（用于多轮上下文） ---
    assistant_output: str | None = None   # 完整的 assistant 回复（含结果摘要+SQL）
    # --- 关联子记录 ---
    steps: list[AgentStep] = field(default_factory=list)
    nodes: list[PlanNode] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    # --- 时间 ---
    total_duration_ms: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_steps(self) -> list[AgentStep]:
        """获取 agent 环节步骤"""
        return self.steps

    @property
    def plan_nodes(self) -> list[PlanNode]:
        """获取 plan 节点步骤"""
        return self.nodes

    def get_nodes_for_step(self, step_id: str) -> list[PlanNode]:
        """获取某个 agent step 下的 plan 节点"""
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
        """生成人类可读的执行摘要"""
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

            # 该 step 下的 plan nodes
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

        if self.summary:
            lines.append(f"\n总结: {self.summary[:200]}...")
        return "\n".join(lines)


# ============================================================
# 向后兼容别名
# ============================================================

# 旧代码可能 import 这些名字，保留别名
ExecutionStep = AgentStep
PlanNodeRecord = PlanNode


# ============================================================
# 数据库层
# ============================================================

def _get_default_db_path() -> Path:
    """获取默认的历史数据库路径"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    db_dir = project_root / "data" / "pilot"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "history.db"


class TaskHistoryDB:
    """
    任务历史数据库（4 张表）
    
    表结构：
    1. tasks        — 任务大宽表（session + query + plan + result + chat history）
    2. agent_steps  — Agent 环节表（semantic_parse / planner / sql_execute / summarize）
    3. plan_nodes   — DAG 计划节点表（ranking_1 等，关联到 agent_step）
    4. llm_calls    — LLM 调用日志
    """

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
        """初始化数据库表（4 张表）"""
        with self._conn() as conn:
            # ---- 表 1: tasks（大宽表）----
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

            # ---- 表 2: agent_steps（Agent 环节表）----
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

            # ---- 表 3: plan_nodes（DAG 计划节点表）----
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

            # ---- 表 4: llm_calls（LLM 调用日志）----
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

            # ---- 索引 ----
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_steps_task ON agent_steps(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_nodes_task ON plan_nodes(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_nodes_step ON plan_nodes(step_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_task ON llm_calls(task_id)")

            # 旧表兼容迁移
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """迁移旧版数据库"""
        # tasks 表新增列
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

        # llm_calls 表新增 node_id 列
        llm_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_calls)").fetchall()}
        if "node_id" not in llm_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN node_id TEXT")

        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        # 从旧版 execution_steps 迁移数据到 agent_steps + plan_nodes
        if "execution_steps" in all_tables:
            # 迁移 agent 类型行 → agent_steps
            conn.execute("""
                INSERT OR IGNORE INTO agent_steps 
                    (step_id, task_id, agent_name, status, input_data, output_data, 
                     error, started_at, finished_at, duration_ms)
                SELECT step_id, task_id, agent_name, status, input_data, output_data,
                       error, started_at, finished_at, duration_ms
                FROM execution_steps WHERE step_type = 'agent'
            """)
            # 迁移 node 类型行 → plan_nodes
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

        # 删除更早期的旧表
        legacy_tables = ["sessions", "runs", "messages"]
        for table in legacy_tables:
            if table in all_tables:
                conn.execute(f"DROP TABLE {table}")  # noqa: S608

    # ============ 任务管理 ============

    def create_task(
        self,
        session_id: str,
        user_query: str,
        metadata: dict | None = None,
    ) -> str:
        """创建新任务"""
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
        """更新任务状态"""
        with self._conn() as conn:
            finished_at = datetime.now().isoformat() if status in (TaskStatus.SUCCESS, TaskStatus.FAILED) else None
            conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ? WHERE task_id = ?",
                (status.value, finished_at, task_id),
            )

    def update_task_rewritten_query(self, task_id: str, rewritten_query: str) -> None:
        """更新改写后的查询"""
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET rewritten_query = ? WHERE task_id = ?", (rewritten_query, task_id))

    def update_task_plan(self, task_id: str, plan_json: str) -> None:
        """更新分析计划"""
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET plan_json = ? WHERE task_id = ?", (plan_json, task_id))

    def update_task_assistant_output(self, task_id: str, assistant_output: str) -> None:
        """更新 assistant 输出（用于多轮对话上下文）"""
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET assistant_output = ? WHERE task_id = ?", (assistant_output, task_id))

    def complete_task(
        self,
        task_id: str,
        final_sql: str | None = None,
        final_result: Any = None,
        summary: str | None = None,
        status: TaskStatus = TaskStatus.SUCCESS,
        total_duration_ms: int | None = None,
    ) -> None:
        """完成任务"""
        with self._conn() as conn:
            result_json = json.dumps(final_result, ensure_ascii=False, default=str) if final_result else None
            conn.execute(
                """UPDATE tasks SET status = ?, final_sql = ?, final_result = ?, summary = ?,
                   total_duration_ms = ?, finished_at = ? WHERE task_id = ?""",
                (status.value, final_sql, result_json, summary, total_duration_ms,
                 datetime.now().isoformat(), task_id),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        """获取完整任务记录（含 steps + nodes + llm_calls）"""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            steps = self._get_steps(conn, task_id)
            nodes = self._get_nodes(conn, task_id)
            llm_calls = self._get_llm_calls(conn, task_id)
            return self._row_to_task(row, steps, nodes, llm_calls)

    def get_tasks_by_session(self, session_id: str, limit: int = 20) -> list[TaskRecord]:
        """获取会话的任务列表"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [
                self._row_to_task(
                    r,
                    self._get_steps(conn, r["task_id"]),
                    self._get_nodes(conn, r["task_id"]),
                    self._get_llm_calls(conn, r["task_id"]),
                )
                for r in rows
            ]

    def get_recent_tasks(self, limit: int = 20) -> list[TaskRecord]:
        """获取最近的任务"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                self._row_to_task(
                    r,
                    self._get_steps(conn, r["task_id"]),
                    self._get_nodes(conn, r["task_id"]),
                    self._get_llm_calls(conn, r["task_id"]),
                )
                for r in rows
            ]

    # ============ 对话历史查询（替代旧 ChatHistoryDB 的功能）============

    def get_chat_history(self, session_id: str, num_runs: int = 3) -> list[dict[str, str]]:
        """
        获取会话的对话历史（chat 格式），用于多轮上下文注入。
        
        返回最近 num_runs 轮对话：
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT user_query, assistant_output FROM tasks 
                   WHERE session_id = ? AND status = 'success' AND assistant_output IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, num_runs),
            ).fetchall()

        # 按时间正序返回
        history = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["user_query"]})
            history.append({"role": "assistant", "content": row["assistant_output"]})
        return history

    def get_chat_history_context(self, session_id: str, num_runs: int = 3) -> str:
        """
        获取格式化的历史上下文字符串（用于注入到 prompt）
        """
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
        """搜索历史任务"""
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

    def add_step(
        self,
        task_id: str,
        agent_name: str,
        input_data: dict[str, Any] | None = None,
        status: TaskStatus = TaskStatus.RUNNING,
    ) -> str:
        """添加 Agent 步骤"""
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

    def complete_step(
        self,
        step_id: str,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        status: TaskStatus | None = None,
    ) -> None:
        """完成 Agent 步骤"""
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

    def add_plan_node(
        self,
        step_id: str,
        task_id: str,
        plan_task_id: str,
        task_type: str,
        description: str,
        depends_on: list[str] | None = None,
        is_parallel: bool = False,
    ) -> str:
        """添加 DAG 计划节点"""
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

    def complete_plan_node(
        self,
        node_id: str,
        sql: str = "",
        row_count: int = 0,
        result_sample: list[dict[str, Any]] | None = None,
        error: str | None = None,
        status: TaskStatus | None = None,
    ) -> None:
        """完成 DAG 计划节点"""
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

    def add_llm_call(
        self,
        task_id: str,
        caller_name: str,
        prompt_preview: str,
        response_preview: str,
        step_id: str | None = None,
        node_id: str | None = None,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
    ) -> str:
        """记录 LLM 调用"""
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
        """获取任务的所有 Agent 步骤"""
        rows = conn.execute(
            "SELECT * FROM agent_steps WHERE task_id = ? ORDER BY started_at", (task_id,)
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def _get_nodes(self, conn: sqlite3.Connection, task_id: str) -> list[PlanNode]:
        """获取任务的所有 Plan 节点"""
        rows = conn.execute(
            "SELECT * FROM plan_nodes WHERE task_id = ? ORDER BY started_at", (task_id,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def _get_llm_calls(self, conn: sqlite3.Connection, task_id: str) -> list[LLMCallRecord]:
        """获取任务的所有 LLM 调用"""
        rows = conn.execute(
            "SELECT * FROM llm_calls WHERE task_id = ? ORDER BY created_at", (task_id,)
        ).fetchall()
        return [self._row_to_llm_call(r) for r in rows]

    def get_steps(self, task_id: str) -> list[AgentStep]:
        """获取任务的所有步骤"""
        with self._conn() as conn:
            return self._get_steps(conn, task_id)

    def get_nodes(self, task_id: str) -> list[PlanNode]:
        """获取任务的所有节点"""
        with self._conn() as conn:
            return self._get_nodes(conn, task_id)

    def _row_to_task(
        self,
        row: sqlite3.Row,
        steps: list[AgentStep],
        nodes: list[PlanNode] | None = None,
        llm_calls: list[LLMCallRecord] | None = None,
    ) -> TaskRecord:
        """行转任务记录"""
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
        """行转 Agent 步骤"""
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
        """行转 Plan 节点"""
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
        """行转 LLM 调用记录"""
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
        """清空会话的任务"""
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
        """清空所有数据"""
        with self._conn() as conn:
            conn.execute("DELETE FROM llm_calls")
            conn.execute("DELETE FROM plan_nodes")
            conn.execute("DELETE FROM agent_steps")
            conn.execute("DELETE FROM tasks")


class TaskTracker:
    """
    任务追踪器
    
    在 Orchestrator 的 process_query 流程中使用，追踪：
    1. 任务级别：一次完整的用户查询
    2. 步骤级别：各 Agent 环节（语义解析 / Planner / SQL 执行 / 总结）
    3. 节点级别：DAG 计划中的每个执行节点
    4. LLM 调用：每次 LLM 请求的 prompt/response/token
    
    用法：
        tracker = TaskTracker(db)
        task_id = tracker.start_task(session_id, query)
        
        # 语义解析
        step_id = tracker.start_step("semantic_parse", {"query": query})
        tracker.end_step(step_id, {"intent": {...}})
        
        # SQL 执行（含 plan nodes）
        step_id = tracker.start_step("sql_execute")
        node_id = tracker.start_node(step_id, "ranking_1", "query", "查询Top3产品")
        tracker.end_node(node_id, sql="SELECT ...", row_count=3, result_sample=[...])
        tracker.end_step(step_id)
        
        tracker.end_task(summary="...")
    """

    def __init__(self, db: TaskHistoryDB | None = None):
        self.db = db or TaskHistoryDB()
        self._log = get_component_logger("TaskTracker")
        self._current_task_id: str | None = None
        self._current_session_id: str | None = None
        self._task_start_time: float = 0.0
        # 当前活跃的 step_id（用于 LLM call 自动关联）
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

    # ---- Task ----

    def start_task(self, session_id: str, user_query: str, metadata: dict | None = None) -> str:
        """开始新任务"""
        self._current_session_id = session_id
        self._task_start_time = time.time()
        self._current_task_id = self.db.create_task(session_id, user_query, metadata)
        self.db.update_task_status(self._current_task_id, TaskStatus.RUNNING)
        self._log.debug(f"任务开始: {self._current_task_id[:8]}... query={user_query[:40]}")
        return self._current_task_id

    def set_rewritten_query(self, rewritten_query: str) -> None:
        """记录改写后的查询"""
        if self._current_task_id:
            self.db.update_task_rewritten_query(self._current_task_id, rewritten_query)

    def set_plan(self, plan_display: str) -> None:
        """记录分析计划"""
        if self._current_task_id:
            self.db.update_task_plan(self._current_task_id, plan_display)

    def set_assistant_output(self, assistant_output: str) -> None:
        """记录 assistant 输出（用于多轮对话上下文）"""
        if self._current_task_id:
            self.db.update_task_assistant_output(self._current_task_id, assistant_output)

    def end_task(
        self,
        final_sql: str | None = None,
        final_result: Any = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """完成任务"""
        if not self._current_task_id:
            return
        duration_ms = int((time.time() - self._task_start_time) * 1000) if self._task_start_time else None
        status = TaskStatus.FAILED if error else TaskStatus.SUCCESS
        self.db.complete_task(
            self._current_task_id, final_sql, final_result, summary, status, duration_ms,
        )
        self._log.debug(f"任务完成: {self._current_task_id[:8]}... status={status.value} duration={duration_ms}ms")
        self._current_task_id = None
        self._active_step_id = None
        self._active_node_id = None

    # ---- Step ----

    def start_step(self, agent_name: str, input_data: dict[str, Any] | None = None) -> str:
        """开始 Agent 步骤"""
        if not self._current_task_id:
            return ""
        step_id = self.db.add_step(self._current_task_id, agent_name, input_data)
        self._active_step_id = step_id
        self._active_node_id = None
        return step_id

    def end_step(
        self,
        step_id: str,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """完成 Agent 步骤"""
        if not step_id:
            return
        self.db.complete_step(step_id, output_data, error)
        if self._active_step_id == step_id:
            self._active_step_id = None
            self._active_node_id = None

    # ---- Plan Node ----

    def start_node(
        self,
        step_id: str,
        plan_task_id: str,
        task_type: str,
        description: str,
        depends_on: list[str] | None = None,
        is_parallel: bool = False,
    ) -> str:
        """开始 DAG 计划节点"""
        if not self._current_task_id or not step_id:
            return ""
        node_id = self.db.add_plan_node(
            step_id, self._current_task_id, plan_task_id,
            task_type, description, depends_on, is_parallel,
        )
        self._active_node_id = node_id
        return node_id

    def end_node(
        self,
        node_id: str,
        sql: str = "",
        row_count: int = 0,
        result_sample: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        """完成 DAG 计划节点"""
        if not node_id:
            return
        self.db.complete_plan_node(node_id, sql, row_count, result_sample, error)
        if self._active_node_id == node_id:
            self._active_node_id = None

    # ---- LLM Call ----

    def log_llm_call(
        self,
        caller_name: str,
        prompt_preview: str,
        response_preview: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
    ) -> str:
        """记录 LLM 调用（自动关联当前活跃的 step/node）"""
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

    # ---- 查询 ----

    def get_current_task(self) -> TaskRecord | None:
        """获取当前任务"""
        if not self._current_task_id:
            return None
        return self.db.get_task(self._current_task_id)

    def get_session_tasks(self, session_id: str | None = None) -> list[TaskRecord]:
        """获取会话的所有任务"""
        sid = session_id or self._current_session_id
        if not sid:
            return []
        return self.db.get_tasks_by_session(sid)
