"""
Agent 间显式消息对象

设计理念：
- Agent 之间通过不可变消息传递信息，而非共享可变状态
- Orchestrator 作为唯一的消息路由器和状态管理者
- 每个消息携带完整的上下文，接收方无需读取全局 state

三种核心消息：
1. TaskRequest:  Orchestrator → SQLAgent（执行指令）
2. TaskResponse: SQLAgent → Orchestrator（执行结果）
3. PlanDecision: Planner → Orchestrator（调度决策）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# TaskRequest: Orchestrator → SQLAgent
# ============================================================

@dataclass(frozen=True)
class TaskRequest:
    """
    Orchestrator 发给 SQLAgent 的任务执行请求

    包含 SQLAgent 执行一个任务所需的全部信息，
    SQLAgent 不再需要从 state 读取 temp_results 等通信字段。
    """
    # 任务描述（来自 Planner 的 AnalysisTask）
    task_id: str
    task_type: str
    description: str
    notes: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    # 上游任务结果摘要（Orchestrator 从 collected_results 构建）
    parent_results_summary: str = ""

    # 之前已收集的结果快照（只读，用于 drilldown 等需要前置结果的场景）
    previous_results: tuple[dict[str, Any], ...] = ()

    # 上游任务的临时表映射 {task_id: {name: str, columns: list, row_count: int}}
    upstream_temp_tables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_planner_task(
        cls,
        task_dict: dict[str, Any],
        *,
        parent_results_summary: str = "",
        previous_results: list[dict[str, Any]] | None = None,
        upstream_temp_tables: dict[str, Any] | None = None,
    ) -> TaskRequest:
        """从 Planner 输出的 task dict 构建"""
        return cls(
            task_id=task_dict.get("id", "unknown"),
            task_type=task_dict.get("type", "basic"),
            description=task_dict.get("description", ""),
            notes=tuple(task_dict.get("notes", [])),
            depends_on=tuple(task_dict.get("depends_on", [])),
            meta=dict(task_dict.get("meta", {})),
            parent_results_summary=parent_results_summary,
            previous_results=tuple(previous_results or []),
            upstream_temp_tables=dict(upstream_temp_tables or {}),
        )


# ============================================================
# TaskResponse: SQLAgent → Orchestrator
# ============================================================

@dataclass
class TaskResultEntry:
    """单个子任务的执行结果"""
    subtask: str
    sql: str = ""
    row_count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    # 完整行数据（供临时表创建使用，不参与序列化）
    all_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask": self.subtask,
            "sql": self.sql,
            "row_count": self.row_count,
            "examples": self.examples,
            "stats": self.stats,
            "issues": self.issues,
        }


@dataclass
class TaskResponse:
    """
    SQLAgent 返回给 Orchestrator 的任务执行结果

    Orchestrator 收到后决定：
    - 写入 collected_results（原 temp_results）
    - 转发给 Planner 做决策
    """
    task_id: str
    success: bool = True
    results: list[TaskResultEntry] = field(default_factory=list)
    error: str | None = None

    def to_results_dicts(self) -> list[dict[str, Any]]:
        """转为 dict 列表（兼容旧 temp_results 格式）"""
        return [r.to_dict() for r in self.results]


# ============================================================
# PlanDecision: Planner → Orchestrator
# ============================================================

@dataclass
class PlanDecision:
    """
    Planner 返回给 Orchestrator 的调度决策

    action 类型：
    - "continue": 按计划执行下一个任务
    - "done":     计划完成，可以生成总结
    - "adjust":   调整策略（修改筛选条件等）
    - "retry":    重试当前任务（附带修复建议）
    """
    action: str  # "continue" | "done" | "adjust" | "retry"
    reason: str = ""

    # action="done" 时的结论
    conclusion: str = ""

    # action="adjust" 时的调整策略
    adjustment: dict[str, Any] = field(default_factory=dict)

    # action="retry" 时的修复建议
    retry_hint: str = ""

    # 原始决策数据（兼容旧逻辑）
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanDecision:
        """从旧格式 dict 构建"""
        action = data.get("action", "done")
        return cls(
            action=action,
            reason=data.get("reason", ""),
            conclusion=data.get("conclusion", ""),
            adjustment=data.get("adjustment", {}),
            retry_hint=data.get("retry_hint", ""),
            raw=data,
        )
