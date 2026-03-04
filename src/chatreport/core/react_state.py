"""
ReportState — 报告级状态管理

对标 chatdb/core/react_state.py 的 ReActState，
但语义粒度是「报告 → 章节 → 子问题」而非「SQL → 执行 → 反思」。

核心数据结构：
- ReportState: 报告全局状态
- SectionSpec: 大纲中的章节规格
- SectionResult: 单章节执行结果
- SubQuestion: 章节内的子问题
- DataResult: 单次数据查询结果
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# 枚举
# ============================================================

class ReportPhase(str, Enum):
    """报告生成阶段"""
    INIT = "init"
    OUTLINE = "outline"                # 大纲生成
    SECTION_PLANNING = "section_plan"  # 章节子问题规划
    DATA_QUERY = "data_query"          # 数据查询
    SECTION_WRITING = "section_write"  # 章节撰写
    VERIFICATION = "verification"      # 质量审计
    ASSEMBLY = "assembly"              # 全局拼装
    DONE = "done"
    ERROR = "error"


class SectionStatus(str, Enum):
    """章节执行状态"""
    PENDING = "pending"
    PLANNING = "planning"      # 正在规划子问题
    QUERYING = "querying"      # 正在查数
    WRITING = "writing"        # 正在撰写
    VERIFYING = "verifying"    # 正在审计
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# 子问题 & 数据结果
# ============================================================

@dataclass
class SubQuestion:
    """章节内的子问题 — 直接交给 ChatDB 查询的自然语言问题"""
    question: str
    purpose: str = ""
    priority: int = 1
    # 查询后回填
    result: DataResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "purpose": self.purpose,
            "priority": self.priority,
            "has_result": self.result is not None,
        }


@dataclass
class DataResult:
    """单次数据查询结果 — ChatDB process_query 的返回封装"""
    success: bool = False
    summary: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)
    sql: str = ""
    row_count: int = 0
    error: str | None = None
    # 元信息
    question: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "sql": self.sql,
            "row_count": self.row_count,
            "error": self.error,
            "question": self.question,
            "elapsed_ms": self.elapsed_ms,
            "data_preview": self.data[:5] if self.data else [],
        }


# ============================================================
# 章节规格 & 结果
# ============================================================

@dataclass
class SectionSpec:
    """大纲中的章节规格 — 由 ReportPlanner.generate_outline 生成"""
    id: str
    title: str
    description: str = ""
    data_needs: str = ""
    dependencies: list[str] = field(default_factory=list)
    analysis_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "data_needs": self.data_needs,
            "dependencies": self.dependencies,
            "analysis_hints": self.analysis_hints,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SectionSpec:
        return cls(
            id=d["id"],
            title=d["title"],
            description=d.get("description", ""),
            data_needs=d.get("data_needs", ""),
            dependencies=d.get("dependencies", []),
            analysis_hints=d.get("analysis_hints", []),
        )


@dataclass
class SectionResult:
    """单章节执行结果"""
    section_id: str
    title: str = ""
    status: SectionStatus = SectionStatus.PENDING
    # 子问题 & 数据
    sub_questions: list[SubQuestion] = field(default_factory=list)
    data_results: list[DataResult] = field(default_factory=list)
    # 产出
    content: str = ""                     # 章节 Markdown 正文
    transition: dict[str, Any] | None = None    # 承上启下上下文
    verification: dict[str, Any] | None = None  # 质量审计结果
    # 元信息
    evidence_ids: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "status": self.status.value,
            "sub_question_count": len(self.sub_questions),
            "data_result_count": len(self.data_results),
            "content_length": len(self.content),
            "has_transition": self.transition is not None,
            "has_verification": self.verification is not None,
            "evidence_ids": self.evidence_ids,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


# ============================================================
# 报告全局状态
# ============================================================

@dataclass
class ReportState:
    """
    报告级状态管理 — 对标 chatdb ReActState

    chatdb ReActState 管理的是「一次 SQL 查询」的状态，
    ReportState 管理的是「一份完整报告」的状态。

    层级关系：
    ReportState
      └── sections: list[SectionResult]
            └── sub_questions: list[SubQuestion]
                  └── result: DataResult
    """
    # 输入
    user_query: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # 阶段
    phase: ReportPhase = ReportPhase.INIT

    # 大纲
    title: str = ""
    outline: dict[str, Any] = field(default_factory=dict)
    section_specs: list[SectionSpec] = field(default_factory=list)

    # 章节执行结果
    sections: dict[str, SectionResult] = field(default_factory=dict)

    # 最终产出
    final_markdown: str = ""
    global_verification: dict[str, Any] | None = None

    # 控制参数
    max_sections: int = 7
    max_sub_questions_per_section: int = 5
    verify_enabled: bool = True

    # 元信息
    start_time: float = field(default_factory=time.time)
    error: str | None = None

    # ---- 便捷方法 ----

    def set_outline(self, outline: dict[str, Any]) -> None:
        """设置大纲并初始化章节状态"""
        self.outline = outline
        self.title = outline.get("title", "")
        self.section_specs = [
            SectionSpec.from_dict(s) for s in outline.get("sections", [])
        ]
        for spec in self.section_specs:
            self.sections[spec.id] = SectionResult(
                section_id=spec.id, title=spec.title
            )
        self.phase = ReportPhase.SECTION_PLANNING

    def get_section(self, section_id: str) -> SectionResult | None:
        return self.sections.get(section_id)

    def update_section_status(self, section_id: str, status: SectionStatus) -> None:
        if section_id in self.sections:
            self.sections[section_id].status = status

    def get_completed_sections(self) -> list[SectionResult]:
        """获取已完成的章节（按大纲顺序）"""
        order = [s.id for s in self.section_specs]
        return [
            self.sections[sid]
            for sid in order
            if sid in self.sections and self.sections[sid].status == SectionStatus.COMPLETED
        ]

    def get_previous_transitions(self, current_section_id: str) -> list[dict[str, Any]]:
        """获取当前章节之前所有已完成章节的 transition"""
        transitions = []
        for spec in self.section_specs:
            if spec.id == current_section_id:
                break
            sr = self.sections.get(spec.id)
            if sr and sr.transition:
                transitions.append(sr.transition)
        return transitions

    @property
    def elapsed_seconds(self) -> float:
        return round(time.time() - self.start_time, 2)

    @property
    def section_count(self) -> int:
        return len(self.section_specs)

    @property
    def evidence_count(self) -> int:
        return sum(len(sr.evidence_ids) for sr in self.sections.values())

    @property
    def all_sections_done(self) -> bool:
        return all(
            sr.status in (SectionStatus.COMPLETED, SectionStatus.FAILED)
            for sr in self.sections.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "session_id": self.session_id,
            "phase": self.phase.value,
            "title": self.title,
            "section_count": self.section_count,
            "evidence_count": self.evidence_count,
            "elapsed_seconds": self.elapsed_seconds,
            "sections": {
                sid: sr.to_dict() for sid, sr in self.sections.items()
            },
            "error": self.error,
        }
