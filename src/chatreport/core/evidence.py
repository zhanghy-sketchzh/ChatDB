"""
EvidenceCollector — 证据链管理

新概念，无 chatdb 直接对标。
为报告中的每个数据引用建立可追溯的证据记录。

核心能力：
1. 记录每次数据查询的证据（SQL、数据摘要、来源表）
2. 按章节组织证据链
3. 支持查询 "某个结论的数据依据是什么"
4. 导出证据链 JSON（用于 scratch 存储 / AG-UI 事件）
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from chatreport.core.react_state import DataResult


@dataclass
class Evidence:
    """单条证据 — 对应一次数据查询"""
    id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:8]}")
    section_id: str = ""
    question: str = ""
    sql: str = ""
    row_count: int = 0
    data_summary: str = ""
    source_table: str = ""
    virtual_fields_used: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "section_id": self.section_id,
            "question": self.question,
            "sql": self.sql,
            "row_count": self.row_count,
            "data_summary": self.data_summary,
            "source_table": self.source_table,
            "virtual_fields_used": self.virtual_fields_used,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_data_result(
        cls,
        data_result: DataResult,
        section_id: str,
    ) -> Evidence:
        """从 DataResult 构建证据"""
        # 从 SQL 中尝试提取表名
        source_table = _extract_table_name(data_result.sql) if data_result.sql else ""

        return cls(
            section_id=section_id,
            question=data_result.question,
            sql=data_result.sql,
            row_count=data_result.row_count,
            data_summary=data_result.summary[:500] if data_result.summary else "",
            source_table=source_table,
        )


class EvidenceCollector:
    """
    证据链收集器 — 按章节管理所有数据查询证据

    使用方式：
        collector = EvidenceCollector()
        eid = collector.record(data_result, section_id="s1")
        section_evidence = collector.get_section_evidence("s1")
    """

    def __init__(self):
        self._evidences: dict[str, Evidence] = {}           # id → Evidence
        self._section_index: dict[str, list[str]] = {}      # section_id → [evidence_id]

    def record(self, data_result: DataResult, section_id: str) -> str:
        """
        记录一条证据

        Args:
            data_result: 数据查询结果
            section_id: 所属章节 ID

        Returns:
            证据 ID
        """
        evidence = Evidence.from_data_result(data_result, section_id)
        self._evidences[evidence.id] = evidence

        if section_id not in self._section_index:
            self._section_index[section_id] = []
        self._section_index[section_id].append(evidence.id)

        return evidence.id

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return self._evidences.get(evidence_id)

    def get_section_evidence(self, section_id: str) -> list[Evidence]:
        """获取某章节的所有证据"""
        ids = self._section_index.get(section_id, [])
        return [self._evidences[eid] for eid in ids if eid in self._evidences]

    def get_all_evidence(self) -> list[Evidence]:
        """获取所有证据（按时间序）"""
        return list(self._evidences.values())

    @property
    def total_count(self) -> int:
        return len(self._evidences)

    def to_dict(self) -> dict[str, Any]:
        """导出完整证据链"""
        result: dict[str, Any] = {}
        for section_id, eids in self._section_index.items():
            result[section_id] = {
                "evidences": [
                    self._evidences[eid].to_dict()
                    for eid in eids
                    if eid in self._evidences
                ]
            }
        return result

    def section_summary(self, section_id: str) -> str:
        """生成章节证据摘要（用于 Writer prompt）"""
        evidences = self.get_section_evidence(section_id)
        if not evidences:
            return "无数据查询证据"

        lines = []
        for i, e in enumerate(evidences, 1):
            lines.append(
                f"  [{i}] Q: {e.question}\n"
                f"      SQL: {e.sql[:100]}{'...' if len(e.sql) > 100 else ''}\n"
                f"      结果: {e.row_count} 行 — {e.data_summary[:150]}"
            )
        return f"共 {len(evidences)} 条数据查询:\n" + "\n".join(lines)


def _extract_table_name(sql: str) -> str:
    """从 SQL 中简单提取第一个 FROM 后的表名"""
    if not sql:
        return ""
    sql_upper = sql.upper()
    idx = sql_upper.find("FROM")
    if idx < 0:
        return ""
    rest = sql[idx + 4:].strip()
    # 取第一个 token
    token = rest.split()[0] if rest.split() else ""
    # 去掉可能的引号
    return token.strip("`\"'")
