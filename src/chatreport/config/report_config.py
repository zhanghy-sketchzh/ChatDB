"""
ReportConfig — 报告配置

对标 chatdb/config/domain_config.py，
管理报告模板、章节上限、质量阈值等参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportConfig:
    """
    报告生成配置 — 控制报告的规模、质量和行为

    对标 chatdb 的 DomainConfig（管控查询行为），
    ReportConfig 管控报告生成行为。
    """

    # ---- 大纲控制 ----
    min_sections: int = 3
    max_sections: int = 7
    default_sections: int = 5

    # ---- 章节控制 ----
    max_sub_questions_per_section: int = 5
    max_query_retries: int = 2           # 单次查询失败重试次数
    query_timeout_seconds: float = 60.0  # 单次查询超时

    # ---- 质量审计 ----
    verify_enabled: bool = True
    verify_section_level: bool = True    # 章节级检查
    verify_global_level: bool = True     # 全局级检查
    max_ungrounded_claims: int = 2       # 允许的最大无证据断言数

    # ---- 报告格式 ----
    include_evidence: bool = True        # 报告末尾附证据链
    include_data_appendix: bool = False  # 报告末尾附原始数据
    include_methodology: bool = True     # 报告开头附数据口径说明
    include_executive_summary: bool = True  # 报告开头附摘要

    # ---- 承上启下 ----
    transition_enabled: bool = True      # 是否生成章节间过渡

    # ---- 并行控制 ----
    max_parallel_sections: int = 3       # 最大并行章节数
    max_parallel_queries: int = 2        # 章节内最大并行查询数（当前设计为串行，预留）

    # ---- LLM 参数 ----
    outline_temperature: float = 0.3     # 大纲生成温度（偏创意）
    writing_temperature: float = 0.2     # 撰写温度（偏严谨）
    verify_temperature: float = 0.0      # 审计温度（完全确定性）

    # ---- 报告模板（预设）----
    report_templates: dict[str, Any] = field(default_factory=lambda: {
        "standard": {
            "sections": ["概述", "趋势分析", "结构分析", "原因分析", "结论与建议"],
            "description": "标准分析报告模板",
        },
        "executive": {
            "sections": ["摘要", "关键发现", "建议"],
            "description": "简洁决策报告",
        },
        "deep_research": {
            "sections": ["研究背景", "数据概览", "多维分析", "交叉分析", "风险提示", "结论"],
            "description": "深度研究报告",
        },
    })

    def get_template(self, name: str) -> dict[str, Any] | None:
        """获取预设模板"""
        return self.report_templates.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_sections": self.min_sections,
            "max_sections": self.max_sections,
            "max_sub_questions_per_section": self.max_sub_questions_per_section,
            "verify_enabled": self.verify_enabled,
            "transition_enabled": self.transition_enabled,
            "max_parallel_sections": self.max_parallel_sections,
            "include_evidence": self.include_evidence,
            "include_methodology": self.include_methodology,
        }
