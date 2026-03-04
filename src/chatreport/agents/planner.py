"""
ReportPlanner — 报告规划 Agent

对标 chatdb/agents/planner.py + chatdb/agents/semantic_parser.py：
- chatdb Planner 将用户问题拆解为 SQL 任务 DAG
- ReportPlanner 将用户问题拆解为报告章节大纲 + 章节内子问题

两个核心方法：
1. generate_outline(): 大纲生成 — 对标 SemanticParser 的意图解析
2. plan_sub_questions(): 章节子问题规划 — 对标 Planner 的任务分解
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from lib.llm import BaseLLM
from lib.utils.common import parse_json, parse_json_array
from lib.utils.logger import get_component_logger

from chatreport.config.report_config import ReportConfig
from chatreport.core.react_state import SectionSpec, SubQuestion

if TYPE_CHECKING:
    pass


class ReportPlanner:
    """
    报告规划 Agent

    职责：
    1. 根据用户查询 + 数据上下文，生成结构化报告大纲
    2. 为每个章节规划可直接查数的自然语言子问题

    设计原则：
    - 大纲中的 dependencies 决定章节间并行/串行（交给 SectionDAG）
    - 子问题是"人话"，直接丢给 ChatDB 的 process_query
    - ReportPlanner 不关心 SQL 怎么写
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: ReportConfig | None = None,
    ):
        self.llm = llm
        self.config = config or ReportConfig()
        self._log = get_component_logger("ReportPlanner")

    # ================================================================
    # Stage 1: 大纲生成
    # ================================================================

    async def generate_outline(
        self,
        query: str,
        data_context: dict[str, Any] | None = None,
        chat_history: list[dict[str, str]] | None = None,
        max_sections: int | None = None,
    ) -> dict[str, Any]:
        """
        生成报告大纲 — 对标 chatdb SemanticParser 的意图解析

        Args:
            query: 用户原始需求（如 "帮我生成一份2024年流水分析报告"）
            data_context: 数据上下文（表元数据、YML 配置等）
            chat_history: 聊天历史
            max_sections: 最大章节数

        Returns:
            大纲 JSON:
            {
                "title": "...",
                "sections": [
                    {
                        "id": "s1",
                        "title": "...",
                        "description": "...",
                        "data_needs": "...",
                        "dependencies": [],
                        "analysis_hints": ["trend", "comparison"]
                    },
                    ...
                ]
            }
        """
        effective_max = max_sections or self.config.max_sections

        # 构建数据上下文描述
        data_context_text = self._format_data_context(data_context)
        history_text = self._format_chat_history(chat_history)

        prompt = f"""## 任务：为以下数据分析需求生成报告大纲

### 用户需求
{query}

{history_text}

### 可用数据上下文
{data_context_text}

### 大纲要求

1. **章节数**：{self.config.min_sections}-{effective_max} 个章节，根据需求复杂度灵活决定
2. **dependencies**：如果某章节的分析需要其他章节的结论作为前提，必须在 dependencies 中声明
   - 无依赖的章节可以并行执行
   - 典型模式：概述/趋势 → 结构分析 → 归因/下钻 → 结论
3. **analysis_hints**：标注每个章节可能涉及的分析类型
   - 可选值：trend, comparison, ranking, ratio, source, drilldown, correlation, anomaly, basic
4. **data_needs**：明确描述该章节需要什么数据，越具体越好
5. **逻辑递进**：章节间应该有清晰的逻辑链条，后面的章节应该在前面的基础上深入

### 输出格式

严格输出 JSON（不要包含任何其他内容）：

```json
{{
  "title": "报告标题",
  "sections": [
    {{
      "id": "s1",
      "title": "章节标题",
      "description": "该章节分析什么",
      "data_needs": "需要什么数据",
      "dependencies": [],
      "analysis_hints": ["trend"]
    }}
  ]
}}
```"""

        self._log.info(f"生成报告大纲: {query[:80]}...")

        response = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是资深数据分析师，擅长规划结构清晰、逻辑递进的数据分析报告。"
                "根据用户需求和可用数据，生成合理的报告大纲。"
                "大纲应该覆盖用户需求的各个方面，章节间有清晰的依赖关系。"
            ),
            caller_name="report_planner_outline",
        )

        outline = parse_json(response)
        outline = self._validate_outline(outline, effective_max)

        self._log.info(
            f"大纲生成完成: {outline.get('title', '')}, "
            f"{len(outline.get('sections', []))} 个章节"
        )
        return outline

    # ================================================================
    # Stage 2: 章节子问题规划
    # ================================================================

    async def plan_sub_questions(
        self,
        section: SectionSpec,
        previous_transitions: list[dict[str, Any]] | None = None,
        outline: dict[str, Any] | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> list[SubQuestion]:
        """
        为章节规划子问题 — 对标 chatdb Planner 的任务分解

        每个子问题是一个可直接交给 ChatDB 查询的自然语言问题。
        子问题按 priority 排序，priority=1 最优先执行。

        Args:
            section: 章节规格
            previous_transitions: 前序章节的承上启下上下文
            outline: 全局大纲（让规划器了解全局视角）
            data_context: 数据上下文

        Returns:
            子问题列表
        """
        # 构建前序上下文
        prev_context = self._format_previous_transitions(previous_transitions)
        outline_context = self._format_outline_context(outline, section.id)
        data_context_text = self._format_data_context(data_context)

        prompt = f"""## 任务：为报告章节规划数据查询子问题

### 章节信息
- **标题**: {section.title}
- **描述**: {section.description}
- **数据需求**: {section.data_needs}
- **分析类型**: {', '.join(section.analysis_hints) if section.analysis_hints else '通用分析'}

### 全局大纲视角
{outline_context}

### 前序章节的关键发现
{prev_context}

### 可用数据上下文
{data_context_text}

### 规划要求

1. 每个子问题是一个**可直接回答的自然语言数据查询**
   - 好的例子: "2024年1-12月每月流水总额是多少？"
   - 差的例子: "分析流水趋势"（太模糊，无法直接查询）
2. 子问题数量: 2-{self.config.max_sub_questions_per_section} 个
3. 每个子问题需要说明 purpose（为什么要查这个数据）
4. priority 表示执行顺序（1 最先执行），后面的查询可能引用前面的结果
5. 如果前序章节已经给出了某些数据，不要重复查询

### 输出格式

严格输出 JSON 数组：

```json
[
  {{
    "question": "具体的数据查询问题",
    "purpose": "查这个数据的目的",
    "priority": 1
  }}
]
```"""

        self._log.info(f"规划章节子问题: [{section.id}] {section.title}")

        response = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是数据分析规划专家。请将章节的分析目标拆解为具体的、可执行的数据查询问题。"
                "每个问题应该足够具体，可以直接用 SQL 查询回答。"
                "注意利用前序章节的发现来避免重复查询，并引导分析方向。"
            ),
            caller_name="report_planner_sub_questions",
        )

        raw_questions = parse_json_array(response)
        questions = self._validate_sub_questions(raw_questions, section.id)

        self._log.info(
            f"子问题规划完成: [{section.id}] {len(questions)} 个子问题"
        )
        return questions

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _format_data_context(self, data_context: dict[str, Any] | None) -> str:
        """格式化数据上下文描述"""
        if not data_context:
            return "（无数据上下文信息）"

        lines: list[str] = []

        # 表元数据
        tables_meta = data_context.get("tables_meta")
        if tables_meta:
            if isinstance(tables_meta, list):
                for tm in tables_meta:
                    tname = tm.get("table_name", "unknown")
                    tdesc = tm.get("table_description", "")
                    cols = tm.get("columns_info", [])
                    lines.append(f"**表: {tname}** — {tdesc}")
                    if cols:
                        col_strs = [
                            f"  - {c.get('name', '')}: {c.get('type', '')}"
                            for c in cols[:30]  # 限制列数
                        ]
                        lines.extend(col_strs)
                        if len(cols) > 30:
                            lines.append(f"  ... 共 {len(cols)} 列")
                    lines.append("")
            else:
                lines.append(str(tables_meta)[:2000])

        # YML 配置
        yml = data_context.get("yml_config")
        if yml:
            lines.append("**YML 配置（可用指标/维度/条件）**:")
            lines.append(str(yml)[:3000])

        return "\n".join(lines) if lines else "（无数据上下文信息）"

    def _format_chat_history(
        self, chat_history: list[dict[str, str]] | None
    ) -> str:
        """格式化聊天历史"""
        if not chat_history:
            return ""
        lines = ["### 聊天历史"]
        for msg in chat_history[-5:]:  # 最多保留最近 5 轮
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            lines.append(f"- **{role}**: {content}")
        return "\n".join(lines)

    def _format_previous_transitions(
        self, transitions: list[dict[str, Any]] | None
    ) -> str:
        """格式化前序章节的 transition 上下文"""
        if not transitions:
            return "（这是第一个章节，无前序上下文）"

        lines: list[str] = []
        for t in transitions:
            summary = t.get("section_summary", "")
            findings = t.get("key_findings", [])
            warnings = t.get("data_warnings", [])
            adjustments = t.get("adjustments", [])

            lines.append(f"**{t.get('section_title', '前序章节')}**:")
            if summary:
                lines.append(f"  摘要: {summary}")
            if findings:
                lines.append("  关键发现:")
                for f in findings:
                    lines.append(f"    - {f}")
            if warnings:
                lines.append("  数据警告:")
                for w in warnings:
                    lines.append(f"    - {w}")
            if adjustments:
                for adj in adjustments:
                    hint = adj.get("hint", "")
                    if hint:
                        lines.append(f"  建议: {hint}")
            lines.append("")

        return "\n".join(lines)

    def _format_outline_context(
        self, outline: dict[str, Any] | None, current_section_id: str
    ) -> str:
        """格式化大纲全局视角"""
        if not outline:
            return "（无全局大纲信息）"

        sections = outline.get("sections", [])
        lines = [f"报告标题: {outline.get('title', '')}"]
        for s in sections:
            marker = " ← 当前" if s.get("id") == current_section_id else ""
            deps = s.get("dependencies", [])
            dep_str = f" (依赖: {', '.join(deps)})" if deps else ""
            lines.append(f"  [{s.get('id', '')}] {s.get('title', '')}{dep_str}{marker}")
        return "\n".join(lines)

    def _validate_outline(
        self, outline: dict[str, Any], max_sections: int
    ) -> dict[str, Any]:
        """验证并修正大纲"""
        if not outline or "sections" not in outline:
            self._log.warn("大纲解析失败，使用默认大纲")
            return {
                "title": "数据分析报告",
                "sections": [
                    {
                        "id": "s1",
                        "title": "数据概览",
                        "description": "整体数据概况",
                        "data_needs": "基础统计数据",
                        "dependencies": [],
                        "analysis_hints": ["basic"],
                    },
                    {
                        "id": "s2",
                        "title": "详细分析",
                        "description": "深入分析",
                        "data_needs": "详细分析数据",
                        "dependencies": ["s1"],
                        "analysis_hints": ["trend"],
                    },
                    {
                        "id": "s3",
                        "title": "结论与建议",
                        "description": "总结发现并给出建议",
                        "data_needs": "基于前序分析",
                        "dependencies": ["s2"],
                        "analysis_hints": ["basic"],
                    },
                ],
            }

        sections = outline["sections"]

        # 确保不超过最大章节数
        if len(sections) > max_sections:
            self._log.warn(
                f"章节数 {len(sections)} 超过上限 {max_sections}，截断"
            )
            sections = sections[:max_sections]
            outline["sections"] = sections

        # 确保每个章节有必要字段
        valid_ids = set()
        for i, s in enumerate(sections):
            if "id" not in s:
                s["id"] = f"s{i + 1}"
            valid_ids.add(s["id"])
            s.setdefault("title", f"章节 {i + 1}")
            s.setdefault("description", "")
            s.setdefault("data_needs", "")
            s.setdefault("dependencies", [])
            s.setdefault("analysis_hints", [])

        # 清理无效依赖
        for s in sections:
            s["dependencies"] = [
                d for d in s["dependencies"] if d in valid_ids and d != s["id"]
            ]

        if "title" not in outline or not outline["title"]:
            outline["title"] = "数据分析报告"

        return outline

    def _validate_sub_questions(
        self, raw: list[Any], section_id: str
    ) -> list[SubQuestion]:
        """验证并构建子问题列表"""
        if not raw:
            self._log.warn(f"[{section_id}] 子问题解析为空，生成默认子问题")
            return [
                SubQuestion(
                    question="请查询该章节相关的基础统计数据",
                    purpose="获取基础数据",
                    priority=1,
                )
            ]

        questions: list[SubQuestion] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            q = item.get("question", "").strip()
            if not q:
                continue
            questions.append(
                SubQuestion(
                    question=q,
                    purpose=item.get("purpose", ""),
                    priority=item.get("priority", len(questions) + 1),
                )
            )

        # 按 priority 排序
        questions.sort(key=lambda x: x.priority)

        # 限制数量
        max_q = self.config.max_sub_questions_per_section
        if len(questions) > max_q:
            self._log.warn(
                f"[{section_id}] 子问题数 {len(questions)} 超过上限 {max_q}，截断"
            )
            questions = questions[:max_q]

        return questions
