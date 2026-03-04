"""
ReportWriter — 报告撰写 Agent

对标 chatdb/agents/sql_agent.py（执行 + 产出角色）和 chatdb/core/summarize.py：
- SQLAgent 执行 SQL 并产出查询结果
- ReportWriter 根据数据结果撰写章节内容 + 全局拼装 + 质量审计

三个核心方法：
1. write_section(): 撰写单章节 Markdown
2. assemble_report(): 全局拼装最终报告
3. verify(): 质量审计（章节级 + 全局级）
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from lib.llm import BaseLLM
from lib.utils.common import parse_json
from lib.utils.logger import get_component_logger

from chatreport.config.report_config import ReportConfig
from chatreport.core.react_state import (
    DataResult,
    ReportState,
    SectionResult,
    SectionSpec,
)

if TYPE_CHECKING:
    from chatreport.core.evidence import EvidenceCollector


class ReportWriter:
    """
    报告撰写 Agent

    职责：
    1. 根据数据查询结果撰写单章节 Markdown
    2. 将所有章节拼装为最终报告
    3. 质量审计：检测无证据断言、跨章节不一致

    对标 chatdb 的 SQLAgent + SummarizeAnswerTool：
    - SQLAgent 是"执行者"（执行 SQL）
    - SummarizeAnswerTool 是"总结者"（总结查询结果）
    - ReportWriter 同时扮演两个角色：执行撰写 + 总结拼装
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: ReportConfig | None = None,
    ):
        self.llm = llm
        self.config = config or ReportConfig()
        self._log = get_component_logger("ReportWriter")

    # ================================================================
    # 章节撰写
    # ================================================================

    async def write_section(
        self,
        section: SectionSpec,
        data_results: list[DataResult],
        previous_transitions: list[dict[str, Any]] | None = None,
        evidence_summary: str = "",
    ) -> str:
        """
        撰写单章节 Markdown — 对标 SQLAgent 的单次 SQL 执行

        Args:
            section: 章节规格
            data_results: 该章节所有子问题的数据查询结果
            previous_transitions: 前序章节的承上启下上下文
            evidence_summary: 证据摘要文本

        Returns:
            章节 Markdown 正文
        """
        # 构建数据上下文
        data_context = self._format_data_results(data_results)
        prev_context = self._format_transitions_for_writing(previous_transitions)

        prompt = f"""## 任务：撰写报告章节

### 章节信息
- **标题**: {section.title}
- **描述**: {section.description}
- **分析类型**: {', '.join(section.analysis_hints) if section.analysis_hints else '通用分析'}

### 前序章节摘要
{prev_context}

### 数据查询结果
{data_context}

### 证据链
{evidence_summary or '无额外证据信息'}

### 撰写要求

1. **数据驱动**：所有结论必须有数据支撑，严禁编造数字
2. **结构清晰**：
   - 以核心发现开篇（1-2 句话）
   - 数据分析展开（使用表格、列表等结构化呈现）
   - 小结（呼应章节主题）
3. **承上启下**：
   - 如有前序章节，自然地引用和衔接
   - 为后续章节埋下伏笔
4. **格式规范**：
   - 输出 Markdown 格式
   - 关键数字加粗
   - 合理使用表格呈现对比数据
   - 不要输出章节标题的 # 标记（标题由拼装阶段统一处理）
5. **单位换算**：
   - 如果数据中提到了单位换算规则（display_divisor/unit），严格遵守
   - 同一指标在不同地方的数值量级必须一致

### 输出

直接输出章节正文 Markdown（不要包含章节标题）"""

        self._log.info(f"撰写章节: [{section.id}] {section.title}")

        content = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是资深数据分析师，擅长将数据分析结果撰写成清晰、专业的报告章节。"
                "严格基于提供的数据撰写，不得编造任何数据。"
                "文风要专业但不晦涩，突出关键数据和洞察。"
            ),
            caller_name="report_writer_section",
        )

        self._log.info(
            f"章节撰写完成: [{section.id}] {len(content)} 字符"
        )
        return content.strip()

    # ================================================================
    # 全局拼装
    # ================================================================

    async def assemble_report(
        self,
        state: ReportState,
        evidence_collector: EvidenceCollector | None = None,
    ) -> str:
        """
        全局拼装最终报告

        将所有章节内容 + 摘要 + 数据口径 + 证据链 组装为完整报告。

        Args:
            state: 报告全局状态
            evidence_collector: 证据链收集器

        Returns:
            完整报告 Markdown
        """
        self._log.info("开始全局拼装报告...")

        # 收集各章节内容
        section_contents: list[str] = []
        for spec in state.section_specs:
            sr = state.get_section(spec.id)
            if sr and sr.content:
                section_contents.append(
                    f"## {spec.title}\n\n{sr.content}"
                )
            else:
                section_contents.append(
                    f"## {spec.title}\n\n（该章节数据不足，暂无内容）"
                )

        body = "\n\n---\n\n".join(section_contents)

        # 构建拼装 prompt
        prompt = f"""## 任务：将以下章节内容拼装为完整的数据分析报告

### 报告标题
{state.title}

### 用户原始需求
{state.user_query}

### 各章节内容

{body}

### 拼装要求

1. **添加摘要**：在报告开头添加 "Executive Summary"（200 字内，概括核心结论）
2. **添加数据口径说明**：简要说明数据来源、分析时间范围、关键口径定义
3. **章节衔接**：确保章节之间有自然的过渡语句，逻辑递进
4. **添加结论**：如果最后一章不是"结论"类型，在末尾添加简洁的结论与建议
5. **格式统一**：
   - 报告标题用 # 一级标题
   - 章节标题用 ## 二级标题
   - 小节用 ### 三级标题
   - 关键数据加粗
6. **不要改动数据**：章节中已有的具体数字不得修改

### 输出

输出完整的 Markdown 报告"""

        # 如需证据链附录
        if self.config.include_evidence and evidence_collector:
            evidence_text = self._format_evidence_appendix(evidence_collector)
            if evidence_text:
                prompt += f"""

### 证据链附录
请在报告末尾附上以下证据链索引：

{evidence_text}"""

        report = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是专业的报告编辑，擅长将分散的章节内容编排成逻辑通顺、格式规范的完整报告。"
                "保持各章节数据的准确性，不得修改具体数字。"
                "添加必要的过渡衔接，使报告读起来流畅自然。"
            ),
            caller_name="report_writer_assembly",
        )

        self._log.info(f"报告拼装完成: {len(report)} 字符")
        return report.strip()

    # ================================================================
    # 质量审计
    # ================================================================

    async def verify(
        self,
        content: str,
        data_results: list[DataResult],
        mode: str = "section",
        all_sections: list[SectionResult] | None = None,
        section_spec: SectionSpec | None = None,
    ) -> dict[str, Any]:
        """
        质量审计 — 对标 chatdb SummarizeAnswerTool 的 _validate_summary

        Args:
            content: 待审计的内容（章节内容或完整报告）
            data_results: 数据查询结果
            mode: "section"（章节级）或 "global"（全局级）
            all_sections: 所有章节结果（仅全局级需要）
            section_spec: 章节规格（仅章节级，用于 data_needs 覆盖检查）

        Returns:
            审计结果 JSON:
            {
                "passed": True/False,
                "issues": [...],
                "suggestions": [],
                "supplementary_queries": []   # 建议补充的查询
            }
        """
        if not self.config.verify_enabled:
            return {"passed": True, "issues": [], "suggestions": [], "supplementary_queries": []}

        if mode == "section" and not self.config.verify_section_level:
            return {"passed": True, "issues": [], "suggestions": [], "supplementary_queries": []}

        if mode == "global" and not self.config.verify_global_level:
            return {"passed": True, "issues": [], "suggestions": [], "supplementary_queries": []}

        data_context = self._format_data_results(data_results)

        if mode == "section":
            prompt = self._build_section_verify_prompt(
                content, data_context, section_spec
            )
        else:
            sections_context = self._format_all_sections(all_sections)
            prompt = self._build_global_verify_prompt(
                content, data_context, sections_context
            )

        self._log.info(f"质量审计 ({mode})...")

        response = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是数据质量审计专家。严格检查报告内容的准确性和一致性。"
                "只基于提供的数据进行验证，不引入外部知识。"
                "以 JSON 格式输出审计结果，不要输出其他文字。"
            ),
            temperature=self.config.verify_temperature,
            caller_name=f"report_writer_verify_{mode}",
        )

        result = parse_json(response)
        if not result:
            # JSON 解析失败，判断是否通过
            passed = "通过" in response or "PASSED" in response.upper()
            result = {
                "passed": passed,
                "issues": [] if passed else [{"type": "parse_error", "detail": "审计结果解析失败"}],
                "suggestions": [],
                "supplementary_queries": [],
            }

        # 确保 supplementary_queries 字段存在
        if "supplementary_queries" not in result:
            result["supplementary_queries"] = []

        # 基于 issues 数量判断 passed（LLM 可能遗漏）
        issues = result.get("issues", [])
        ungrounded = [i for i in issues if i.get("type") == "ungrounded_claim"]
        if len(ungrounded) > self.config.max_ungrounded_claims:
            result["passed"] = False

        self._log.info(
            f"审计完成 ({mode}): {'通过' if result.get('passed') else '有问题'}, "
            f"{len(issues)} 个问题, "
            f"{len(result['supplementary_queries'])} 条补充查询建议"
        )
        return result

    # ================================================================
    # 内容修正（审计后）
    # ================================================================

    async def revise_section(
        self,
        original_content: str,
        verification: dict[str, Any],
        supplementary_data: list[DataResult] | None = None,
        section_spec: SectionSpec | None = None,
    ) -> str:
        """
        根据审计结果修正章节内容

        Args:
            original_content: 原始章节内容
            verification: verify() 返回的审计结果
            supplementary_data: 补充查询的数据结果
            section_spec: 章节规格

        Returns:
            修正后的章节 Markdown
        """
        issues = verification.get("issues", [])
        if not issues:
            return original_content

        # 格式化问题列表
        issues_text = "\n".join(
            f"  {i + 1}. [{iss.get('type', 'unknown')}] {iss.get('detail', '')}"
            + (f"\n     建议: {iss['suggestion']}" if iss.get("suggestion") else "")
            for i, iss in enumerate(issues)
        )

        # 格式化补充数据
        supp_data_text = ""
        if supplementary_data:
            supp_data_text = self._format_data_results(supplementary_data)

        section_title = section_spec.title if section_spec else "（未知章节）"

        prompt = f"""## 任务：根据审计结果修正章节内容

### 章节: {section_title}

### 原始内容
{original_content}

### 审计发现的问题
{issues_text}

### 补充查询数据
{supp_data_text or '（无补充数据）'}

### 修正要求

1. **修正无证据断言**：删除或修改没有数据支撑的具体数字/结论
2. **补充遗漏内容**：如果有补充数据，将其融入章节中
3. **保持原有结构和文风**：修正应最小化改动，不要重写整个章节
4. **不引入新的无据断言**：修正后的内容同样必须有数据支撑

### 输出

直接输出修正后的章节正文 Markdown（不要包含章节标题）"""

        self._log.info(f"修正章节: {section_title}, {len(issues)} 个问题")

        revised = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是数据报告编辑，擅长根据审计反馈精确修正报告内容。"
                "保持最小化修改原则，只修正有问题的部分。"
            ),
            caller_name="report_writer_revise",
        )

        self._log.info(
            f"章节修正完成: {section_title}, "
            f"{len(original_content)} → {len(revised)} 字符"
        )
        return revised.strip()

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _format_data_results(self, data_results: list[DataResult]) -> str:
        """格式化数据查询结果"""
        if not data_results:
            return "（无数据查询结果）"

        lines: list[str] = []
        for i, dr in enumerate(data_results, 1):
            lines.append(f"**查询 {i}**: {dr.question}")
            if dr.success:
                lines.append(f"  SQL: `{dr.sql[:150]}{'...' if len(dr.sql) > 150 else ''}`")
                lines.append(f"  结果: {dr.row_count} 行")
                lines.append(f"  摘要: {dr.summary[:300]}")
                # 展示前几行数据
                if dr.data:
                    preview = dr.data[:5]
                    lines.append(f"  数据预览: {_format_data_preview(preview)}")
            else:
                lines.append(f"  失败: {dr.error or '未知错误'}")
            lines.append("")

        return "\n".join(lines)

    def _format_transitions_for_writing(
        self, transitions: list[dict[str, Any]] | None
    ) -> str:
        """格式化前序 transition 用于写作"""
        if not transitions:
            return "（这是报告的第一个章节）"

        lines: list[str] = []
        for t in transitions:
            summary = t.get("section_summary", "")
            findings = t.get("key_findings", [])
            if summary:
                lines.append(f"- {summary}")
            for f in findings[:3]:
                lines.append(f"  - {f}")
        return "\n".join(lines) if lines else "（无前序章节摘要）"

    def _format_evidence_appendix(
        self, collector: EvidenceCollector
    ) -> str:
        """格式化证据链附录"""
        all_evidence = collector.get_all_evidence()
        if not all_evidence:
            return ""

        lines = ["### 数据来源证据链\n"]
        lines.append("| # | 章节 | 查询问题 | SQL 概要 | 结果行数 |")
        lines.append("|---|------|----------|----------|----------|")
        for i, e in enumerate(all_evidence, 1):
            sql_brief = e.sql[:60] + "..." if len(e.sql) > 60 else e.sql
            lines.append(
                f"| {i} | {e.section_id} | {e.question[:40]} | `{sql_brief}` | {e.row_count} |"
            )
        return "\n".join(lines)

    def _format_all_sections(
        self, sections: list[SectionResult] | None
    ) -> str:
        """格式化所有章节（用于全局审计）"""
        if not sections:
            return "（无章节数据）"

        lines: list[str] = []
        for sr in sections:
            lines.append(f"### [{sr.section_id}] {sr.title}")
            lines.append(sr.content[:500] if sr.content else "（无内容）")
            lines.append("")
        return "\n".join(lines)

    def _build_section_verify_prompt(
        self, content: str, data_context: str,
        section_spec: SectionSpec | None = None,
    ) -> str:
        """构建章节级审计 prompt"""
        data_needs_text = ""
        if section_spec and section_spec.data_needs:
            data_needs_text = f"""
### 章节数据需求（大纲定义）
{section_spec.data_needs}
"""

        return f"""## 质量审计：章节级检查

### 章节内容
{content}

### 数据查询结果（作为事实依据）
{data_context}
{data_needs_text}
### 检查项目

1. **无证据断言检测**：章节中出现的具体数字，是否都能在数据查询结果中找到对应？
   - 如果有具体数字无法对应到查询结果，标记为 "ungrounded_claim"
   - 包括百分比、金额、增长率等所有量化数据
2. **数据覆盖检查**：数据查询结果中的重要发现，是否在章节中被充分利用？
   - 如果大纲的 data_needs 中提到的数据需求未被覆盖，标记为 "data_gap"
3. **逻辑一致性**：章节内部的陈述是否前后一致？
   - 例如前面说增长，后面又说下降
4. **单位一致性**：同一指标在不同地方的单位和量级是否一致？

### 输出格式

严格输出以下 JSON，不要有其他文字：

```json
{{
  "passed": true/false,
  "issues": [
    {{
      "type": "ungrounded_claim/data_gap/inconsistency/unit_mismatch",
      "detail": "具体描述",
      "suggestion": "改进建议"
    }}
  ],
  "suggestions": ["可选的改进建议"],
  "supplementary_queries": [
    "如果有数据缺口，这里列出建议补充查询的自然语言问题"
  ]
}}
```"""

    def _build_global_verify_prompt(
        self, content: str, data_context: str, sections_context: str
    ) -> str:
        """构建全局级审计 prompt"""
        return f"""## 质量审计：全局级检查

### 完整报告
{content[:5000]}

### 各章节概要
{sections_context}

### 数据查询结果汇总
{data_context[:3000]}

### 检查项目

1. **跨章节数据一致性**：不同章节引用同一指标的数字是否一致？
   - 重点检查：总量值、百分比、增长率等在不同章节中是否矛盾
   - 例如：A 章节说总流水 100 亿，B 章节说总流水 120 亿
2. **结论矛盾检测**：不同章节的结论是否存在逻辑冲突？
   - 例如：A 章节说销售上升，B 章节说市场萎缩
3. **遗漏检测**：用户需求中的关键方面是否都被覆盖？
4. **过渡连贯性**：章节间的逻辑过渡是否自然？
5. **数字精度**：报告中引用的数字精度是否一致（小数位数等）？

### 输出格式

严格输出以下 JSON，不要有其他文字：

```json
{{
  "passed": true/false,
  "issues": [
    {{
      "type": "inconsistency/contradiction/gap/coherence/precision",
      "detail": "具体描述",
      "sections": ["涉及的章节 ID"],
      "suggestion": "改进建议"
    }}
  ],
  "suggestions": ["可选的全局改进建议"],
  "supplementary_queries": [
    "如果有数据缺口需要补充查询，在这里列出"
  ]
}}
```"""


def _format_data_preview(data: list[dict[str, Any]], max_rows: int = 5) -> str:
    """格式化数据预览为紧凑文本"""
    if not data:
        return "无数据"

    sample = data[:max_rows]
    if not sample:
        return "无数据"

    keys = list(sample[0].keys())
    # 表头
    lines = [" | ".join(str(k) for k in keys)]
    lines.append(" | ".join("---" for _ in keys))
    # 数据行
    for row in sample:
        values = [str(row.get(k, ""))[:30] for k in keys]
        lines.append(" | ".join(values))

    if len(data) > max_rows:
        lines.append(f"... 共 {len(data)} 行")

    return "\n" + "\n".join(lines)
