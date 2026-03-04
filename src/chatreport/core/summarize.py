"""
TransitionPlanner — 章节承上启下与上下文串联

对标 chatdb/core/summarize.py 的 SummarizeAnswerTool：
- chatdb 的 summarize 将多阶段分析结果汇总为自然语言回答
- TransitionPlanner 为每个已完成章节生成结构化的 transition 上下文

transition 的作用：
1. 为后续章节提供前序章节的关键发现
2. 动态调整后续章节的分析方向
3. 保证报告整体逻辑连贯

输出格式（参见 todo.md 九章）：
{
    "section_id": "s1",
    "section_title": "...",
    "section_summary": "...",
    "key_findings": [...],
    "data_warnings": [...],
    "adjustments": [{"target_section_id": "s3", "hint": "..."}],
    "coherence_notes": "..."
}
"""

from __future__ import annotations

from typing import Any

from lib.llm import BaseLLM
from lib.utils.common import parse_json
from lib.utils.logger import get_component_logger

from chatreport.config.report_config import ReportConfig
from chatreport.core.react_state import DataResult, SectionSpec


class TransitionPlanner:
    """
    章节承上启下规划器

    核心能力：
    1. 从已完成章节中提炼结构化摘要（key_findings + data_warnings）
    2. 为后续章节生成动态调整建议（adjustments）
    3. 生成衔接说明（coherence_notes），帮助 Writer 保持行文连贯

    使用方式：
        planner = TransitionPlanner(llm)
        transition = await planner.plan_transition(
            section=section_spec,
            section_content="...",
            data_results=[...],
            outline=outline_dict,
        )
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: ReportConfig | None = None,
    ):
        self.llm = llm
        self.config = config or ReportConfig()
        self._log = get_component_logger("TransitionPlanner")

    async def plan_transition(
        self,
        section: SectionSpec,
        section_content: str,
        data_results: list[DataResult],
        outline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        为已完成的章节生成 transition 上下文

        Args:
            section: 章节规格
            section_content: 已完成的章节正文
            data_results: 该章节的数据查询结果
            outline: 全局大纲

        Returns:
            结构化 transition JSON
        """
        if not self.config.transition_enabled:
            return self._minimal_transition(section, section_content)

        # 后续章节信息
        downstream_info = self._get_downstream_sections(outline, section.id)
        data_summary = self._format_data_summary(data_results)

        prompt = f"""## 任务：为已完成的报告章节生成承上启下上下文

### 已完成章节
- **ID**: {section.id}
- **标题**: {section.title}
- **描述**: {section.description}

### 章节正文
{section_content[:2000]}

### 数据查询摘要
{data_summary}

### 后续章节
{downstream_info}

### 要求

请提取该章节的核心发现和上下文信息，供后续章节参考。

1. **section_summary**: 一句话概括本章核心结论
2. **key_findings**: 3-5 个关键数据发现（每个一句话，必须包含具体数字）
3. **data_warnings**: 数据质量问题或分析局限（如样本不完整、异常值等）
4. **adjustments**: 对后续章节的分析建议
   - target_section_id: 目标章节 ID
   - hint: 具体建议（如 "应重点关注Q3-Q4的加速下滑"）
5. **coherence_notes**: 与后续章节的逻辑衔接说明

### 输出格式

```json
{{
  "section_id": "{section.id}",
  "section_title": "{section.title}",
  "section_summary": "...",
  "key_findings": ["...", "..."],
  "data_warnings": ["..."],
  "adjustments": [
    {{"target_section_id": "...", "hint": "..."}}
  ],
  "coherence_notes": "..."
}}
```"""

        self._log.info(f"生成章节 transition: [{section.id}] {section.title}")

        response = await self.llm.chat(
            prompt=prompt,
            system_prompt=(
                "你是报告编辑专家，擅长从已完成的分析章节中提炼关键发现，"
                "并为后续章节提供有价值的上下文和方向建议。"
                "提取的信息要具体（含数据），建议要可操作。"
            ),
            caller_name="transition_planner",
        )

        transition = parse_json(response)
        transition = self._validate_transition(transition, section)

        self._log.info(
            f"Transition 完成: [{section.id}] "
            f"{len(transition.get('key_findings', []))} 个发现, "
            f"{len(transition.get('adjustments', []))} 个建议"
        )
        return transition

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _minimal_transition(
        self, section: SectionSpec, content: str
    ) -> dict[str, Any]:
        """transition 关闭时生成最小化上下文"""
        # 取正文前 200 字符作为摘要
        summary = content[:200].strip()
        if len(content) > 200:
            summary += "..."
        return {
            "section_id": section.id,
            "section_title": section.title,
            "section_summary": summary,
            "key_findings": [],
            "data_warnings": [],
            "adjustments": [],
            "coherence_notes": "",
        }

    def _get_downstream_sections(
        self, outline: dict[str, Any] | None, current_id: str
    ) -> str:
        """获取后续章节信息"""
        if not outline:
            return "（无全局大纲信息）"

        sections = outline.get("sections", [])
        found_current = False
        downstream: list[str] = []

        for s in sections:
            if s.get("id") == current_id:
                found_current = True
                continue
            if found_current:
                deps = s.get("dependencies", [])
                dep_str = f" (依赖: {', '.join(deps)})" if deps else ""
                downstream.append(
                    f"- [{s['id']}] {s.get('title', '')}: {s.get('description', '')}{dep_str}"
                )

        return "\n".join(downstream) if downstream else "（当前是最后一个章节）"

    def _format_data_summary(self, data_results: list[DataResult]) -> str:
        """格式化数据查询摘要"""
        if not data_results:
            return "（无数据查询结果）"

        lines: list[str] = []
        for i, dr in enumerate(data_results, 1):
            if dr.success:
                lines.append(
                    f"[{i}] {dr.question} → {dr.row_count} 行: {dr.summary[:200]}"
                )
            else:
                lines.append(f"[{i}] {dr.question} → 失败: {dr.error}")
        return "\n".join(lines)

    def _validate_transition(
        self, transition: dict[str, Any], section: SectionSpec
    ) -> dict[str, Any]:
        """验证并补全 transition"""
        if not transition:
            return self._minimal_transition(section, "")

        # 确保必要字段
        transition.setdefault("section_id", section.id)
        transition.setdefault("section_title", section.title)
        transition.setdefault("section_summary", "")
        transition.setdefault("key_findings", [])
        transition.setdefault("data_warnings", [])
        transition.setdefault("adjustments", [])
        transition.setdefault("coherence_notes", "")

        # 确保 key_findings 是 list[str]
        transition["key_findings"] = [
            str(f) for f in transition["key_findings"] if f
        ]
        # 确保 adjustments 格式正确
        valid_adjustments = []
        for adj in transition.get("adjustments", []):
            if isinstance(adj, dict) and "target_section_id" in adj:
                valid_adjustments.append(adj)
        transition["adjustments"] = valid_adjustments

        return transition
