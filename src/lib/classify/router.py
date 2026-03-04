"""
QueryRouter — 统一查询分类器

决定请求走 ChatDB（数据查询）还是 ChatReport（报告生成）。

分类策略（由快到慢）：
1. 关键词快速匹配 → 命中即返回
2. 若提供 LLM 则做精确分类（类似 chatdb 的 _quick_classify）
3. 兜底 → 默认走 chatdb
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lib.llm import BaseLLM


@dataclass
class RouteDecision:
    """路由决策结果"""

    target: Literal["chatdb", "chatreport"]  # 目标引擎
    query_type: str  # 细分类型 (basic/comparison/ranking/trend/report/chat/...)
    confidence: float  # 置信度 0-1


# ── 关键词集合 ──────────────────────────────────────────────

_REPORT_KEYWORDS: list[str] = [
    "报告", "分析报告", "研究报告", "深度分析", "深度报告",
    "总结报告", "调研报告", "全面分析", "综合报告", "详细分析",
    "趋势报告", "写一份", "生成报告", "出一份", "做一个报告",
    "report", "analysis report", "deep research", "comprehensive analysis",
    "generate a report", "write a report",
]

_REPORT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(帮我|请|给我).{0,8}(写|生成|出|做).{0,4}(报告|分析|研究)", re.IGNORECASE),
    re.compile(r"(全面|深度|详细|综合).{0,4}(分析|研究|报告)", re.IGNORECASE),
    re.compile(r"write.{0,10}report", re.IGNORECASE),
    re.compile(r"generate.{0,10}(report|analysis)", re.IGNORECASE),
]

# chatdb 典型关键词（优先级低于 report）
_QUERY_KEYWORDS: list[str] = [
    "查询", "多少", "哪些", "排名", "top", "最高", "最低",
    "平均", "总计", "合计", "统计", "增长率", "同比", "环比",
    "select", "count", "sum", "avg", "group by",
]

# ── LLM 分类 prompt ─────────────────────────────────────────

_CLASSIFY_PROMPT = """\
你是查询分类器。根据用户输入判断意图类型，仅输出一个JSON：

分类规则：
- "report"：要求生成报告、深度分析、研究报告、综合分析等结构化长文
- "query"：数据查询、统计、排名、趋势查询等短回答
- "chat"：闲聊、常识问答、与数据分析无关

用户输入：{query}

仅输出JSON，不要其他文字：
{{"type": "report"|"query"|"chat", "confidence": 0.0-1.0}}"""


class QueryRouter:
    """
    统一查询分类器 — 决定请求走 ChatDB 还是 ChatReport。

    用法::

        router = QueryRouter()
        decision = await router.classify("帮我写一份销售分析报告")
        # RouteDecision(target="chatreport", query_type="report", confidence=0.95)

        decision = await router.classify("今年销售额最高的产品是什么")
        # RouteDecision(target="chatdb", query_type="query", confidence=0.90)
    """

    def __init__(self, report_keywords: list[str] | None = None):
        self._report_keywords = report_keywords or _REPORT_KEYWORDS

    async def classify(
        self,
        query: str,
        llm: "BaseLLM | None" = None,
    ) -> RouteDecision:
        """
        分类入口。

        Parameters
        ----------
        query : str
            用户自然语言输入
        llm : BaseLLM | None
            若提供则做精确 LLM 分类；否则仅用规则

        Returns
        -------
        RouteDecision
        """
        # 1. 关键词快速匹配
        decision = self._keyword_classify(query)
        if decision is not None:
            return decision

        # 2. LLM 精确分类
        if llm is not None:
            decision = await self._llm_classify(query, llm)
            if decision is not None:
                return decision

        # 3. 兜底 → chatdb
        return RouteDecision(target="chatdb", query_type="query", confidence=0.5)

    # ── 内部方法 ────────────────────────────────────────

    def _keyword_classify(self, query: str) -> RouteDecision | None:
        """基于关键词和正则的快速分类"""
        q_lower = query.lower().strip()

        # 检查 report 正则
        for pat in _REPORT_PATTERNS:
            if pat.search(q_lower):
                return RouteDecision(
                    target="chatreport", query_type="report", confidence=0.90
                )

        # 检查 report 关键词
        for kw in self._report_keywords:
            if kw.lower() in q_lower:
                return RouteDecision(
                    target="chatreport", query_type="report", confidence=0.85
                )

        # 检查典型查询关键词（增强信心但不做最终决策 — 留给 LLM）
        query_hit = any(kw.lower() in q_lower for kw in _QUERY_KEYWORDS)
        if query_hit:
            return RouteDecision(
                target="chatdb", query_type="query", confidence=0.80
            )

        return None

    async def _llm_classify(
        self, query: str, llm: "BaseLLM"
    ) -> RouteDecision | None:
        """使用 LLM 做精确分类"""
        import json as _json

        prompt = _CLASSIFY_PROMPT.format(query=query)
        try:
            response = await llm.chat(
                prompt=prompt,
                temperature=0.0,
                caller_name="query_router",
            )
            text = response.strip()
            # 提取 JSON
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                data = _json.loads(json_str)
            else:
                return None

            qtype = data.get("type", "query")
            conf = float(data.get("confidence", 0.7))

            if qtype == "report":
                return RouteDecision(
                    target="chatreport", query_type="report", confidence=conf
                )
            elif qtype == "chat":
                return RouteDecision(
                    target="chatdb", query_type="chat", confidence=conf
                )
            else:
                return RouteDecision(
                    target="chatdb", query_type="query", confidence=conf
                )
        except Exception:
            return None
