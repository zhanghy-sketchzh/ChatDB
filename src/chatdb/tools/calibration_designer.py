"""
口径设计器 (Calibration Designer) - LLM 驱动的智能口径设计

核心职责：
- 接收 StructuredIntent（含 filter_refs）和用户原始问题
- 由 LLM 判断这是什么类型的计算（single / ratio / trend / comparison）
- 对于 ratio 类型，智能区分"分子筛选"和"分母筛选"
- 输出结构化的 CalibrationPlan，供 SQL 生成器使用

设计理念：
- 不依赖关键词硬判断（如"包含'占比'字样"）
- 让 LLM 理解业务语义，做智能决策
- YAML 只提供筛选器的元信息（group, merge_mode），不决定流程
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import json
import re

from chatdb.llm.base import BaseLLM
from chatdb.utils.logger import get_component_logger

if TYPE_CHECKING:
    from chatdb.agents.semantic_parser import StructuredIntent


@dataclass
class CalibrationPlan:
    """
    口径设计方案
    
    核心字段：
    - calculation_type: 计算类型（single/ratio/trend/comparison/aggregation）
    - numerator_filters: 分子专用筛选（仅用于 CASE WHEN）
    - denominator_filters: 分母筛选（如果是 ratio 类型）
    - global_filters: 全局筛选（WHERE 条件，影响分子和分母）
    - sql_pattern: SQL 模式建议（case_when_ratio / simple_agg / window_func）
    """
    calculation_type: str = "single"  # single / ratio / trend / comparison / aggregation
    
    # 筛选条件角色划分
    numerator_filters: list[dict[str, Any]] = field(default_factory=list)
    denominator_filters: list[dict[str, Any]] = field(default_factory=list)
    global_filters: list[dict[str, Any]] = field(default_factory=list)
    
    # SQL 生成提示
    sql_pattern: str = "simple_agg"  # simple_agg / case_when_ratio / window_func / subquery
    sql_hint: str = ""  # LLM 给 SQL 生成的额外建议
    
    # 元信息
    reasoning: str = ""  # LLM 的推理过程
    confidence: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_type": self.calculation_type,
            "numerator_filters": self.numerator_filters,
            "denominator_filters": self.denominator_filters,
            "global_filters": self.global_filters,
            "sql_pattern": self.sql_pattern,
            "sql_hint": self.sql_hint,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationPlan":
        return cls(
            calculation_type=data.get("calculation_type", "single"),
            numerator_filters=data.get("numerator_filters", []),
            denominator_filters=data.get("denominator_filters", []),
            global_filters=data.get("global_filters", []),
            sql_pattern=data.get("sql_pattern", "simple_agg"),
            sql_hint=data.get("sql_hint", ""),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 1.0),
        )
    
    def is_ratio(self) -> bool:
        """是否是占比类计算"""
        return self.calculation_type == "ratio"
    
    def get_case_when_filters(self) -> list[dict[str, Any]]:
        """获取需要放入 CASE WHEN 的筛选条件（分子专用）"""
        return self.numerator_filters
    
    def get_where_filters(self) -> list[dict[str, Any]]:
        """获取需要放入 WHERE 的筛选条件（全局）"""
        return self.global_filters


class CalibrationDesigner:
    """
    口径设计器 - 让 LLM 智能决定筛选条件的角色
    
    使用场景：
    - SQL 生成前，先调用 design() 获取 CalibrationPlan
    - SQLTool 根据 CalibrationPlan 构造正确的 SQL 结构
    
    核心改进：
    - 不再用关键词判断"是否是占比计算"
    - 让 LLM 理解"手游在IEG总流水中的占比"的语义
    - LLM 输出结构化的口径划分方案
    """
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._log = get_component_logger("CalibrationDesigner")
    
    async def design(
        self,
        user_query: str,
        intent: "StructuredIntent",
        yml_config: dict[str, Any],
        task_description: str = "",
        required_filters: list[dict[str, Any]] = None,
    ) -> CalibrationPlan:
        """
        设计口径方案
        
        Args:
            user_query: 用户原始问题
            intent: 结构化意图（含 filter_refs）
            yml_config: YAML 配置（含筛选器定义）
            task_description: 当前任务描述（如果有）
            required_filters: 已解析的筛选条件列表（来自 _inject_metric_definition）
                            ★ 这是完整的筛选列表，包含了 metric 定义中的筛选
        
        Returns:
            CalibrationPlan: 口径设计方案
        """
        self._log.info(f"设计口径: {user_query[:50]}...")
        
        # 1. 准备筛选器信息（供 LLM 参考）
        # ★ 优先使用 required_filters（更完整），兜底用 intent.filter_refs
        if required_filters:
            filters_info = self._prepare_filters_from_required(required_filters, yml_config)
        else:
            filters_info = self._prepare_filters_info(intent, yml_config)
        
        # 2. 构建 prompt，让 LLM 做口径设计
        prompt = self._build_design_prompt(
            user_query=user_query,
            task_description=task_description,
            filters_info=filters_info,
            metrics_info=self._prepare_metrics_info(intent, yml_config),
        )
        
        # 3. 调用 LLM
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                caller_name="calibration_design",
            )
            
            # 4. 解析 LLM 输出
            plan = self._parse_response(response, filters_info)
            self._log.observe(
                f"口径设计完成: type={plan.calculation_type}, "
                f"num_filters={len(plan.numerator_filters)}, "
                f"global_filters={len(plan.global_filters)}"
            )
            return plan
            
        except Exception as e:
            self._log.error(f"口径设计失败: {e}")
            # 降级：返回简单方案（所有筛选都放 WHERE）
            return self._fallback_plan(filters_info)
    
    def _prepare_filters_from_required(
        self,
        required_filters: list[dict[str, Any]],
        yml_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        从 required_filters（已解析的筛选列表）准备筛选器信息
        
        ★ 这是新增的方法，用于处理 _inject_metric_definition 生成的筛选列表
        相比 _prepare_filters_info，这个方法能看到：
        - 组合筛选（如 product_category_combined）
        - metric 定义中的隐含筛选（如 base_valid_data）
        """
        filters_config = yml_config.get("filters", {})
        filters_info = []
        
        for f in required_filters:
            fid = f.get("id", "")
            label = f.get("label", "")
            expr = f.get("expr", "")
            
            # 尝试从 yml 获取更多元信息
            group = "default"
            merge_mode = "intersect"
            is_slice = False
            is_global = False
            
            # 检查是否是组合筛选（如 product_category_combined）
            if fid.endswith("_combined"):
                base_group = fid.replace("_combined", "")
                group = base_group
                merge_mode = "union"
                # 组合筛选通常是切片（如多个产品类别的 OR）
                is_slice = base_group in ("product_category", "product")
            elif fid in filters_config:
                f_def = filters_config[fid]
                group = f_def.get("group", "default")
                merge_mode = f_def.get("merge_mode", "intersect")
                is_slice = group in ("product_category", "product") and merge_mode == "union"
                is_global = group in ("data_source", "organization", "time_year") or merge_mode == "exclusive"
            else:
                # 未知筛选器，根据名称推断
                if "valid" in fid.lower() or "base" in fid.lower():
                    is_global = True
                    group = "base"
            
            filters_info.append({
                "id": fid,
                "label": label,
                "expr": expr,
                "description": f.get("description", ""),
                "group": group,
                "merge_mode": merge_mode,
                "is_slice_hint": is_slice,
                "is_global_hint": is_global,
            })
        
        return filters_info
    
    def _prepare_filters_info(
        self,
        intent: "StructuredIntent",
        yml_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        准备筛选器信息，供 LLM 参考
        
        提取每个筛选器的：
        - id, label, expr
        - group: 筛选器组（如 product_category, data_source, metric）
        - merge_mode: 合并模式（exclusive / union）
        - is_slice: 是否是"切片类"筛选（通常只影响分子）
        """
        filters_config = yml_config.get("filters", {})
        filter_refs = intent.filter_refs if intent else []
        
        filters_info = []
        for fref in filter_refs:
            if fref not in filters_config:
                continue
            
            f_def = filters_config[fref]
            group = f_def.get("group", "default")
            merge_mode = f_def.get("merge_mode", "intersect")
            
            # 判断是否是"切片类"筛选（基于 group 和 merge_mode）
            # - product_category (手游/端游) → 切片
            # - product (王者荣耀/和平精英) → 切片
            # - data_source (实际/预测/预算) → 全局口径
            # - metric (流水/利润) → 指标筛选
            # - time_year → 全局口径
            is_slice = group in ("product_category", "product") and merge_mode == "union"
            is_global = group in ("data_source", "organization", "time_year") or merge_mode == "exclusive"
            
            filters_info.append({
                "id": fref,
                "label": f_def.get("label", fref),
                "expr": f_def.get("expr", ""),
                "description": f_def.get("description", ""),
                "group": group,
                "merge_mode": merge_mode,
                "is_slice_hint": is_slice,  # 提示 LLM 这可能是切片筛选
                "is_global_hint": is_global,  # 提示 LLM 这可能是全局筛选
            })
        
        return filters_info
    
    def _prepare_metrics_info(
        self,
        intent: "StructuredIntent",
        yml_config: dict[str, Any],
    ) -> str:
        """准备指标信息"""
        metrics_config = yml_config.get("metrics", {})
        metric_ids = intent.metrics if intent else []
        
        if not metric_ids:
            return "未指定具体指标"
        
        lines = []
        for mid in metric_ids:
            if mid in metrics_config:
                m = metrics_config[mid]
                lines.append(f"- {mid}: {m.get('label', '')} ({m.get('type', 'atomic')})")
        
        return "\n".join(lines) if lines else "未指定具体指标"
    
    def _build_design_prompt(
        self,
        user_query: str,
        task_description: str,
        filters_info: list[dict[str, Any]],
        metrics_info: str,
    ) -> str:
        """构建口径设计 prompt"""
        
        # 格式化筛选器信息
        filters_text = ""
        for f in filters_info:
            hint = ""
            if f.get("is_slice_hint"):
                hint = " [可能是切片筛选]"
            elif f.get("is_global_hint"):
                hint = " [可能是全局口径]"
            
            filters_text += f"""
- **{f['id']}**: {f['label']}{hint}
  - 表达式: `{f['expr']}`
  - 组: {f['group']}
  - 描述: {f['description']}
"""
        
        if not filters_text:
            filters_text = "（无预定义筛选器）"
        
        return f"""请分析以下查询，设计数据口径方案。

## 用户问题
{user_query}

## 当前任务描述（如果有）
{task_description or "（无）"}

## 涉及指标
{metrics_info}

## 候选筛选器（已从用户问题中识别）
{filters_text}

## 你的任务

1. **判断计算类型**：
   - `single`: 简单聚合（如"今年流水是多少"）
   - `ratio`: 占比/比例计算（如"X 在 Y 中的占比"、"X 占 Y 的百分比"）
   - `trend`: 趋势分析（如"流水的年度变化"）
   - `comparison`: 对比分析（如"同比增长"、"A vs B"）
   - `aggregation`: 分组聚合（如"按维度拆解"）

2. **对于 ratio 类型**，智能划分筛选器角色：
   - `numerator_filters`: **分子专用**（如"手游"只影响分子，不影响分母）
   - `global_filters`: **全局口径**（如"实际数据"、"2025年"影响分子和分母）
   
   **关键判断**：
   - 如果用户说"X 在 Y 中的占比"，X 相关的筛选是分子专用的
   - Y 相关的筛选是全局口径
   - 例如"手游在IEG总流水中的占比"：
     - "手游"(category_mobile) → 分子专用
     - "IEG总流水"的筛选（如数据来源、年份）→ 全局口径

3. **对于其他类型**，所有筛选器都放入 `global_filters`

## 输出格式（JSON）

```json
{{
  "calculation_type": "single|ratio|trend|comparison|aggregation",
  "numerator_filters": ["只有 ratio 类型才需要，放分子专用的筛选器 ID"],
  "global_filters": ["放全局口径的筛选器 ID"],
  "sql_pattern": "simple_agg|case_when_ratio|window_func|subquery",
  "sql_hint": "给 SQL 生成的建议",
  "reasoning": "一句话解释你的判断理由"
}}
```

只输出 JSON，不要其他文字。"""
    
    def _get_system_prompt(self) -> str:
        return """你是数据分析口径设计专家。

你的职责是分析用户的数据查询需求，智能设计数据口径方案：
1. 判断这是什么类型的计算（简单聚合 / 占比 / 趋势 / 对比）
2. 对于占比类问题，精确区分哪些筛选条件只影响分子、哪些是全局口径
3. 输出结构化的口径设计方案

关键原则：
- 占比计算 = 分子 / 分母，分子和分母的口径可能不同
- "X 在 Y 中的占比"：X 的筛选只影响分子，Y 的筛选是全局口径
- 不要用关键词硬匹配，要理解业务语义

只输出 JSON，不要解释。"""
    
    def _parse_response(
        self,
        response: str,
        filters_info: list[dict[str, Any]],
    ) -> CalibrationPlan:
        """解析 LLM 输出"""
        # 尝试解析 JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                self._log.warn(f"无法解析 LLM 输出: {response[:200]}")
                return self._fallback_plan(filters_info)
        
        # 构建 id -> filter_info 映射
        filters_map = {f["id"]: f for f in filters_info}
        
        # 解析筛选器角色
        numerator_filter_ids = data.get("numerator_filters", [])
        global_filter_ids = data.get("global_filters", [])
        
        # 转换为完整的筛选器信息
        numerator_filters = [
            filters_map[fid] for fid in numerator_filter_ids
            if fid in filters_map
        ]
        global_filters = [
            filters_map[fid] for fid in global_filter_ids
            if fid in filters_map
        ]
        
        # 兜底：如果 LLM 遗漏了某些筛选器，根据 hint 自动分配
        assigned_ids = set(numerator_filter_ids + global_filter_ids)
        for f in filters_info:
            if f["id"] not in assigned_ids:
                if f.get("is_slice_hint") and data.get("calculation_type") == "ratio":
                    numerator_filters.append(f)
                else:
                    global_filters.append(f)
        
        return CalibrationPlan(
            calculation_type=data.get("calculation_type", "single"),
            numerator_filters=numerator_filters,
            denominator_filters=[],  # 分母通常不需要额外筛选
            global_filters=global_filters,
            sql_pattern=data.get("sql_pattern", "simple_agg"),
            sql_hint=data.get("sql_hint", ""),
            reasoning=data.get("reasoning", ""),
            confidence=0.9,
        )
    
    def _fallback_plan(self, filters_info: list[dict[str, Any]]) -> CalibrationPlan:
        """降级方案：所有筛选都放 global"""
        return CalibrationPlan(
            calculation_type="single",
            numerator_filters=[],
            denominator_filters=[],
            global_filters=filters_info,
            sql_pattern="simple_agg",
            sql_hint="降级方案，所有筛选放 WHERE",
            reasoning="LLM 解析失败，使用降级方案",
            confidence=0.5,
        )


# 工厂函数
def create_calibration_designer(llm: BaseLLM) -> CalibrationDesigner:
    """创建口径设计器"""
    return CalibrationDesigner(llm)
