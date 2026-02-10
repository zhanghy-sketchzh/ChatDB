"""
SQLAgent - SQL 分析 Agent（具备 ReAct 能力）

设计理念：
- 接收 Planner 的高层任务（只有描述，没有具体参数）
- 内部进行 ReAct 拆解：分析任务 → 生成 SQL → 执行 → 评估
- 结果写入 state.temp_results 供 Planner 查看

核心接口：
- run_task(state, task): 接收 Planner 任务，内部 ReAct 执行
- temp_results 结构: {task_id: [{subtask, sql, row_count, examples, stats, issues}]}

架构特点：
- SQLTaskType 枚举：类型安全的任务类型定义
- SQLTaskMeta：每种任务类型的元信息（需要时间/维度/上游结果等）
- Handler 注册表：插件式任务处理器，新增类型只需 @register_sql_task_handler
- 统一的 stats/issues 字段约定，便于 Planner 决策

ReAct 流程示例：
```
[THINK] 收到任务: source_analysis - 从主要维度拆解流水来源
[STEP 1] 确定优先维度: 国内/海外, 投资公司标签
[STEP 2] 生成 GROUP BY SQL
[STEP 3] 执行并计算贡献度
[OBSERVE] 完成，国内占比 70%
[写入 temp_results]
```
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    from chatdb.agents.sql_agent import SQLAgent

from chatdb.agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from chatdb.core.react_state import ReActState
from chatdb.database.base import BaseDatabaseConnector
from chatdb.llm.base import BaseLLM
from chatdb.tools.sql import SQLTool
from chatdb.tools.calibration_designer import CalibrationDesigner, CalibrationPlan
from chatdb.utils.logger import get_component_logger


# =============================================================================
# 第一步：SQLTaskType 枚举 - 类型安全的任务类型
# =============================================================================

class SQLTaskType(str, Enum):
    """SQL 任务类型枚举（与 Planner 的 TaskType 对齐）"""
    TREND = "trend"           # 趋势分析（按时间聚合）
    SOURCE = "source"         # 来源分析（按维度拆解）
    DRILLDOWN = "drilldown"   # 下钻分析（对 top 结果细分）
    COMPARISON = "comparison" # 对比分析（同比/环比）
    BASIC = "basic"           # 基础查询（简单聚合）
    SUMMARY = "summary"       # 生成总结（由 Orchestrator 处理）
    ANOMALY = "anomaly"       # 异常检测（预留）
    COHORT = "cohort"         # 队列分析（预留）
    CORRELATION = "correlation"  # 相关性分析（预留）
    VALIDATION = "validation" # 数据验证/诊断（诊断空结果、检查字段）


# =============================================================================
# 第二步：SQLTaskMeta - 任务类型元信息
# =============================================================================

@dataclass
class SQLTaskMeta:
    """
    SQL 任务类型的结构化元信息
    
    用于：
    - 前置参数校验（缺少时间/维度时自动补全）
    - SQLTool 生成 SQL 时的语义提示
    - Planner 决策时的条件判断
    """
    label: str                              # 显示名称
    description: str                        # 任务描述
    requires_time: bool = False             # 是否需要时间参数
    requires_dimension: bool = False        # 是否需要维度参数
    requires_previous_results: bool = False # 是否依赖上游结果
    default_time_granularity: str = ""      # 默认时间粒度
    intent_hint_template: str = ""          # SQL 生成提示模板
    stats_fields: list[str] = field(default_factory=list)  # 标准 stats 字段
    issue_fields: list[str] = field(default_factory=list)  # 标准 issue 字段


# 任务类型元信息注册表
SQL_TASK_META: dict[SQLTaskType, SQLTaskMeta] = {
    SQLTaskType.TREND: SQLTaskMeta(
        label="趋势分析",
        description="按时间聚合做趋势，观察指标随时间的变化",
        requires_time=True,
        default_time_granularity="year",
        intent_hint_template="按{granularity}聚合{metric}，观察时间变化趋势，需要 GROUP BY 时间列并按时间排序",
        stats_fields=["available_years", "year_count", "growth_rate"],
        issue_fields=["only_single_year", "missing_time_range"],
    ),
    SQLTaskType.SOURCE: SQLTaskMeta(
        label="来源分析",
        description="按维度拆解来源构成，找出主要贡献者",
        requires_dimension=True,
        intent_hint_template="按维度「{dimension}」拆解来源构成，GROUP BY 该维度并按数值降序排序，找出主要贡献来源",
        stats_fields=["top_contributor", "top_ratio", "category_count"],
        issue_fields=["single_category", "low_coverage"],
    ),
    SQLTaskType.DRILLDOWN: SQLTaskMeta(
        label="下钻分析",
        description="基于上一步 top 结果进一步细分",
        requires_previous_results=True,
        intent_hint_template="在上一步结果基础上进一步细分，增加筛选条件或更细粒度的维度",
        stats_fields=["parent_top_ratio", "drilldown_depth"],
        issue_fields=["no_previous_result", "drilldown_exhausted"],
    ),
    SQLTaskType.COMPARISON: SQLTaskMeta(
        label="对比分析",
        description="对比两个时间段或条件下的指标差异",
        requires_time=True,
        intent_hint_template="对比两个时间段或条件下的指标，计算差值或增长率",
        stats_fields=["delta", "growth_rate", "comparison_base"],
        issue_fields=["insufficient_comparison_data"],
    ),
    SQLTaskType.BASIC: SQLTaskMeta(
        label="基础查询",
        description="执行基础查询，获取核心指标",
        intent_hint_template="执行基础查询，获取核心指标",
        stats_fields=["total_value"],
        issue_fields=[],
    ),
    SQLTaskType.SUMMARY: SQLTaskMeta(
        label="生成总结",
        description="汇总分析结果，生成报告（由 Orchestrator 处理）",
        requires_previous_results=True,
        stats_fields=[],
        issue_fields=[],
    ),
    SQLTaskType.ANOMALY: SQLTaskMeta(
        label="异常检测",
        description="检测数据中的异常点或显著变化",
        requires_time=True,
        intent_hint_template="检测指标的异常波动，识别显著偏离正常范围的数据点",
        stats_fields=["anomaly_count", "anomaly_severity"],
        issue_fields=["no_anomaly_found", "insufficient_data"],
    ),
    SQLTaskType.COHORT: SQLTaskMeta(
        label="队列分析",
        description="按队列（时间/属性分组）分析行为模式",
        requires_time=True,
        requires_dimension=True,
        intent_hint_template="按队列分组分析，观察不同群体的行为模式差异",
        stats_fields=["cohort_count", "retention_rate"],
        issue_fields=["insufficient_cohort_size"],
    ),
    SQLTaskType.CORRELATION: SQLTaskMeta(
        label="相关性分析",
        description="分析多个指标之间的相关关系",
        intent_hint_template="分析指标之间的相关性，计算相关系数",
        stats_fields=["correlation_coefficient", "p_value"],
        issue_fields=["weak_correlation"],
    ),
    SQLTaskType.VALIDATION: SQLTaskMeta(
        label="数据验证",
        description="诊断空结果原因、验证字段存在性、检查数据分布",
        intent_hint_template="执行诊断查询，检查数据是否存在、字段值分布、条件匹配情况",
        stats_fields=["row_count", "distinct_values", "sample_values"],
        issue_fields=["no_data", "field_not_found", "value_mismatch"],
    ),
}


def get_task_meta(task_type: SQLTaskType) -> SQLTaskMeta:
    """获取任务类型的元信息（带默认值）"""
    return SQL_TASK_META.get(task_type, SQLTaskMeta(
        label="未知任务",
        description="未定义的任务类型",
    ))


# =============================================================================
# Handler 注册表 - 插件式任务处理器
# =============================================================================

# Handler 类型签名
SQLTaskHandler = Callable[
    ["SQLAgent", ReActState, AgentContext, dict[str, Any]],
    Awaitable[None]
]

# 全局 handler 注册表
_SQL_TASK_HANDLERS: dict[SQLTaskType, SQLTaskHandler] = {}


def register_sql_task_handler(task_type: SQLTaskType):
    """
    装饰器：注册 SQL 任务处理器
    
    用法：
        @register_sql_task_handler(SQLTaskType.TREND)
        async def handle_trend_task(agent, state, context, task):
            ...
    """
    def decorator(func: SQLTaskHandler) -> SQLTaskHandler:
        _SQL_TASK_HANDLERS[task_type] = func
        return func
    return decorator


def get_task_handler(task_type: SQLTaskType) -> Optional[SQLTaskHandler]:
    """获取任务类型对应的 handler"""
    return _SQL_TASK_HANDLERS.get(task_type)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class SQLAgentThought:
    """SQLAgent 的思考记录"""
    step: int
    action: str  # think / step / act / observe / reflect
    content: str
    result: Optional[str] = None


@dataclass
class TaskResult:
    """任务执行结果（写入 temp_results）"""
    subtask: str
    sql: str = ""
    row_count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)  # 前 N 行样例
    stats: dict[str, Any] = field(default_factory=dict)           # 描述性统计
    issues: list[str] = field(default_factory=list)               # 问题/警告
    
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
class DomainConfig:
    """领域配置（从 YAML 加载）"""
    name: str
    display_name: str
    description: str
    table_name: str
    value_column: str
    
    # 领域知识
    business_terms: list[dict[str, Any]]
    dimensions: dict[str, dict]
    filters: dict[str, dict]
    metrics: dict[str, dict]
    examples: list[dict]
    rules: list[dict]
    
    # 行为规则
    behavior_rules: dict[str, Any]
    
    # 维度优先级
    priority_dimensions: dict[str, list[dict]]
    
    # 回答模板
    summary_templates: dict[str, str]
    
    # 原始 dict
    raw_config: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DomainConfig":
        """从 YAML 文件加载配置"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML 配置文件不存在: {yaml_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "DomainConfig":
        """从字典加载配置"""
        meta = data.get("meta", {})
        
        return cls(
            name=meta.get("table_name", ""),
            display_name=meta.get("display_name", ""),
            description=meta.get("description", ""),
            table_name=meta.get("table_name", ""),
            value_column=meta.get("value_column", ""),
            business_terms=data.get("business_terms", []),
            dimensions=data.get("dimensions", {}),
            filters=data.get("filters", {}),
            metrics=data.get("metrics", {}),
            examples=data.get("examples", []),
            rules=data.get("rules", []),
            behavior_rules=data.get("behavior_rules", {}),
            priority_dimensions=data.get("priority_dimensions", {}),
            summary_templates=data.get("summary_templates", {}),
            raw_config=data,
        )
    
    def get_priority_dimensions(self, task_type: str = "source_analysis") -> list[str]:
        """获取指定任务类型的优先维度列表"""
        dims = self.priority_dimensions.get(task_type, [])
        return [d.get("column") or d.get("dimension", "") for d in dims if d.get("column") or d.get("dimension")]
    
    def get_dimension_column(self, dim_id: str) -> Optional[str]:
        """获取维度对应的列名"""
        dim = self.dimensions.get(dim_id, {})
        return dim.get("column") or dim.get("label")


class SQLAgent(BaseAgent):
    """
    SQL 分析 Agent（具备 ReAct 能力）
    
    核心设计：
    1. 接收 Planner 的高层任务（只有描述，没有具体参数）
    2. 内部 ReAct 拆解：理解任务 → 规划子步骤 → 生成 SQL → 执行 → 评估
    3. 结果写入 state.temp_results 供 Planner 查看和决策
    
    支持的任务类型：
    - trend: 趋势分析（按时间聚合）
    - source: 来源分析（按维度拆解）
    - drilldown: 下钻分析（对 top 结果细分）
    - comparison: 对比分析（同比/环比）
    - basic: 基础查询（简单聚合）
    - summary: 生成总结（由 Orchestrator 处理）
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        db_connector: BaseDatabaseConnector,
        yml_config: Optional[Union[str, Path, dict]] = None,
    ):
        super().__init__(
            name="SQLAgent",
            llm=llm,
            description="SQL 分析专家：接收高层任务，内部 ReAct 拆解执行",
        )
        self.db_connector = db_connector
        self._sql_tool = SQLTool(llm, db_connector)
        self._calibration_designer = CalibrationDesigner(llm)  # ★ 新增：口径设计器
        self._log = get_component_logger("SQLAgent")
        self._thoughts: list[SQLAgentThought] = []
        
        # 加载领域配置
        self.config: Optional[DomainConfig] = None
        if yml_config:
            self._load_config(yml_config)
    
    @property
    def display_name(self) -> str:
        """显示名称"""
        return self.config.display_name if self.config else "SQLAgent"
    
    def _load_config(self, yml_config: Union[str, Path, dict]) -> None:
        """加载 YAML 配置"""
        try:
            if isinstance(yml_config, dict):
                self.config = DomainConfig.from_dict(yml_config)
            else:
                self.config = DomainConfig.from_yaml(yml_config)
            self._log.info(f"已加载领域配置: {self.config.display_name or self.config.name}")
        except Exception as e:
            self._log.warn(f"加载领域配置失败: {e}")
            self.config = None

    # ============================================================
    # 核心接口：run_task（接收 Planner 任务）
    # ============================================================

    def _build_task_context(
        self,
        task: dict[str, Any],
        task_type: SQLTaskType,
        state: ReActState,
        *,
        time_granularity: str = "",
        intent_hint: str = "",
        current_dimension: str = "",
        parent_results_summary: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        构建任务上下文（传给 SQLTool，影响 SQL 生成 prompt）
        
        这是 SQLAgent ReAct 思路进入 generate_sql prompt 的关键桥梁。
        
        增强：
        - 注入 metric_id、required_filters
        - 注入 task_meta（任务类型元信息）
        - 注入 retry_hint（来自 Planner 的 SQL 修复建议）
        - SQLTool 可基于这些信息做模板化 SQL 生成
        """
        # 获取任务元信息
        meta = get_task_meta(task_type)
        
        # 如果没有指定 intent_hint，使用元信息中的模板
        if not intent_hint and meta.intent_hint_template:
            intent_hint = meta.intent_hint_template.format(
                granularity=time_granularity or meta.default_time_granularity or "year",
                dimension=current_dimension or "维度",
                metric=getattr(state, "current_metric", "指标"),
            )
        
        # 如果需要时间但没指定粒度，使用默认
        if meta.requires_time and not time_granularity:
            time_granularity = meta.default_time_granularity
        
        # ★ 提取 Planner 的 SQL 修复建议
        task_meta = task.get("meta", {})
        retry_hint = task_meta.get("retry_hint", "")
        retry_count = task_meta.get("retry_count", 0)
        
        ctx = {
            # 基础任务信息
            "task_id": task.get("id", ""),
            "task_type": task_type.value,
            "description": task.get("description", ""),
            "notes": task.get("notes", []),
            "depends_on": task.get("depends_on", []),
            
            # SQLAgent 补充的执行提示
            "time_granularity": time_granularity,
            "intent_hint": intent_hint,
            "current_dimension": current_dimension,
            "parent_results_summary": parent_results_summary,
            
            # ★ Planner 的 SQL 修复建议（重试时使用）
            "retry_hint": retry_hint,
            "retry_count": retry_count,
            
            # 新增：结构化元信息（供 SQLTool 使用）
            "metric_id": getattr(state, "current_metric", ""),
            "required_filters": getattr(state, "required_filters", []),
            "task_meta": {
                "label": meta.label,
                "description": meta.description,
                "requires_time": meta.requires_time,
                "requires_dimension": meta.requires_dimension,
                "requires_previous_results": meta.requires_previous_results,
                "stats_fields": meta.stats_fields,
                "issue_fields": meta.issue_fields,
            },
        }
        if extra:
            ctx.update(extra)
        return ctx

    def _expand_derived_metric(
        self, 
        metric_id: str, 
        metrics_config: dict[str, Any],
        filters_config: dict[str, Any],
    ) -> tuple[str, list[str]]:
        """
        展开派生指标公式为 SQL 表达式
        
        派生指标的 formula 引用其他指标 ID，例如：
        - profit_margin.formula = "total_profit_post / total_gross * 100"
        
        展开逻辑：
        1. 解析 formula 中的指标 ID
        2. 将原子指标替换为 CASE WHEN 表达式（因为不同指标有不同的 filter_refs）
        3. 返回完整的 SQL 聚合表达式
        
        Returns:
            (agg_expr, filter_refs): 展开后的聚合表达式和需要的筛选器
        """
        metric_def = metrics_config.get(metric_id, {})
        
        # 原子指标：直接返回 agg
        if metric_def.get("type") != "derived":
            agg = metric_def.get("agg", "")
            filter_refs = metric_def.get("filter_refs", [])
            return agg, filter_refs
        
        formula = metric_def.get("formula", "")
        if not formula:
            return "", []
        
        # 收集所有需要的筛选器（派生指标依赖的所有原子指标的公共筛选器）
        all_filter_refs: set[str] = set()
        
        # 解析公式中引用的指标 ID
        # 使用正则匹配标识符（字母、数字、下划线组成，以字母开头）
        import re
        tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
        
        # 构建替换映射：指标 ID -> CASE WHEN 表达式
        expanded_formula = formula
        for token in tokens:
            if token not in metrics_config:
                continue
            
            ref_metric = metrics_config[token]
            ref_type = ref_metric.get("type", "atomic")
            
            if ref_type == "derived":
                # 递归展开（暂不支持多层嵌套，简化处理）
                self._log.warn(f"派生指标 {metric_id} 引用了另一个派生指标 {token}，暂不支持嵌套展开")
                continue
            
            # 获取原子指标的 filter_refs，找出区分不同指标的筛选器
            ref_filter_refs = ref_metric.get("filter_refs", [])
            
            # 找出区分报表项的筛选器（metric_* 开头的）
            metric_filter = None
            for fref in ref_filter_refs:
                if fref.startswith("metric_"):
                    metric_filter = fref
                    break
            
            # 构建 CASE WHEN 表达式
            if metric_filter and metric_filter in filters_config:
                filter_def = filters_config[metric_filter]
                filter_expr = filter_def.get("expr", "")
                # 提取筛选值（如 "大盘报表项" = '递延后利润'）
                # filter_expr 格式如: "大盘报表项" = '递延后利润'
                case_expr = f'SUM(CASE WHEN {filter_expr} THEN "ieg口径金额-人民币" ELSE 0 END)'
            else:
                # 没有特定筛选器，使用通用聚合
                case_expr = ref_metric.get("agg", 'SUM("ieg口径金额-人民币")')
            
            # 替换公式中的指标 ID
            # 使用单词边界确保精确匹配
            expanded_formula = re.sub(rf'\b{token}\b', f'({case_expr})', expanded_formula)
            
            # 收集公共筛选器（排除 metric_* 的，因为已经在 CASE WHEN 中处理了）
            for fref in ref_filter_refs:
                if not fref.startswith("metric_"):
                    all_filter_refs.add(fref)
        
        # 添加 ROUND 和 NULLIF 防止除零
        round_digits = metric_def.get("round", 2)
        # 检测是否有除法，添加 NULLIF
        if "/" in expanded_formula:
            # 简单处理：用 NULLIF 包装分母
            # 更复杂的场景需要解析表达式树
            pass
        
        final_expr = f"ROUND({expanded_formula}, {round_digits})"
        
        # 添加派生指标自己声明的 filter_refs
        for fref in metric_def.get("filter_refs", []):
            all_filter_refs.add(fref)
        
        return final_expr, list(all_filter_refs)

    def _inject_metric_definition(self, state: ReActState) -> None:
        """
        ★ 核心方法：把 YAML 里的 metric/filter 结构化灌入 state
        
        支持两种指标类型：
        1. 原子指标（atomic）：直接使用 agg 表达式
        2. 派生指标（derived）：展开 formula 为 SQL 表达式
        
        这样 SQLTool 在生成 SQL 时，能直接拿到：
        - current_metric: 当前指标 ID
        - current_metric_def: 指标定义（含展开后的 agg）
        - required_filters: 必须包含的筛选条件（已解析为 SQL 片段）
        """
        if not state.yml_config:
            return
        
        yml_config = state.yml_config
        metrics_config = yml_config.get("metrics", {})
        filters_config = yml_config.get("filters", {})
        
        # 从 intent 获取当前指标，或使用默认
        metric_ids = []
        if state.intent and state.intent.metrics:
            metric_ids = state.intent.metrics
        
        # 如果没有指定指标，尝试从 filter_refs 推断
        if not metric_ids and state.intent and state.intent.filter_refs:
            for fref in state.intent.filter_refs:
                # 根据 filter_refs 反查 metrics
                for mid, mdef in metrics_config.items():
                    if fref in mdef.get("filter_refs", []):
                        metric_ids.append(mid)
                        break
        
        # 默认使用 total_flow
        if not metric_ids:
            metric_ids = ["total_flow"]
        
        # 取第一个指标作为主指标
        metric_name = metric_ids[0]
        metric_def = metrics_config.get(metric_name, {}).copy()  # 复制以免修改原配置
        
        # ★ 关键：检测是否是派生指标，展开公式
        metric_type = metric_def.get("type", "atomic")
        if metric_type == "derived":
            expanded_agg, expanded_filter_refs = self._expand_derived_metric(
                metric_name, metrics_config, filters_config
            )
            # 将展开后的表达式写入 metric_def
            metric_def["agg"] = expanded_agg
            metric_def["_expanded"] = True
            metric_def["_expanded_filter_refs"] = expanded_filter_refs
            self._log.info(f"派生指标 {metric_name} 展开为: {expanded_agg[:100]}...")
        
        # 写入 state
        state.current_metric = metric_name
        state.current_metric_def = metric_def
        
        # 解析 filter_refs，构建必须包含的 WHERE 片段
        required_filters = []
        
        # 派生指标使用展开后的 filter_refs
        if metric_def.get("_expanded"):
            filter_refs = metric_def.get("_expanded_filter_refs", [])
        else:
            filter_refs = metric_def.get("filter_refs", [])
        
        # 同时加入 intent 中的 filter_refs
        intent_filter_refs = []
        if state.intent and state.intent.filter_refs:
            intent_filter_refs = state.intent.filter_refs
        
        # ========================================
        # 通用筛选器合并逻辑（基于 YAML 元数据）
        # ========================================
        # 
        # 每个筛选器可以有 group 和 merge_mode 属性：
        # - group: 同组筛选器会被一起处理
        # - merge_mode: 
        #   - "exclusive": 互斥，用户指定的优先，忽略指标定义中的同组筛选器
        #   - "union": 并集，同组筛选器用 OR 连接
        #   - "intersect" (默认): 交集，同组筛选器用 AND 连接
        #
        # 这样避免硬编码 metric_*、time_q* 等规则
        
        def get_filter_group(fref: str) -> tuple[str, str]:
            """获取筛选器的 group 和 merge_mode"""
            if fref in filters_config:
                f_def = filters_config[fref]
                group = f_def.get("group", "default")
                merge_mode = f_def.get("merge_mode", "intersect")
                return group, merge_mode
            return "default", "intersect"
        
        # 1. 先收集 intent 中各 group 的筛选器
        intent_groups: dict[str, list[str]] = {}
        for fref in intent_filter_refs:
            group, _ = get_filter_group(fref)
            if group not in intent_groups:
                intent_groups[group] = []
            intent_groups[group].append(fref)
        
        # 2. 合并筛选器，处理 exclusive 逻辑
        merged_filter_refs = []
        for fref in filter_refs:
            group, merge_mode = get_filter_group(fref)
            
            # exclusive 模式：如果 intent 已有同组筛选器，跳过指标定义中的
            if merge_mode == "exclusive" and group in intent_groups:
                continue
            
            # 跳过派生指标已处理的筛选器（通过 group="metric" 识别）
            if metric_def.get("_expanded"):
                f_group, _ = get_filter_group(fref)
                if f_group == "metric":
                    continue
            
            merged_filter_refs.append(fref)
        
        # 3. 添加 intent 中的筛选器（避免重复）
        for fref in intent_filter_refs:
            if fref not in merged_filter_refs:
                merged_filter_refs.append(fref)
        
        # 4. 按 group 分组，准备合并
        groups: dict[str, list[str]] = {}
        for fref in merged_filter_refs:
            group, _ = get_filter_group(fref)
            if group not in groups:
                groups[group] = []
            groups[group].append(fref)
        
        # 5. 根据 merge_mode 生成最终筛选条件
        for group_name, frefs in groups.items():
            if not frefs:
                continue
            
            # 获取该组的 merge_mode（取第一个筛选器的设置）
            _, merge_mode = get_filter_group(frefs[0])
            
            if merge_mode == "union" and len(frefs) > 1:
                # union 模式：多个筛选器用 OR 连接
                exprs = []
                labels = []
                for fref in frefs:
                    if fref in filters_config:
                        f_def = filters_config[fref]
                        expr = f_def.get("expr", "")
                        if expr:
                            exprs.append(f"({expr})")
                            labels.append(f_def.get("label", fref))
                if exprs:
                    combined_expr = " OR ".join(exprs)
                    required_filters.append({
                        "id": f"{group_name}_combined",
                        "label": f"{', '.join(labels)}",
                        "expr": f"({combined_expr})",
                        "description": f"筛选 {', '.join(labels)} 的数据",
                    })
            else:
                # intersect 或 exclusive 模式：每个筛选器独立（AND 连接）
                for fref in frefs:
                    if fref in filters_config:
                        f_def = filters_config[fref]
                        expr = f_def.get("expr", "")
                        if expr:
                            required_filters.append({
                                "id": fref,
                                "label": f_def.get("label", fref),
                                "expr": expr,
                                "description": f_def.get("description", ""),
                            })
        
        state.required_filters = required_filters
        
        agg_preview = metric_def.get('agg', '')[:50] if metric_def.get('agg') else ''
        self._log.info(f"注入指标定义: {metric_name} ({metric_type}), agg={agg_preview}..., "
                       f"required_filters={len(required_filters)}个")

    async def _design_calibration(
        self,
        state: ReActState,
        task_description: str = "",
    ) -> Optional[CalibrationPlan]:
        """
        ★ 调用口径设计器，智能划分筛选条件角色
        
        这是架构改造的核心：
        - 不再用关键词判断"是否是占比计算"
        - 让 LLM 理解业务语义，智能决定筛选条件的角色
        - 输出 CalibrationPlan，用于指导 SQL 生成
        
        ★ 重要：必须在 _inject_metric_definition 之后调用
        这样口径设计器能看到完整的 required_filters 列表
        
        Args:
            state: ReAct 状态（含 intent、yml_config、required_filters）
            task_description: 当前任务描述
        
        Returns:
            CalibrationPlan 或 None（如果不需要特殊处理）
        """
        if not state.intent or not state.yml_config:
            return None
        
        # 调用口径设计器
        try:
            # ★ 重要修复：传入 required_filters，让口径设计器看到完整的筛选列表
            plan = await self._calibration_designer.design(
                user_query=state.user_query,
                intent=state.intent,
                yml_config=state.yml_config,
                task_description=task_description,
                required_filters=state.required_filters,  # 新增参数
            )
            
            # 保存到 state，供后续使用
            state.calibration_plan = plan
            
            self._log.info(
                f"口径设计完成: type={plan.calculation_type}, "
                f"numerator={len(plan.numerator_filters)}, "
                f"global={len(plan.global_filters)}"
            )
            
            return plan
            
        except Exception as e:
            self._log.warn(f"口径设计失败，使用默认逻辑: {e}")
            return None

    def _apply_calibration_plan(self, state: ReActState, plan: CalibrationPlan) -> None:
        """
        ★ 根据口径设计方案，重新划分 required_filters
        
        核心逻辑：
        - 如果是 ratio 类型，把 numerator_filters 从 required_filters 中移出
        - numerator_filters 会被放入 state.numerator_filters，供 SQLTool 生成 CASE WHEN
        - global_filters 保留在 required_filters 中，放入 WHERE
        
        ★ 重要修复：
        - CalibrationDesigner 输出的是原始 filter ID（如 "category_mobile"）
        - 但 _inject_metric_definition 可能生成了组合 ID（如 "product_category_combined"）
        - 需要同时匹配原始 ID 和组合 ID
        """
        if not plan.is_ratio():
            # 非占比计算，不需要特殊处理
            return
        
        self._log.info("应用占比口径设计...")
        
        # ★ 修复：获取分子筛选的原始 ID 集合
        numerator_filter_ids = {f["id"] for f in plan.numerator_filters}
        
        # ★ 修复：同时记录分子筛选的 group（用于匹配 _combined 形式的 ID）
        numerator_groups = set()
        filters_config = state.yml_config.get("filters", {}) if state.yml_config else {}
        for fid in numerator_filter_ids:
            if fid in filters_config:
                group = filters_config[fid].get("group", "default")
                numerator_groups.add(group)
                numerator_groups.add(f"{group}_combined")  # 预防性添加组合 ID
        
        self._log.info(f"  分子筛选 ID: {numerator_filter_ids}")
        self._log.info(f"  分子筛选组: {numerator_groups}")
        
        # 从 required_filters 中分离出分子筛选
        new_required_filters = []
        numerator_filters_for_sql = []
        
        for f in (state.required_filters or []):
            fid = f.get("id", "")
            flabel = f.get("label", "")
            
            # ★ 修复：多种匹配方式
            is_numerator = False
            
            # 1. 精确匹配原始 ID
            if fid in numerator_filter_ids:
                is_numerator = True
            
            # 2. 匹配组合 ID（如 product_category_combined）
            elif fid in numerator_groups:
                is_numerator = True
            
            # 3. 匹配 _combined 后缀的组名
            elif fid.endswith("_combined"):
                base_group = fid.replace("_combined", "")
                if base_group in numerator_groups or f"{base_group}_combined" in numerator_groups:
                    is_numerator = True
            
            # 4. 检查 label 是否包含分子筛选的 label（兜底）
            # 例如：label="手游, 端游" 包含 "手游"
            if not is_numerator:
                for nf in plan.numerator_filters:
                    nlabel = nf.get("label", "")
                    if nlabel and nlabel in flabel:
                        is_numerator = True
                        break
            
            if is_numerator:
                # 这是分子专用筛选，放入 numerator_filters
                numerator_filters_for_sql.append(f)
                self._log.info(f"  分子筛选: {fid} ({flabel})")
            else:
                # 这是全局筛选，保留在 WHERE
                new_required_filters.append(f)
                self._log.info(f"  全局筛选: {fid} ({flabel})")
        
        # 更新 state
        state.required_filters = new_required_filters
        state.numerator_filters = numerator_filters_for_sql
        
        # 设置 SQL 模式提示
        state.sql_pattern = plan.sql_pattern
        state.sql_hint = plan.sql_hint
        
        self._log.info(
            f"口径划分完成: WHERE 筛选 {len(new_required_filters)} 个, "
            f"CASE WHEN 筛选 {len(numerator_filters_for_sql)} 个"
        )
        
        # ★ 重要校验：如果 LLM 识别出是 ratio 但没有分离出分子筛选，打印警告
        if not numerator_filters_for_sql:
            self._log.warn(
                f"⚠️ 口径设计为 ratio 类型，但未能从 required_filters 中分离出分子筛选！"
                f"numerator_filter_ids={numerator_filter_ids}, "
                f"required_filter_ids={[f.get('id') for f in state.required_filters or []]}"
            )

    async def run_task(self, state: ReActState, context: AgentContext, task: dict[str, Any]) -> None:
        """
        执行 Planner 给的分析任务（核心接口）
        
        重构后使用注册表分发，支持插件式扩展。
        
        Args:
            state: ReAct 状态
            context: Agent 上下文
            task: Planner 任务
                {
                    "id": "source_analysis",
                    "type": "source",
                    "description": "从主要维度拆解流水来源，找出贡献最大的部分",
                    "notes": ["优先尝试国内/海外维度", "关注 top 贡献者"]
                }
        
        结果写入 state.temp_results[task_id]
        """
        self._thoughts = []
        task_id = task.get("id", "unknown")
        raw_type = task.get("type", "basic")
        description = task.get("description", "")
        notes = task.get("notes", [])
        
        self._think(f"收到任务: {task_id} - {description[:50]}...", state)
        if notes:
            self._think(f"注意事项: {notes[0][:50]}...", state)
        
        # 注入领域配置
        if self.config:
            state.yml_config = self.config.raw_config
            context.yml_config = self.config.raw_config
            if not state.table_name:
                state.table_name = self.config.table_name
        
        # ★ 关键：检查是否是重试任务
        task_meta = task.get("meta", {})
        retry_count = task_meta.get("retry_count", 0)
        retry_hint = task_meta.get("retry_hint", "")
        
        # 初始化或清空 temp_results
        # ★ 如果是重试任务（retry_count > 0），清空旧结果，避免 Planner 看到旧的错误信息
        if retry_count > 0:
            self._log.info(f"重试任务 {task_id}（第 {retry_count} 次），清除旧结果")
            state.temp_results[task_id] = []
            # 同时清理可能残留的错误状态
            state.execution_error = None
            state.error = None
        elif task_id not in state.temp_results:
            state.temp_results[task_id] = []
        
        # 类型解析：字符串 → 枚举（类型安全）
        try:
            task_type = SQLTaskType(raw_type)
        except ValueError:
            self._log.warn(f"未知任务类型: {raw_type}，使用 basic")
            task_type = SQLTaskType.BASIC
        
        # 获取 handler（注册表分发）
        handler = get_task_handler(task_type)
        if not handler:
            self._log.warn(f"任务类型 {task_type.value} 未注册 handler，降级 basic")
            handler = get_task_handler(SQLTaskType.BASIC)
        
        # 执行 handler
        try:
            if handler:
                await handler(self, state, context, task)
            else:
                # 兜底：直接执行基础查询
                self._log.error("无法找到任何 handler，执行最简查询")
                await self._fallback_basic_query(state, context, task)
        except Exception as e:
            self._observe(f"任务执行失败: {e}", state)
            result = TaskResult(subtask="error", issues=[str(e)])
            state.temp_results[task_id].append(result.to_dict())
        
        self._observe(f"任务 {task_id} 完成，{len(self._thoughts)} 步思考", state)
    
    async def _fallback_basic_query(
        self, state: ReActState, context: AgentContext, task: dict[str, Any]
    ) -> None:
        """兜底的基础查询（当没有注册 handler 时）"""
        task_id = task.get("id", "basic")
        self._inject_metric_definition(state)
        context.current_task = self._build_task_context(
            task, SQLTaskType.BASIC, state,
            intent_hint="执行基础查询，获取核心指标",
        )
        await self._sql_tool.run_workflow(state, context)
        result = self._collect_result("basic_query", state)
        state.temp_results[task_id].append(result.to_dict())

    # ============================================================
    # 任务执行器（已迁移到模块级 handler 函数）
    # 以下方法保留为内部工具方法
    # ============================================================

    def _get_parent_results_summary(self, state: ReActState, task: dict[str, Any]) -> str:
        """获取上游任务结果摘要（供子节点参考）"""
        depends_on = task.get("depends_on", [])
        if not depends_on or not state.temp_results:
            return ""
        
        summaries = []
        for parent_id in depends_on:
            parent_results = state.temp_results.get(parent_id, [])
            for r in parent_results:
                stats = r.get("stats", {})
                if stats.get("available_years"):
                    summaries.append(f"可用年份: {stats['available_years']}")
                if stats.get("top_contributor"):
                    summaries.append(f"top贡献: {stats['top_contributor']}")
                row_count = r.get("row_count", 0)
                if row_count:
                    summaries.append(f"上游返回 {row_count} 行")
        
        return "; ".join(summaries) if summaries else ""

    # ============================================================
    # 结果收集与统计
    # ============================================================

    def _collect_result(self, subtask: str, state: ReActState) -> TaskResult:
        """
        收集 SQL 执行结果，生成 TaskResult
        
        包含：样例行 + 描述性统计 + 问题诊断
        """
        result = TaskResult(subtask=subtask)
        result.sql = state.current_sql or ""
        
        # ★ 重要：先检查执行错误，即使 execute_result 为空也要记录错误信息
        if state.execution_error:
            result.issues.append(f"sql_error:{state.execution_error}")
        if state.error:
            result.issues.append(f"error:{state.error}")
        
        if not state.execute_result:
            result.issues.append("no_execute_result")
            return result
        
        rows = state.execute_result.get("rows", [])
        result.row_count = len(rows)
        
        # 样例行（前 5 行）
        result.examples = rows[:5] if rows else []
        
        # 描述性统计
        if rows:
            result.stats = self._compute_stats(rows)
        
        # 问题诊断
        if result.row_count == 0:
            result.issues.append("empty_result")
        elif result.row_count == 1:
            result.issues.append("single_row")
        
        return result

    def _compute_stats(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """计算描述性统计"""
        if not rows:
            return {}
        
        stats: dict[str, Any] = {
            "row_count": len(rows),
        }
        
        # 找到数值列
        numeric_cols = []
        for key, val in rows[0].items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_cols.append(key)
        
        # 对每个数值列计算统计
        for col in numeric_cols[:3]:  # 最多 3 个数值列
            values = [row.get(col, 0) for row in rows if row.get(col) is not None]
            if values:
                stats[f"{col}_sum"] = sum(values)
                stats[f"{col}_min"] = min(values)
                stats[f"{col}_max"] = max(values)
                if len(values) > 0:
                    stats[f"{col}_avg"] = sum(values) / len(values)
        
        return stats

    def _get_numeric_value(self, row: dict[str, Any]) -> float:
        """从行中获取数值（用于计算贡献度）"""
        for val in row.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        return 0.0

    def _get_previous_results(self, state: ReActState, exclude_task: str = "") -> list[dict[str, Any]]:
        """获取之前的任务结果"""
        results = []
        for task_id, task_results in state.temp_results.items():
            if task_id != exclude_task:
                results.extend(task_results)
        return results

    # ============================================================
    # 兼容旧接口
    # ============================================================

    async def run(
        self,
        state: ReActState,
        context: AgentContext,
        instructions: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        执行 SQL 分析流程（兼容旧接口）
        
        如果有 instructions，转换为 task 格式调用 run_task
        """
        self._thoughts = []
        
        # 确保有基本信息
        if self.config:
            state.yml_config = self.config.raw_config
            context.yml_config = self.config.raw_config
            if not state.table_name:
                state.table_name = self.config.table_name
        
        # 有指令时，转换为 task 格式
        if instructions:
            task: dict[str, Any] = {
                "id": instructions.get("step", "task"),
                "type": self._map_task_type(instructions.get("task", "")),
                "description": instructions.get("task", ""),
                "notes": [],
            }
            # 如果有 instructions 子字段，加入 notes
            if instructions.get("instructions"):
                notes_list: list[str] = task["notes"]
                notes_list.append(str(instructions["instructions"]))
            
            await self.run_task(state, context, task)
            return
        
        # 无指令时，使用基础流程
        self._think(f"开始 SQL 分析: {state.user_query[:50]}...", state)
        await self._sql_tool.run_workflow(state, context)
        
        if state.has_result:
            self._observe(f"执行成功，返回 {state.execute_result.get('row_count', 0) if state.execute_result else 0} 行", state)
        state.observe(f"[SQLAgent] 完成，{len(self._thoughts)} 步思考")

    def _map_task_type(self, task_name: str) -> str:
        """将任务名映射为任务类型"""
        mapping = {
            "计算整体指标": "basic",
            "计算趋势": "trend",
            "按维度拆解": "source",
            "对比分析": "comparison",
            "下钻分析": "drilldown",
            "生成总结": "summary",
        }
        return mapping.get(task_name, "basic")

    async def execute(self, context: AgentContext) -> AgentResult:
        """兼容 BaseAgent 接口"""
        return AgentResult(status=AgentStatus.SUCCESS, message="SQLAgent executed")

    async def run_workflow(self, state: ReActState, context: AgentContext) -> None:
        """执行完整 workflow（别名）"""
        await self.run(state, context)

    async def run_generate(self, state: ReActState, context: AgentContext) -> None:
        """仅执行 SQL 生成"""
        await self._sql_tool.run_generate(state, context)

    async def run_execute_and_evaluate(self, state: ReActState, context: AgentContext) -> None:
        """仅执行 SQL 并评估"""
        await self._sql_tool.run_execute_and_evaluate(state, context)

    # ============================================================
    # ReAct 思考记录
    # ============================================================
    
    def _think(self, content: str, state: ReActState | None = None) -> None:
        """记录思考"""
        self._thoughts.append(SQLAgentThought(len(self._thoughts)+1, "think", content))
        self._log.info(f"[THINK] {content}")
        # 同步到 state（如果提供）
        if state:
            state.think(content)
    
    def _step(self, content: str, state: ReActState | None = None) -> None:
        """记录执行步骤"""
        self._thoughts.append(SQLAgentThought(len(self._thoughts)+1, "step", content))
        self._log.info(f"[STEP] {content}")
        # step 作为 think 同步
        if state:
            state.think(content)
    
    def _observe(self, content: str, state: ReActState | None = None) -> None:
        """记录观察"""
        self._thoughts.append(SQLAgentThought(len(self._thoughts)+1, "observe", content))
        self._log.info(f"[OBSERVE] {content}")
        # 同步到 state
        if state:
            state.observe(content)
    
    def _reflect(self, content: str, state: ReActState | None = None) -> None:
        """记录反思"""
        self._thoughts.append(SQLAgentThought(len(self._thoughts)+1, "reflect", content))
        self._log.info(f"[REFLECT] {content}")
        # 同步到 state
        if state:
            state.reflect(content)
    
    def _act(self, action: str, content: str, result: str = "") -> None:
        """记录行动"""
        self._thoughts.append(SQLAgentThought(len(self._thoughts)+1, action, content, result))
        self._log.info(f"[ACT] {content}")
        if result:
            self._log.info(f"  → {result[:100]}...")

    def get_thoughts_display(self) -> str:
        """获取思考过程的可读展示"""
        lines = ["SQLAgent 思考过程:"]
        for t in self._thoughts:
            lines.append(f"  [{t.action.upper()}] {t.content}")
            if t.result:
                lines.append(f"    → {t.result[:80]}...")
        return "\n".join(lines)


# =============================================================================
# 工厂函数
# =============================================================================

def create_sql_agent(
    llm: BaseLLM,
    db_connector: BaseDatabaseConnector,
    yml_config: Optional[Union[str, Path, dict]] = None,
) -> SQLAgent:
    """创建 SQL Agent"""
    return SQLAgent(llm, db_connector, yml_config)


# =============================================================================
# 注册式 Handler 函数（模块级）
# =============================================================================

@register_sql_task_handler(SQLTaskType.BASIC)
async def handle_basic_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """执行基础查询任务
    
    ★ 架构改造：在 SQL 生成前调用口径设计器
    
    根据 Intent 中的时间粒度提示 LLM 如何分组。
    LLM 会根据 column_profiles（列元信息）自行推理合适的分组列。
    """
    task_id = task.get("id", "basic")
    task_desc = task.get("description", "")
    
    agent._step("1. 分析查询需求", state)
    
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # ★ 新增：调用口径设计器，智能划分筛选条件角色
    agent._step("2. 设计数据口径", state)
    plan = await agent._design_calibration(state, task_desc)
    if plan:
        agent._apply_calibration_plan(state, plan)
        agent._observe(f"口径类型: {plan.calculation_type}, 推理: {plan.reasoning[:50]}...", state)
    
    agent._step("3. 生成并执行 SQL", state)
    
    # 构建任务上下文（LLM 根据 Intent 和 column_profiles 自行推理分组方式）
    context.current_task = agent._build_task_context(
        task, SQLTaskType.BASIC, state,
    )
    
    # 执行 SQL
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("basic_query", state)
    
    # 统一 stats 字段
    if result.examples:
        total = sum(agent._get_numeric_value(ex) for ex in result.examples)
        result.stats["total_value"] = total
    
    state.temp_results[task_id].append(result.to_dict())
    agent._observe(f"基础查询完成: {result.row_count} 行", state)


@register_sql_task_handler(SQLTaskType.TREND)
async def handle_trend_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """执行趋势分析任务"""
    task_id = task.get("id", "trend")
    
    agent._step("1. 确定时间维度（年）", state)
    agent._step("2. 构造年度聚合 SQL", state)
    agent._step("3. 执行并分析趋势", state)
    
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 构建任务上下文
    context.current_task = agent._build_task_context(
        task, SQLTaskType.TREND, state,
        time_granularity="year",
        intent_hint="按年聚合总流水，观察年度变化趋势，需要 GROUP BY 时间列并按时间排序",
    )
    
    # 适度修正 Intent：趋势分析强制按年聚合
    if state.intent and state.intent.time:
        state.intent.time["granularity"] = "year"
        state.intent.time.pop("comparison", None)
    
    # 执行 SQL
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果并计算统计
    result = agent._collect_result("trend_analysis", state)
    
    # 统一 stats 字段（趋势特有）
    if result.examples:
        years = [ex.get("年") or ex.get("year") for ex in result.examples if ex.get("年") or ex.get("year")]
        result.stats["available_years"] = years
        result.stats["year_count"] = len(years)  # 统一字段名
        
        # 计算增长率（如果有多年数据）
        if len(result.examples) >= 2:
            values = [agent._get_numeric_value(ex) for ex in result.examples]
            if values[0] > 0:
                result.stats["growth_rate"] = round((values[-1] - values[0]) / values[0], 4)
        
        if len(years) <= 1:
            result.issues.append("only_single_year")
            agent._observe("⚠️ 只有单一年份数据，无法做多年度趋势分析", state)
    
    state.temp_results[task_id].append(result.to_dict())
    agent._observe(f"趋势分析完成: {result.row_count} 个时间点", state)


@register_sql_task_handler(SQLTaskType.SOURCE)
async def handle_source_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """执行来源分析任务（按维度拆解）"""
    task_id = task.get("id", "source")
    
    agent._step("1. 获取优先维度列表", state)
    
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 从配置获取优先维度
    priority_dims = []
    if agent.config:
        priority_dims = agent.config.get_priority_dimensions("source_analysis")
    if not priority_dims:
        priority_dims = ["国内/海外", "投资公司标签", "产品大类"]
    
    agent._step(f"2. 候选维度: {priority_dims[:3]}", state)
    
    # 获取上游任务结果摘要
    parent_summary = agent._get_parent_results_summary(state, task)
    
    # 对每个维度尝试分析
    for dim in priority_dims[:3]:
        if dim in state.explored_dimensions:
            agent._step(f"   维度 [{dim}] 已探索，跳过", state)
            continue
        
        agent._step(f"3. 分析维度: {dim}", state)
        
        # 构建任务上下文
        context.current_task = agent._build_task_context(
            task, SQLTaskType.SOURCE, state,
            current_dimension=dim,
            intent_hint=f"按维度「{dim}」拆解来源构成，GROUP BY 该维度并按数值降序排序，找出主要贡献来源",
            parent_results_summary=parent_summary,
        )
        
        # 适度修正 Intent
        if state.intent:
            if state.intent.time:
                state.intent.time.pop("comparison", None)
            if not state.intent.dimensions:
                state.intent.dimensions = []
            if dim not in state.intent.dimensions:
                state.intent.dimensions.append(dim)
        
        # 执行维度分析
        await agent._sql_tool.run_workflow(state, context)
        
        # 标记已探索
        state.explored_dimensions.add(dim)
        
        # 收集结果
        result = agent._collect_result(f"source_{dim}", state)
        
        # 统一 stats 字段（来源分析特有）
        result.stats["category_count"] = result.row_count
        
        if result.examples and len(result.examples) > 1:
            total = sum(agent._get_numeric_value(ex) for ex in result.examples)
            if total > 0:
                for ex in result.examples:
                    val = agent._get_numeric_value(ex)
                    ex["contribution_ratio"] = round(val / total, 4)
                
                # top 贡献者
                top = max(result.examples, key=lambda x: x.get("contribution_ratio", 0))
                result.stats["top_contributor"] = top
                result.stats["top_ratio"] = top.get("contribution_ratio", 0)
        elif result.row_count == 1:
            result.issues.append("single_category")
        
        state.temp_results[task_id].append(result.to_dict())
        agent._observe(f"维度 [{dim}] 分析完成: {result.row_count} 类别", state)
        
        # 如果有有效结果，可以停止
        if result.row_count > 1:
            break
    
    agent._reflect(f"来源分析完成，探索了 {len(state.explored_dimensions)} 个维度", state)


@register_sql_task_handler(SQLTaskType.DRILLDOWN)
async def handle_drilldown_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """执行下钻分析任务"""
    task_id = task.get("id", "drilldown")
    
    agent._step("1. 获取上一步的 top 结果", state)
    
    # 从 temp_results 获取上一步结果
    prev_results = agent._get_previous_results(state, exclude_task=task_id)
    parent_summary = agent._get_parent_results_summary(state, task)
    
    if not prev_results:
        agent._observe("⚠️ 没有上一步结果，无法下钻", state)
        result = TaskResult(subtask="drilldown", issues=["no_previous_result"])
        state.temp_results[task_id].append(result.to_dict())
        return
    
    agent._step("2. 选择下钻目标", state)
    agent._step("3. 执行下钻 SQL", state)
    
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 构建任务上下文
    context.current_task = agent._build_task_context(
        task, SQLTaskType.DRILLDOWN, state,
        intent_hint="在上一步结果基础上进一步细分，增加筛选条件或更细粒度的维度",
        parent_results_summary=parent_summary,
        extra={"previous_results": prev_results},
    )
    
    # 执行
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("drilldown", state)
    
    # 统一 stats 字段（下钻特有）
    # 尝试从上游获取 top_ratio 作为 parent_top_ratio
    for pr in prev_results:
        if pr.get("stats", {}).get("top_ratio"):
            result.stats["parent_top_ratio"] = pr["stats"]["top_ratio"]
            break
    
    state.temp_results[task_id].append(result.to_dict())
    agent._observe(f"下钻分析完成: {result.row_count} 行", state)


@register_sql_task_handler(SQLTaskType.COMPARISON)
async def handle_comparison_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """执行对比分析任务"""
    task_id = task.get("id", "comparison")
    
    agent._step("1. 确定对比基准", state)
    agent._step("2. 构造对比 SQL", state)
    agent._step("3. 计算差异", state)
    
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 获取上游结果摘要
    parent_summary = agent._get_parent_results_summary(state, task)
    
    # 构建任务上下文
    context.current_task = agent._build_task_context(
        task, SQLTaskType.COMPARISON, state,
        intent_hint="对比两个时间段或条件下的指标，计算差值或增长率",
        parent_results_summary=parent_summary,
    )
    
    # 执行
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("comparison", state)
    
    # 统一 stats 字段（对比特有）
    if result.examples and len(result.examples) >= 2:
        values = [agent._get_numeric_value(ex) for ex in result.examples]
        if len(values) >= 2:
            result.stats["delta"] = values[1] - values[0]
            if values[0] > 0:
                result.stats["growth_rate"] = round((values[1] - values[0]) / values[0], 4)
            result.stats["comparison_base"] = values[0]
    
    state.temp_results[task_id].append(result.to_dict())
    agent._observe(f"对比分析完成: {result.row_count} 行", state)


@register_sql_task_handler(SQLTaskType.SUMMARY)
async def handle_summary_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """处理总结任务（由 Orchestrator 处理，此处仅记录）"""
    task_id = task.get("id", "summary")
    agent._observe("总结任务，跳过 SQL 执行，由 Orchestrator 处理", state)
    
    # 记录一个空结果，标明此任务已处理
    result = TaskResult(subtask="summary", issues=["handled_by_orchestrator"])
    state.temp_results[task_id].append(result.to_dict())


@register_sql_task_handler(SQLTaskType.VALIDATION)
async def handle_validation_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    task: dict[str, Any],
) -> None:
    """
    执行数据验证/诊断任务
    
    目的：诊断空结果原因，不是生成复杂的业务查询
    策略：
    1. 先查总行数（无任何条件）
    2. 逐步添加条件，看哪个条件导致数据为空
    3. 检查关键字段的值分布
    """
    task_id = task.get("id", "validation")
    description = task.get("description", "")
    
    agent._think(f"执行诊断任务: {description}", state)
    agent._step("1. 检查基础数据量", state)
    
    table_name = state.table_name
    
    # 诊断 SQL 列表（简单查询，不是业务计算）
    diagnostic_sqls = []
    
    # 1. 总行数
    diagnostic_sqls.append({
        "name": "total_rows",
        "sql": f'SELECT COUNT(*) as cnt FROM "{table_name}"',
        "desc": "表总行数"
    })
    
    # 2. 检查关键筛选字段的值分布
    # 从 Intent 中获取筛选条件相关的字段
    filter_fields = set()
    
    # 从 Intent 的 filter_refs 提取字段名
    if state.intent and hasattr(state.intent, 'filter_refs'):
        for ref in state.intent.filter_refs:
            if hasattr(ref, 'column') and ref.column:
                filter_fields.add(ref.column)
    
    # 从 available_columns 获取列名列表
    available_col_names = set()
    if state.available_columns:
        for col in state.available_columns:
            col_name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
            if col_name:
                available_col_names.add(col_name)
    
    # 常见的可能导致空结果的字段（只添加存在的字段）
    common_filter_fields = ["产品大类", "数据集来源", "大盘报表项", "统一剔除标签", "考核口径", "特殊口径"]
    for f in common_filter_fields:
        if f in available_col_names:
            filter_fields.add(f)
    
    # 为每个筛选字段生成值分布查询
    for field in list(filter_fields)[:5]:  # 最多检查 5 个字段
        diagnostic_sqls.append({
            "name": f"values_{field}",
            "sql": f'SELECT "{field}", COUNT(*) as cnt FROM "{table_name}" GROUP BY "{field}" ORDER BY cnt DESC LIMIT 10',
            "desc": f"字段 [{field}] 的值分布"
        })
    
    # 执行诊断查询
    results = []
    for diag in diagnostic_sqls:
        agent._step(f"检查: {diag['desc']}", state)
        try:
            rows = await agent.db_connector.execute_query(diag["sql"])
            result = TaskResult(
                subtask=diag["name"],
                sql=diag["sql"],
                row_count=len(rows),
                examples=rows[:10] if rows else [],
            )
            results.append(result)
            
            # 记录诊断发现
            if diag["name"] == "total_rows" and rows:
                total = rows[0].get("cnt", 0)
                if total == 0:
                    result.issues.append("table_empty")
                    agent._observe("⚠️ 表为空！", state)
                else:
                    agent._observe(f"表总行数: {total}", state)
                result.stats["total_rows"] = total
            elif diag["name"].startswith("values_"):
                if rows:
                    distinct_values = [str(r.get(list(r.keys())[0], "")) for r in rows[:5]]
                    result.stats["distinct_values"] = distinct_values
                    agent._observe(f"字段值示例: {distinct_values}", state)
                else:
                    result.issues.append("no_values")
                    
        except Exception as e:
            result = TaskResult(
                subtask=diag["name"],
                sql=diag["sql"],
                row_count=0,
                issues=[f"query_error: {str(e)[:100]}"],
            )
            results.append(result)
            agent._observe(f"查询失败: {e}", state)
    
    # 汇总诊断结果
    for r in results:
        state.temp_results[task_id].append(r.to_dict())
    
    agent._observe(f"诊断完成: 执行了 {len(results)} 个检查", state)
