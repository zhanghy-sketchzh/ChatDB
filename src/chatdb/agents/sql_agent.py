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
from chatdb.config.metrics_loader import preprocess_yaml_config
from chatdb.core.messages import TaskRequest, TaskResponse, TaskResultEntry
from chatdb.core.react_state import ErrorType, ReActState
from chatdb.database.base import BaseDatabaseConnector
from chatdb.llm.base import BaseLLM
from chatdb.tools.sql import SQLTool
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
    RANKING = "ranking"       # 排名分析（TopN）
    RATIO = "ratio"           # 占比分析（结构占比）
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
    SQLTaskType.RANKING: SQLTaskMeta(
        label="排名分析",
        description="按指标排序，找出 TopN 或 BottomN",
        requires_dimension=True,
        intent_hint_template="按{metric}排序，找出 Top{limit}，需要 GROUP BY 维度 ORDER BY 指标 DESC LIMIT N",
        stats_fields=["top_items", "top_value", "bottom_value"],
        issue_fields=["insufficient_data", "single_item"],
    ),
    SQLTaskType.RATIO: SQLTaskMeta(
        label="占比分析",
        description="计算部分占总体的比例",
        intent_hint_template="计算{numerator}在{denominator}中的占比，分子条件用 CASE WHEN，全局条件用 WHERE",
        stats_fields=["numerator", "denominator", "ratio"],
        issue_fields=["zero_denominator", "missing_numerator"],
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
    """获取任务类型的元信息"""
    if task_type not in SQL_TASK_META:
        raise KeyError(f"未注册的任务类型: {task_type.value}，请在 SQL_TASK_META 中添加定义")
    return SQL_TASK_META[task_type]


# =============================================================================
# Handler 注册表 - 插件式任务处理器
# =============================================================================

# Handler 类型签名
SQLTaskHandler = Callable[
    ["SQLAgent", ReActState, AgentContext, TaskRequest, TaskResponse],
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
    """任务执行结果（兼容旧接口，内部已迁移到 TaskResultEntry）"""
    subtask: str
    sql: str = ""
    row_count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask": self.subtask,
            "sql": self.sql,
            "row_count": self.row_count,
            "examples": self.examples,
            "stats": self.stats,
            "issues": self.issues,
        }

    def to_entry(self) -> TaskResultEntry:
        """转换为 TaskResultEntry"""
        return TaskResultEntry(
            subtask=self.subtask,
            sql=self.sql,
            row_count=self.row_count,
            examples=self.examples,
            stats=self.stats,
            issues=self.issues,
        )


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
        # 预处理：展开模板、表驱动生成
        data = preprocess_yaml_config(data)
        
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
        request: TaskRequest,
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
        
        # intent_hint 优先级：调用方显式指定 > 任务描述（description 已足够具体） > 元信息模板
        # 避免用刚性模板覆盖任务的具体语义
        if not intent_hint:
            # 任务描述已经足够具体时，直接用描述作为执行意图
            task_desc = request.description or ""
            if task_desc and len(task_desc) > 10:
                intent_hint = task_desc
            elif meta.intent_hint_template:
                intent_hint = meta.intent_hint_template.format(
                    granularity=time_granularity or meta.default_time_granularity or "year",
                    dimension=current_dimension or "维度",
                    metric=getattr(state, "current_metric", "指标"),
                )
        
        # 如果需要时间但没指定粒度，使用默认
        if meta.requires_time and not time_granularity:
            time_granularity = meta.default_time_granularity
        
        # ★ 提取 Planner 的 SQL 修复建议
        retry_hint = request.meta.get("retry_hint", "")
        retry_count = request.meta.get("retry_count", 0)
        
        # 优先使用 request 中的 parent_results_summary
        if not parent_results_summary and request.parent_results_summary:
            parent_results_summary = request.parent_results_summary
        
        ctx = {
            # 基础任务信息
            "task_id": request.task_id,
            "task_type": task_type.value,
            "description": request.description,
            "notes": list(request.notes),
            "depends_on": list(request.depends_on),
            
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

    @staticmethod
    def _resolve_dimension_column(dim_id: str, yml_config: dict[str, Any]) -> str:
        """将维度 ID 解析为实际列名（如 dim_product → 考核产品）"""
        dims = yml_config.get("dimensions", {})
        if dim_id in dims:
            return dims[dim_id].get("column", dim_id)
        return dim_id

    @staticmethod
    def _custom_condition_to_sql(col: str, op: str, val: Any, yml_config: dict[str, Any]) -> str | None:
        """
        将 custom condition 转换为 DuckDB SQL 表达式。
        
        自动将维度 ID（如 dim_product）解析为真实列名（如 考核产品）。
        """
        real_col = SQLAgent._resolve_dimension_column(col, yml_config)
        quoted = f'"{real_col}"'
        op = (op or "=").upper()

        if op == "IN":
            if isinstance(val, str):
                values = [v.strip() for v in val.split(",") if v.strip()]
            elif isinstance(val, (list, tuple)):
                values = [str(v) for v in val]
            else:
                values = [str(val)]
            if not values:
                return None
            vals_str = ", ".join(f"'{v}'" for v in values)
            return f"{quoted} IN ({vals_str})"
        elif op in ("=", "!=", "<>", ">", "<", ">=", "<="):
            op_str = "!=" if op == "<>" else op
            return f"{quoted} {op_str} '{val}'"
        elif op == "LIKE":
            return f"{quoted} LIKE '{val}'"
        return None

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
        
        # ★ 追加 intent 中的 custom 条件（如多轮对话指代消解产生的 IN 条件）
        if state.intent:
            for cond in state.intent.get_custom_filters():
                col = cond.get("column", "")
                op = (cond.get("op") or cond.get("operator", "=")).upper()
                val = cond.get("value")
                if not col or val is None:
                    continue
                sql_expr = self._custom_condition_to_sql(col, op, val, yml_config)
                if sql_expr:
                    required_filters.append({
                        "id": f"custom_{col}",
                        "label": f"用户指定筛选: {col}",
                        "expr": sql_expr,
                        "description": "来自用户查询的自定义筛选条件（如多轮对话指代消解）",
                    })

        state.required_filters = required_filters

        agg_preview = metric_def.get('agg', '')[:50] if metric_def.get('agg') else ''
        self._log.info(f"注入指标定义: {metric_name} ({metric_type}), agg={agg_preview}..., "
                       f"required_filters={len(required_filters)}个")

    async def run_task(
        self,
        state: ReActState,
        context: AgentContext,
        request: TaskRequest,
    ) -> TaskResponse:
        """
        执行 Planner 给的分析任务（核心接口）
        
        通过显式消息通信：
        - 输入: TaskRequest（Orchestrator 构建，包含任务描述和上游结果摘要）
        - 输出: TaskResponse（SQLAgent 返回，Orchestrator 决定如何存储）
        
        SQLAgent 不再直接写入 state.temp_results，
        由 Orchestrator 收到 TaskResponse 后集中管理。
        
        Args:
            state: ReAct 状态（SQLAgent 只读数据层字段 + 写执行期临时状态）
            context: Agent 上下文（API 兼容层）
            request: 任务请求消息
        
        Returns:
            TaskResponse: 执行结果消息
        """
        self._thoughts = []
        response = TaskResponse(task_id=request.task_id)
        
        # 注入领域配置
        if self.config:
            state.yml_config = self.config.raw_config
            if not state.table_name:
                state.table_name = self.config.table_name
        
        # ★ 每个新任务开始时，清除上一个任务的残留执行状态
        # 防止上一个任务的成功结果干扰 run_workflow 的重试循环判断
        state.execute_result = None
        state.execution_error = None
        state.error = None
        state.error_type = ErrorType.NONE
        state.error_context = {}
        state.current_sql = ""
        state.final_sql = ""
        state.refine_attempts = 0
        
        # 重试时的额外日志
        retry_count = request.meta.get("retry_count", 0)
        if retry_count > 0:
            self._log.info(f"重试任务 {request.task_id}（第 {retry_count} 次）")
        
        # 类型解析：字符串 → 枚举（类型安全）
        try:
            task_type = SQLTaskType(request.task_type)
        except ValueError:
            raise ValueError(f"未知任务类型: {request.task_type}，有效类型: {[t.value for t in SQLTaskType]}")
        
        # 获取 handler（注册表分发）
        handler = get_task_handler(task_type)
        if not handler:
            raise RuntimeError(f"任务类型 {task_type.value} 未注册 handler，请使用 @register_sql_task_handler 注册")
        
        # THINK: 记录分析思路
        state.think(request.description)
        
        # 执行 handler（handler 将结果追加到 response.results）
        try:
            await handler(self, state, context, request, response)
        except Exception as e:
            self._log.error(f"任务执行失败: {e}")
            response.success = False
            response.error = str(e)
            response.results.append(TaskResultEntry(subtask="error", issues=[str(e)]))
        
        # 进入下一个 ReAct 步骤
        state.next_step()
        return response
    
    # ============================================================
    # 任务执行器（已迁移到模块级 handler 函数）
    # 以下方法保留为内部工具方法
    # ============================================================

    def _get_parent_results_summary(self, request: TaskRequest) -> str:
        """从 TaskRequest 获取上游任务结果摘要"""
        # 优先使用 Orchestrator 预构建的摘要
        if request.parent_results_summary:
            return request.parent_results_summary
        
        # 从 previous_results 构建（兜底）
        if not request.previous_results:
            return ""
        
        summaries = []
        for r in request.previous_results:
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

    def _collect_result(self, subtask: str, state: ReActState) -> TaskResultEntry:
        """
        收集 SQL 执行结果，生成 TaskResultEntry
        
        包含：样例行 + 描述性统计 + 问题诊断
        """
        result = TaskResultEntry(subtask=subtask)
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
        """对所有数值列计算描述性统计（sum/min/max/avg）"""
        if not rows:
            return {}
        
        stats: dict[str, Any] = {
            "row_count": len(rows),
        }
        
        # 找到所有数值列
        numeric_cols = []
        for key, val in rows[0].items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_cols.append(key)
        
        # 对每个数值列计算统计
        for col in numeric_cols:
            values = [row.get(col, 0) for row in rows if row.get(col) is not None]
            if values:
                stats[f"{col}_sum"] = sum(values)
                stats[f"{col}_min"] = min(values)
                stats[f"{col}_max"] = max(values)
                stats[f"{col}_avg"] = sum(values) / len(values)
        
        return stats

    def _get_numeric_value(self, row: dict[str, Any]) -> float:
        """从行中获取数值（用于计算贡献度）"""
        for val in row.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        return 0.0

    def _get_previous_results(self, request: TaskRequest) -> list[dict[str, Any]]:
        """从 TaskRequest 获取之前的任务结果"""
        return list(request.previous_results)

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
        
        如果有 instructions，转换为 TaskRequest 格式调用 run_task
        """
        self._thoughts = []
        
        # 确保有基本信息
        if self.config:
            state.yml_config = self.config.raw_config
            if not state.table_name:
                state.table_name = self.config.table_name
        
        # 有指令时，转换为 TaskRequest 格式
        if instructions:
            task_type_str = instructions.get("task_type", instructions.get("type", ""))
            if not task_type_str:
                raise ValueError("instructions 中缺少 task_type 字段")
            request = TaskRequest(
                task_id=instructions.get("step", "task"),
                task_type=task_type_str,
                description=instructions.get("task", ""),
                notes=tuple([str(instructions["instructions"])]) if instructions.get("instructions") else (),
            )
            
            response = await self.run_task(state, context, request)
            # 兼容：将结果写回 state.temp_results
            if response.results:
                state.temp_results[request.task_id] = response.to_results_dicts()
            return
        
        # 无指令时，使用基础流程
        self._think(f"开始 SQL 分析: {state.user_query[:50]}...", state)
        await self._sql_tool.run_workflow(state, context)



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
    
    def _act(self, action: str, content: str, result: str = "", state: ReActState | None = None, tool: str = "") -> None:
        """
        记录行动
        
        Args:
            action: 行动类型（如 "execute_sql", "call_llm"）
            content: 行动描述
            result: 执行结果（可选）
            state: ReActState（可选，用于同步）
            tool: 工具名称（可选）
        """
        self._thoughts.append(SQLAgentThought(len(self._thoughts)+1, action, content, result))
        self._log.info(f"[ACT] {content}")
        if result:
            self._log.info(f"  → {result[:100]}...")
        # 同步到 state
        if state:
            state.act(content, tool=tool or action)

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
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """执行基础查询任务
    
    根据 Intent 中的时间粒度提示 LLM 如何分组。
    LLM 会根据 column_profiles（列元信息）自行推理合适的分组列。
    """
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 构建任务上下文（LLM 根据 Intent 和 column_profiles 自行推理分组方式）
    state.current_task = agent._build_task_context(
        request, SQLTaskType.BASIC, state,
    )
    
    # 执行 SQL
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("basic_query", state)
    
    # 统一 stats 字段
    if result.examples:
        total = sum(agent._get_numeric_value(ex) for ex in result.examples)
        result.stats["total_value"] = total
    
    response.results.append(result)


@register_sql_task_handler(SQLTaskType.TREND)
async def handle_trend_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """执行趋势分析任务"""
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 构建任务上下文（不硬编码 intent_hint，让 description 优先逻辑生效）
    state.current_task = agent._build_task_context(
        request, SQLTaskType.TREND, state,
        time_granularity="year",
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
        result.stats["year_count"] = len(years)
        
        # 计算增长率（如果有多年数据）
        if len(result.examples) >= 2:
            values = [agent._get_numeric_value(ex) for ex in result.examples]
            if values[0] > 0:
                result.stats["growth_rate"] = round((values[-1] - values[0]) / values[0], 4)
        
        if len(years) <= 1:
            result.issues.append("only_single_year")
    
    response.results.append(result)


@register_sql_task_handler(SQLTaskType.SOURCE)
async def handle_source_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """执行来源分析任务（按维度拆解）"""
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # ★ 收集所有可用维度（不选择，全部传给 LLM 判断）
    yml_config = state.yml_config or {}
    available_dims: list[str] = []
    
    # 从 intent.dimensions 解析
    if state.intent and state.intent.dimensions:
        for dim_id in state.intent.dimensions:
            col = SQLAgent._resolve_dimension_column(dim_id, yml_config)
            if col and col not in available_dims:
                available_dims.append(col)
    
    # 从配置补充
    if agent.config:
        for col in agent.config.get_priority_dimensions("source_analysis"):
            if col not in available_dims:
                available_dims.append(col)
    
    # 获取上游任务结果摘要
    parent_summary = agent._get_parent_results_summary(request)
    
    # ★ 将所有可用维度传入 context，由 SQL 生成 LLM 根据任务描述自行选择
    state.current_task = agent._build_task_context(
        request, SQLTaskType.SOURCE, state,
        parent_results_summary=parent_summary,
        extra={"available_dimensions": available_dims},
    )
    
    # 清理 intent 中的时间对比标记（source 不做对比）
    if state.intent and state.intent.time:
        state.intent.time.pop("comparison", None)
    
    # 执行维度分析
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("source", state)
    
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
    
    response.results.append(result)


@register_sql_task_handler(SQLTaskType.DRILLDOWN)
async def handle_drilldown_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """执行下钻分析任务"""
    # 从 TaskRequest 获取上游结果
    prev_results = agent._get_previous_results(request)
    parent_summary = agent._get_parent_results_summary(request)
    
    if not prev_results:
        response.results.append(TaskResultEntry(subtask="drilldown", issues=["no_previous_result"]))
        return
    
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 构建任务上下文（不硬编码 intent_hint，让 description 优先逻辑生效）
    state.current_task = agent._build_task_context(
        request, SQLTaskType.DRILLDOWN, state,
        parent_results_summary=parent_summary,
        extra={"previous_results": prev_results},
    )
    
    # 执行
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("drilldown", state)
    
    # 统一 stats 字段（下钻特有）
    for pr in prev_results:
        if pr.get("stats", {}).get("top_ratio"):
            result.stats["parent_top_ratio"] = pr["stats"]["top_ratio"]
            break
    
    response.results.append(result)


@register_sql_task_handler(SQLTaskType.COMPARISON)
async def handle_comparison_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """执行对比分析任务"""
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 获取上游结果摘要
    parent_summary = agent._get_parent_results_summary(request)
    
    # 构建任务上下文（不硬编码 intent_hint，让 description 优先逻辑生效）
    state.current_task = agent._build_task_context(
        request, SQLTaskType.COMPARISON, state,
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
    
    response.results.append(result)


@register_sql_task_handler(SQLTaskType.SUMMARY)
async def handle_summary_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """处理总结任务（由 Orchestrator 处理，此处仅记录）"""
    response.results.append(TaskResultEntry(subtask="summary", issues=["handled_by_orchestrator"]))


@register_sql_task_handler(SQLTaskType.VALIDATION)
async def handle_validation_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """
    执行数据验证/诊断任务
    
    目的：诊断空结果原因，不是生成复杂的业务查询
    策略：
    1. 先查总行数（无任何条件）
    2. 逐步添加条件，看哪个条件导致数据为空
    3. 检查关键字段的值分布
    """
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
    filter_fields = set()
    
    if state.intent and hasattr(state.intent, 'filter_refs'):
        for ref in state.intent.filter_refs:
            if hasattr(ref, 'column') and ref.column:
                filter_fields.add(ref.column)
    
    available_col_names = set()
    if state.available_columns:
        for col in state.available_columns:
            col_name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
            if col_name:
                available_col_names.add(col_name)
    
    for fld in list(filter_fields)[:5]:
        diagnostic_sqls.append({
            "name": f"values_{fld}",
            "sql": f'SELECT "{fld}", COUNT(*) as cnt FROM "{table_name}" GROUP BY "{fld}" ORDER BY cnt DESC LIMIT 10',
            "desc": f"字段 [{fld}] 的值分布"
        })
    
    # 执行诊断查询
    for diag in diagnostic_sqls:
        try:
            rows = await agent.db_connector.execute_query(diag["sql"])
            result = TaskResultEntry(
                subtask=diag["name"],
                sql=diag["sql"],
                row_count=len(rows),
                examples=rows[:10] if rows else [],
            )
            
            if diag["name"] == "total_rows" and rows:
                total = rows[0].get("cnt", 0)
                if total == 0:
                    result.issues.append("table_empty")
                result.stats["total_rows"] = total
            elif diag["name"].startswith("values_"):
                if rows:
                    distinct_values = [str(r.get(list(r.keys())[0], "")) for r in rows[:5]]
                    result.stats["distinct_values"] = distinct_values
                else:
                    result.issues.append("no_values")
                    
        except Exception as e:
            result = TaskResultEntry(
                subtask=diag["name"],
                sql=diag["sql"],
                row_count=0,
                issues=[f"query_error: {str(e)[:100]}"],
            )
        response.results.append(result)


@register_sql_task_handler(SQLTaskType.RATIO)
async def handle_ratio_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """
    执行占比分析任务
    
    核心目标：计算"部分 / 总体"的比例
    
    SQL 模式：
    ```sql
    SELECT 
        SUM(CASE WHEN 分子条件 THEN 金额 ELSE 0 END) as 分子,
        SUM(金额) as 分母,
        ROUND(分子/分母 * 100, 2) as 占比
    FROM 表
    WHERE 全局条件
    ```
    """
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 构建任务上下文
    state.current_task = agent._build_task_context(
        request, SQLTaskType.RATIO, state,
        intent_hint="计算占比，LLM 需智能判断：分子条件用 CASE WHEN，全局条件用 WHERE，输出占比百分比",
    )
    
    # 执行 SQL
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("ratio_analysis", state)
    
    # 统一 stats 字段（占比特有）：直接将所有数值列存入 stats
    if result.examples:
        for ex in result.examples:
            for key, val in ex.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    result.stats[key] = val
    
    response.results.append(result)


@register_sql_task_handler(SQLTaskType.RANKING)
async def handle_ranking_task(
    agent: SQLAgent,
    state: ReActState,
    context: AgentContext,
    request: TaskRequest,
    response: TaskResponse,
) -> None:
    """
    执行排名分析任务
    
    核心目标：按指标排序，找出 TopN 或 BottomN
    """
    # 注入指标定义
    agent._inject_metric_definition(state)
    
    # 获取排名参数（从 Intent 中）
    order_by = None
    limit = 10
    
    if state.intent:
        order_by = getattr(state.intent, "order_by", None)
        limit = getattr(state.intent, "limit", None) or 10
    
    direction = "DESC"
    if order_by and isinstance(order_by, dict):
        direction = order_by.get("direction", "DESC")
    
    # 构建任务上下文
    state.current_task = agent._build_task_context(
        request, SQLTaskType.RANKING, state,
        intent_hint=f"按指标{'降序' if direction == 'DESC' else '升序'}排序，返回 Top{limit}，需要 GROUP BY 维度 ORDER BY 指标 {direction} LIMIT {limit}",
        extra={"limit": limit, "order_direction": direction},
    )
    
    # 执行 SQL
    await agent._sql_tool.run_workflow(state, context)
    
    # 收集结果
    result = agent._collect_result("ranking_analysis", state)
    
    # 统一 stats 字段（排名特有）
    result.stats["limit"] = limit
    result.stats["direction"] = direction
    
    if result.examples:
        result.stats["top_items"] = len(result.examples)
        
        for ex in result.examples:
            for key, val in ex.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if "top_value" not in result.stats:
                        result.stats["top_value"] = val
                    result.stats["bottom_value"] = val
                    break
    
    response.results.append(result)
