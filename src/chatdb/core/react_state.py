"""
ReActState - 增强版状态机

支持 Planner 驱动的循环调度，包含：
- 阶段状态 (phase)
- 需求标记 (need_*)
- 错误分类 (error_type)
- 反思日志 (reflections)
- 分析切片 (AnalysisSlice) - 多步分析的中间结果
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chatdb.utils.logger import get_component_logger


# ============================================================
# 分析切片 - 多步分析的结构化中间结果
# ============================================================

class AnalysisPhase(str, Enum):
    """分析阶段类型"""
    BASE = "base"           # 基础查询结果
    EXPLORE = "explore"     # 维度探索
    DRILL_DOWN = "drill"    # 下钻分析
    COMPARE = "compare"     # 对比分析
    FILTER = "filter"       # 条件筛选


@dataclass
class AnalysisSlice:
    """
    分析切片 - 一次分析步骤的结构化结果
    
    设计目的：
    1. 存储每步分析的中间结果，支持增量分析
    2. 为 Planner 提供决策依据（已分析哪些维度、还有哪些可选）
    3. 支持多轮追问时复用已有结果
    
    Example:
        AnalysisSlice(
            phase=AnalysisPhase.EXPLORE,
            dimension="投资公司标签",
            sql="SELECT 投资公司标签, SUM(流水) ...",
            filters={"product": "王者荣耀", "year": 2025},
            rows=[{"投资公司标签": "IEG本部", "total": 68850}],
            row_count=1,
            insight="流水集中在 IEG本部，占比 100%",
        )
    """
    phase: AnalysisPhase
    sql: str = ""                                   # 执行的 SQL
    dimension: str | None = None                    # 拆解维度
    filters: dict[str, Any] = field(default_factory=dict)  # 筛选条件快照
    rows: list[dict[str, Any]] = field(default_factory=list)  # 结果数据（可裁剪）
    row_count: int = 0                              # 总行数
    insight: str | None = None                      # 局部洞察（LLM 生成）
    
    # 元信息
    step: int = 0                                   # 执行步骤号
    parent_slice_id: int | None = None              # 父切片 ID（用于追踪分析链路）
    meta: dict[str, Any] = field(default_factory=dict)  # 扩展字段
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "phase": self.phase.value,
            "dimension": self.dimension,
            "sql": self.sql,
            "filters": self.filters,
            "rows": self.rows[:10],  # 只保留前 10 行
            "row_count": self.row_count,
            "insight": self.insight,
            "step": self.step,
            "parent_slice_id": self.parent_slice_id,
        }
    
    def get_top_contributors(self, value_col: str | None = None, n: int = 3) -> list[dict]:
        """获取 Top N 贡献者"""
        if not self.rows:
            return []
        # 自动检测数值列
        if value_col is None:
            for row in self.rows[:1]:
                for k, v in row.items():
                    if isinstance(v, (int, float)) and k != self.dimension:
                        value_col = k
                        break
        if value_col is None:
            return self.rows[:n]
        return sorted(self.rows, key=lambda x: x.get(value_col, 0), reverse=True)[:n]
    
    def summary_line(self) -> str:
        """生成单行摘要（用于日志/Planner prompt）"""
        if self.phase == AnalysisPhase.BASE:
            return f"[BASE] {self.row_count} rows"
        elif self.phase == AnalysisPhase.EXPLORE:
            top = self.get_top_contributors(n=1)
            top_str = f", top={list(top[0].values())[0]}" if top else ""
            return f"[EXPLORE:{self.dimension}] {self.row_count} categories{top_str}"
        elif self.phase == AnalysisPhase.DRILL_DOWN:
            return f"[DRILL:{self.dimension}] {self.row_count} rows"
        elif self.phase == AnalysisPhase.COMPARE:
            return f"[COMPARE:{self.dimension}] {self.row_count} rows"
        return f"[{self.phase.value}] {self.row_count} rows"


class ReActPhase(str, Enum):
    """ReAct 阶段（统一新旧版本）"""
    # 新版阶段
    INIT = "init"                    # 初始化
    SEMANTIC_PARSE = "semantic"       # 语义解析
    SCHEMA_RESOLVE = "schema"         # Schema 补全
    SQL_BUILD = "sql_build"           # SQL 生成
    EXECUTE = "execute"               # 执行 SQL
    CRITIQUE = "critique"             # 评估/诊断
    REFINE = "refine"                 # 修正 SQL
    DONE = "done"                     # 完成
    GIVE_UP = "give_up"               # 放弃
    # 兼容旧版阶段
    PLAN = "plan"                    # 兼容: 等同于 INIT
    PARSE = "parse"                  # 兼容: 等同于 SEMANTIC_PARSE
    GENERATE = "generate"            # 兼容: 等同于 SQL_BUILD
    EVALUATE = "evaluate"            # 兼容: 等同于 CRITIQUE


class ErrorType(str, Enum):
    """错误类型分类"""
    NONE = "none"                          # 无错误
    UNKNOWN_COLUMN = "unknown_column"       # 列不存在
    TYPE_MISMATCH = "type_mismatch"         # 类型不匹配
    SYNTAX_ERROR = "syntax_error"           # SQL 语法错误
    SEMANTIC_GAP = "semantic_gap"           # 业务口径缺失
    AMBIGUOUS_INTENT = "ambiguous_intent"   # 意图不明确
    NO_DATA = "no_data"                      # 无数据返回
    TIMEOUT = "timeout"                      # 执行超时
    OTHER = "other"                          # 其他错误


@dataclass
class ReActState:
    """
    增强版 ReAct 状态
    
    核心设计：
    - need_* 标记：告诉 Planner 下一步需要什么
    - error_type：错误分类，决定回退策略
    - flags：业务标记（如 missing_time_dimension）
    - reflections：思考/反思日志
    """
    
    # ===== 输入 =====
    user_query: str
    
    # ===== 表选择 =====
    table_name: str | None = None
    schema_text: str = ""
    available_columns: list[dict[str, Any]] = field(default_factory=list)
    
    # ===== 语义解析 =====
    intent: Any = None                     # StructuredIntent
    yml_config: dict[str, Any] = field(default_factory=dict)
    
    # 语义补全中间结果
    extracted_terms: list[str] = field(default_factory=list)    # 提取的业务术语
    resolved_columns: dict[str, str] = field(default_factory=dict)  # 术语 -> 列名映射
    
    # ===== SQL 生成 =====
    sql_candidates: list[dict[str, Any]] = field(default_factory=list)
    current_sql: str = ""
    final_sql: str = ""
    
    # ===== 执行结果 =====
    execute_result: dict[str, Any] | None = None
    execution_error: str | None = None
    
    # ===== 输出 =====
    summary: str = ""
    
    # ===== 状态控制 =====
    phase: ReActPhase = ReActPhase.INIT
    step: int = 0
    max_steps: int = 10
    
    # ===== 需求标记（Planner 决策依据）=====
    need_intent: bool = True               # 需要语义解析
    need_schema_resolve: bool = False      # 需要 Schema 补全
    need_sql: bool = False                 # 需要生成 SQL
    need_execute: bool = False             # 需要执行 SQL
    need_critique: bool = False            # 需要评估
    need_refine: bool = False              # 需要修正
    
    # ===== 错误处理 =====
    error: str | None = None
    error_type: ErrorType = ErrorType.NONE
    error_context: dict[str, Any] = field(default_factory=dict)  # 错误上下文
    refine_attempts: int = 0
    max_refine_attempts: int = 3
    
    # ===== 业务标记 =====
    flags: dict[str, bool] = field(default_factory=dict)
    # 例如：
    # - missing_time_dimension: True  (没有时间列)
    # - accept_no_time_filter: False  (是否接受无时间条件)
    
    # ===== 显式任务计划（ToDoList）=====
    plan: list[dict[str, Any]] = field(default_factory=list)  # [{"step", "action", "goal", "status"}, ...]
    plan_index: int = 0                    # 当前执行到计划的第几步（从 0 开始）
    
    # ===== 分析型任务：结构化中间结果 =====
    need_more_analysis: bool = False       # LLM 评估：当前结果尚不足以完整回答用户问题
    analysis_slices: list[AnalysisSlice] = field(default_factory=list)  # 结构化分析切片
    explored_dimensions: set[str] = field(default_factory=set)  # 已探索过的维度（用于 Planner 决策）
    
    # ===== temp_results: Planner <-> SQL Agent 共享记忆 =====
    # 结构: {task_id: [{"subtask", "sql", "row_count", "examples", "stats", "issues"}, ...]}
    temp_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # ===== YAML 指标定义注入（SQLAgent -> SQLTool）=====
    current_metric: str = ""                 # 当前指标 ID
    current_metric_def: dict[str, Any] = field(default_factory=dict)  # 指标定义（含 agg, filter_refs）
    required_filters: list[dict[str, Any]] = field(default_factory=list)  # 必须包含的 WHERE 筛选条件
    
    # ===== 口径设计（CalibrationDesigner -> SQLTool）=====
    calibration_plan: Any = None             # CalibrationPlan 对象
    numerator_filters: list[dict[str, Any]] = field(default_factory=list)  # 分子专用筛选（CASE WHEN）
    sql_pattern: str = ""                    # SQL 模式建议（case_when_ratio / simple_agg）
    sql_hint: str = ""                       # LLM 给 SQL 生成的额外建议

    # ===== 思考/反思日志 =====
    thoughts: list[str] = field(default_factory=list)        # 思考日志
    observations: list[str] = field(default_factory=list)    # 观察日志
    reflections: list[str] = field(default_factory=list)     # 反思日志
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    
    # ===== 方法 =====
    
    def _get_logger(self):
        """获取组件日志器（延迟加载避免循环导入）"""
        return get_component_logger("ReAct")
    
    def think(self, thought: str) -> None:
        """记录思考"""
        self.thoughts.append(f"[Step {self.step}] THINK: {thought}")
        self._get_logger().think(f"(Step {self.step}) {thought}")
    
    def observe(self, observation: str) -> None:
        """记录观察"""
        self.observations.append(f"[Step {self.step}] OBSERVE: {observation}")
        self._get_logger().observe(f"(Step {self.step}) {observation}")
    
    def reflect(self, reflection: str) -> None:
        """记录反思"""
        self.reflections.append(f"[Step {self.step}] REFLECT: {reflection}")
        self._get_logger().reflect(f"(Step {self.step}) {reflection}")
    
    def act(self, action: str, result: dict[str, Any] | None = None) -> None:
        """记录行动"""
        self.tool_log.append({
            "step": self.step,
            "action": action,
            "result": result,
        })
        self._get_logger().act(action)
    
    # ===== 兼容旧版方法 =====
    
    def add_thought(self, thought: str) -> None:
        """兼容旧版: 等同于 think()"""
        self.think(thought)
    
    def log_tool(self, tool_name: str, input_data: dict, output_data: dict, duration_ms: float = 0) -> None:
        """兼容旧版: 记录工具调用"""
        self.tool_log.append({
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms,
        })
    
    def set_error(self, error: str, error_type: ErrorType, context: dict[str, Any] | None = None) -> None:
        """设置错误"""
        self.error = error
        self.error_type = error_type
        self.error_context = context or {}
        self.reflect(f"错误 ({error_type.value}): {error}")
    
    def clear_error(self) -> None:
        """清除错误"""
        self.error = None
        self.error_type = ErrorType.NONE
        self.error_context = {}
    
    def mark_need(self, **needs: bool) -> None:
        """
        设置需求标记
        
        Example:
            state.mark_need(need_intent=False, need_sql=True)
        """
        for key, value in needs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def clear_all_needs(self) -> None:
        """清除所有需求标记"""
        self.need_intent = False
        self.need_schema_resolve = False
        self.need_sql = False
        self.need_execute = False
        self.need_critique = False
        self.need_refine = False
    
    @property
    def is_done(self) -> bool:
        """是否完成"""
        return self.phase in (ReActPhase.DONE, ReActPhase.GIVE_UP)
    
    @property
    def can_continue(self) -> bool:
        """是否可以继续"""
        return not self.is_done and self.step < self.max_steps
    
    @property
    def has_valid_sql(self) -> bool:
        """是否有有效 SQL"""
        return bool(self.current_sql or self.final_sql)
    
    @property
    def has_result(self) -> bool:
        """是否有执行结果"""
        return self.execute_result is not None and self.execute_result.get("rows") is not None
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "user_query": self.user_query,
            "table_name": self.table_name,
            "intent": self.intent.to_dict() if self.intent and hasattr(self.intent, 'to_dict') else None,
            "phase": self.phase.value,
            "step": self.step,
            "current_sql": self.current_sql,
            "final_sql": self.final_sql,
            "error": self.error,
            "error_type": self.error_type.value,
            "flags": self.flags,
            "needs": {
                "intent": self.need_intent,
                "schema_resolve": self.need_schema_resolve,
                "sql": self.need_sql,
                "execute": self.need_execute,
                "critique": self.need_critique,
                "refine": self.need_refine,
            },
            "summary": self.summary,
        }
    
    def get_debug_info(self) -> dict[str, Any]:
        """获取调试信息"""
        return {
            "plan": self.plan,
            "plan_display": self.get_plan_display(),
            "reasoning_trace": self.get_reasoning_trace(),
            "thoughts": self.thoughts,
            "observations": self.observations,
            "reflections": self.reflections,
            "tool_log": self.tool_log,
            "flags": self.flags,
            "refine_attempts": self.refine_attempts,
        }

    def get_plan_display(self) -> str:
        """格式化计划为可读文本（用于 debug 输出）"""
        if not self.plan:
            return ""
        lines = ["Plan:"]
        for i, item in enumerate(self.plan):
            step = item.get("step", i + 1)
            goal = item.get("goal", "")
            action = item.get("action", "")
            status = item.get("status", "pending")
            marker = "→" if i == self.plan_index else ("✓" if status == "done" else " ")
            lines.append(f"  {marker} {step}. [{action}] {goal}")
        return "\n".join(lines)

    def get_current_plan_step(self) -> dict[str, Any] | None:
        """获取当前计划步骤"""
        if 0 <= self.plan_index < len(self.plan):
            return self.plan[self.plan_index]
        return None

    def advance_plan(self) -> None:
        """将计划推进到下一步"""
        if self.plan and self.plan_index < len(self.plan):
            self.plan[self.plan_index]["status"] = "done"
            self.plan_index += 1

    def insert_plan_step(self, action: str, goal: str, position: int | None = None) -> None:
        """在计划中插入新步骤（用于动态调整计划）"""
        new_step = {
            "step": len(self.plan) + 1,
            "action": action,
            "goal": goal,
            "status": "pending",
        }
        if position is None or position >= len(self.plan):
            # 插入到当前位置之后
            insert_pos = self.plan_index + 1
        else:
            insert_pos = position
        self.plan.insert(insert_pos, new_step)
        # 重新编号
        for i, p in enumerate(self.plan):
            p["step"] = i + 1

    def get_reasoning_trace(self) -> str:
        """
        将 THINK/OBSERVE/REFLECT 按步数合并为一段「ReAct 过程回放」
        
        格式优化：按步骤分组展示，THINK → OBSERVE → REFLECT 形成完整思考链
        """
        entries: list[tuple[int, str, str]] = []  # (step, kind, msg)
        
        # 定义排序优先级：THINK → OBSERVE → REFLECT
        kind_order = {"THINK": 0, "OBSERVE": 1, "REFLECT": 2}
        
        for s in self.thoughts:
            m = re.match(r"\[Step (\d+)\] THINK: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "THINK", m.group(2).strip()))
        for s in self.observations:
            m = re.match(r"\[Step (\d+)\] OBSERVE: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "OBSERVE", m.group(2).strip()))
        for s in self.reflections:
            m = re.match(r"\[Step (\d+)\] REFLECT: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "REFLECT", m.group(2).strip()))
        
        # 按 (step, kind_order) 排序
        entries.sort(key=lambda x: (x[0], kind_order.get(x[1], 9)))
        
        if not entries:
            return ""
        
        # 按步骤分组输出
        lines = ["ReAct 过程回放："]
        current_step = -1
        
        for step, kind, msg in entries:
            if step != current_step:
                current_step = step
                lines.append(f"\n[Step {step}]")
            
            # 根据类型添加前缀符号
            prefix = {"THINK": "💭", "OBSERVE": "👁️", "REFLECT": "🔄"}.get(kind, "•")
            lines.append(f"  {prefix} {kind}: {msg}")
        
        return "\n".join(lines)

    # ===== 分析切片管理 =====
    
    def add_analysis_slice(
        self,
        phase: AnalysisPhase,
        sql: str = "",
        dimension: str | None = None,
        filters: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
        row_count: int = 0,
        insight: str | None = None,
        parent_slice_id: int | None = None,
        **meta,
    ) -> AnalysisSlice:
        """
        添加分析切片
        
        Args:
            phase: 分析阶段
            sql: 执行的 SQL
            dimension: 拆解维度
            filters: 筛选条件
            rows: 结果数据
            row_count: 总行数
            insight: 局部洞察
            parent_slice_id: 父切片 ID
            **meta: 其他元信息
        
        Returns:
            创建的 AnalysisSlice
        """
        slice_obj = AnalysisSlice(
            phase=phase,
            sql=sql,
            dimension=dimension,
            filters=filters or {},
            rows=rows or [],
            row_count=row_count or len(rows or []),
            insight=insight,
            step=self.step,
            parent_slice_id=parent_slice_id,
            meta=meta,
        )
        self.analysis_slices.append(slice_obj)
        
        # 记录已探索维度
        if dimension and phase in (AnalysisPhase.EXPLORE, AnalysisPhase.DRILL_DOWN):
            self.explored_dimensions.add(dimension)
        
        # 日志
        self._get_logger().observe(f"Analysis: {slice_obj.summary_line()}")
        
        return slice_obj
    
    def get_base_slice(self) -> AnalysisSlice | None:
        """获取基础查询切片"""
        for s in self.analysis_slices:
            if s.phase == AnalysisPhase.BASE:
                return s
        return None
    
    def get_slices_by_dimension(self, dimension: str) -> list[AnalysisSlice]:
        """获取指定维度的所有切片"""
        return [s for s in self.analysis_slices if s.dimension == dimension]
    
    def get_unexplored_dimensions(self, available_dimensions: list[str]) -> list[str]:
        """获取尚未探索的维度"""
        return [d for d in available_dimensions if d not in self.explored_dimensions]
    
    def get_analysis_summary(self) -> str:
        """
        生成分析摘要（用于 Planner prompt 或 summary）
        
        Example output:
            ## 已有分析结果
            - 基础结果: total_flow=606.72亿, 1 rows
            - 已按以下维度拆解:
              - 投资公司标签: 1 类 (top: IEG本部)
            - 未探索维度: 国内/海外, 产品大类
        """
        if not self.analysis_slices:
            return ""
        
        lines = ["## 已有分析结果"]
        
        # 基础结果
        base = self.get_base_slice()
        if base:
            if base.rows:
                preview = ", ".join(f"{k}={v}" for k, v in list(base.rows[0].items())[:3])
                lines.append(f"- 基础结果: {preview} ({base.row_count} rows)")
            else:
                lines.append(f"- 基础结果: {base.row_count} rows")
        
        # 探索/下钻结果
        explores = [s for s in self.analysis_slices if s.phase in (AnalysisPhase.EXPLORE, AnalysisPhase.DRILL_DOWN)]
        if explores:
            lines.append("- 已按以下维度拆解:")
            for s in explores:
                top = s.get_top_contributors(n=1)
                top_str = f" (top: {list(top[0].values())[0]})" if top else ""
                lines.append(f"  - {s.dimension}: {s.row_count} 类{top_str}")
        
        return "\n".join(lines)

    # ===== 兼容旧版 analysis_results =====
    
    @property
    def analysis_results(self) -> list[dict[str, Any]]:
        """兼容旧版: 返回 dict 格式的分析结果"""
        return [s.to_dict() for s in self.analysis_slices]

