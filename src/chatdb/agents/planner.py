"""
PlannerAgent：规划决策 Agent，将用户问题拆解为数据分析任务（DAG）。

核心理念：让 LLM 做智能决策，减少规则层复杂度
- 生成计划：LLM 直接输出 tasks
- 执行决策：LLM 根据 temp_results 决定继续/调整/结束
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from chatdb.agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from chatdb.core.react_state import ReActState
from chatdb.llm.base import BaseLLM, _extract_json_from_text as extract_json
from chatdb.utils.logger import get_component_logger


# ============================================================
# 任务类型枚举
# ============================================================

class TaskType(str, Enum):
    """任务类型枚举"""
    TREND = "trend"
    SOURCE = "source"
    DRILLDOWN = "drilldown"
    COMPARISON = "comparison"
    BASIC = "basic"
    SUMMARY = "summary"
    VALIDATION = "validation"
    ANOMALY = "anomaly"
    COHORT = "cohort"
    CORRELATION = "correlation"
    CLARIFY = "clarify"
    META = "meta"

    @property
    def label(self) -> str:
        labels = {
            self.TREND: "趋势分析", self.SOURCE: "来源分析",
            self.DRILLDOWN: "下钻分析", self.COMPARISON: "对比分析",
            self.BASIC: "基础查询", self.SUMMARY: "结果总结",
            self.VALIDATION: "SQL/数据验证", self.ANOMALY: "异常检测",
            self.COHORT: "队列分析", self.CORRELATION: "关联分析",
            self.CLARIFY: "人工澄清", self.META: "元任务",
        }
        return labels.get(self, self.value)

    @property
    def description(self) -> str:
        descriptions = {
            self.TREND: "按时间聚合，分析变化趋势",
            self.SOURCE: "按维度拆解，找出贡献来源",
            self.DRILLDOWN: "对 top 结果进一步细分",
            self.COMPARISON: "两个时间段/条件的对比",
            self.BASIC: "简单的聚合/求和/计数",
            self.SUMMARY: "整合前面结果，回答用户问题",
            self.VALIDATION: "验证 SQL 正确性、检查口径一致性、诊断空结果原因",
            self.ANOMALY: "发现数据中的异常点或突变",
            self.COHORT: "按用户群/时间群分析行为",
            self.CORRELATION: "分析指标间的关联关系",
            self.CLARIFY: "口径歧义、无法自动解决时请求用户确认",
            self.META: "读取多个任务结果做二次计算/判断/择优",
        }
        return descriptions.get(self, "")

    @property
    def priority_boost(self) -> int:
        return 10 if self == self.ANOMALY else 0


def get_task_types_for_prompt(for_generation: bool = True) -> str:
    """生成任务类型说明文本（用于 LLM prompt）"""
    # 生成阶段不含 validation/clarify
    exclude = {TaskType.VALIDATION, TaskType.CLARIFY} if for_generation else set()
    lines = []
    for t in TaskType:
        if t in exclude:
            continue
        lines.append(f'- "{t.value}": {t.label} - {t.description}')
    return "\n".join(lines)


# ============================================================
# 分析任务
# ============================================================

@dataclass
class AnalysisTask:
    """单条分析任务"""
    id: str
    type: TaskType
    description: str
    notes: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    skip_reason: str = ""
    strategy_group: str = ""
    priority: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            try:
                self.type = TaskType(self.type)
            except ValueError:
                self.type = TaskType.BASIC
        if self.priority == 0 and isinstance(self.type, TaskType):
            self.priority = self.type.priority_boost

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, TaskType) else self.type,
            "description": self.description,
            "notes": self.notes,
            "depends_on": self.depends_on,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "strategy_group": self.strategy_group,
            "priority": self.priority,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisTask":
        valid_fields = {"id", "type", "description", "notes", "depends_on", 
                        "status", "skip_reason", "strategy_group", "priority", "meta"}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ============================================================
# 分析计划
# ============================================================

@dataclass
class AnalysisPlan:
    """分析计划（DAG 拓扑结构）"""
    tasks: list[AnalysisTask] = field(default_factory=list)

    @property
    def task_map(self) -> dict[str, AnalysisTask]:
        return {t.id: t for t in self.tasks}

    def get_task(self, task_id: str) -> Optional[AnalysisTask]:
        return self.task_map.get(task_id)

    def add_task(self, task: AnalysisTask) -> None:
        self.tasks.append(task)

    def mark_completed(self, task_id: str) -> None:
        if task := self.task_map.get(task_id):
            task.status = "completed"

    def mark_skipped(self, task_id: str, reason: str = "") -> None:
        if task := self.task_map.get(task_id):
            task.status = "skipped"
            task.skip_reason = reason

    def mark_failed(self, task_id: str, error: str = "") -> None:
        if task := self.task_map.get(task_id):
            task.status = "failed"
            task.skip_reason = error

    def skip_tasks_by_type(self, task_types: list[TaskType], reason: str) -> list[str]:
        skipped = []
        for task in self.tasks:
            if task.type in task_types and task.status == "pending":
                task.status = "skipped"
                task.skip_reason = reason
                skipped.append(task.id)
        return skipped

    def is_done(self) -> bool:
        return all(t.status in ("completed", "failed", "skipped") for t in self.tasks)

    def get_ready_tasks(self, temp_results: Optional[dict] = None) -> list[AnalysisTask]:
        """获取依赖已满足的 pending 任务"""
        ready = []
        for task in self.tasks:
            if task.status != "pending":
                continue
            deps_ok = all(
                (dep := self.task_map.get(dep_id)) and dep.status in ("completed", "skipped")
                for dep_id in task.depends_on
            )
            if deps_ok:
                ready.append(task)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def to_display(self, ready_task_ids: Optional[set[str]] = None) -> str:
        """格式化展示计划"""
        lines = []
        icons = {"pending": "○", "in_progress": "●", "completed": "✓", "failed": "✗", "skipped": "⊘"}
        ready_ids = ready_task_ids or set()

        for t in self.tasks:
            marker = "→" if t.id in ready_ids and t.status == "pending" else " "
            desc = t.description[:50] + "..." if len(t.description) > 50 else t.description
            line = f"  {marker}{icons.get(t.status, '?')} [{t.type.value}] {desc}"
            if t.depends_on:
                line += f"  (deps: {', '.join(t.depends_on)})"
            lines.append(line)
            if t.status == "skipped" and t.skip_reason:
                lines.append(f"      ⚠️ 跳过原因: {t.skip_reason}")
        return "\n".join(lines)


# ============================================================
# 兼容旧代码的导出
# ============================================================

DEFAULT_TASK_TYPES = {t.value: t.description for t in TaskType}
_task_types: dict[str, str] = dict(DEFAULT_TASK_TYPES)

def get_task_types() -> dict[str, str]:
    return _task_types

def load_task_types_from_config(config: dict[str, Any]) -> dict[str, str]:
    global _task_types
    custom_types = config.get("planner", {}).get("task_types", {})
    if custom_types:
        _task_types = {**DEFAULT_TASK_TYPES, **custom_types}
    return _task_types


# ============================================================
# PlannerAgent
# ============================================================

class PlannerAgent(BaseAgent):
    """
    规划决策 Agent
    
    核心能力：
    1. 生成分析计划（LLM 直接输出任务 DAG）
    2. 查看执行结果，动态决策下一步
    3. 调整计划（跳过/插入/重试任务）
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="Planner",
            llm=llm,
            description="规划者：生成数据分析任务，查看执行结果，动态调整策略",
        )
        self._log = get_component_logger("Planner")
        self._analysis_plan: Optional[AnalysisPlan] = None
        self._data_constraints: dict[str, Any] = {}

    def clear_history(self) -> None:
        self._analysis_plan = None
        self._data_constraints = {}

    # ============================================================
    # 生成计划
    # ============================================================

    async def generate_analysis_plan(self, state: ReActState, context: AgentContext) -> AnalysisPlan:
        """生成分析计划"""
        if not state.intent:
            self._log.warn("无 Intent，使用默认计划")
            return self._get_default_plan()

        if state.yml_config:
            load_task_types_from_config(state.yml_config)

        if state.intent.is_meta_query():
            self._log.info("meta 模式，跳过分析流程")
            return self._get_meta_plan(state)

        plan = await self._generate_plan_with_llm(state, context)
        if plan and plan.tasks:
            self._analysis_plan = plan
            return plan

        self._log.warn("LLM 生成失败，使用默认计划")
        return self._get_default_plan()

    async def _generate_plan_with_llm(
        self, state: ReActState, context: AgentContext
    ) -> Optional[AnalysisPlan]:
        """LLM 生成分析任务 DAG"""
        intent = state.intent
        intent_summary = {
            "metrics": intent.metrics,
            "dimensions": intent.dimensions,
            "time": intent.time,
            "filter_refs": intent.filter_refs,
        }

        resolved_filters = self._resolve_filter_refs(state, intent.filter_refs)
        schema_info = self._build_schema_info(state, context)
        yml_info = self._build_yml_info(state)
        task_types_desc = get_task_types_for_prompt(for_generation=True)

        prompt = f"""## 角色
你是资深数据分析师，需要为用户问题设计**优雅、鲁棒**的分析思路。

## 用户问题
{state.user_query}

## 已解析的意图
{json.dumps(intent_summary, ensure_ascii=False, indent=2)}

## 已绑定的数据约束（自动应用到 SQL）
{resolved_filters}

## 数据 Schema
{schema_info}

## 业务配置
{yml_info or "（无）"}

## 可用的任务类型
{task_types_desc}

---

## 核心原则：尽量一步到位！

### 原则一：能一步完成的绝不拆分
- **占比/比例计算**：一个 SQL 用子查询或 CTE 即可完成，**不要拆成分子和分母两个任务**
- **同比/环比**：一个 SQL 用 LAG/LEAD 或子查询即可，**不要拆分**
- **简单聚合**：一个 basic 任务搞定

### 原则二：只有必须分步时才拆分
- 后续分析依赖前一步的**具体结果**
- 需要根据中间结果**动态决策**
- 多维度并行分析后需要**汇总对比**

---

## 输出格式

```json
{{
  "analysis_approach": "一句话说明分析思路",
  "tasks": [
    {{
      "id": "唯一标识",
      "type": "basic|trend|source|...",
      "description": "完整的分析指令",
      "depends_on": ["依赖的任务ID"],
      "notes": ["执行提示"]
    }}
  ]
}}
```

## 规则
1. **占比、同比、环比等计算型问题 → 1 个 basic 任务**
2. 简单问题不要拆解，1 个 basic 解决
3. 复杂问题一般不超过 5 个任务
4. **不要使用 validation/clarify**，它们是执行时遇到问题才用的"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是数据分析规划专家。设计优雅的分析思路，用原子任务编排。输出 JSON。",
                caller_name="planner",
            )

            json_str = extract_json(response)
            if not json_str:
                return None

            data = json.loads(json_str)
            tasks_data = data.get("tasks", [])
            
            approach = data.get("analysis_approach", "")
            if approach:
                self._log.info(f"分析思路: {approach}")

            self._log.info(f"LLM 生成 {len(tasks_data)} 个任务")
            return self._parse_tasks(tasks_data) if tasks_data else None

        except Exception as e:
            self._log.warn(f"规划失败: {e}")
            return None

    def _resolve_filter_refs(self, state: ReActState, filter_refs: list[str]) -> str:
        if not filter_refs:
            return "（无预定义筛选）"

        filters_config = (state.yml_config or {}).get("filters", {})
        lines = []
        for fid in filter_refs:
            if fid in filters_config:
                f = filters_config[fid]
                label = f.get("label", fid)
                expr = f.get("expr", "").strip()
                if expr:
                    expr_short = " ".join(line.strip() for line in expr.split("\n"))[:60]
                    lines.append(f"- **{fid}**: {label} → `{expr_short}`")
                else:
                    lines.append(f"- **{fid}**: {label}")
            else:
                lines.append(f"- {fid}: （未找到定义）")
        return "\n".join(lines)

    def _build_schema_info(self, state: ReActState, context: AgentContext) -> str:
        available_tables = context.available_tables or []
        if not available_tables:
            return state.schema_text[:800] if state.schema_text else "（无 Schema 信息）"

        target = next((t for t in available_tables if t.get("table_name") == state.table_name), None)
        if not target and available_tables:
            target = available_tables[0]
        if not target:
            return "（无 Schema 信息）"

        lines = [
            f"表名: {target.get('table_name', 'unknown')}",
            f"行数: {target.get('row_count', 0):,}",
            "",
            "列信息:",
        ]
        columns = target.get("columns_info") or target.get("columns", [])
        for col in columns:
            col_name = col.get("name", col.get("column_name", ""))
            col_type = col.get("type", col.get("column_type", ""))
            lines.append(f"  - {col_name} ({col_type})")
        return "\n".join(lines)

    def _build_yml_info(self, state: ReActState) -> str:
        if not state.yml_config:
            return ""

        lines = []
        metrics = state.yml_config.get("metrics", {})
        if metrics:
            lines.append("可用指标:")
            for mid, m in list(metrics.items())[:6]:
                label = m.get("label", mid)
                expr = m.get("expr") or m.get("agg", "")
                lines.append(f"  - {mid}: {label}" + (f" = {expr}" if expr else ""))

        dimensions = state.yml_config.get("dimensions", {})
        if dimensions:
            lines.append("可用维度:")
            for did, d in list(dimensions.items())[:8]:
                lines.append(f"  - {did}: {d.get('label', did)}")

        return "\n".join(lines)

    def _parse_tasks(self, tasks_data: list[dict]) -> Optional[AnalysisPlan]:
        if not tasks_data:
            return None

        tasks = []
        for i, t in enumerate(tasks_data):
            raw = {
                "id": t.get("id") or f"task_{i + 1}",
                "type": t.get("type", "basic"),
                "description": t.get("description", ""),
                "notes": t.get("notes", []),
                "depends_on": t.get("depends_on", []),
                "strategy_group": t.get("strategy_group", ""),
                "priority": t.get("priority", 0),
                "meta": t.get("meta", {}),
            }
            tasks.append(AnalysisTask.from_dict(raw))
        return AnalysisPlan(tasks=tasks) if tasks else None

    def _get_meta_plan(self, state: ReActState) -> AnalysisPlan:
        tasks = [AnalysisTask(
            id="meta_response",
            type=TaskType.META,
            description="处理元指令请求",
            notes=[f"meta_request: {getattr(state.intent, 'meta_request', '未指定')}"],
            meta={"action": "meta_response"},
        )]
        plan = AnalysisPlan(tasks=tasks)
        self._analysis_plan = plan
        return plan

    def _get_default_plan(self) -> AnalysisPlan:
        tasks = [
            AnalysisTask(id="basic_query", type=TaskType.BASIC,
                         description="执行基础查询，获取核心指标"),
            AnalysisTask(id="final_summary", type=TaskType.SUMMARY,
                         description="总结查询结果，回答用户问题"),
        ]
        plan = AnalysisPlan(tasks=tasks)
        self._analysis_plan = plan
        return plan

    # ============================================================
    # 查看执行结果
    # ============================================================

    def inspect_temp_results(self, state: ReActState) -> str:
        """查看 temp_results"""
        if not state.temp_results:
            return "（尚无）"

        lines = []
        for task_id, results in state.temp_results.items():
            lines.append(f"### 任务: {task_id}")
            for i, r in enumerate(results):
                subtask = r.get("subtask", f"步骤{i+1}")
                row_count = r.get("row_count", 0)
                examples = r.get("examples", [])
                stats = r.get("stats", {})
                issues = r.get("issues", [])
                sql = r.get("sql", "")

                lines.append(f"  [{subtask}] 返回 {row_count} 行")

                # SQL 错误时显示失败的 SQL
                has_sql_error = any("error:" in issue for issue in issues)
                if has_sql_error and sql:
                    sql_preview = sql[:300] + "..." if len(sql) > 300 else sql
                    lines.append(f"  **失败的SQL**: `{sql_preview}`")

                if examples:
                    lines.append("  示例:")
                    for ex in examples[:3]:
                        items = list(ex.items())[:4]
                        lines.append(f"    - {', '.join(f'{k}={v}' for k, v in items)}")
                if stats:
                    lines.append(f"  统计: {', '.join(f'{k}={v}' for k, v in list(stats.items())[:5])}")
                if issues:
                    lines.append(f"  备注: {', '.join(issues)}")
            lines.append("")
        return "\n".join(lines).strip()

    def summarize_results_for_planner(self, state: ReActState) -> str:
        """为决策 LLM 生成数据摘要"""
        if not state.temp_results:
            return "暂无执行结果"
        return "## 已执行任务的数据摘要\n" + self.inspect_temp_results(state)

    # ============================================================
    # 决策与调整
    # ============================================================

    async def decide_next_action(self, state: ReActState, context: AgentContext) -> dict[str, Any]:
        """根据 temp_results 决定下一步"""
        if not self._analysis_plan:
            return {"action": "done", "reason": "no_plan"}

        if self._analysis_plan.is_done():
            return {"action": "done", "reason": "plan_completed"}

        ready_tasks = self._analysis_plan.get_ready_tasks(state.temp_results)
        current_task = ready_tasks[0] if ready_tasks else None

        if not current_task:
            return {"action": "done", "reason": "no_current_task"}

        # 无结果时直接继续
        if not state.temp_results:
            return {"action": "continue", "task": current_task}

        # 调用 LLM 决策
        data = await self._llm_decide(state, current_task, ready_tasks)
        return self._apply_decision(data, current_task, state)

    async def _llm_decide(
        self, state: ReActState, current_task: AnalysisTask, ready_tasks: list[AnalysisTask]
    ) -> dict[str, Any]:
        """LLM 决策"""
        data_summary = self.summarize_results_for_planner(state)
        ready_ids = {t.id for t in ready_tasks}
        plan_display = self._analysis_plan.to_display(ready_ids) if self._analysis_plan else ""
        issues = self._detect_issues(data_summary)

        system_prompt = """你是数据分析决策专家。根据执行结果决定下一步。
关键原则：
1. 空结果 ≠ 失败，先诊断是 SQL 问题还是数据真的为空
2. SQL 报错时分析错误信息，给出修复建议
3. 输出 JSON"""

        prompt = f"""## 用户问题
{state.user_query}

## 当前计划状态
{plan_display}

{data_summary}

{issues}

---

## 决策选项

**A. 继续**：结果正常，按计划执行下一任务
**B. 插入任务**：遇到问题，需要插入 validation/clarify 任务处理
**C. 跳过任务**：前提不成立，跳过部分后续任务
**D. 结束**：可以给出结论

## 输出格式

选 A: {{"decision": "A", "reason": "..."}}
选 B（插入）: {{"decision": "B", "reason": "...", "adjustment": {{"insert_task": {{"type": "validation", "description": "..."}}}}}}
选 B（重试）: {{"decision": "B", "reason": "...", "adjustment": {{"retry_task": {{"task_id": "...", "fix_hint": "..."}}}}}}
选 C: {{"decision": "C", "reason": "...", "adjustment": {{"skip_tasks": ["task_id"]}}}}
选 D: {{"decision": "D", "reason": "...", "conclusion": "..."}}"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                caller_name="planner_decide",
            )
            json_str = extract_json(response)
            if json_str:
                data = json.loads(json_str)
                self._log.info(f"决策: {data.get('decision', '?')} - {data.get('reason', '')}")
                return data
        except Exception as e:
            self._log.warn(f"决策解析失败: {e}")
        return {"decision": "A", "reason": "解析失败，默认继续"}

    def _detect_issues(self, data_summary: str) -> str:
        """检测执行结果中的问题"""
        if not data_summary or data_summary == "暂无执行结果":
            return ""

        lines = []
        if "返回 0 行" in data_summary:
            lines.append("## ⚠️ 检测到空结果\n请判断是 SQL 问题、数据问题还是口径问题")

        # 提取 SQL 错误
        sql_errors = []
        for pattern in [r'error:([^\n,]+)', r'(Parser Error[^\n]*)', r'(Binder Error[^\n]*)']:
            for match in re.findall(pattern, data_summary, re.IGNORECASE):
                if match and match not in sql_errors:
                    sql_errors.append(match.strip())

        if sql_errors:
            error_details = "\n".join(f"- {err}" for err in sql_errors)
            lines.append(f"## ⚠️ SQL 执行错误\n{error_details}\n请分析错误原因，在 retry_task.fix_hint 中给出修复建议")

        return "\n\n".join(lines)

    def _apply_decision(
        self, data: dict[str, Any], current_task: AnalysisTask, state: ReActState
    ) -> dict[str, Any]:
        """应用决策结果"""
        decision = data.get("decision", "A")
        reason = data.get("reason", "")

        if decision == "A":
            next_task = self._get_next_task(state.temp_results)
            if next_task:
                return {"action": "continue", "task": next_task}
            return {"action": "done", "reason": "plan_completed"}

        elif decision == "B":
            adjustment = data.get("adjustment", {})
            adjust_result = self.apply_adjustment(adjustment, state)

            if adjust_result.get("inserted_task"):
                inserted = self._analysis_plan.get_task(adjust_result["inserted_task"])
                if inserted:
                    return {"action": "continue", "task": inserted, "adjustment_applied": adjust_result}

            if adjust_result.get("retry_task"):
                retry = self._analysis_plan.get_task(adjust_result["retry_task"])
                if retry:
                    return {"action": "retry", "task": retry, 
                            "retry_hint": retry.meta.get("retry_hint", ""),
                            "adjustment_applied": adjust_result}

            next_task = self._get_next_task(state.temp_results)
            if next_task:
                return {"action": "continue", "task": next_task, "adjustment_applied": adjust_result}
            return {"action": "done", "reason": "调整后无可执行任务"}

        elif decision == "C":
            adjustment = data.get("adjustment", {})
            self.apply_adjustment(adjustment, state)
            next_task = self._get_next_task(state.temp_results)
            if next_task:
                return {"action": "continue", "task": next_task}
            return {"action": "done", "reason": "跳过后无可执行任务"}

        else:  # D
            return {
                "action": "done",
                "reason": reason,
                "conclusion": data.get("conclusion", reason),
            }

    def apply_adjustment(self, adjustment: dict[str, Any], state: ReActState) -> dict[str, Any]:
        """执行计划调整"""
        if not self._analysis_plan:
            return {"applied": False, "message": "无计划可调整"}

        result: dict[str, Any] = {"applied": False, "skipped_tasks": [], 
                                   "inserted_task": None, "retry_task": None}
        reason = adjustment.get("reason", "LLM 决策调整")

        # 按任务类型跳过
        if adjustment.get("skip_types"):
            skip_types = [TaskType(t) for t in adjustment["skip_types"] if t in TaskType.__members__.values()]
            if skip_types:
                result["skipped_tasks"].extend(
                    self._analysis_plan.skip_tasks_by_type(skip_types, reason=reason)
                )

        # 按任务 ID 跳过
        if adjustment.get("skip_tasks"):
            for task_id in adjustment["skip_tasks"]:
                task = self._analysis_plan.get_task(task_id)
                if task and task.status == "pending":
                    self._analysis_plan.mark_skipped(task_id, reason=reason)
                    result["skipped_tasks"].append(task_id)

        # 插入任务
        if adjustment.get("insert_task"):
            inserted = self._insert_task(adjustment["insert_task"])
            if inserted:
                result["inserted_task"] = inserted.id
                result["applied"] = True

        # 重试任务
        if adjustment.get("retry_task"):
            retry_info = adjustment["retry_task"]
            task_id = retry_info.get("task_id")
            if task_id:
                task = self._analysis_plan.get_task(task_id)
                if task:
                    task.status = "pending"
                    task.meta["retry_hint"] = retry_info.get("fix_hint", "")
                    task.meta["retry_count"] = task.meta.get("retry_count", 0) + 1
                    result["retry_task"] = task_id
                    result["applied"] = True

        if result["skipped_tasks"]:
            result["applied"] = True

        return result

    def _insert_task(self, task_spec: dict[str, Any]) -> Optional[AnalysisTask]:
        """插入新任务"""
        if not self._analysis_plan:
            return None

        task_type_str = task_spec.get("type", "validation")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.VALIDATION

        # 防止死循环：同类型任务最多插入 2 个
        inserted_count = sum(1 for t in self._analysis_plan.tasks 
                             if t.id.startswith(f"inserted_{task_type.value}"))
        if inserted_count >= 2:
            self._log.warn(f"已插入 {inserted_count} 个 {task_type.value} 任务，不再插入")
            return None

        existing_ids = {t.id for t in self._analysis_plan.tasks}
        task_id = f"inserted_{task_type.value}"
        counter = 1
        while task_id in existing_ids:
            task_id = f"inserted_{task_type.value}_{counter}"
            counter += 1

        new_task = AnalysisTask(
            id=task_id,
            type=task_type,
            description=task_spec.get("description", f"动态插入的 {task_type.label}"),
            notes=task_spec.get("notes", []),
            depends_on=task_spec.get("depends_on", []),
            meta=task_spec.get("meta", {}),
            priority=15,
        )
        self._analysis_plan.add_task(new_task)
        self._log.info(f"插入任务: {task_id}")
        return new_task

    # ============================================================
    # 计划管理
    # ============================================================

    def _get_next_task(self, temp_results: Optional[dict] = None) -> Optional[AnalysisTask]:
        if self._analysis_plan:
            ready = self._analysis_plan.get_ready_tasks(temp_results)
            return ready[0] if ready else None
        return None

    def get_current_task(self, temp_results: Optional[dict] = None) -> Optional[AnalysisTask]:
        return self._get_next_task(temp_results)

    def get_ready_tasks(self, temp_results: Optional[dict] = None) -> list[AnalysisTask]:
        if self._analysis_plan:
            return self._analysis_plan.get_ready_tasks(temp_results)
        return []

    def advance_plan(self, temp_results: Optional[dict] = None) -> None:
        if self._analysis_plan:
            current = self._get_next_task(temp_results)
            if current:
                self._analysis_plan.mark_completed(current.id)

    def mark_task_failed(self, error: str, temp_results: Optional[dict] = None) -> None:
        if self._analysis_plan:
            current = self._get_next_task(temp_results)
            if current:
                self._analysis_plan.mark_failed(current.id, error)

    def is_plan_done(self) -> bool:
        return self._analysis_plan.is_done() if self._analysis_plan else True

    def get_plan_display(self, temp_results: Optional[dict] = None) -> str:
        if self._analysis_plan:
            ready = self._analysis_plan.get_ready_tasks(temp_results)
            return self._analysis_plan.to_display({t.id for t in ready})
        return "（无计划）"

    @property
    def analysis_plan(self) -> Optional[AnalysisPlan]:
        return self._analysis_plan

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(status=AgentStatus.SUCCESS, message="Planner executed")

    def get_system_prompt(self) -> str:
        return """你是数据分析规划专家，负责把用户的业务问题拆解为一组分析任务。
输出仅包含「要分析什么」——即任务类型、描述和依赖。请严格输出符合约定结构的 JSON。"""

    # ============================================================
    # 高级任务处理（简化版）
    # ============================================================

    def handle_clarify_task(self, task: AnalysisTask, state: ReActState) -> dict[str, Any]:
        """处理交互确认任务"""
        question = task.description
        options = task.meta.get("options", [])
        return {
            "needs_clarification": True,
            "question": question,
            "options": options[:5],
            "task_id": task.id,
        }

    def handle_meta_task(self, task: AnalysisTask, state: ReActState) -> dict[str, Any]:
        """处理元任务"""
        action = task.meta.get("action", "")
        if action == "select_best_strategy":
            group_name = task.meta.get("group", "")
            if not group_name or not self._analysis_plan:
                return {"success": False, "message": "未指定策略分组"}
            
            tasks = [t for t in self._analysis_plan.tasks if t.strategy_group == group_name]
            completed = [t for t in tasks if t.status == "completed"]
            if not completed:
                return {"success": False, "message": f"策略分组 {group_name} 无可用结果"}
            
            best = completed[0]
            return {"success": True, "action": action, "best_task": best.id, "group": group_name}
        
        return {"success": True, "action": action}
