"""
SQL 工具 - 单一 SQLTool 类：验证、执行、生成、执行与评估

对外暴露：
- SQLTool(llm?, db?): 统一入口，支持 validate_sql / execute_sql / generate_sql / execute_and_evaluate
- run_workflow / run_generate / run_execute_and_evaluate: 供 Orchestrator 与 ReAct 流程调用
- ValidateSQLTool / ExecuteSQLTool / GenerateSQLTool / ExecuteAndEvaluateTool: 薄包装，供 Registry 注册
- SQLWorkflowTool: 生成→验证→执行与评估 的完整流程
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.duckdb.syntax_rules import get_duckdb_syntax_rules
from chatdb.llm.base import BaseLLM
from chatdb.tools.base import BaseTool, ToolParameter, ToolResult
from chatdb.utils.logger import get_component_logger
from chatdb.utils.common import parse_json, clean_sql as _clean_sql_util, format_rows

from chatdb.core.react_state import ReActState, ReActPhase, ErrorType, AnalysisPhase

if TYPE_CHECKING:
    from chatdb.agents.base import AgentContext
    from chatdb.agents.semantic_parser import StructuredIntent


# ---------- 公共逻辑 ----------

DANGEROUS_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "CREATE",
]


def check_sql_readonly(sql: str) -> list[str]:
    """只读安全检查：必须 SELECT，禁止写操作关键词。返回错误列表，空表示通过。"""
    errors: list[str] = []
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT"):
        errors.append("只支持 SELECT 查询")
    for kw in DANGEROUS_KEYWORDS:
        if kw in sql_upper:
            errors.append(f"禁止使用 {kw} 语句")
    return errors


def check_sql_syntax(sql: str) -> list[str]:
    """基础语法检查：SELECT、FROM、括号匹配。返回错误列表，空表示通过。"""
    errors: list[str] = []
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT"):
        errors.append("SQL 必须以 SELECT 开头")
    if "FROM" not in sql_upper:
        errors.append("SQL 必须包含 FROM 子句")
    if sql.count("(") != sql.count(")"):
        errors.append("括号不匹配")
    return errors


def check_sql_safety(sql: str) -> list[str]:
    """安全检查（危险关键词）。返回错误列表。"""
    errors: list[str] = []
    sql_upper = sql.upper()
    for kw in DANGEROUS_KEYWORDS:
        if kw in sql_upper:
            errors.append(f"禁止使用 {kw} 语句")
    return errors


def error_type_from_str(s: str) -> ErrorType:
    """将 Agent/工具返回的 error_type 字符串映射为 ErrorType 枚举。"""
    _map = {
        "unknown_column": ErrorType.UNKNOWN_COLUMN,
        "type_mismatch": ErrorType.TYPE_MISMATCH,
        "syntax_error": ErrorType.SYNTAX_ERROR,
        "no_data": ErrorType.NO_DATA,
    }
    return _map.get(s, ErrorType.OTHER)


# ---------- 数据类与错误模式 ----------


@dataclass
class SQLCandidate:
    """SQL 候选项"""
    sql: str
    reason: str
    confidence: float = 1.0
    validation_status: str = "pending"
    validation_error: str | None = None


@dataclass
class EvaluationResult:
    """评估结果"""
    sql: str
    execution_success: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_error: str | None = None
    diagnosis: str = ""
    error_type: ErrorType = ErrorType.NONE
    error_context: dict[str, Any] = field(default_factory=dict)
    refined: bool = False
    refined_sql: str = ""
    refinement_reason: str = ""
    summary: str = ""


ERROR_PATTERNS: dict[ErrorType, list[str]] = {
    ErrorType.UNKNOWN_COLUMN: [
        r"Binder Error.*column.*not found",
        r"Unknown column",
        r"no such column",
        r"does not exist",
        r"Referenced column.*not found",
    ],
    ErrorType.TYPE_MISMATCH: [
        r"Type mismatch",
        r"cannot compare",
        r"incompatible types",
        r"Conversion Error",
        r"Could not convert",
    ],
    ErrorType.SYNTAX_ERROR: [
        r"Parser Error",
        r"Syntax error",
        r"unexpected token",
        r"near \".*\"",
    ],
}

def _check_business(sql: str, yml_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """业务规则检查。返回 (errors, warnings)。"""
    errors: list[str] = []
    warnings: list[str] = []
    filters = yml_config.get("filters", {})
    rules = yml_config.get("rules", [])
    base_filter = filters.get("base_valid_data", {})
    if base_filter:
        expr = base_filter.get("expr", "")
        for field in ["统一剔除标签", "is_valid"]:
            if field in expr and field not in sql:
                warnings.append(f"建议添加基础筛选条件（{field}）")
    for rule in rules:
        if rule.get("type") == "validate_filter":
            must_include = rule.get("must_include")
            if must_include:
                filter_def = filters.get(must_include, {})
                if filter_def.get("expr") and filter_def["expr"] not in sql:
                    warnings.append(f"建议包含 {filter_def.get('label', must_include)} 筛选")
    return errors, warnings


# ---------- 单一 SQLTool 类 ----------


class SQLTool:
    """
    统一 SQL 工具：验证、执行、生成、执行与评估。

    - validate_sql: 语法与业务规则校验（无需 llm/db）
    - execute_sql: 只读执行并返回结果（需 db_connector）
    - generate_sql: 根据意图生成 SQL（需 llm）
    - execute_and_evaluate: 执行并诊断/修正（需 llm + db_connector）
    """

    MAX_REFINE_ATTEMPTS = 2

    def __init__(
        self,
        llm: BaseLLM | None = None,
        db_connector: BaseDatabaseConnector | None = None,
    ):
        self.llm = llm
        self.db_connector = db_connector
        self._log = get_component_logger("SQLTool")

    def validate_sql(
        self,
        sql: str,
        yml_config: dict[str, Any] | None = None,
    ) -> ToolResult:
        """验证 SQL 语法与业务规则。"""
        self._log.info(f"验证: {sql[:50]}...")
        errors: list[str] = []
        warnings: list[str] = []
        errors.extend(check_sql_syntax(sql))
        errors.extend(check_sql_safety(sql))
        if yml_config:
            biz_errors, biz_warnings = _check_business(sql, yml_config)
            errors.extend(biz_errors)
            warnings.extend(biz_warnings)
        is_valid = len(errors) == 0
        return ToolResult.ok(
            data={
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
            },
            message="验证通过" if is_valid else f"验证失败: {len(errors)} 个错误",
        )

    async def execute_sql(self, sql: str, limit: int = 100) -> ToolResult:
        """执行只读 SQL 并返回结果。"""
        if not self.db_connector:
            return ToolResult.fail("未配置数据库连接")
        self._log.info(f"执行: {sql[:60]}...")
        errs = check_sql_readonly(sql)
        if errs:
            return ToolResult.fail(errs[0])
        try:
            rows = await self.db_connector.execute_query(sql)
            columns = list(rows[0].keys()) if rows else []
            return ToolResult.ok(
                data={
                    "rows": rows[:limit],
                    "row_count": len(rows),
                    "columns": columns,
                },
                message=f"查询返回 {len(rows)} 行",
            )
        except Exception as e:
            self._log.error(f"执行失败: {e}")
            return ToolResult.fail(f"SQL 执行失败: {e}")

    # ---------- 生成 SQL（原 SQLGenerator 逻辑） ----------

    def _get_table_schema(
        self,
        table_name: str,
        available_tables: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """获取表的 Schema 信息
        
        返回：
            - table_name: 表名
            - columns: 列信息
            - row_count: 行数
            - create_table_sql: 建表 SQL
            - column_profiles: 丰富的列元信息（如果有）
        """
        if not available_tables:
            return {"table_name": table_name, "columns": [], "row_count": 0}
        for table in available_tables:
            if table.get("table_name") == table_name:
                columns = table.get("columns_info") or table.get("columns", [])
                return {
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": table.get("row_count", 0),
                    "create_table_sql": table.get("create_table_sql", ""),
                    # 新增：column_profiles 包含唯一值数量、高频值、统计信息
                    "column_profiles": table.get("column_profiles", []),
                }
        return {"table_name": table_name, "columns": [], "row_count": 0}

    def _format_columns_for_prompt(
        self, 
        columns: list[dict[str, Any]],
        column_profiles: list[dict[str, Any]] | None = None,
    ) -> str:
        """格式化列信息供 LLM 理解
        
        Args:
            columns: 基础列信息 [{"name": "月份", "type": "BIGINT"}, ...]
            column_profiles: 丰富的列元信息（来自 meta_data.db），包含：
                - unique_count: 唯一值数量
                - summary: 统计摘要（范围/高频值）
                - top_values: 高频值列表
                - stats: 数值统计（min/max/mean）
        
        这些信息让 LLM 能够推理出如何分组、筛选，而不需要硬编码规则。
        例如：LLM 看到 "月份: 整数, 12个唯一值, 范围[202501~202512]" 
        就能推理出这是月度数据，可以按季度分组。
        """
        if not columns:
            return "无列信息"
        
        # 构建 column_profiles 的索引
        profiles_map: dict[str, dict] = {}
        if column_profiles:
            for p in column_profiles:
                profiles_map[p.get("name", "")] = p
        
        lines = []
        for col in columns:
            col_name = col.get("name", col.get("column_name", ""))
            col_type = col.get("type", col.get("column_type", ""))
            
            # 基础信息
            line = f'- "{col_name}" ({col_type})'
            
            # 尝试从 column_profiles 获取丰富信息
            profile = profiles_map.get(col_name)
            if profile:
                # 唯一值数量
                unique_count = profile.get("unique_count")
                if unique_count is not None:
                    line += f" [唯一值:{unique_count}]"
                
                # 统计摘要（最有价值的信息）
                summary = profile.get("summary", "")
                if summary:
                    # 截断过长的摘要
                    if len(summary) > 100:
                        summary = summary[:100] + "..."
                    line += f" -- {summary}"
            else:
                # 没有 column_profiles，不提供额外信息
                pass
            
            lines.append(line)
        
        return "\n".join(lines)

    def _get_metrics_info(
        self,
        metric_ids: list[str],
        metrics_config: dict[str, Any],
        exclude_metric_id: str | None = None,
    ) -> str:
        """指标参考信息；exclude_metric_id 用于避免与上文「指标约束」重复。"""
        if not metric_ids and not metrics_config:
            return "无指定指标"
        if exclude_metric_id and metric_ids == [exclude_metric_id] and not metrics_config:
            return "（当前指标见上文「指标约束」）"
        lines = []
        for mid in metric_ids:
            if mid == exclude_metric_id:
                continue
            if mid in metrics_config:
                m = metrics_config[mid]
                lines.append(f"- {mid}:")
                lines.append(f"  label: {m.get('label', '')}")
                expr = m.get('expr') or m.get('agg', '')
                if expr:
                    lines.append(f"  expr: {expr}")
                default_filters = m.get("default_filters") or m.get("filter_refs", [])
                if default_filters:
                    lines.append(f"  default_filters: {default_filters}")
                if m.get("description"):
                    lines.append(f"  description: {m.get('description')}")
        if not lines and metrics_config:
            for mid, m in metrics_config.items():
                if mid == exclude_metric_id:
                    continue
                expr = m.get('expr') or m.get('agg', '')
                lines.append(f"- {mid}: {m.get('label', '')} = {expr}")
        if exclude_metric_id and not lines:
            return "（当前指标见上文「指标约束」）"
        return "\n".join(lines) if lines else "无匹配指标"

    def _get_dimensions_info(self, dim_ids: list[str], dims_config: dict[str, Any]) -> str:
        if not dim_ids and not dims_config:
            return "无指定维度"
        lines = []
        for did in dim_ids:
            if did in dims_config:
                d = dims_config[did]
                lines.append(f"- {did}:")
                lines.append(f"  label: {d.get('label', '')}")
                lines.append(f"  column: {d.get('column', '')}")
        if not lines and dims_config:
            lines.append("可用维度：")
            for did, d in dims_config.items():
                col = d.get("column", did)
                lines.append(f'- {did}: {d.get("label", "")} -> 列 "{col}"')
        return "\n".join(lines) if lines else "无匹配维度"

    def _get_filter_refs_info(self, filter_refs: list[str], filters_config: dict[str, Any]) -> str:
        if not filter_refs and not filters_config:
            return "无预定义筛选器"
        lines = []
        for fid in filter_refs:
            if fid in filters_config:
                f = filters_config[fid]
                lines.append(f"- {fid}:")
                lines.append(f"  label: {f.get('label', '')}")
                lines.append(f"  expr: {f.get('expr', '')}")
                lines.append(f"  description: {f.get('description', '')}")
        if not lines and filters_config:
            lines.append("可用筛选器：")
            for fid, f in filters_config.items():
                lines.append(f"- {fid}: {f.get('label', '')} = {f.get('expr', '')}")
        return "\n".join(lines) if lines else "无匹配筛选器"

    def _extract_metrics_from_filter_refs(
        self,
        filter_refs: list[str],
        filters_config: dict[str, Any],
        metrics_config: dict[str, Any],
    ) -> str:
        """
        从 filter_refs 中提取指标相关的筛选表达式。
        
        很多 YAML 配置把指标定义放在 filters 节点下（如 metric_flow, metric_gross），
        而不是 metrics 节点。这个方法识别这些"指标型筛选器"并提取其表达式。
        """
        if not filter_refs or not filters_config:
            return ""
        
        lines = []
        for fid in filter_refs:
            # 识别指标型筛选器（以 metric_ 开头，基于 YAML 配置结构识别）
            if fid.startswith("metric_"):
                if fid in filters_config:
                    f = filters_config[fid]
                    expr = f.get("expr", "")
                    label = f.get("label", "")
                    if expr:
                        lines.append(f"- {label}（{fid}）: WHERE {expr}")
        
        # 同时检查 metrics 中是否有 filter_refs 引用
        for metric_id, metric_def in metrics_config.items():
            metric_filter_refs = metric_def.get("filter_refs", [])
            for mfr in metric_filter_refs:
                if mfr in filters_config and mfr not in [l.split("（")[1].split("）")[0] for l in lines if "（" in l]:
                    f = filters_config[mfr]
                    expr = f.get("expr", "")
                    if expr:
                        lines.append(f"- {metric_def.get('label', metric_id)} 需要: WHERE {expr}")
        
        return "\n".join(lines) if lines else ""

    def _get_filters_info(self, filters: list[dict], dims_config: dict[str, Any]) -> str:
        if not filters:
            return "无筛选条件"
        lines = []
        for f in filters:
            if "_agg_column" in f:
                lines.append(f"- 聚合: {f.get('_agg_func', 'SUM')}(\"{f.get('_agg_column')}\")")
                continue
            dim_id = f.get("dimension", f.get("column", ""))
            value = f.get("value", "")
            operator = f.get("operator", "=")
            if dim_id in dims_config:
                dim_def = dims_config[dim_id]
                column = dim_def.get("column", dim_id)
                terms = dim_def.get("terms", {})
                term_filter = None
                for term_id, term_def in terms.items():
                    if term_def.get("term") == value or value in term_def.get("synonyms", []):
                        term_filter = term_def.get("filter", "")
                        break
                if term_filter:
                    lines.append(f"- {dim_id}.{value} -> {term_filter}")
                else:
                    lines.append(f'- "{column}" {operator} \'{value}\'')
            else:
                lines.append(f'- "{dim_id}" {operator} \'{value}\'')
        return "\n".join(lines)

    def _clean_sql(self, sql: str) -> str:
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*", "", sql)
        sql = sql.strip()
        
        # 修复不完整的 CASE 语句（LLM 有时会截断 END 关键字）
        sql = self._fix_incomplete_case(sql)
        
        if not sql.endswith(";"):
            sql += ";"
        return sql
    
    def _fix_incomplete_case(self, sql: str) -> str:
        """
        修复 LLM 生成的不完整 CASE 语句
        
        问题场景：LLM 在 ORDER BY 等子句中生成 CASE WHEN...THEN... 后可能截断 END
        例如：ORDER BY CASE WHEN x THEN 'Q1' WHEN y THEN 'Q2'  (缺少 END)
        """
        # 统计 CASE 和 END 的数量（忽略大小写）
        sql_upper = sql.upper()
        case_count = len(re.findall(r'\bCASE\b', sql_upper))
        end_count = len(re.findall(r'\bEND\b', sql_upper))
        
        if case_count > end_count:
            # 有未闭合的 CASE，尝试在末尾补充 END
            missing = case_count - end_count
            self._log.warn(f"检测到 {missing} 个未闭合的 CASE 语句，尝试修复")
            
            # 移除末尾分号（如果有）
            sql_trimmed = sql.rstrip(';').rstrip()
            
            # 添加缺少的 END
            sql = sql_trimmed + ' END' * missing
        
        return sql

    def _parse_candidates(self, response: str) -> list[SQLCandidate]:
        candidates = []
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response)
            data = json.loads(json_match.group()) if json_match else {}
        for c in data.get("candidates", []):
            sql = (c.get("sql", "") or "").strip()
            reason = c.get("reason", "")
            if sql:
                sql = self._clean_sql(sql)
                candidates.append(SQLCandidate(sql=sql, reason=reason, confidence=0.9))
        return candidates

    def _build_rule_based_sql(
        self,
        intent: Any,
        yml_config: dict[str, Any],
        table_schema: dict[str, Any] | None = None,
        state: Any = None,  # ReActState，用于获取预组合的 required_filters
    ) -> str | None:
        if not intent.table_name:
            return None
        metrics_config = yml_config.get("metrics", {})
        dims_config = yml_config.get("dimensions", {})
        filters_config = yml_config.get("filters", {})
        available_columns = set()
        if table_schema and table_schema.get("columns"):
            for col in table_schema["columns"]:
                cn = col.get("name", col.get("column_name", ""))
                if cn:
                    available_columns.add(cn)
        
        # ★ 核心改动：获取当前指标的 agg 表达式
        agg_expr = None
        if state and hasattr(state, "current_metric_def") and state.current_metric_def:
            agg_expr = state.current_metric_def.get("agg", "")
        
        select_parts = []
        for dim_id in intent.dimensions:
            if dim_id in dims_config:
                col = dims_config[dim_id].get("column", dim_id)
                if col in available_columns or not available_columns:
                    select_parts.append(f'"{col}"')
            elif dim_id in available_columns:
                select_parts.append(f'"{dim_id}"')
        time_config = dims_config.get("time", {})
        col_map = time_config.get("column_map", {"year": "年", "quarter": "季度", "month": "月份"})
        year_col = col_map.get("year", "年")
        if year_col in available_columns or not available_columns:
            if intent.time.get("granularity", "year") in ("year", "quarter", "month"):
                select_parts.append(f'"{year_col}"')
        
        # ★ 使用 agg 表达式（如果有）
        if agg_expr:
            metric_label = ""
            if state and hasattr(state, "current_metric_def"):
                metric_label = state.current_metric_def.get("label", "指标")
            select_parts.append(f'{agg_expr} AS "{metric_label}"')
        else:
            # 回退到原逻辑
            for metric_id in intent.metrics:
                if metric_id in metrics_config:
                    m = metrics_config[metric_id]
                    expr = m.get("agg", m.get("expr", ""))
                    label = m.get("label", metric_id)
                    if expr:
                        select_parts.append(f'{expr} AS "{label}"')
        
        for f in intent.filters:
            if "_agg_column" in f:
                agg_col = f.get("_agg_column")
                agg_func = f.get("_agg_func", "SUM")
                if agg_col in available_columns or not available_columns:
                    select_parts.append(f'{agg_func}("{agg_col}") AS "{agg_col}_{agg_func}"')
        if not select_parts:
            return None
        
        # ★ 核心改动：优先使用预组合的 required_filters
        where_parts = []
        if state and hasattr(state, "required_filters") and state.required_filters:
            for f in state.required_filters:
                expr = f.get("expr", "")
                if expr:
                    # 处理多行 expr
                    expr_lines = [line.strip() for line in expr.strip().split("\n") if line.strip()]
                    where_parts.append(" ".join(expr_lines))
        
        # 补充时间筛选
        if intent.time.get("year") and (year_col in available_columns or not available_columns):
            year_filter = f'"{year_col}" = {intent.time["year"]}'
            if year_filter not in " ".join(where_parts):
                where_parts.append(year_filter)
        
        # 补充 intent 中的其他筛选（但不能与 required_filters 冲突）
        for f in intent.filters:
            if "_agg_column" in f:
                continue
            col = f.get("column", f.get("dimension", ""))
            val = f.get("value", "")
            op = f.get("operator", "=")
            if col and val and (col in available_columns or not available_columns):
                if isinstance(val, str):
                    filter_expr = f'"{col}" {op} \'{val}\''
                else:
                    filter_expr = f'"{col}" {op} {val}'
                # 避免重复
                if filter_expr not in " ".join(where_parts):
                    where_parts.append(filter_expr)
        
        sql = f'SELECT\n    {", ".join(select_parts)}\nFROM "{intent.table_name}"'
        if where_parts:
            sql += f"\nWHERE {' AND '.join(where_parts)}"
        group_cols = [p for p in select_parts if "AS" not in p and "(" not in p]
        if group_cols:
            sql += f"\nGROUP BY {', '.join(group_cols)}"
        return sql + ";"

    def _build_sql_hard_rules(self, has_required_where: bool = False, has_task_description: bool = False) -> str:
        """SQL 生成的技术约束，根据上下文动态调整规则。"""
        rules = [
            '**列名必须从"表结构"中选择，不能发明不存在的列**',
            '使用双引号包裹列名：SELECT "列名1", "列名2"',
            '字符串值使用单引号：WHERE "列名" = \'值\'',
            '**只生成 1 个 SQL，必须与当前任务类型匹配**',
            '**若上文指定了指标聚合表达式，SELECT 中必须使用该表达式**',
        ]
        if has_required_where:
            rules.append(
                '**WHERE 条件已确定**：上文"建议 WHERE 条件"是数据约束的完整定义，直接使用，可追加 AND，不可删改'
            )
            rules.append(
                '**禁止推断额外筛选**：任务描述只描述分析动作，不包含筛选逻辑。不要根据描述中的词汇添加额外 WHERE 条件'
            )
        if has_task_description:
            rules.append(
                '**任务描述解读**：description 中的词汇（如"市场费""今年"）是上下文说明，其对应的筛选条件已在"建议 WHERE 条件"中，不要重复添加'
            )
            rules.append(
                '**禁止越权**：SQL 只实现"任务描述"中明确要求的动作。不要自作主张做额外分析（找极值、下钻、排名等）'
            )
        rules.append("只输出 JSON，不要其他文字")
        return "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))

    def _collect_relevant_columns(
        self,
        intent: Any,
        yml_config: dict[str, Any],
        state: Any,
    ) -> set[str]:
        """收集 intent 涉及的所有真实列名，用于筛选 table schema。"""
        cols: set[str] = set()
        dims_config = yml_config.get("dimensions", {})
        metrics_config = yml_config.get("metrics", {})

        # 维度 → 列名
        for did in (intent.dimensions or []):
            if did in dims_config:
                cols.add(dims_config[did].get("column", did))

        # 指标 agg 中引用的列
        for mid in (intent.metrics or []):
            m = metrics_config.get(mid, {})
            agg = m.get("agg", "")
            for match in re.findall(r'"([^"]+)"', agg):
                cols.add(match)

        # conditions 中引用的列
        for c in (intent.conditions or []):
            col_id = c.get("column", "")
            if col_id in dims_config:
                cols.add(dims_config[col_id].get("column", col_id))
            elif col_id:
                cols.add(col_id)

        # required_filters 中引用的列
        if state and getattr(state, "required_filters", None):
            for f in state.required_filters:
                for match in re.findall(r'"([^"]+)"', f.get("expr", "")):
                    cols.add(match)

        # current_metric_def 中的列
        if state and getattr(state, "current_metric_def", None):
            agg = state.current_metric_def.get("agg", "")
            for match in re.findall(r'"([^"]+)"', agg):
                cols.add(match)

        return cols

    def _build_generation_prompt(
        self,
        intent: Any,
        yml_config: dict[str, Any],
        schema_text: str | None,
        table_schema: dict[str, Any] | None,
        current_task: dict[str, Any] | None = None,
        state: Any = None,
    ) -> str:
        # ── 1. 列信息：只保留 intent 相关列 ──
        relevant_cols = self._collect_relevant_columns(intent, yml_config, state)
        if table_schema and table_schema.get("columns"):
            all_cols = table_schema["columns"]
            profiles = table_schema.get("column_profiles", [])
            if relevant_cols:
                key_cols = [c for c in all_cols if c.get("name", c.get("column_name", "")) in relevant_cols]
                other_cols = [c for c in all_cols if c.get("name", c.get("column_name", "")) not in relevant_cols]
                columns_info = "### 关键列（与本次查询直接相关）\n"
                columns_info += self._format_columns_for_prompt(key_cols, column_profiles=profiles)
                columns_info += f"\n\n### 其他可用列（共 {len(other_cols)} 列）\n"
                columns_info += ", ".join(
                    f'"{c.get("name", c.get("column_name", ""))}"'
                    for c in other_cols
                )
            else:
                columns_info = self._format_columns_for_prompt(all_cols, column_profiles=profiles)
        elif schema_text:
            columns_info = schema_text
        else:
            columns_info = "无列信息"

        # ── 2. 指标约束（强制使用聚合表达式）──
        metric_section = ""
        if state and getattr(state, "current_metric_def", None):
            md = state.current_metric_def
            metric_name = getattr(state, "current_metric", "")
            agg_expr = md.get("agg", "")
            filter_refs = md.get("filter_refs", [])
            metric_section = f"""## 指标约束
- 指标ID: {metric_name}
- 含义: {md.get('label', '')}
- 聚合表达式（必须使用）: {agg_expr}
- 默认筛选器: {filter_refs}"""

        # ── 3. WHERE 条件（已解析的 SQL 片段）──
        has_required_where = bool(state and getattr(state, "required_filters", None))
        where_section = ""
        if has_required_where:
            where_parts = []
            for f in state.required_filters:
                where_parts.append(f"  -- {f['id']}: {f['label']}\n  ({f['expr']})")
            where_section = f"## 建议 WHERE 条件\n以下条件直接用于 WHERE 子句（可追加 AND，不可删改）：\n\n{chr(10).join(where_parts)}"

        # ── 4. 结构化意图摘要 ──
        # ★ task_type 优先使用当前任务的类型（而非全局 intent 的主类型）
        effective_task_type = (
            current_task.get("task_type", intent.task_type)
            if current_task
            else intent.task_type
        )
        intent_section = f"""## 结构化意图
- task_type: {effective_task_type}
- metrics: {intent.metrics}
- dimensions: {intent.dimensions}"""

        # ── 5. 任务指令 ──
        user_query = getattr(state, "user_query", "") if state else ""
        task_instruction = self._build_task_instruction(current_task, user_query)

        # ── 6. 用户查询（多任务时精简，避免泄露完整意图导致越权）──
        rewritten = getattr(intent, "rewritten_query", "") or ""
        if current_task and current_task.get("description"):
            # 多步计划：只展示当前任务描述作为查询上下文
            query_section = f"## 用户查询（当前任务视角）\n{current_task['description']}"
        elif rewritten and rewritten != intent.raw_query:
            query_section = f"## 用户查询\n{rewritten}\n（原始: {intent.raw_query}）"
        else:
            query_section = f"## 用户查询\n{intent.raw_query}"

        # ── 7. 检索增强：值匹配提示 ──
        value_hint_section = ""
        rc = getattr(state, "retrieval_context", None) if state else None
        if rc is not None and hasattr(rc, "format_value_hint"):
            value_hint_section = rc.format_value_hint()

        return f"""请根据以下信息生成可执行的 SQL。
{task_instruction}
{query_section}

## 表名
{intent.table_name}

## 列信息（只能使用这些列！）
{columns_info}

{intent_section}

{metric_section}

{where_section}

{value_hint_section}

## DuckDB SQL 规范
{get_duckdb_syntax_rules()}

### 硬约束
{self._build_sql_hard_rules(has_required_where=has_required_where, has_task_description=bool(current_task))}

## 输出要求
严格输出 JSON，**只生成 1 个最匹配当前任务的 SQL**：
{{
  "candidates": [
    {{ "sql": "SELECT ... FROM ... WHERE ... GROUP BY ...", "reason": "一句话解释业务逻辑和口径选择" }}
  ]
}}
"""

    def _build_task_instruction(self, current_task: dict[str, Any] | None, user_query: str) -> str:
        """构建任务类型指令（完整版，保留各类型的详细规则）。"""
        if not current_task:
            return ""

        task_type = current_task.get("task_type", current_task.get("type", ""))
        task_id = current_task.get("task_id", current_task.get("id", ""))
        task_desc = current_task.get("description", "")
        task_notes = current_task.get("notes", [])
        current_dim = current_task.get("current_dimension", "")
        available_dims = current_task.get("available_dimensions", [])
        time_granularity = current_task.get("time_granularity", "")
        intent_hint = current_task.get("intent_hint", "")
        parent_summary = current_task.get("parent_results_summary", "")
        depends_on = current_task.get("depends_on", [])
        retry_hint = current_task.get("retry_hint", "")
        retry_count = current_task.get("retry_count", 0)

        inst = f"""
## 当前分析任务
- 任务 ID: {task_id}
- 任务类型: {task_type}
- 任务描述: {task_desc}

### ★ 任务边界（必须遵守）
**只完成上述"任务描述"中的动作，禁止越权：**
- 不要做任务描述之外的额外分析（如找极值、下钻明细、对比等）
- 不要用 CTE 串联多个不同目的的查询
- 后续分析由其他任务负责，不需要你在这一步完成
"""
        if task_notes:
            inst += f"- 注意事项: {'; '.join(str(n) for n in task_notes)}\n"
        if current_dim:
            inst += f"- 当前分析维度: {current_dim}\n"
        if time_granularity:
            inst += f"- 时间粒度: {time_granularity}\n"
        if depends_on:
            inst += f"- 依赖任务: {', '.join(depends_on)}\n"
        if parent_summary:
            # ★ 分离 Planner 分析结论（高优先级）和原始上游数据（参考）
            planner_conclusion_lines = []
            upstream_data_lines = []
            for line in parent_summary.split("\n"):
                if line.startswith("[Planner 分析结论]"):
                    planner_conclusion_lines.append(line.replace("[Planner 分析结论] ", ""))
                else:
                    upstream_data_lines.append(line)
            
            if planner_conclusion_lines:
                inst += f"""
### ★ Planner 承上启下分析（务必参考）
{chr(10).join(planner_conclusion_lines)}
"""
            if upstream_data_lines:
                upstream_text = "\n".join(upstream_data_lines).strip()
                if upstream_text:
                    inst += f"- 上游结果摘要: {upstream_text}\n"

        if retry_hint:
            inst += f"""
### SQL 修复建议（第 {retry_count} 次重试）
上一次 SQL 执行失败，请根据以下建议修复：
{retry_hint}
"""
        if intent_hint:
            inst += f"""
### 执行意图
{intent_hint}
"""

        # ── 各任务类型的参考规则（指导性，根据具体任务描述灵活调整） ──
        if task_type == "trend":
            inst += """
### 趋势分析参考
- 通常按时间维度（年/季/月/周/日）GROUP BY
- 通常按时间升序排列（ORDER BY 时间列 ASC），但具体排序方向以任务描述为准
- 只生成 1 个 SQL
"""
        elif task_type == "source":
            dims_hint = "、".join(f"「{d}」" for d in available_dims) if available_dims else "任务描述中指定的维度"
            inst += f"""
### 来源/构成分析参考
- 可用的分组维度: {dims_hint}
- 根据任务描述选择最合适的维度进行 GROUP BY
- 通常按数值降序排序（看 top 贡献），但具体排序以任务描述为准
- 只生成 1 个 SQL
"""
        elif task_type == "comparison":
            inst += """
### 对比分析参考
- 对比两个时间段或两个条件的差异
- 计算差值和/或增长率：(当期 - 基期) / NULLIF(基期, 0) * 100
- 结果应包含：基期值、当期值、变化值/增长率
- 可用 LAG/LEAD 窗口函数或子查询实现
- ★ 排序方向必须严格匹配任务描述的语义：
  - "下降最多/跌幅最大" → ORDER BY 变化值 ASC（取最负的值）
  - "增长最多/涨幅最大" → ORDER BY 变化值 DESC（取最正的值）
  - 禁止使用 ABS() 排序，因为 ABS 会混淆增长和下降的方向
"""
        elif task_type == "drilldown":
            inst += """
### 下钻分析参考
- 在上一步结果基础上进一步细分
- 增加更细粒度的维度或筛选条件
- 保留上游的筛选条件
"""
        elif task_type == "ratio":
            inst += f"""
### 占比分析规则

**用户问题**：{user_query}

请根据用户问题判断使用哪种占比模式：

**模式1 - 分别占比**（用户说"分别""各自""每个"时使用）：按维度 GROUP BY，每行一个占比
```sql
SELECT "维度列",
  SUM("数值列") AS 值,
  ROUND(SUM("数值列") * 100.0 / (SELECT SUM("数值列") FROM 表名 WHERE <全局条件>), 2) AS 占比
FROM 表名
WHERE <全局条件> AND <分组条件>
GROUP BY "维度列"
ORDER BY 值 DESC
```

**模式2 - 合计占比**（用户问一组对象合起来占多少时使用）：CASE WHEN 输出单行
```sql
SELECT
  ROUND(SUM(CASE WHEN <分子条件> THEN "数值列" ELSE 0 END) * 100.0 / SUM("数值列"), 2) AS 占比
FROM 表名 WHERE <全局条件>
```

**模式3 - TOP N 占比**（分子条件涉及聚合排序时）：CTE 取 TOP N 再 CASE WHEN IN
```sql
WITH top_n AS (
  SELECT "维度列" FROM 表名 WHERE <全局条件>
  GROUP BY "维度列" ORDER BY SUM("数值列") DESC LIMIT N
)
SELECT
  SUM(CASE WHEN "维度列" IN (SELECT "维度列" FROM top_n) THEN "数值列" ELSE 0 END) AS 分子,
  SUM("数值列") AS 分母,
  ROUND(... * 100.0 / ..., 2) AS 占比
FROM 表名 WHERE <全局条件>
```

**关键判断**：
- 用户说"**分别**占多少"→ 模式1（GROUP BY）
- 用户说"**一共**占多少"→ 模式2（CASE WHEN）
- 涉及"前N名""TOP N"→ 模式3（CTE + CASE WHEN IN）
"""
        elif task_type == "ranking":
            inst += """
### 排名分析参考
- 按指标 GROUP BY 维度后排序
- 使用 ORDER BY 指标 DESC/ASC（根据任务描述决定方向）
- 使用 LIMIT N 限制返回数量
"""
        return inst

    async def _generate_candidates(
        self,
        intent: Any,
        yml_config: dict[str, Any],
        schema_text: str | None = None,
        table_schema: dict[str, Any] | None = None,
        current_task: dict[str, Any] | None = None,
        state: Any = None,  # ReActState
    ) -> list[SQLCandidate]:
        prompt = self._build_generation_prompt(intent, yml_config, schema_text, table_schema, current_task, state)
        system = "你是 SQL 生成专家。根据用户查询和表结构生成 DuckDB SQL。核心原则：以用户查询为准，业务配置仅供参考；只使用提供的列名；列名用双引号，字符串值用单引号。严格输出 JSON。"
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt=system,
                caller_name="generate_sql",
            )
            candidates = self._parse_candidates(response)
            if candidates:
                self._log.observe(f"生成 {len(candidates)} 个候选 SQL")
                return candidates
        except Exception as e:
            self._log.error(f"LLM 生成失败: {e}")
        return []

    async def generate_sql(
        self,
        intent: "StructuredIntent | dict",
        schema_text: str = "",
        yml_config: dict[str, Any] | None = None,
        available_tables: list[dict] | None = None,
        current_task: dict[str, Any] | None = None,
        state: Any = None,  # ReActState
        **kwargs: Any,
    ) -> ToolResult:
        """根据结构化意图生成 SQL。"""
        if not self.llm:
            return ToolResult.fail("未配置 LLM，无法生成 SQL")
        from chatdb.agents.semantic_parser import StructuredIntent
        intent_obj = StructuredIntent.from_dict(intent) if isinstance(intent, dict) else intent
        table_schema = self._get_table_schema(intent_obj.table_name, available_tables or [])
        candidates = await self._generate_candidates(
            intent_obj,
            yml_config or {},
            schema_text or None,
            table_schema,
            current_task,
            state,  # 传入 state
        )
        if not candidates:
            return ToolResult.fail("无法生成有效 SQL")
        c = candidates[0]
        return ToolResult.ok(
            data={
                "sql": c.sql,
                "reason": c.reason,
                "candidates": [
                    {"sql": x.sql, "reason": x.reason, "confidence": x.confidence}
                    for x in candidates
                ],
            },
            message="SQL 生成成功",
        )

    # ---------- 执行与评估（原 ResultEvaluator 逻辑） ----------

    def _classify_error(self, error_msg: str) -> ErrorType:
        for error_type, patterns in ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    return error_type
        return ErrorType.OTHER

    def _extract_error_context(
        self,
        error_msg: str,
        error_type: ErrorType,
        state: ReActState | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if error_type == ErrorType.UNKNOWN_COLUMN:
            match = re.search(r'column\s+"?([^"]+)"?\s+not found', error_msg, re.IGNORECASE)
            if match:
                context["wrong_column"] = match.group(1)
                if state and state.available_columns:
                    alternatives = self._find_alternatives(match.group(1), state.available_columns)
                    context["alternatives"] = alternatives
                    context["has_alternative"] = bool(alternatives)
        elif error_type == ErrorType.TYPE_MISMATCH:
            match = re.search(r"cannot compare\s+(\w+)\s+and\s+(\w+)", error_msg, re.IGNORECASE)
            if match:
                context["type1"], context["type2"] = match.group(1), match.group(2)
        return context

    def _find_alternatives(self, wrong_col: str, columns: list[dict[str, Any]]) -> list[str]:
        alternatives = []
        wrong_lower = wrong_col.lower()
        for col in columns:
            col_name = col.get("name", col.get("column_name", ""))
            if wrong_lower in col_name.lower() or col_name.lower() in wrong_lower:
                alternatives.append(col_name)
        return list(set(alternatives))[:5]

    async def _execute_sql_internal(self, eval_result: EvaluationResult) -> EvaluationResult:
        if not self.db_connector:
            eval_result.execution_success = False
            eval_result.execution_error = "未配置数据库连接"
            return eval_result
        try:
            rows = await self.db_connector.execute_query(eval_result.sql)
            eval_result.rows = rows
            eval_result.row_count = len(rows)
            eval_result.execution_success = True
            eval_result.execution_error = None
            eval_result.error_type = ErrorType.NONE
            self._log.observe(f"执行成功: {len(rows)} 行")
        except Exception as e:
            eval_result.execution_success = False
            eval_result.execution_error = str(e)
            eval_result.error_type = self._classify_error(str(e))
            self._log.warn(f"执行失败: {e}")
        return eval_result

    def _build_diagnose_prompt(
        self,
        eval_result: EvaluationResult,
        intent: Any,
        schema_text: str | None,
    ) -> str:
        prompt = f"""请诊断以下 SQL 的问题并给出最小修正。

## 当前 SQL
{eval_result.sql}

## 错误类型
{eval_result.error_type.value}

## 数据库错误
{eval_result.execution_error}
"""
        if eval_result.error_context:
            if eval_result.error_context.get("wrong_column"):
                prompt += f"\n## 出错元素\n{eval_result.error_context['wrong_column']}\n"
            if eval_result.error_context.get("alternatives"):
                prompt += f"\n## 可用替代\n{', '.join(eval_result.error_context['alternatives'])}\n"
        if intent:
            prompt += f"\n## 用户原始查询\n{intent.raw_query}\n"
        if schema_text:
            prompt += f"\n## 表 Schema\n{schema_text[:2000]}\n"
        prompt += """
## 输出要求
```json
{ "diagnosis": "一句话说明问题", "refined_sql": "修正后的完整 SQL" }
```
原则：只改出错部分，保持其他结构不变。只输出 JSON。"""
        return prompt

    async def _diagnose_and_refine(
        self,
        eval_result: EvaluationResult,
        intent: Any,
        yml_config: dict[str, Any],
        schema_text: str | None,
    ) -> EvaluationResult:
        prompt = self._build_diagnose_prompt(eval_result, intent, schema_text)
        system = "你是 SQL 调试专家。诊断错误原因，给出最小修正。只改出错部分，输出清晰 JSON。"
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt=system,
                caller_name="diagnose_and_refine",
            )
            result = parse_json(response)
            eval_result.diagnosis = result.get("diagnosis", "")
            refined_sql = result.get("refined_sql", "")
            if refined_sql and refined_sql != eval_result.sql:
                eval_result.refined = True
                eval_result.refined_sql = _clean_sql_util(refined_sql)
                eval_result.refinement_reason = eval_result.diagnosis
                self._log.reflect(f"诊断: {eval_result.diagnosis[:100]}")
            else:
                eval_result.refined = False
        except Exception as e:
            self._log.error(f"诊断失败: {e}")
        return eval_result

    async def _generate_summary_internal(
        self,
        eval_result: EvaluationResult,
        user_query: str,
        intent: Any,
    ) -> EvaluationResult:
        if eval_result.row_count == 0:
            eval_result.summary = "查询未返回结果。"
            return eval_result
        sample = eval_result.rows[:10]
        prompt = f"""请根据查询结果回答用户问题。

用户问题: {user_query}

查询结果（共 {eval_result.row_count} 行，前 {len(sample)} 行）:
{format_rows(sample)}

请用简洁的语言总结，突出关键数据。"""
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是数据分析专家，请简洁回答。",
                caller_name="generate_summary",
            )
            eval_result.summary = response.strip()
        except Exception as e:
            self._log.error(f"总结失败: {e}")
            eval_result.summary = f"查询返回 {eval_result.row_count} 行结果。"
        return eval_result

    async def execute_and_evaluate(
        self,
        sql: str,
        schema_text: str = "",
        intent: Any = None,
        yml_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行 SQL 并诊断/修正。"""
        if not self.llm or not self.db_connector:
            return ToolResult.fail("未配置 LLM 或数据库连接，无法执行与评估")
        eval_result = EvaluationResult(sql=sql)
        eval_result = await self._execute_sql_internal(eval_result)
        attempts = 0
        while not eval_result.execution_success and attempts < self.MAX_REFINE_ATTEMPTS:
            self._log.info(f"尝试修正 (第 {attempts + 1} 次)")
            eval_result = await self._diagnose_and_refine(
                eval_result, intent, yml_config or {}, schema_text or None,
            )
            if eval_result.refined:
                eval_result.sql = eval_result.refined_sql
                eval_result = await self._execute_sql_internal(eval_result)
            attempts += 1
        if eval_result.execution_success:
            eval_result = await self._generate_summary_internal(
                eval_result,
                intent.raw_query if intent and hasattr(intent, "raw_query") else "",
                intent,
            )
        return ToolResult.ok(
            data={
                "sql": eval_result.sql,
                "rows": eval_result.rows,
                "row_count": eval_result.row_count,
                "execution_success": eval_result.execution_success,
                "execution_error": eval_result.execution_error,
                "diagnosis": eval_result.diagnosis,
                "error_type": eval_result.error_type.value,
                "refined": eval_result.refined,
                "summary": eval_result.summary,
            },
            message="评估完成" if eval_result.execution_success else "执行失败",
        )

    # ---------- ReAct 流程：critique / refine / diagnose_no_data ----------

    async def _build_no_data_guidance(self, user_query: str, diagnosis: dict[str, Any]) -> str:
        prompt = f"""用户问题: {user_query}

空结果诊断:
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

请用中文给出一段友好说明：1. 一句话解释为什么查不到数据 2. 说明是数据不存在还是条件写错 3. 1～2 条下一步建议。不超过 200 字。"""
        try:
            text = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是数据分析顾问，向业务用户解释为什么查不到数据。",
                caller_name="no_data_guidance",
            )
            return text.strip()
        except Exception as e:
            self._log.error(f"生成 no_data 用户指导失败: {e}")
            return "当前条件下查不到任何数据，可能是数据尚未入库或筛选条件过于严格。"

    def _format_result_for_observe(self, rows: list[dict]) -> str:
        """
        格式化查询结果用于 OBSERVE 展示
        
        示例输出：
        total_gross: 550983403643.47
        或
        年份 | 流水
        2023 | 100亿
        2024 | 120亿
        """
        if not rows:
            return "无结果"
        
        # 单行结果：直接展示 key: value
        if len(rows) == 1:
            row = rows[0]
            parts = []
            for k, v in row.items():
                if isinstance(v, float):
                    # 格式化数字，保留2位小数
                    parts.append(f"{k}: {v:,.2f}")
                elif isinstance(v, int):
                    parts.append(f"{k}: {v:,}")
                else:
                    parts.append(f"{k}: {v}")
            return " | ".join(parts)
        
        # 多行结果：表格形式，最多展示前5行
        max_rows = 5
        display_rows = rows[:max_rows]
        
        # 获取列名
        columns = list(rows[0].keys())
        
        # 构建表格
        lines = []
        lines.append(" | ".join(columns))
        
        for row in display_rows:
            values = []
            for col in columns:
                v = row.get(col, "")
                if isinstance(v, float):
                    values.append(f"{v:,.2f}")
                elif isinstance(v, int):
                    values.append(f"{v:,}")
                else:
                    values.append(str(v) if v is not None else "")
            lines.append(" | ".join(values))
        
        if len(rows) > max_rows:
            lines.append(f"... 共 {len(rows)} 行")
        
        return "\n".join(lines)

    async def _assess_answer_sufficiency(self, state: ReActState) -> None:
        rows = (state.execute_result or {}).get("rows", [])
        if not rows:
            return
        sample = rows[:5] if len(rows) > 5 else rows
        try:
            prompt = f"""用户问题：{state.user_query}

当前查询结果行数：{len(rows)}。前几行示例：{json.dumps(sample, ensure_ascii=False)}

请判断：仅凭当前结果是否足以完整、准确地回答用户问题？只输出一个词：sufficient 或 insufficient。"""
            resp = await self.llm.chat(
                prompt=prompt,
                system_prompt="根据问题语义与结果内容判断是否足以回答问题，只输出 sufficient 或 insufficient。",
                caller_name="assess_answer_sufficiency",
            )
            if "insufficient" in resp.strip().lower():
                state.need_more_analysis = True
                state.reflect("LLM评估：当前结果不足以完整回答问题，需进一步分析")
        except Exception as e:
            self._log.warn(f"评估结果充分性失败: {e}")

    def _pick_dimension_and_metric(self, state: ReActState) -> tuple[str | None, str | None]:
        cols = state.available_columns or []
        if not cols:
            return None, None
        dim_col = None
        metric_col = None
        for c in cols:
            name = c.get("name") or c.get("column_name") or ""
            typ = (c.get("type") or c.get("column_type") or "").upper()
            if not dim_col and ("VARCHAR" in typ or "TEXT" in typ or "STRING" in typ or "CHAR" in typ):
                dim_col = name
            if not metric_col and ("DOUBLE" in typ or "DECIMAL" in typ or "INT" in typ or "NUMERIC" in typ or "FLOAT" in typ or "BIGINT" in typ):
                metric_col = name
            if dim_col and metric_col:
                break
        if not dim_col and cols:
            dim_col = cols[0].get("name") or cols[0].get("column_name")
        return dim_col, metric_col

    async def _generate_probe_queries(
        self, sql: str, table_name: str, schema_text: str | None,
    ) -> list[dict[str, str]]:
        prompt = f"""分析以下 SQL 的 WHERE 条件，生成数据探测查询来验证各条件是否有数据。

## 原始 SQL
{sql}

## 表名
{table_name}

## 表结构
{schema_text[:1500] if schema_text else "未提供"}

## 任务
1. 提取 SQL 中的每个筛选条件
2. 为每个关键条件生成一个探测查询
3. 输出 JSON: {{ "probe_queries": [ {{ "purpose": "...", "field": "...", "query": "SELECT ..." }} ] }}"""
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是 SQL 分析专家，擅长诊断数据问题。",
                caller_name="generate_probe_queries",
            )
            result = parse_json(response)
            return result.get("probe_queries", [])
        except Exception as e:
            self._log.error(f"生成探测查询失败: {e}")
            return []

    async def _execute_probe_queries(
        self, probe_queries: list[dict[str, str]], table_name: str,
    ) -> list[dict[str, Any]]:
        results = []
        for probe in probe_queries:
            query = probe.get("query", "")
            if not query:
                continue
            try:
                rows = await self.db_connector.execute_query(query)
                results.append({
                    "purpose": probe.get("purpose", ""),
                    "field": probe.get("field", ""),
                    "query": query,
                    "success": True,
                    "rows": rows[:20],
                    "row_count": len(rows),
                })
            except Exception as e:
                results.append({
                    "purpose": probe.get("purpose", ""),
                    "field": probe.get("field", ""),
                    "query": query,
                    "success": False,
                    "error": str(e),
                })
        return results

    async def _analyze_probe_results(
        self, original_sql: str, probe_results: list[dict[str, Any]], user_query: str, intent: Any,
    ) -> dict[str, Any]:
        probe_summary = []
        for r in probe_results:
            if r.get("success"):
                values = [list(row.values())[0] if row else None for row in r.get("rows", [])]
                probe_summary.append(f"- {r['purpose']}: 找到 {r['row_count']} 个值，示例: {values[:5]}")
            else:
                probe_summary.append(f"- {r['purpose']}: 查询失败 ({r.get('error', '')})")
        prompt = f"""根据数据探测结果，诊断为什么原始查询返回空结果。

## 用户问题
{user_query}

## 原始 SQL（返回空结果）
{original_sql}

## 数据探测结果
{chr(10).join(probe_summary)}

## 任务
1. 分析哪个条件导致了空结果
2. 判断是"数据确实不存在"还是"条件写错了"
3. 如果是条件错误，给出修正建议

## 输出格式 JSON
{{ "conclusion": "...", "root_cause": "no_data_exists|wrong_condition|too_strict|unknown", "details": [], "can_fix": true/false, "fix_reason": "...", "suggested_sql": "..." }}"""
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是数据诊断专家，请客观分析，不要猜测。",
                caller_name="analyze_probe_results",
            )
            return parse_json(response)
        except Exception as e:
            self._log.error(f"分析探测结果失败: {e}")
            return {"conclusion": f"分析失败: {e}", "root_cause": "unknown", "can_fix": False}

    async def _critique(self, state: ReActState) -> None:
        state.phase = ReActPhase.CRITIQUE
        sql = state.current_sql or state.final_sql
        if not sql:
            state.set_error("缺少 SQL", ErrorType.OTHER)
            return
        if not self.db_connector:
            state.set_error("未配置数据库连接", ErrorType.OTHER)
            return
        try:
            rows = await self.db_connector.execute_query(sql)
            state.execute_result = {"rows": rows, "row_count": len(rows)}
            state.execution_error = None
            state.clear_error()
            if len(rows) == 0:
                state.set_error("查询返回空结果", ErrorType.NO_DATA)
                state.mark_need(need_critique=True)
            else:
                state.clear_all_needs()
                await self._assess_answer_sufficiency(state)
        except Exception as e:
            state.execution_error = str(e)
            error_type = self._classify_error(str(e))
            context = self._extract_error_context(str(e), error_type, state)
            state.set_error(str(e), error_type, context)
            state.refine_attempts += 1

    async def _refine_sql(self, state: ReActState) -> None:
        state.phase = ReActPhase.REFINE
        eval_result = EvaluationResult(
            sql=state.current_sql,
            execution_error=state.execution_error or state.error,
            error_type=state.error_type,
            error_context=state.error_context,
        )
        eval_result = await self._diagnose_and_refine(
            eval_result, state.intent, state.yml_config, state.schema_text,
        )
        if eval_result.refined and eval_result.refined_sql != state.current_sql:
            state.current_sql = eval_result.refined_sql
            state.reflect(f"SQL 已修正: {eval_result.diagnosis}")
            state.mark_need(need_execute=True)
        else:
            state.reflect("无法修正 SQL")

    async def _diagnose_no_data(self, state: ReActState) -> None:
        state.phase = ReActPhase.CRITIQUE
        sql = state.current_sql or state.final_sql
        if not sql or not state.table_name:
            state.reflect("缺少 SQL 或表名，无法诊断")
            return
        try:
            probe_queries = await self._generate_probe_queries(sql, state.table_name, state.schema_text)
            probe_results = await self._execute_probe_queries(probe_queries, state.table_name)
            diagnosis = await self._analyze_probe_results(
                sql, probe_results, state.user_query, state.intent,
            )
            state.error_context["no_data_diagnosis"] = diagnosis
            state.error_context["no_data_diagnosis_done"] = True
            state.error_context["root_cause"] = diagnosis.get("root_cause")
            state.error_context["can_fix"] = diagnosis.get("can_fix", False)
            state.error_context["suggested_sql"] = diagnosis.get("suggested_sql", "")
            state.error_context["fix_reason"] = diagnosis.get("fix_reason", "")
            state.error_context["user_guidance"] = await self._build_no_data_guidance(
                state.user_query, diagnosis,
            )
            state.reflect(f"空结果诊断: {diagnosis.get('conclusion', '未知原因')}")
            if diagnosis.get("suggested_sql") and diagnosis.get("can_fix"):
                state.current_sql = _clean_sql_util(diagnosis["suggested_sql"])
                state.mark_need(need_execute=True)
                state.reflect(f"建议修正: {diagnosis.get('fix_reason', '')}")
        except Exception as e:
            self._log.error(f"空结果诊断失败: {e}")
            state.reflect(f"诊断失败: {e}")

    async def run_generate(self, state: ReActState, context: Any) -> None:
        """运行 SQL 生成并写回 state。供 Orchestrator 调用。"""
        state.phase = ReActPhase.SQL_BUILD
        if not state.intent:
            state.set_error("缺少意图，无法生成 SQL", ErrorType.AMBIGUOUS_INTENT)
            return
        if not self.llm:
            state.set_error("未配置 LLM，无法生成 SQL", ErrorType.OTHER)
            return
        
        # 获取当前任务上下文（来自 state，由 Planner/SQLAgent 设置）
        current_task = state.current_task
        
        result = await self.generate_sql(
            intent=state.intent,
            schema_text=state.schema_text,
            yml_config=state.yml_config,
            available_tables=state.available_tables or None,
            current_task=current_task,  # 传递任务上下文
            state=state,  # ★ 传递 state，用于获取注入的指标定义
        )
        if result.success:
            state.current_sql = result.data.get("sql", "")
            state.sql_candidates = result.data.get("candidates", [])
            # ACT: 记录生成的 SQL
            state.act(state.current_sql, tool="SQL")
            state.mark_need(need_sql=False, need_execute=True)
            if hasattr(context, "generated_sql"):
                context.generated_sql = state.current_sql
        else:
            state.set_error(result.error or "SQL 生成失败", ErrorType.OTHER)

    async def run_execute_and_evaluate(self, state: ReActState, context: Any) -> None:
        """执行一步评估流程：先执行（若需要），再 critique/refine/diagnose_no_data。供 Orchestrator 调用。"""
        if state.need_execute and not state.execute_result:
            if not self.db_connector:
                state.set_error("未配置数据库连接", ErrorType.OTHER)
                return
            state.phase = ReActPhase.EXECUTE
            sql = state.current_sql
            if not sql:
                state.set_error("缺少 SQL", ErrorType.OTHER)
                return
            try:
                rows = await self.db_connector.execute_query(sql)
                state.execute_result = {"rows": rows, "row_count": len(rows)}
                state.final_sql = sql
                state.current_sql = sql
                state.clear_error()
                # OBSERVE: 展示实际查询结果
                state.observe(self._format_result_for_observe(rows))
                if len(rows) == 0:
                    state.set_error("查询返回空结果", ErrorType.NO_DATA)
                    state.mark_need(need_critique=True)
                else:
                    state.clear_all_needs()
                if hasattr(context, "generated_sql"):
                    context.generated_sql = state.final_sql
            except Exception as e:
                state.execution_error = str(e)
                state.observe(f"失败: {str(e)[:50]}")
                error_type = self._classify_error(str(e))
                ctx = self._extract_error_context(str(e), error_type, state)
                state.set_error(str(e), error_type, ctx)
                state.refine_attempts += 1
                state.mark_need(need_execute=False, need_critique=True)
            return
        if state.execution_error:
            await self._critique(state)
            if state.need_refine:
                await self._refine_sql(state)
            return
        if state.error_type == ErrorType.NO_DATA:
            if not state.error_context.get("no_data_diagnosis_done"):
                await self._diagnose_no_data(state)
            return
        if state.has_result:
            state.clear_all_needs()

    async def run_workflow(self, state: ReActState, context: Any) -> None:
        """
        完整流程：生成 SQL → 执行 → 评估 → (失败时 critique → refine → 重新执行)
        
        ReAct 重试循环：
        1. 生成 SQL
        2. 执行 SQL
        3. 执行失败 → critique → refine → 重新执行（最多 max_refine 次）
        4. 执行成功 → 结束
        """
        max_refine = 3
        
        await self.run_generate(state, context)
        if state.error:
            return
        
        for attempt in range(max_refine + 1):
            # 执行 SQL
            await self._execute_sql(state, context)
            
            # 执行成功，结束
            if state.has_result:
                return
            
            # 空结果，诊断后结束（不重试）
            if state.error_type == ErrorType.NO_DATA:
                if not state.error_context.get("no_data_diagnosis_done"):
                    await self._diagnose_no_data(state)
                    # 诊断后如果有修正 SQL，继续重试
                    if state.current_sql and state.need_execute:
                        state.execute_result = None
                        state.execution_error = None
                        continue
                return
            
            # SQL 执行报错，尝试 refine（LLM 诊断错误并生成修正 SQL）
            if state.execution_error and attempt < max_refine:
                self._log.info(f"SQL 执行失败，尝试修正 (第 {attempt + 1}/{max_refine} 次)")
                await self._refine_sql(state)
                if state.current_sql and state.need_execute:
                    # refine 产生了新 SQL，记录 ACT，清除旧执行结果，继续循环
                    state.act(state.current_sql, tool="SQL")
                    state.execute_result = None
                    state.execution_error = None
                    continue
                # 无法修正，结束
                self._log.warn("SQL 无法修正，停止重试")
                return
            
            # 其他错误或超过最大重试，结束
            return
    
    async def _execute_sql(self, state: ReActState, context: Any) -> None:
        """执行 SQL 并记录结果（不含 critique/refine 逻辑）"""
        if not self.db_connector:
            state.set_error("未配置数据库连接", ErrorType.OTHER)
            return
        state.phase = ReActPhase.EXECUTE
        sql = state.current_sql
        if not sql:
            state.set_error("缺少 SQL", ErrorType.OTHER)
            return
        try:
            rows = await self.db_connector.execute_query(sql)
            state.execute_result = {"rows": rows, "row_count": len(rows)}
            state.final_sql = sql
            state.current_sql = sql
            state.clear_error()
            # OBSERVE: 展示实际查询结果
            state.observe(self._format_result_for_observe(rows))
            if len(rows) == 0:
                state.set_error("查询返回空结果", ErrorType.NO_DATA)
            else:
                state.clear_all_needs()
            if hasattr(context, "generated_sql"):
                context.generated_sql = state.final_sql
        except Exception as e:
            # ★ 执行失败时清除 execute_result，确保 has_result 返回 False
            # 防止上一次成功的结果残留，导致 run_workflow 重试循环被跳过
            state.execute_result = None
            state.execution_error = str(e)
            state.observe(f"失败: {str(e)[:80]}")
            error_type = self._classify_error(str(e))
            ctx = self._extract_error_context(str(e), error_type, state)
            state.set_error(str(e), error_type, ctx)
            state.refine_attempts += 1


# ---------- 薄包装（供 Registry 注册，构造签名不变） ----------


class ValidateSQLTool(BaseTool):
    """验证 SQL。委托 SQLTool.validate_sql。"""

    def __init__(self):
        super().__init__(None)
        self._impl = SQLTool(None, None)
        self._log = get_component_logger("ValidateSQLTool")

    @property
    def name(self) -> str:
        return "validate_sql"

    @property
    def description(self) -> str:
        return """验证 SQL 语句的语法和业务逻辑。

使用场景：执行前安全检查、业务规则校验。
输入：sql, yml_config(可选)。输出：is_valid, errors, warnings。"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="sql", type="string", description="待验证的 SQL 语句", required=True),
            ToolParameter(name="yml_config", type="object", description="业务配置", required=False),
        ]

    async def execute(
        self,
        sql: str,
        yml_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        return self._impl.validate_sql(sql=sql, yml_config=yml_config)


class ExecuteSQLTool(BaseTool):
    """执行只读 SQL。委托 SQLTool.execute_sql。"""

    def __init__(self, db_connector: BaseDatabaseConnector):
        super().__init__(None)
        self._impl = SQLTool(None, db_connector)
        self._log = get_component_logger("ExecuteSQLTool")

    @property
    def name(self) -> str:
        return "execute_sql"

    @property
    def description(self) -> str:
        return """执行 SQL 查询并返回结果。仅支持 SELECT。
输入：sql, limit(可选)。输出：rows, row_count, columns。"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="sql", type="string", description="要执行的 SQL 语句", required=True),
            ToolParameter(name="limit", type="number", description="返回行数限制", required=False, default=100),
        ]

    async def execute(self, sql: str, limit: int = 100, **kwargs: Any) -> ToolResult:
        return await self._impl.execute_sql(sql=sql, limit=limit)


class GenerateSQLTool(BaseTool):
    """生成 SQL。委托 SQLTool.generate_sql。"""

    def __init__(self, llm: BaseLLM):
        super().__init__(None)
        self._impl = SQLTool(llm, None)
        self._log = get_component_logger("GenerateSQLTool")

    @property
    def name(self) -> str:
        return "generate_sql"

    @property
    def description(self) -> str:
        return """根据结构化意图生成可执行的 SQL。输出：主 SQL、多候选、业务解释。"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="intent", type="object", description="结构化意图", required=True),
            ToolParameter(name="schema_text", type="string", description="表结构", required=False),
            ToolParameter(name="yml_config", type="object", description="YAML 配置", required=False),
        ]

    async def execute(
        self,
        intent: "StructuredIntent | dict",
        schema_text: str = "",
        yml_config: dict[str, Any] | None = None,
        available_tables: list[dict] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        self._log.info("生成 SQL...")
        try:
            return await self._impl.generate_sql(
                intent=intent,
                schema_text=schema_text,
                yml_config=yml_config,
                available_tables=available_tables or [],
                **kwargs,
            )
        except Exception as e:
            self._log.error(f"生成失败: {e}")
            return ToolResult.fail(str(e))

    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        from chatdb.core.react_state import ReActPhase, ErrorType
        state.phase = ReActPhase.SQL_BUILD
        if not state.intent:
            state.set_error("缺少意图，无法生成 SQL", ErrorType.AMBIGUOUS_INTENT)
            return
        result = await self.execute(
            intent=state.intent,
            schema_text=state.schema_text,
            yml_config=state.yml_config,
            available_tables=state.available_tables or None,
        )
        if result.success:
            state.current_sql = result.data.get("sql", "")
            state.sql_candidates = result.data.get("candidates", [])
            state.mark_need(need_sql=False, need_execute=True)
            if hasattr(context, "generated_sql"):
                context.generated_sql = state.current_sql
        else:
            state.set_error(result.error or "SQL 生成失败", ErrorType.OTHER)


class ExecuteAndEvaluateTool(BaseTool):
    """执行 SQL 并诊断/修正。委托 SQLTool.execute_and_evaluate。"""

    def __init__(self, llm: BaseLLM, db_connector: BaseDatabaseConnector):
        super().__init__(None)
        self._impl = SQLTool(llm, db_connector)
        self._log = get_component_logger("ExecuteAndEvaluateTool")

    @property
    def name(self) -> str:
        return "execute_and_evaluate"

    @property
    def description(self) -> str:
        return """执行 SQL、诊断错误、做最小修正。输出：执行结果、错误诊断、修正建议。"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="sql", type="string", description="要执行的 SQL", required=True),
            ToolParameter(name="schema_text", type="string", description="表结构", required=False),
        ]

    async def execute(
        self,
        sql: str,
        schema_text: str = "",
        intent: Any = None,
        yml_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        self._log.info(f"执行: {sql[:50]}...")
        try:
            return await self._impl.execute_and_evaluate(
                sql=sql,
                schema_text=schema_text,
                intent=intent,
                yml_config=yml_config or {},
                **kwargs,
            )
        except Exception as e:
            self._log.error(f"执行失败: {e}")
            return ToolResult.fail(str(e))

    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        from chatdb.core.react_state import ReActPhase, ErrorType
        state.phase = ReActPhase.EXECUTE
        sql = state.current_sql
        if not sql:
            state.set_error("缺少 SQL", ErrorType.OTHER)
            return
        result = await self.execute(
            sql=sql,
            schema_text=state.schema_text,
            intent=state.intent,
            yml_config=state.yml_config,
        )
        if result.data.get("execution_success"):
            rows = result.data.get("rows", [])
            state.execute_result = {"rows": rows, "row_count": len(rows)}
            state.final_sql = result.data.get("sql", sql)
            state.current_sql = state.final_sql
            state.clear_error()
            if len(rows) == 0:
                state.set_error("查询返回空结果", ErrorType.NO_DATA)
                state.mark_need(need_critique=True)
            else:
                state.clear_all_needs()
            if hasattr(context, "generated_sql"):
                context.generated_sql = state.final_sql
        else:
            error = result.data.get("execution_error", "执行失败")
            error_type = error_type_from_str(result.data.get("error_type", "other"))
            state.execution_error = error
            state.set_error(error, error_type)
            state.mark_need(need_execute=False, need_critique=True)
            if result.data.get("refined") and result.data.get("sql") != sql:
                state.current_sql = result.data.get("sql")
                state.mark_need(need_execute=True)


# ---------- 完整流程工具 ----------


class SQLWorkflowTool(BaseTool):
    """完整流程：生成 → 验证 → 执行与评估。内部使用一个 SQLTool。"""

    def __init__(self, llm: BaseLLM, db_connector: BaseDatabaseConnector):
        super().__init__(None)
        self._impl = SQLTool(llm, db_connector)
        self._log = get_component_logger("SQLWorkflowTool")

    @property
    def name(self) -> str:
        return "sql_workflow"

    @property
    def description(self) -> str:
        return """SQL 完整流程：根据意图生成 SQL、验证、执行并评估。"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="intent", type="object", description="结构化意图", required=True),
            ToolParameter(name="schema_text", type="string", description="表结构", required=False),
            ToolParameter(name="yml_config", type="object", description="YAML 配置", required=False),
        ]

    async def execute(
        self,
        intent: "StructuredIntent | dict",
        schema_text: str = "",
        yml_config: dict[str, Any] | None = None,
        available_tables: list[dict] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        self._log.info("SQL Workflow: 生成 → 验证 → 执行")
        gen_result = await self._impl.generate_sql(
            intent=intent,
            schema_text=schema_text,
            yml_config=yml_config,
            available_tables=available_tables or [],
            **kwargs,
        )
        if not gen_result.success:
            return gen_result
        sql = gen_result.data.get("sql", "")
        if not sql:
            return ToolResult.fail("SQL 生成结果为空")
        val_result = self._impl.validate_sql(sql=sql, yml_config=yml_config)
        exec_result = await self._impl.execute_and_evaluate(
            sql=sql,
            schema_text=schema_text,
            intent=intent if hasattr(intent, "raw_query") else None,
            yml_config=yml_config or {},
            **kwargs,
        )
        data = {
            "sql": exec_result.data.get("sql", sql),
            "validation": val_result.data,
            "execution_success": exec_result.data.get("execution_success", False),
            "rows": exec_result.data.get("rows", []),
            "row_count": exec_result.data.get("row_count", 0),
            "diagnosis": exec_result.data.get("diagnosis", ""),
            "refined": exec_result.data.get("refined", False),
            "error_type": exec_result.data.get("error_type", "none"),
        }
        if not exec_result.data.get("execution_success"):
            data["error"] = exec_result.data.get("execution_error") or exec_result.error
        return ToolResult.ok(
            data=data,
            message="Workflow 完成" if data["execution_success"] else "执行未成功，已记录诊断",
        )

    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        from chatdb.core.react_state import ReActPhase, ErrorType
        state.phase = ReActPhase.SQL_BUILD
        if not state.intent:
            state.set_error("缺少意图，无法执行 SQL 流程", ErrorType.AMBIGUOUS_INTENT)
            return
        result = await self.execute(
            intent=state.intent,
            schema_text=state.schema_text,
            yml_config=state.yml_config,
            available_tables=state.available_tables or None,
        )
        if not result.success:
            state.set_error(result.error or "SQL 流程失败", ErrorType.OTHER)
            return
        d = result.data
        state.current_sql = d.get("sql", "")
        state.final_sql = state.current_sql
        state.sql_candidates = []
        if d.get("execution_success"):
            rows = d.get("rows", [])
            state.execute_result = {"rows": rows, "row_count": d.get("row_count", 0)}
            state.clear_error()
            if len(rows) == 0:
                state.set_error("查询返回空结果", ErrorType.NO_DATA)
                state.mark_need(need_critique=True)
            else:
                state.clear_all_needs()
            if hasattr(context, "generated_sql"):
                context.generated_sql = state.final_sql
        else:
            state.execution_error = d.get("error", "执行失败")
            state.set_error(state.execution_error, error_type_from_str(d.get("error_type", "other")))
            state.mark_need(need_execute=False, need_critique=True)
        state.mark_need(need_sql=False)
