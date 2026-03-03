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
from chatdb.tools.virtual_field_converter import (
    build_mappings_from_tables_info,
    expand_virtual_fields,
)

if TYPE_CHECKING:
    from chatdb.agents.base import AgentContext
    from chatdb.agents.semantic_parser import StructuredIntent


# ---------- dict → ColumnStats 转换 ----------

def _dicts_to_column_stats(
    stats_dicts: list[dict[str, Any]],
) -> list:
    """将旧格式 dict 列表转换为 ColumnStats 对象列表。"""
    from chatdb.database.column_stats_provider import ColumnStats

    result = []
    for d in stats_dicts:
        cs = ColumnStats(
            name=d.get("name", ""),
            dtype=d.get("type", ""),
            null_pct=d.get("null_pct", 0.0),
            unique_count=d.get("unique_count", 0),
        )
        stats = d.get("stats")
        if stats:
            cs.min_val = stats.get("min")
            cs.max_val = stats.get("max")
            cs.mean_val = stats.get("mean")
            cs.median_val = stats.get("median")
        if d.get("top_values"):
            cs.top_values = d["top_values"]
        result.append(cs)
    return result


# ---------- 公共逻辑 ----------

DANGEROUS_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "CREATE",
]


def check_sql_validity(sql: str, checks: str = "all") -> list[str]:
    """统一的 SQL 检查函数。
    
    Args:
        sql: 待检查的 SQL 语句
        checks: 检查级别
            - "readonly": 只读检查（SELECT + 禁止写操作）
            - "syntax": 语法检查（SELECT + FROM + 括号匹配）
            - "safety": 安全检查（危险关键词）
            - "all": 全部检查（默认）
    
    Returns:
        错误列表，空列表表示通过
    """
    errors: list[str] = []
    sql_upper = sql.upper().strip()
    
    # 只读检查
    if checks in ("readonly", "all"):
        if not sql_upper.startswith("SELECT"):
            errors.append("只支持 SELECT 查询")
        for kw in DANGEROUS_KEYWORDS:
            if kw in sql_upper:
                errors.append(f"禁止使用 {kw} 语句")
    
    # 语法检查
    if checks in ("syntax", "all"):
        if not sql_upper.startswith("SELECT"):
            errors.append("SQL 必须以 SELECT 开头")
        if "FROM" not in sql_upper:
            errors.append("SQL 必须包含 FROM 子句")
        if sql.count("(") != sql.count(")"):
            errors.append("括号不匹配")
    
    # 安全检查
    if checks == "safety":
        for kw in DANGEROUS_KEYWORDS:
            if kw in sql_upper:
                errors.append(f"禁止使用 {kw} 语句")
    
    return errors


# 向后兼容的包装函数
def check_sql_readonly(sql: str) -> list[str]:
    """只读安全检查（向后兼容）"""
    return check_sql_validity(sql, checks="readonly")


def check_sql_syntax(sql: str) -> list[str]:
    """基础语法检查（向后兼容）"""
    return check_sql_validity(sql, checks="syntax")


def check_sql_safety(sql: str) -> list[str]:
    """安全检查（向后兼容）"""
    return check_sql_validity(sql, checks="safety")


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
        skill_registry: Any = None,
    ):
        self.llm = llm
        self.db_connector = db_connector
        self._log = get_component_logger("SQLTool")
        self._skill_registry = skill_registry  # SkillRegistry 实例（可选）

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
        column_stats: list[dict[str, Any]] | None = None,
    ) -> str:
        """格式化列信息供 LLM 理解（委托 ColumnStatsProvider）。"""
        from chatdb.database.column_stats_provider import ColumnStatsProvider, ColumnStats

        stats_objs = _dicts_to_column_stats(column_stats) if column_stats else []
        return ColumnStatsProvider.format_columns(
            stats_objs, columns=columns, relevant_cols=None,
        )

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

    def _parse_candidates(self, response: str) -> tuple[list[SQLCandidate], list[str]]:
        """解析 LLM 返回的候选 SQL。

        Returns:
            (candidates, virtual_fields) — virtual_fields 为 LLM 声明使用的虚拟字段列表
        """
        candidates = []
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response)
            data = json.loads(json_match.group()) if json_match else {}
        virtual_fields: list[str] = data.get("virtual_fields", [])
        for c in data.get("candidates", []):
            sql = (c.get("sql", "") or "").strip()
            reason = c.get("reason", "")
            if sql:
                sql = self._clean_sql(sql)
                candidates.append(SQLCandidate(sql=sql, reason=reason, confidence=0.9))
        return candidates, virtual_fields

    def _expand_virtual_fields_in_candidates(
        self,
        candidates: list[SQLCandidate],
        virtual_fields: list[str],
        tables_info: list[dict[str, Any]] | None,
        state: Any = None,
    ) -> None:
        """使用 AST 转换引擎替换候选 SQL 中的虚拟字段。原地修改 candidates。"""
        mappings = build_mappings_from_tables_info(tables_info, state)
        if not mappings:
            return
        all_replaced: set[str] = set()
        for c in candidates:
            result = expand_virtual_fields(c.sql, mappings)
            if result.replaced_fields:
                self._log.info(
                    f"虚拟字段转换: {result.replaced_fields}"
                    f"{' (正则兜底)' if result.used_fallback else ''}"
                )
                for msg in result.messages:
                    self._log.debug(msg)
            c.sql = result.sql
            all_replaced.update(result.replaced_fields)
        # 检查 LLM 声明但未替换的字段（column 类型跳过：它的作用域在 Planner 层面，SQL 中直接用真实列名）
        mapping_by_id = {m.field_id: m for m in mappings}
        for fid in virtual_fields:
            if fid not in all_replaced and fid in mapping_by_id:
                m = mapping_by_id[fid]
                if m.field_type == "column":
                    self._log.debug(f"虚拟字段 '{fid}' 为 column 类型，SQL 中直接使用真实列名，跳过占位符检查")
                else:
                    self._log.warn(f"LLM 声明了虚拟字段 '{fid}' 但 SQL 中未找到占位符")

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
            '**SELECT 中所有聚合表达式和虚拟字段必须用 AS 起一个简短的别名**：'
            '如 `total_flow AS "总流水"`、`SUM("金额") AS "总金额"`。'
            '禁止 SELECT 中出现无别名的聚合表达式或虚拟字段裸名称，否则列名会变成整段表达式导致下游引用失败',
        ]
        if has_required_where:
            rules.append(
                '**虚拟字段用法**：条件型虚拟字段是**完整布尔条件**，独立作为 AND 子句，'
                '禁止追加 `=`/`IN`/`>` 等操作符；'
                '指标型虚拟字段用**裸名称**引用（如 `total_flow`），系统自动包裹聚合函数。'
                '即使写了 `SUM(total_flow)`，系统也会安全跳过重复包裹，推荐用裸名称保持简洁。'
                '额外筛选用真实列名写独立 AND 条件，不要修改虚拟字段本身'
            )
            rules.append(
                '**同组互斥字段不能 AND**：同一互斥组的虚拟字段操作同一列的不同枚举值，AND 连接永远返回 0 行。'
                '对比场景应将它们分别放在不同子查询的 WHERE 中'
            )
            rules.append(
                '**必选条件不可遗漏**：标记为"必选条件"的虚拟字段控制数据口径，遗漏会导致查询到错误范围的数据'
            )
        if has_task_description:
            rules.append(
                '**任务描述解读**：description 中的词汇（如"市场费""今年"）是上下文说明，其对应的筛选条件已在虚拟字段中，不要重复添加'
            )
            rules.append(
                '**禁止越权**：SQL 只实现"任务描述"中明确要求的动作。不要自作主张做额外分析（找极值、下钻、排名等）'
            )
        rules.append(
            '**SQL 必须查询真实数据表**：禁止生成不含 FROM 子句的纯常量 SQL（如 `SELECT 字面值 UNION ALL SELECT ...`）。'
            '即使上下文中提供了具体数值，也必须通过 FROM 子句从表中查询获取数据，不得将数值硬编码为 SELECT 常量'
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

        # current_metric_def 中的列（+ all_metric_defs）
        if state:
            all_defs = getattr(state, "all_metric_defs", {}) or {}
            if not all_defs and getattr(state, "current_metric_def", None):
                all_defs = {"_": state.current_metric_def}
            for mdef in all_defs.values():
                agg = mdef.get("agg", "")
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
        column_stats: list[dict[str, Any]] | None = None,
        tables_info: list[dict[str, Any]] | None = None,
    ) -> str:
        # ★ 统一表模型：所有选中的表一视同仁，每个表自带自己的上下文
        # 不区分源表/临时表，选了谁就用谁的信息

        # ── 1. 构建所有表的完整信息段（列信息 + 业务说明 + 指标约束 + WHERE 条件）──
        tables_sections = self._build_tables_sections(
            intent, yml_config, schema_text, table_schema, column_stats,
            state, tables_info,
        )

        # ── 2. 判断是否有任何表带了 required_filters（用于硬约束规则）──
        has_required_where = False
        if tables_info:
            for tbl in tables_info:
                if tbl.get("required_filters"):
                    has_required_where = True
                    break
        elif state and getattr(state, "required_filters", None):
            # 兼容旧路径
            has_required_where = True

        # ── 3. 结构化意图摘要 ──
        effective_task_type = (
            current_task.get("task_type", intent.task_type)
            if current_task
            else intent.task_type
        )
        # 从所有表的 yml_config 中汇集 dimensions 信息
        all_dims_config: dict[str, Any] = {}
        if tables_info:
            for tbl in tables_info:
                tbl_yml = tbl.get("yml_config") or {}
                all_dims_config.update(tbl_yml.get("dimensions", {}))
        if not all_dims_config:
            all_dims_config = (yml_config or {}).get("dimensions", {})
        
        dim_display = []
        for did in (intent.dimensions or []):
            if did in all_dims_config:
                col = all_dims_config[did].get("column", did)
                dim_display.append(f'{did} → 列名 `{col}`')
            else:
                dim_display.append(did)
        intent_section = f"""## 结构化意图
- task_type: {effective_task_type}
- metrics: {intent.metrics}
- dimensions: {', '.join(dim_display) if dim_display else '[]'}
- SQL 中必须使用真实列名，不要使用维度ID"""

        # ── 4. 任务指令 ──
        task_instruction = self._build_task_instruction(current_task)

        # ── 5. 用户查询 ──
        rewritten = getattr(intent, "rewritten_query", "") or ""
        if current_task and current_task.get("description"):
            query_section = f"## 用户查询（当前任务视角）\n{current_task['description']}"
        elif rewritten and rewritten != intent.raw_query:
            query_section = f"## 用户查询\n{rewritten}\n（原始: {intent.raw_query}）"
        else:
            query_section = f"## 用户查询\n{intent.raw_query}"

        # ── 6. 检索增强：值匹配提示（通用，不绑定特定表）──
        value_hint_section = ""
        rc = getattr(state, "retrieval_context", None) if state else None
        if rc is not None and hasattr(rc, "format_value_hint"):
            value_hint_section = rc.format_value_hint()

        # ── 7. 虚拟字段检索召回 ──
        # ★ 已移除：检索召回的虚拟字段段落（vf_retrieval_section）与上方
        # _build_virtual_fields_section() 大量重复，后者已包含完整的虚拟字段表格、
        # 互斥组说明、scope 标注和用法示例，信息更精确。去除重复可缩减 prompt 长度。
        vf_retrieval_section = ""

        # ── 8. 构建动态输出格式示例 ──
        output_section = self._build_output_section(tables_info, state, has_required_where)

        # ── 9. 统一组装 prompt ──
        sections = [
            "请根据以下信息生成可执行的 SQL。",
            task_instruction,
            query_section,
            "",
            tables_sections,
            "",
            intent_section,
            "",
            value_hint_section,
            "",
            vf_retrieval_section,
            "",
            f"## DuckDB SQL 规范\n{get_duckdb_syntax_rules()}",
            "",
            f"### 硬约束\n{self._build_sql_hard_rules(has_required_where=has_required_where, has_task_description=bool(current_task))}",
            "",
            output_section,
        ]
        # 过滤掉空段落，避免多余空行
        return "\n".join(s for s in sections if s)

    def _build_output_section(
        self,
        tables_info: list[dict[str, Any]] | None,
        state: Any,
        has_required_where: bool,
    ) -> str:
        """构建输出格式说明。

        ★ 不硬编码具体 SQL 示例，只规定 JSON 结构和虚拟字段声明规则，
        让 LLM 根据上文的虚拟字段说明、任务指令、Skill SQL 示例自由生成 SQL。
        """
        # 收集所有虚拟字段 ID
        vf_ids: list[str] = []
        if tables_info:
            for tbl in tables_info:
                for f in (tbl.get("required_filters") or []):
                    fid = f.get("id", "")
                    if fid and fid not in vf_ids:
                        vf_ids.append(fid)
        elif state and getattr(state, "required_filters", None):
            for f in state.required_filters:
                fid = f.get("id", "")
                if fid and fid not in vf_ids:
                    vf_ids.append(fid)

        lines = [
            "## 输出要求",
            "严格输出 JSON，**只生成 1 个最匹配当前任务的 SQL**：",
            "```json",
            "{",
            '  "candidates": [',
            '    { "sql": "<你生成的完整 SQL>", "reason": "一句话解释" }',
            "  ],",
            f'  "virtual_fields": {json.dumps(vf_ids, ensure_ascii=False) if vf_ids else "[]"}',
            "}",
            "```",
        ]

        if vf_ids:
            lines.append(f"**virtual_fields 必须包含所有使用到的虚拟字段**（上方共 {len(vf_ids)} 个）。")
            lines.append("SQL 中如何使用虚拟字段，请参照上方「虚拟字段」章节的规则和示例。")
            lines.append("若需额外筛选条件，用真实列名单独添加 AND 子句。")

        return "\n".join(lines)

    def _build_tables_sections(
        self,
        intent: Any,
        yml_config: dict[str, Any],
        schema_text: str | None,
        table_schema: dict[str, Any] | None,
        column_stats: list[dict[str, Any]] | None,
        state: Any,
        tables_info: list[dict[str, Any]] | None,
    ) -> str:
        """为所有选中表构建统一格式的完整信息段。

        ★ 核心设计：所有表一视同仁，每个表自带自己的上下文。
        选了哪个表，就用哪个表的 understanding、yml、metric、filters。

        每张表的格式：
        ## 表: "表名"  (N 行)
        ### 业务说明
        ...（该表自己的 table_understanding）
        ### 列信息
        - "列名" (类型) [缺失:X%] [唯一值:N] -- 摘要
        ### 指标约束（如该表有 yml 配置）
        ...
        ### 建议 WHERE 条件（如该表有 required_filters）
        ...
        """
        # ★ 有 tables_info 时使用统一模型
        if tables_info:
            parts = []
            for tbl in tables_info:
                parts.append(self._build_single_table_section(tbl, intent, state))

            if not parts:
                return "## 可用表\n无表信息"

            result = "\n\n".join(parts)
            if len(parts) > 1:
                result += "\n\n★ 多个表已列出，请根据任务需求选择合适的表进行查询。"
            return result

        # ★ 兼容旧路径：无 tables_info 时退化到单表逻辑
        if table_schema and table_schema.get("columns"):
            tbl_yml = yml_config or {}
            relevant_cols = self._collect_relevant_columns(intent, tbl_yml, state)
            all_cols = table_schema["columns"]
            if relevant_cols:
                col_info = self._format_columns_with_highlight(all_cols, relevant_cols, column_stats)
            else:
                col_info = self._format_columns_for_prompt(all_cols, column_stats=column_stats)
            source_table = getattr(intent, "table_name", "") or ""
            return f'## 表: `"{source_table}"`\n{col_info}'
        elif schema_text:
            return f"## 列信息\n{schema_text}"
        else:
            return "## 列信息\n无列信息"

    def _build_single_table_section(
        self,
        tbl: dict[str, Any],
        intent: Any,
        state: Any,
    ) -> str:
        """为单个表构建完整的 prompt 段落。
        
        每个表自带：schema、column_stats、understanding、yml_config、
        metric_def、required_filters —— 有什么就展示什么，没有就跳过。
        """
        tbl_name = tbl["table_name"]
        tbl_schema = tbl.get("schema") or {}
        tbl_stats = tbl.get("column_stats")
        tbl_cols = tbl_schema.get("columns", [])
        row_count = tbl_schema.get("row_count", 0)
        tbl_yml = tbl.get("yml_config") or {}
        tbl_understanding = tbl.get("understanding") or ""
        tbl_metric_def = tbl.get("metric_def")
        tbl_metric_name = tbl.get("metric_name") or ""
        tbl_all_metric_defs = tbl.get("all_metric_defs") or {}
        tbl_required_filters = tbl.get("required_filters") or []
        tbl_metric_union_info = tbl.get("metric_union_info") or []

        # 表头
        header = f'## 表: `"{tbl_name}"`'
        if row_count:
            header += f"  ({row_count} 行)"

        sub_sections = [header]

        # 业务说明（该表自己的）
        if tbl_understanding:
            sub_sections.append(f"### 业务说明\n{tbl_understanding}")

        # 列信息
        if tbl_cols:
            # 有 yml 配置时区分关键列/其他列；无 yml 时展示全部列
            if tbl_yml and tbl_yml.get("metrics"):
                relevant_cols = self._collect_relevant_columns(intent, tbl_yml, state)
                col_info = self._format_columns_with_highlight(tbl_cols, relevant_cols, tbl_stats)
            else:
                col_info = self._format_columns_for_prompt(tbl_cols, column_stats=tbl_stats)
            sub_sections.append(f"### 列信息\n{col_info}")
        else:
            sub_sections.append("### 列信息\n无列信息")

        # 指标约束（展示该任务涉及的所有指标定义）
        if tbl_all_metric_defs:
            # 多指标：逐个展示
            metric_lines = ["### 指标约束"]
            for mid, mdef in tbl_all_metric_defs.items():
                agg_expr = mdef.get("agg", "")
                f_refs = mdef.get("filter_refs", [])
                role = "（主指标）" if mid == tbl_metric_name else ""
                # 已展开的指标：metric_* 筛选器已内置在 CASE WHEN 中，不再展示
                if mdef.get("_expanded"):
                    f_refs = [f for f in f_refs if not f.startswith("metric_")]
                    metric_lines.append(
                        f"- **{mid}**{role}: {mdef.get('label', '')}\n"
                        f"  - 聚合表达式（已含指标筛选）: `{agg_expr}`\n"
                        f"  - 公共筛选器: {f_refs}"
                    )
                else:
                    metric_lines.append(
                        f"- **{mid}**{role}: {mdef.get('label', '')}\n"
                        f"  - 聚合表达式: `{agg_expr}`\n"
                        f"  - 默认筛选器: {f_refs}"
                    )
            sub_sections.append("\n".join(metric_lines))
        elif tbl_metric_def:
            # 向后兼容：单指标
            agg_expr = tbl_metric_def.get("agg", "")
            filter_refs = tbl_metric_def.get("filter_refs", [])
            sub_sections.append(f"""### 指标约束
- 指标ID: {tbl_metric_name}
- 含义: {tbl_metric_def.get('label', '')}
- 聚合表达式（必须使用）: {agg_expr}
- 默认筛选器: {filter_refs}""")

        # ★ 多原子指标 CASE WHEN 引导（当 WHERE 中互斥条件已用 OR 合并时）
        if tbl_metric_union_info and tbl_all_metric_defs:
            sub_sections.append(self._build_metric_union_hint(
                tbl_metric_union_info, tbl_all_metric_defs,
            ))

        # 虚拟字段占位符（条件型 + 列型）
        if tbl_required_filters:
            sub_sections.append(self._build_virtual_fields_section(tbl_required_filters, tbl_metric_name))

        return "\n".join(sub_sections)

    def _build_metric_union_hint(
        self,
        metric_union_info: list[dict[str, Any]],
        all_metric_defs: dict[str, dict[str, Any]],
    ) -> str:
        """构建多原子指标聚合引导段落。

        当多个原子指标的互斥筛选条件（如"报表项=利润" vs "报表项=流水"）
        被 OR 合并到 WHERE 中时，SQL 的 SELECT 必须使用各指标预定义的聚合表达式，
        这些表达式已经内含 CASE WHEN 逻辑，LLM 不需要自行构造。
        """
        lines = [
            "### ★ 多指标聚合模式",
            "",
            "本任务涉及**多个互斥指标**，WHERE 中已用 OR 合并其筛选条件以保证数据完整。",
            "**你必须直接使用上方「指标约束」中给出的聚合表达式**，不要自行构造 CASE WHEN。",
            "",
            "每个指标的聚合表达式已经内含了正确的 CASE WHEN 逻辑，直接在 SELECT 中引用即可：",
            "",
        ]

        for mid, mdef in all_metric_defs.items():
            agg_expr = mdef.get("agg", "")
            label = mdef.get("label", mid)
            if agg_expr:
                lines.append(f'- **{label}** (`{mid}`): `{agg_expr}`')

        lines.append("")
        lines.append("**⚠️ 禁止自行构造 CASE WHEN**，直接复制上方聚合表达式使用。")

        return "\n".join(lines)

    def _build_virtual_fields_section(
        self,
        required_filters: list[dict[str, Any]],
        metric_name: str = "",
    ) -> str:
        """构建虚拟字段说明段落（委托给 VirtualFieldPromptBuilder）。"""
        from chatdb.config.virtual_field import VirtualFieldPromptBuilder
        return VirtualFieldPromptBuilder.build(required_filters, metric_name)

    def _format_columns_with_highlight(
        self,
        all_cols: list[dict[str, Any]],
        relevant_cols: set[str],
        column_stats: list[dict[str, Any]] | None,
    ) -> str:
        """格式化列信息，区分关键列和其他列（委托 ColumnStatsProvider）。"""
        from chatdb.database.column_stats_provider import ColumnStatsProvider

        stats_objs = _dicts_to_column_stats(column_stats) if column_stats else []
        return ColumnStatsProvider.format_columns(
            stats_objs, columns=all_cols,
            relevant_cols=relevant_cols if relevant_cols else None,
        )

    def _build_task_instruction(self, current_task: dict[str, Any] | None) -> str:
        """构建任务类型指令，SQL 生成规则统一从 SkillRegistry 加载。"""
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
            # ★ 对分组维度相关的 note 加强调，避免 LLM 遗漏 GROUP BY 维度
            formatted_notes = []
            for n in task_notes:
                note_str = str(n)
                if any(kw in note_str for kw in ['分组维度', 'GROUP BY', 'group by']):
                    formatted_notes.append(f"  - ★ **{note_str}**（GROUP BY 必须包含全部列，缺少任何一个都会导致数据被错误聚合）")
                else:
                    formatted_notes.append(f"  - {note_str}")
            inst += "- 注意事项:\n" + "\n".join(formatted_notes) + "\n"
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

        # ── 各任务类型的参考规则（统一从 SkillRegistry 加载）──
        if self._skill_registry:
            skill_instruction = self._skill_registry.get_sql_instruction(task_type)

            if skill_instruction:
                # 动态参数注入：source 类型需要可用维度
                if task_type == "source" and available_dims:
                    dims_hint = "、".join(f"「{d}」" for d in available_dims)
                    inst += f"\n- 可用的分组维度: {dims_hint}\n"

                inst += f"\n{skill_instruction}\n"

                # 注入 SQL 示例（从 SkillRegistry 加载）
                sql_examples = self._skill_registry.get_sql_examples(task_type)
                if sql_examples:
                    inst += "\n### SQL 参考示例\n"
                    inst += "以下是同类任务的 SQL 模板参考（需根据实际表名和列名调整）：\n"
                    for ex in sql_examples:
                        desc = ex.get("description", "")
                        notes = ex.get("notes", [])
                        sql_tpl = ex.get("sql_template", "")
                        inst += f"\n**{desc}**\n"
                        if notes:
                            inst += f"- 要点: {'; '.join(str(n) for n in notes)}\n"
                        if sql_tpl:
                            inst += f"```sql\n{sql_tpl}\n```\n"

        return inst

    async def _generate_candidates(
        self,
        intent: Any,
        yml_config: dict[str, Any],
        schema_text: str | None = None,
        table_schema: dict[str, Any] | None = None,
        current_task: dict[str, Any] | None = None,
        state: Any = None,  # ReActState
        column_stats: list[dict[str, Any]] | None = None,
        tables_info: list[dict[str, Any]] | None = None,
    ) -> list[SQLCandidate]:
        prompt = self._build_generation_prompt(
            intent, yml_config, schema_text, table_schema,
            current_task, state, column_stats=column_stats,
            tables_info=tables_info,
        )
        system = "你是 SQL 生成专家。根据任务描述和表结构生成 DuckDB SQL。核心原则：以任务描述为准，业务配置仅供参考；只使用提供的列名；列名用双引号，字符串值用单引号。严格输出 JSON。"
        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt=system,
                caller_name="generate_sql",
            )
            candidates, virtual_fields = self._parse_candidates(response)
            # 虚拟字段替换（AST 转换引擎）
            if candidates:
                self._expand_virtual_fields_in_candidates(
                    candidates, virtual_fields, tables_info, state,
                )
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
        """根据结构化意图生成 SQL。
        
        ★ 核心设计：所有选中的表一视同仁。
        每个表自带自己的完整上下文（understanding、yml_config、metric_def、filters），
        不依赖 state 上的全局"源表"字段。
        """
        if not self.llm:
            return ToolResult.fail("未配置 LLM，无法生成 SQL")
        from chatdb.agents.semantic_parser import StructuredIntent
        intent_obj = StructuredIntent.from_dict(intent) if isinstance(intent, dict) else intent

        # ★ 统一表模型：确定需要查询的表
        selected = getattr(state, "selected_tables", None) if state else None
        source_table = intent_obj.table_name or ""

        if selected:
            tables_to_query: list[str] = list(selected)
        else:
            tables_to_query = [source_table] if source_table else []
            if current_task:
                for dep_id, tbl_info in current_task.get("upstream_temp_tables", {}).items():
                    tbl_name = tbl_info.get("name", "") if isinstance(tbl_info, dict) else tbl_info
                    if tbl_name and tbl_name not in tables_to_query:
                        tables_to_query.append(tbl_name)

        # ★ 为每张表构建完整上下文（schema + stats + understanding + yml + metric + filters）
        tables_info: list[dict[str, Any]] = []
        for tbl_name in tables_to_query:
            tbl_entry = await self._build_table_context(
                tbl_name, source_table, available_tables, state,
            )
            tables_info.append(tbl_entry)

        candidates = await self._generate_candidates(
            intent_obj,
            yml_config or {},
            schema_text or None,
            None,
            current_task,
            state,
            column_stats=None,
            tables_info=tables_info,
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

    async def _build_table_context(
        self,
        tbl_name: str,
        source_table: str,
        available_tables: list[dict] | None,
        state: Any,
    ) -> dict[str, Any]:
        """为单个表构建完整上下文。
        
        ★ 核心方法：每个表自带自己的全部信息，不区分源表/临时表。
        有 yml 配置就带上指标和 filters，有 understanding 就带上，没有就不带。
        
        Returns:
            {
                "table_name": str,
                "schema": dict,           # 列信息
                "column_stats": list,      # 实时列统计
                "understanding": str,      # 该表的业务说明
                "yml_config": dict,        # 该表的 yml 配置（如有）
                "metric_name": str,        # 该表的当前指标 ID（如有）
                "metric_def": dict,        # 该表的指标定义（如有）
                "required_filters": list,  # 该表的 WHERE 条件（如有）
            }
        """
        # 1. Schema
        tbl_schema = self._get_table_schema(tbl_name, available_tables or [])
        if not tbl_schema.get("columns"):
            tbl_schema = await self._fetch_table_schema_from_db(tbl_name)
        
        # 2. 实时列统计
        tbl_stats = await self._fetch_column_stats(tbl_name)
        
        # 3. 该表自己的上下文（understanding、yml_config、metric、filters）
        # 从 state 获取（state 上挂的是当前任务注入的信息）
        tbl_understanding = ""
        tbl_yml_config: dict[str, Any] = {}
        tbl_metric_name = ""
        tbl_metric_def: dict[str, Any] | None = None
        tbl_all_metric_defs: dict[str, dict[str, Any]] = {}
        tbl_required_filters: list[dict[str, Any]] = []
        tbl_metric_union_info: list[dict[str, Any]] = []
        
        if state:
            # ★ 如果该表是 state.table_name 指向的表，继承 state 上的所有上下文
            # （这些上下文由 SQLAgent._inject_metric_definition 注入，与该表绑定）
            if tbl_name == getattr(state, "table_name", ""):
                tbl_understanding = getattr(state, "table_understanding", "") or ""
                tbl_yml_config = getattr(state, "yml_config", {}) or {}
                tbl_metric_name = getattr(state, "current_metric", "") or ""
                tbl_metric_def = getattr(state, "current_metric_def", None)
                tbl_all_metric_defs = getattr(state, "all_metric_defs", {}) or {}
                tbl_required_filters = getattr(state, "required_filters", []) or []
                tbl_metric_union_info = getattr(state, "metric_union_info", []) or []
            elif tbl_name.startswith("temp_"):
                # ★ 临时表：不继承源表的虚拟字段/指标/filters
                # 临时表是上游任务的预聚合结果，列名来自上游 SELECT，
                # 不包含源表原始列，注入虚拟字段展开会引用不存在的列导致 BinderException
                tbl_understanding = (
                    f"这是上游任务生成的临时表，包含预聚合结果。"
                    f"直接使用该表现有的列名查询，不需要重新聚合或过滤。"
                )
            else:
                # 不是 state.table_name 的表：尝试从 MetaDataStore 加载 understanding
                tbl_understanding = self._load_table_understanding_sync(tbl_name)
        
        return {
            "table_name": tbl_name,
            "schema": tbl_schema,
            "column_stats": tbl_stats,
            "understanding": tbl_understanding,
            "yml_config": tbl_yml_config,
            "metric_name": tbl_metric_name,
            "metric_def": tbl_metric_def,
            "all_metric_defs": tbl_all_metric_defs,
            "required_filters": tbl_required_filters,
            "metric_union_info": tbl_metric_union_info,
        }

    def _load_table_understanding_sync(self, table_name: str) -> str:
        """同步加载表的业务说明（从 MetaDataStore 缓存读取）。
        
        用于非源表（如临时表）在没有预加载 understanding 时的回退。
        """
        try:
            from chatdb.storage import MetaDataStore
            meta_store = MetaDataStore()
            meta = meta_store.get_by_table_name(table_name)
            if meta and meta.get("table_understanding"):
                return meta["table_understanding"]
        except Exception:
            pass
        return ""

    async def _fetch_column_stats(self, table_name: str) -> list[dict[str, Any]] | None:
        """从数据库实时获取列描述统计，失败时返回 None 而非中断。"""
        if not self.db_connector or not table_name:
            return None
        try:
            from chatdb.database.duckdb.duckdb import DuckDBConnector
            if isinstance(self.db_connector, DuckDBConnector):
                return await self.db_connector.get_column_stats_async(table_name)
        except Exception as e:
            self._log.warn(f"获取列描述统计失败: {e}")
        return None

    async def _fetch_table_schema_from_db(self, table_name: str) -> dict[str, Any]:
        """从 DuckDB 实时获取表的 schema（列名 + 类型 + 行数）。
        
        用于临时表等不在 available_tables 中的表。
        """
        result: dict[str, Any] = {"table_name": table_name, "columns": [], "row_count": 0}
        if not self.db_connector or not table_name:
            return result
        try:
            from chatdb.database.duckdb.duckdb import DuckDBConnector
            if isinstance(self.db_connector, DuckDBConnector):
                from sqlalchemy import text
                import asyncio
                def _describe() -> dict[str, Any]:
                    with self.db_connector._engine.connect() as conn:
                        desc = conn.execute(text(f'DESCRIBE "{table_name}"'))
                        cols = [{"name": r[0], "type": r[1]} for r in desc.fetchall()]
                        cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                        row_count = cnt.fetchone()[0] or 0
                        return {"table_name": table_name, "columns": cols, "row_count": row_count}
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, _describe)
        except Exception as e:
            self._log.warn(f"从 DB 获取 schema 失败 ({table_name}): {e}")
        return result

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
        required_filters: list[dict[str, Any]] | None = None,
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

        # ★ 注入虚拟字段信息，让修正 LLM 了解虚拟字段语义
        if required_filters:
            vf_section = self._build_virtual_fields_section(required_filters)
            if vf_section:
                prompt += f"\n{vf_section}\n"

        # ★ 根据错误类型给出针对性的诊断指南
        error_guide = self._get_error_diagnosis_guide(eval_result.error_type)
        if error_guide:
            prompt += f"\n{error_guide}\n"

        prompt += """
## 输出要求
```json
{ "diagnosis": "一句话说明问题", "refined_sql": "修正后的完整 SQL" }
```
原则：只改出错部分，保持其他结构不变。修正后的 SQL 中必须保留所有虚拟字段占位符（如 "base_valid_data"、"source_actual"、total_flow 等），不要将它们手动展开为真实表达式，系统会自动处理展开。只输出 JSON。"""
        return prompt

    @staticmethod
    def _get_error_diagnosis_guide(error_type: ErrorType) -> str:
        """根据错误类型返回针对性的诊断指南和常见修复方法"""
        guides = {
            ErrorType.SYNTAX_ERROR: """## 常见语法错误模式与修复方法

**1. 括号不匹配**（最常见）
- 症状: `syntax error at or near ","` 或 `syntax error at or near ")"`
- 诊断方法: 逐一数每对括号（子查询括号、函数括号、IN 列表括号、AND/OR 分组括号），确保左右配对
- 常见场景: ROUND(... / (SELECT ...), 2) 中子查询的右括号与 ROUND 的逗号、右括号交织
- 修复: 格式化 SQL 对齐括号层级，找到缺失的括号位置补齐
- ★ 关键检查: IN (...) 的右括号后面如果紧跟 AND，确保 AND 前的 `)` 数量正确

**2. 子查询中 AND/OR 优先级错误**
- 症状: 括号看似配对，但 WHERE 逻辑不对
- 诊断: 检查 `A AND B OR C` 是否应该是 `A AND (B OR C)` 或 `(A AND B) OR C`
- 修复: 给 AND/OR 混用的地方加明确括号

**3. 中文别名未加双引号**
- 症状: `syntax error` 在中文词附近
- 修复: 列名/别名含中文时必须用双引号包裹

**4. ROUND 函数参数错误**
- 症状: `syntax error at or near ","` 在 ROUND 附近
- 原因: ROUND 的第一个参数（表达式）内部的子查询括号不完整，导致逗号被误解析
- 修复: 确保 ROUND( <完整表达式> , 2) 中第一个参数是完整闭合的表达式""",

            ErrorType.UNKNOWN_COLUMN: """## 常见列名错误模式与修复方法

**1. 使用了维度 ID 而非真实列名**
- 症状: `column "dim_product_category" not found`
- 原因: 误用了结构化意图中的维度 ID（dim_xxx）作为列名
- 修复: 查看"表 Schema"中的实际列名，例如 dim_product_category → "产品大类"

**2. 列名拼写错误或使用了不存在的列**
- 症状: `column "xxx" not found`，且有 `Candidate bindings` 提示
- 修复: 使用 Candidate bindings 中推荐的列名

**3. CTE 内层列名在外层不可见**
- 症状: 外层 SELECT 引用的列在 CTE 中未被 SELECT
- 修复: 确保外层引用的列在 CTE 的 SELECT 列表中""",

            ErrorType.TYPE_MISMATCH: """## 常见类型不匹配模式与修复方法

**1. BIGINT 列与字符串值比较**
- 症状: `cannot compare BIGINT and VARCHAR`
- 场景: WHERE "年" IN ('2024', '2025')，但"年"列是 BIGINT 类型
- 修复: 去掉单引号，改为 WHERE "年" IN (2024, 2025)

**2. 日期函数用在整数列上**
- 症状: `QUARTER()/MONTH() 只接受 DATE/TIMESTAMP`
- 场景: 月份列是 BIGINT（如 202501），不能直接用 QUARTER()
- 修复: 使用 CASE WHEN 映射或数学运算提取季度

**3. 聚合函数参数类型错误**
- 症状: `cannot apply SUM to VARCHAR`
- 修复: 确保 SUM/AVG 等聚合函数作用于数值列""",
        }
        return guides.get(error_type, "")

    async def _diagnose_and_refine(
        self,
        eval_result: EvaluationResult,
        intent: Any,
        yml_config: dict[str, Any],
        schema_text: str | None,
        required_filters: list[dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        prompt = self._build_diagnose_prompt(eval_result, intent, schema_text, required_filters)
        system = ("你是 SQL 调试专家。严格按以下步骤修正：\n"
                  "1. 精确定位错误位置（不要只说'括号不匹配'，要指出具体在哪个函数/子查询处缺少了什么）\n"
                  "2. 参考诊断指南中的常见模式匹配错误类型\n"
                  "3. 只改出错部分，给出可直接执行的完整 SQL\n"
                  "4. 修正后的 SQL 必须与原 SQL 有实质性差异，不能原样返回\n"
                  "输出清晰 JSON。")
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
        required_filters = kwargs.get("required_filters", None)
        eval_result = EvaluationResult(sql=sql)
        eval_result = await self._execute_sql_internal(eval_result)
        attempts = 0
        while not eval_result.execution_success and attempts < self.MAX_REFINE_ATTEMPTS:
            self._log.info(f"尝试修正 (第 {attempts + 1} 次)")
            eval_result = await self._diagnose_and_refine(
                eval_result, intent, yml_config or {}, schema_text or None,
                required_filters=required_filters,
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
            required_filters=state.required_filters or None,
        )
        if eval_result.refined and eval_result.refined_sql != state.current_sql:
            # ★ 修复后的 SQL 可能仍含虚拟字段占位符，需要再做一次展开
            refined = eval_result.refined_sql
            mappings = build_mappings_from_tables_info(None, state)
            if mappings:
                result = expand_virtual_fields(refined, mappings)
                if result.replaced_fields:
                    self._log.info(f"[refine] 虚拟字段展开: {result.replaced_fields}")
                refined = result.sql
            state.current_sql = refined
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
                suggested = _clean_sql_util(diagnosis["suggested_sql"])
                # ★ 诊断修复的 SQL 可能仍含虚拟字段占位符，需要再做一次展开
                mappings = build_mappings_from_tables_info(None, state)
                if mappings:
                    result = expand_virtual_fields(suggested, mappings)
                    if result.replaced_fields:
                        self._log.info(f"[no_data_fix] 虚拟字段展开: {result.replaced_fields}")
                    suggested = result.sql
                state.current_sql = suggested
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
            # ACT: 记录生成的 SQL（条件校验延后到执行成功后进行）
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
        完整流程：生成 SQL → 执行 → (失败时 refine → 重新执行) → 执行成功后条件校验 → 再执行

        核心设计：条件校验（AST）延后到 SQL 可执行之后再做。
        因为 LLM 生成的 SQL 可能有语法错误（如括号不匹配），AST 解析会失败。
        让数据库 + refine 先修好语法，再用 AST 做条件完整性校验。

        流程：
        1. LLM 生成 SQL
        2. 执行 SQL
        3. 执行失败 → refine → 重新执行（最多 max_refine 次）
        4. 执行成功 → 条件校验/注入 → 如果 SQL 变了则再执行一次
        """
        max_refine = 3
        
        await self.run_generate(state, context)
        if state.error:
            return
        
        for attempt in range(max_refine + 1):
            # 执行 SQL
            await self._execute_sql(state, context)
            
            # 执行成功 → 直接返回（虚拟字段已在生成阶段替换，无需后置校验）
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
