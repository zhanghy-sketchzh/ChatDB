"""
虚拟字段转换引擎 — 基于 sqlglot AST，将 SQL 中的虚拟字段替换为真实表达式。

架构（参考流程图）：

    输入 SQL + 字段映射
           │
    ┌──────▼──────┐
    │  预处理阶段   │ → 处理特殊语法（namespace::table）、创建反向映射、过滤相关映射
    └──────┬──────┘
           │
    ┌──────▼──────┐      ┌──────────────┐
    │ AST 解析阶段 │ ←──→ │ 错误处理      │
    │ sqlglot 解析 │      │ fallback→正则 │
    │ 构建 AST 树  │      └──────────────┘
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 递归转换阶段  │ → 深度优先遍历
    │             │ → 处理子查询和 CTE
    │             │ → 转换 SELECT 语句
    │             │ → 处理各种子句 (WHERE/GROUP BY/ORDER BY/HAVING)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  后处理阶段   │ → 生成格式化 SQL → 恢复特殊语法 → 应用反向映射
    └──────┬──────┘
           │
    输出转换后的 SQL

核心能力：
- 虚拟字段可出现在 SQL 任意位置（SELECT / WHERE / GROUP BY / ORDER BY / HAVING / 子查询 / CTE）
- 根据上下文智能替换：SELECT 中保留别名、WHERE 中直接内联、GROUP BY/ORDER BY 中展开
- AST 优先、正则兜底：sqlglot 解析失败时退化为安全的正则替换
- 支持两种占位符格式：{field_id}（花括号）和裸引用 "field_id"（双引号列名）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from lib.utils.logger import get_component_logger

_log = get_component_logger("VirtualFieldConverter")

_DIALECT = "duckdb"

# ─────────────────────── 数据结构 ───────────────────────


@dataclass
class VirtualFieldMapping:
    """单个虚拟字段的映射描述。"""
    field_id: str          # 虚拟字段标识符，如 "base_valid_data"、"total_flow"
    expr: str              # 真实 SQL 表达式（metric 类型为纯值表达式，不含聚合函数）
    field_type: str = "condition"  # "condition"（WHERE 条件）/ "metric"（聚合表达式）
    label: str = ""        # 人类可读描述
    description: str = ""  # 业务说明（如"仅筛选流水类报表项"）
    agg_type: str = ""     # 聚合类型：SUM/AVG/COUNT/MAX/MIN/EXPR（仅 metric 类型有效）
                           # SUM 等：系统自动包裹 SUM(expr)
                           # EXPR：自包含表达式，不需要外层聚合（派生指标）
                           # 空字符串：向后兼容，不做聚合包裹


@dataclass
class ConversionResult:
    """转换结果。"""
    sql: str                              # 转换后的 SQL
    replaced_fields: list[str] = field(default_factory=list)  # 已替换的虚拟字段 ID
    messages: list[str] = field(default_factory=list)         # 日志消息
    used_fallback: bool = False           # 是否使用了正则兜底


# ─────────────────────── 预处理 ───────────────────────


def _normalize_expr(expr: str) -> str:
    """多行表达式合并为单行。"""
    lines = [line.strip() for line in expr.strip().split("\n") if line.strip()]
    return " ".join(lines)


def _build_field_maps(
    mappings: list[VirtualFieldMapping],
) -> tuple[dict[str, VirtualFieldMapping], dict[str, VirtualFieldMapping], dict[str, VirtualFieldMapping]]:
    """构建三种映射索引。

    Returns:
        (condition_map, column_map, metric_map) — 分别存放条件型、其他型和指标型虚拟字段
        注意：column 类型已从 YML 配置中移除，column_map 正常情况下应为空。
    """
    cond_map: dict[str, VirtualFieldMapping] = {}
    col_map: dict[str, VirtualFieldMapping] = {}
    metric_map: dict[str, VirtualFieldMapping] = {}
    for m in mappings:
        if m.field_type == "condition":
            cond_map[m.field_id] = m
        elif m.field_type == "metric":
            metric_map[m.field_id] = m
        else:
            col_map[m.field_id] = m
    return cond_map, col_map, metric_map


# ─────────────────────── AST 工具 ───────────────────────


def _parse_sql(sql: str) -> exp.Expression:
    """解析 SQL 为 AST，失败抛异常。"""
    stmts = sqlglot.parse(sql, dialect=_DIALECT)
    if not stmts or stmts[0] is None:
        raise ValueError(f"sqlglot 无法解析: {sql[:120]}…")
    return stmts[0]


def _parse_expr(expr_sql: str) -> exp.Expression:
    """将 SQL 表达式片段解析为 AST 节点。

    对条件类表达式：包装为 SELECT 1 WHERE {expr} 后提取 Where.this
    对列类表达式：包装为 SELECT {expr} AS __vf__ 后提取第一个 expression
    """
    # 先尝试作为列表达式解析
    try:
        wrapper = f"SELECT {expr_sql} AS __vf__"
        tree = _parse_sql(wrapper)
        sel = tree.find(exp.Select)
        if sel and sel.expressions:
            alias_node = sel.expressions[0]
            # 返回别名的 this（去掉 AS __vf__）
            if isinstance(alias_node, exp.Alias):
                return alias_node.this
            return alias_node
    except Exception:
        pass

    # 再尝试作为条件表达式解析
    try:
        wrapper = f"SELECT 1 WHERE {expr_sql}"
        tree = _parse_sql(wrapper)
        where = tree.find(exp.Where)
        if where:
            return where.this
    except Exception:
        pass

    raise ValueError(f"无法解析表达式: {expr_sql[:80]}…")


def _parse_condition_expr(expr_sql: str) -> exp.Expression:
    """将 WHERE 条件片段解析为 AST 节点。"""
    wrapper = f"SELECT 1 WHERE {expr_sql}"
    tree = _parse_sql(wrapper)
    where = tree.find(exp.Where)
    if not where:
        raise ValueError(f"无法解析条件: {expr_sql[:80]}…")
    return where.this


def _parse_column_expr(expr_sql: str) -> exp.Expression:
    """将列表达式解析为 AST 节点。"""
    wrapper = f"SELECT {expr_sql} AS __vf__"
    tree = _parse_sql(wrapper)
    sel = tree.find(exp.Select)
    if sel and sel.expressions:
        node = sel.expressions[0]
        if isinstance(node, exp.Alias):
            return node.this
        return node
    raise ValueError(f"无法解析列表达式: {expr_sql[:80]}…")


# ─────────────────────── AST 转换核心 ───────────────────────


class VirtualFieldConverter:
    """基于 sqlglot AST 的虚拟字段转换引擎。

    使用方式：
        converter = VirtualFieldConverter(mappings)
        result = converter.convert(sql)
    """

    def __init__(self, mappings: list[VirtualFieldMapping]) -> None:
        # ★ 检测重复 field_id 并告警（后者覆盖前者可能导致条件丢失）
        seen: dict[str, str] = {}
        for m in mappings:
            if m.field_id in seen:
                _log.warn(
                    f"虚拟字段 ID 重复: '{m.field_id}'，"
                    f"已有 expr='{seen[m.field_id][:60]}', "
                    f"新 expr='{m.expr[:60]}' → 后者覆盖前者"
                )
            seen[m.field_id] = m.expr
        self._mappings = {m.field_id: m for m in mappings}
        self._cond_map, self._col_map, self._metric_map = _build_field_maps(mappings)
        # 预解析所有表达式 AST（延迟构建，缓存结果）
        self._parsed_cache: dict[str, exp.Expression] = {}
        self._replaced: list[str] = []
        self._messages: list[str] = []
        # ★ 作用域信息（在 _transform_tree 中初始化）
        self._cte_names: set[str] = set()           # CTE 名称集合
        self._table_alias_to_source: dict[str, str] = {}  # 表别名 → 源名称

    def convert(self, sql: str) -> ConversionResult:
        """主入口：转换 SQL 中的虚拟字段。"""
        self._replaced = []
        self._messages = []

        if not self._mappings:
            return ConversionResult(sql=sql)

        # ── 预处理：花括号占位符 → 双引号列名（使 sqlglot 可解析）──
        preprocessed, placeholder_ids = self._preprocess(sql)

        # ── AST 路径 ──
        try:
            tree = _parse_sql(preprocessed)
            self._transform_tree(tree)
            converted = tree.sql(dialect=_DIALECT)
            # 后处理
            converted = self._postprocess(converted)
            return ConversionResult(
                sql=converted,
                replaced_fields=list(set(self._replaced)),
                messages=self._messages,
                used_fallback=False,
            )
        except Exception as e:
            _log.warn(f"AST 转换失败，降级为正则替换: {e}")
            # ── 正则兜底 ──
            fallback_sql = self._regex_fallback(sql)
            return ConversionResult(
                sql=fallback_sql,
                replaced_fields=list(set(self._replaced)),
                messages=self._messages,
                used_fallback=True,
            )

    # ── 预处理 ──

    def _preprocess(self, sql: str) -> tuple[str, list[str]]:
        """将虚拟字段引用统一转为 __vf__ 前缀列名，使 sqlglot 能解析。

        支持两种输入格式（向后兼容）：
        1. 花括号占位符: WHERE {base_valid_data} AND ...
        2. 双引号列名:   WHERE "base_valid_data" AND ...

        两种都转为: WHERE "__vf__base_valid_data" AND ...

        Returns:
            (preprocessed_sql, list_of_placeholder_field_ids)
        """
        found: list[str] = []
        result = sql
        for fid in self._mappings:
            safe_name = f"__vf__{fid}"
            # 1. 花括号占位符（向后兼容）
            placeholder = "{" + fid + "}"
            if placeholder in result:
                result = result.replace(placeholder, f'"{safe_name}"')
                found.append(fid)
                continue
            # 2. 双引号列名（新格式）—— 已经是合法 SQL 列引用
            #    只需加 __vf__ 前缀以区分真实列名
            quoted = f'"{fid}"'
            if quoted in result:
                result = result.replace(quoted, f'"{safe_name}"')
                found.append(fid)
        return result, found

    def _postprocess(self, sql: str) -> str:
        """后处理：清理残留的 __vf__ 前缀标记。

        ★ 注意区分虚拟字段类型和 SQL 位置：
        - column 类型：直接替换为列名（不加括号），AS 位置也安全
        - condition/metric 类型：
          - AS 位置：不加括号
          - 其他位置：用括号包裹确保优先级正确
        """
        for fid in self._mappings:
            safe_name = f"__vf__{fid}"
            quoted_safe = f'"{safe_name}"'
            if safe_name not in sql:
                continue
            self._messages.append(f"警告: 虚拟字段 '{fid}' 未被 AST 替换，已用正则兜底清理")
            mapping = self._mappings[fid]
            expr = _normalize_expr(mapping.expr)

            if mapping.field_type == "column":
                # column 类型：简单列名映射，直接替换（不加括号）
                # 注意：postprocess 无法恢复 table 前缀（a./b.），
                # 这应该在 AST 阶段处理
                sql = sql.replace(quoted_safe, expr)
            else:
                # condition/metric 类型
                # 1. AS 别名位置：不加括号
                as_pattern = re.compile(
                    rf'AS\s+{re.escape(quoted_safe)}',
                    re.IGNORECASE,
                )
                sql = as_pattern.sub(f'AS {quoted_safe.replace(safe_name, fid)}', sql)
                # 2. 其他位置：括号包裹
                sql = sql.replace(quoted_safe, f"({expr})")
            self._replaced.append(fid)
        return sql

    # ── AST 遍历与替换 ──

    def _transform_tree(self, tree: exp.Expression) -> None:
        """深度优先遍历 AST，替换所有虚拟字段引用。"""
        # ★ 收集作用域信息：CTE 名称和表别名映射
        self._collect_scope_info(tree)
        # 使用 sqlglot 的 transform 机制
        # transform 会深度优先遍历，对每个节点调用回调
        tree.transform(self._transform_node, copy=False)

    def _collect_scope_info(self, tree: exp.Expression) -> None:
        """收集 AST 中的 CTE 名称和表别名 → 源名称映射。

        用于判断某个 Column 引用是否来自派生数据源（CTE / 子查询），
        从而决定是否跳过虚拟字段展开。
        """
        self._cte_names = set()
        self._table_alias_to_source = {}

        # 1. 收集所有 CTE 名称
        for cte_node in tree.find_all(exp.CTE):
            alias = cte_node.alias
            if alias:
                self._cte_names.add(alias)

        # 2. 收集表别名 → 源名称映射
        #    FROM grouped AS g  →  g → "grouped"
        #    FROM (subquery) AS s  →  s → "__subquery__"
        for ta in tree.find_all(exp.TableAlias):
            parent = ta.parent
            alias_name = (
                ta.this.name if isinstance(ta.this, exp.Identifier) else str(ta.this)
            )
            if isinstance(parent, exp.Table):
                self._table_alias_to_source[alias_name] = parent.name
            elif isinstance(parent, exp.Subquery):
                self._table_alias_to_source[alias_name] = "__subquery__"

    def _is_from_derived_source(self, node: exp.Column) -> bool:
        """判断 Column 节点是否引用了派生数据源（CTE 或子查询）的输出列。

        判断逻辑：
        1. 有表前缀 (如 g.dim_product_category)：
           查找 g → 源名称 → 如果是 CTE 名称或子查询 → True
        2. 无表前缀 (如 dim_product_category)：
           找到所属 SELECT 的 FROM/JOIN 数据源，如果**所有**直接数据源
           都是 CTE 或子查询 → True（该 SELECT 层不直接接触原表）

        Returns:
            True 表示该 Column 引用的是派生源，不应展开虚拟字段
        """
        table_prefix = node.table

        if table_prefix:
            # 有表前缀：解析别名链
            source = self._table_alias_to_source.get(table_prefix, table_prefix)
            return source in self._cte_names or source == "__subquery__"

        # 无表前缀：检查所属 SELECT 的所有直接 FROM/JOIN 源
        enclosing_select = self._find_enclosing_select(node)
        if enclosing_select is None:
            return False

        sources = self._get_direct_from_sources(enclosing_select)
        if not sources:
            return False

        return all(
            self._is_derived_source_name(s) for s in sources
        )

    def _is_derived_source_name(self, name: str) -> bool:
        """判断一个数据源名称是否为派生源（CTE 或子查询）。"""
        if name in self._cte_names or name == "__subquery__":
            return True
        # 也检查别名映射：如果 name 本身是一个别名，解析其源
        source = self._table_alias_to_source.get(name, name)
        return source in self._cte_names or source == "__subquery__"

    @staticmethod
    def _find_enclosing_select(node: exp.Expression) -> exp.Select | None:
        """向上查找最近的 Select 祖先节点。"""
        parent = node.parent
        while parent is not None:
            if isinstance(parent, exp.Select):
                return parent
            parent = parent.parent
        return None

    @staticmethod
    def _get_direct_from_sources(select: exp.Select) -> list[str]:
        """获取 SELECT 的直接 FROM 和 JOIN 数据源名称。

        只检查直接的源（不穿透子查询内部），返回源名称列表：
        - Table → 表名
        - Subquery → "__subquery__"
        """
        sources: list[str] = []

        # FROM 子句
        frm = select.find(exp.From)
        if frm and frm.this:
            source_node = frm.this
            if isinstance(source_node, exp.Table):
                sources.append(source_node.name)
            elif isinstance(source_node, exp.Subquery):
                sources.append("__subquery__")

        # JOIN 子句（直接子节点）
        for join in select.find_all(exp.Join):
            # 只处理当前 SELECT 层的 JOIN，不穿透子查询
            # 检查 join 是否是 select 的直接子 JOIN
            if join.parent is not select and not isinstance(join.parent, exp.From):
                # 向上检查是否属于当前 select
                p = join.parent
                is_direct = False
                while p is not None:
                    if p is select:
                        is_direct = True
                        break
                    if isinstance(p, (exp.Select, exp.Subquery)):
                        # 穿过了另一个 SELECT 边界
                        break
                    p = p.parent
                if not is_direct:
                    continue

            join_source = join.this
            if isinstance(join_source, exp.Table):
                sources.append(join_source.name)
            elif isinstance(join_source, exp.Subquery):
                sources.append("__subquery__")

        return sources

    def _transform_node(self, node: exp.Expression) -> exp.Expression:
        """对单个 AST 节点进行转换。

        检查节点是否是虚拟字段引用（Column 类型且名称匹配映射表），
        如果是，替换为对应的真实表达式 AST。

        ★ 作用域感知：如果该 Column 引用的数据源是 CTE 或子查询（派生源），
        则跳过展开——因为虚拟字段名在派生源中可能是 AS 别名，
        展开后会破坏外层对 CTE 别名的引用。
        """
        if not isinstance(node, exp.Column):
            return node

        col_name = node.name
        if not col_name:
            return node

        # 检查 __vf__ 前缀标记（来自预处理的花括号占位符）
        fid = None
        if col_name.startswith("__vf__"):
            fid = col_name[6:]  # 去掉 __vf__ 前缀
        elif col_name in self._mappings:
            fid = col_name

        if fid is None or fid not in self._mappings:
            return node

        # ★ 作用域检查：如果 Column 引用的是派生数据源，跳过展开
        if self._is_from_derived_source(node):
            self._messages.append(
                f"跳过虚拟字段 '{fid}' 展开: 引用来自 CTE/子查询的输出列"
            )
            # 还原 __vf__ 前缀为原始字段名（保持 AS 别名正确）
            if col_name.startswith("__vf__"):
                node.set("this", exp.to_identifier(fid, quoted=True))
            return node

        mapping = self._mappings[fid]
        expr_str = _normalize_expr(mapping.expr)

        # 根据上下文位置决定替换策略
        context = self._detect_context(node)
        replacement = self._build_replacement(mapping, expr_str, context, node)

        if replacement is not None:
            self._replaced.append(fid)
            self._messages.append(
                f"替换虚拟字段 '{fid}' (位置: {context}, 类型: {mapping.field_type})"
            )
            return replacement

        return node

    def _detect_context(self, node: exp.Expression) -> str:
        """检测节点在 SQL 中的位置上下文。

        Returns:
            "select" / "where" / "group" / "order" / "having" / "join" / "other"
        """
        parent = node.parent
        while parent is not None:
            if isinstance(parent, exp.Where):
                return "where"
            if isinstance(parent, exp.Having):
                return "having"
            if isinstance(parent, exp.Group):
                return "group"
            if isinstance(parent, exp.Order):
                return "order"
            if isinstance(parent, exp.Join):
                return "join"
            if isinstance(parent, exp.Select):
                # 判断 node 在 SELECT 的哪个部分
                # 如果在 expressions 列表中，则是 SELECT 列
                sel_exprs = parent.expressions
                if sel_exprs and self._is_descendant_of_any(node, sel_exprs):
                    return "select"
                return "other"
            parent = parent.parent
        return "other"

    @staticmethod
    def _is_descendant_of_any(
        node: exp.Expression,
        ancestors: list[exp.Expression],
    ) -> bool:
        """检查 node 是否是 ancestors 中任一节点的后代。"""
        for anc in ancestors:
            current = node
            while current is not None:
                if current is anc:
                    return True
                current = current.parent
        return False

    def _build_replacement(
        self,
        mapping: VirtualFieldMapping,
        expr_str: str,
        context: str,
        original_node: exp.Column,
    ) -> exp.Expression | None:
        """根据虚拟字段类型和上下文位置，构建替换 AST 节点。"""
        fid = mapping.field_id

        # 获取或缓存解析后的 AST
        if fid not in self._parsed_cache:
            try:
                if mapping.field_type == "condition":
                    self._parsed_cache[fid] = _parse_condition_expr(expr_str)
                else:
                    # metric 和 column 都是列/聚合表达式
                    self._parsed_cache[fid] = _parse_column_expr(expr_str)
            except Exception as e:
                _log.warn(f"解析虚拟字段 '{fid}' 的表达式失败: {e}")
                return None

        parsed = self._parsed_cache[fid]

        if mapping.field_type == "condition":
            return self._replace_condition(parsed, context, original_node)
        elif mapping.field_type == "metric":
            return self._replace_metric(parsed, context, original_node)
        else:
            return self._replace_column(parsed, context, original_node)

    def _replace_condition(
        self,
        parsed_expr: exp.Expression,
        context: str,
        original_node: exp.Column,
    ) -> exp.Expression | None:
        """替换条件型虚拟字段。

        条件型字段（如 base_valid_data）通常出现在 WHERE/HAVING 中。
        用 Paren 包裹确保优先级正确。
        """
        # 深拷贝避免 AST 节点被多次引用时出问题
        replacement = parsed_expr.copy()
        return exp.Paren(this=replacement)

    def _replace_column(
        self,
        parsed_expr: exp.Expression,
        context: str,
        original_node: exp.Column,
    ) -> exp.Expression | None:
        """替换列型虚拟字段。

        列型字段（如 dim_product → "考核产品"）本质上是列名映射，
        替换时需要保留原始节点的 table 前缀（如 a./b.），
        确保 JOIN ON 条件中的表引用不丢失。
        """
        replacement = parsed_expr.copy()

        # ★ 如果替换结果本身是 Column 节点，保留原始节点的 table 前缀
        if isinstance(replacement, exp.Column) and original_node.table:
            replacement.set("table", exp.to_identifier(original_node.table))

        # 所有位置都直接返回（不加 Paren），column 类型是简单列名映射
        return replacement

    def _replace_metric(
        self,
        parsed_expr: exp.Expression,
        context: str,
        original_node: exp.Column,
    ) -> exp.Expression | None:
        """替换指标型虚拟字段。

        指标型字段（如 total_flow → CASE WHEN ... THEN col ELSE 0 END）是纯值表达式，
        系统根据 agg_type 自动包裹聚合函数（如 SUM）。

        聚合包裹策略：
        - agg_type = SUM/AVG/COUNT/MAX/MIN → 自动包裹（如 SUM(expr)），但若外层已有
          同名聚合函数则不重复包裹（防嵌套聚合）
        - agg_type = EXPR → 自包含表达式（派生指标），不做聚合包裹
        - agg_type = ""（空）→ 向后兼容，不做聚合包裹（expr 本身已含聚合函数）

        ★ 如果 original_node 带有 table 前缀（如 a."total_flow"），
        则递归为展开后表达式中所有 Column 节点添加相同的 table 前缀，
        确保 JOIN 场景下列引用不丢失表别名。
        """
        replacement = parsed_expr.copy()
        mapping = self._mappings.get(original_node.name.replace("__vf__", ""))
        if mapping is None:
            # fallback: 用 field_id 查找
            for fid, m in self._mappings.items():
                if original_node.name == fid or original_node.name == f"__vf__{fid}":
                    mapping = m
                    break

        # ★ 保留表别名前缀
        if original_node.table:
            table_id = exp.to_identifier(original_node.table)
            for col_node in replacement.find_all(exp.Column):
                if not col_node.table:
                    col_node.set("table", table_id.copy())

        # ★ 根据 agg_type 自动包裹聚合函数
        agg_type = (mapping.agg_type if mapping else "").upper().strip()
        if agg_type and agg_type != "EXPR":
            # 检测外层是否已有聚合函数包裹（防止 SUM(SUM(...)) 嵌套）
            outer_has_agg = self._outer_has_aggregate(original_node, agg_type)
            if outer_has_agg:
                self._messages.append(
                    f"检测到外层已有 {agg_type}() 包裹虚拟字段 "
                    f"'{original_node.name.replace('__vf__', '')}', 跳过自动聚合包裹"
                )
            else:
                # 包裹聚合函数：SUM(expr) / AVG(expr) / ...
                agg_func_cls = self._get_agg_func_class(agg_type)
                if agg_func_cls:
                    replacement = agg_func_cls(this=replacement)

        if context == "select":
            parent = original_node.parent
            if isinstance(parent, exp.Alias):
                return replacement
            return replacement

        # 其他位置（算术表达式、HAVING、ORDER BY 等）→ 括号包裹
        return exp.Paren(this=replacement)

    @staticmethod
    def _outer_has_aggregate(node: exp.Expression, agg_name: str) -> bool:
        """检测节点的直接父节点是否是指定的聚合函数调用。

        用于防止 SUM(metric_field) 展开后变成 SUM(SUM(CASE WHEN ...)) 的嵌套聚合问题。
        检查逻辑：向上遍历父节点链，跳过 Paren/Alias 等透明节点，
        如果遇到同名的聚合函数节点则返回 True。
        """
        agg_upper = agg_name.upper()
        parent = node.parent
        # 向上查找，跳过 Paren / Alias / Anonymous 等透明包裹
        while parent is not None:
            if isinstance(parent, (exp.Paren, exp.Alias)):
                parent = parent.parent
                continue
            # 检查是否是聚合函数
            if isinstance(parent, exp.AggFunc):
                parent_name = type(parent).__name__.upper()
                # sqlglot 中 SUM → exp.Sum, AVG → exp.Avg 等
                if parent_name == agg_upper:
                    return True
            # 不再继续向上（只检查直接的聚合包裹层）
            break
        return False

    @staticmethod
    def _get_agg_func_class(agg_type: str) -> type | None:
        """根据 agg_type 名称获取 sqlglot 的聚合函数节点类。"""
        agg_map = {
            "SUM": exp.Sum,
            "AVG": exp.Avg,
            "COUNT": exp.Count,
            "MAX": exp.Max,
            "MIN": exp.Min,
        }
        return agg_map.get(agg_type.upper())

    # ── 正则兜底 ──

    def _regex_fallback(self, sql: str) -> str:
        """AST 解析失败时的正则兜底替换。"""
        self._messages.append("使用正则兜底替换")
        result = sql

        for fid, mapping in self._mappings.items():
            expr = _normalize_expr(mapping.expr)

            # 1. 花括号占位符 {field_id}（向后兼容）
            placeholder = "{" + fid + "}"
            if placeholder in result:
                result = result.replace(placeholder, f"({expr})")
                self._replaced.append(fid)
                continue

            # 2. 双引号列引用 "field_id"（条件型、列型、指标型都支持）
            pattern = re.compile(
                rf'(?<!\bAS\s)"{re.escape(fid)}"',
                re.IGNORECASE,
            )
            if pattern.search(result):
                result = pattern.sub(f"({expr})", result)
                self._replaced.append(fid)
                continue

            # 3. 裸名称引用（metric / column 类型常见：LLM 直接写 total_flow 而非 "total_flow"）
            #    用 word boundary 匹配，排除 AS 后面的别名
            if mapping.field_type in ("metric", "column"):
                bare_pattern = re.compile(
                    rf'(?<!\bAS\s)(?<!")(?<!\w){re.escape(fid)}(?!\w)(?!")',
                    re.IGNORECASE,
                )
                if bare_pattern.search(result):
                    result = bare_pattern.sub(f"({expr})", result)
                    self._replaced.append(fid)

        return result


# ─────────────────────── 便捷函数 ───────────────────────


def build_mappings_from_filters(
    required_filters: list[dict[str, Any]],
) -> list[VirtualFieldMapping]:
    """从 required_filters 构建虚拟字段映射列表（向后兼容）。"""
    result: list[VirtualFieldMapping] = []
    for f in required_filters:
        fid = f.get("id", "")
        expr = f.get("expr", "")
        if fid and expr:
            result.append(VirtualFieldMapping(
                field_id=fid,
                expr=expr,
                field_type=f.get("field_type", "condition"),
                label=f.get("label", ""),
                description=f.get("description", ""),
                agg_type=f.get("agg_type", ""),
            ))
    return result


def build_mappings_from_tables_info(
    tables_info: list[dict[str, Any]] | None,
    state: Any = None,
) -> list[VirtualFieldMapping]:
    """从 tables_info / state 构建虚拟字段映射列表。
    
    扫描 required_filters 中的所有节点（condition / metric / column），
    为每种类型构建对应的 VirtualFieldMapping。
    """
    filters: list[dict[str, Any]] = []
    if tables_info:
        for tbl in tables_info:
            for f in (tbl.get("required_filters") or []):
                filters.append(f)
    elif state and getattr(state, "required_filters", None):
        filters = state.required_filters

    return build_mappings_from_filters(filters)


def expand_virtual_fields(
    sql: str,
    mappings: list[VirtualFieldMapping],
) -> ConversionResult:
    """一步完成：转换 SQL 中的虚拟字段。"""
    if not mappings:
        return ConversionResult(sql=sql)
    converter = VirtualFieldConverter(mappings)
    return converter.convert(sql)
