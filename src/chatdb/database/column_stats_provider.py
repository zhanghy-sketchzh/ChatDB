"""
列统计信息提供者（Column Stats Provider）

统一封装「实时扫描列 Schema + 统计信息」的能力，供各 Agent 统一调用。
取代散落在 DuckDBConnector / SQLTool / SemanticParser / Orchestrator 中的重复逻辑。

设计原则：
- 扫描能力与数据库引擎无关（通过 execute_fn 注入）
- 格式化输出统一、可复用
- 分类字段只展示高频 top-N 常见值
- 数值字段展示范围/均值/中位数 + 缺失率
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from lib.utils.logger import get_component_logger

_log = get_component_logger("ColumnStatsProvider")

# ─────────────────────── 数据结构 ───────────────────────

# 被视为缺失的字符串值
_NULL_STRINGS = ("None", "none", "null", "NULL", "Null", "", "为空", "NA", "N/A", "na", "n/a")

# DuckDB / 通用 SQL 类型分类
_TEXT_TYPES = {"VARCHAR", "TEXT", "STRING"}
_NUMERIC_TYPES = {
    "BIGINT", "INTEGER", "SMALLINT", "TINYINT",
    "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT", "NUMERIC",
}
_DATE_TYPES = {"DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "DATETIME"}


@dataclass
class ColumnStats:
    """单列的统计信息（不可变值对象）。"""

    name: str
    dtype: str  # 原始数据库类型，如 VARCHAR、BIGINT

    # 基础统计
    null_pct: float = 0.0  # 缺失率 (%)，含伪空值
    unique_count: int = 0

    # 数值字段
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    median_val: float | None = None

    # 分类字段：高频常见值 [(value, count), ...]
    top_values: list[tuple[str, int]] = field(default_factory=list)

    # ── 便捷属性 ──

    @property
    def is_numeric(self) -> bool:
        return self._base_type in _NUMERIC_TYPES

    @property
    def is_text(self) -> bool:
        return self._base_type in _TEXT_TYPES

    @property
    def is_date(self) -> bool:
        return self._base_type in _DATE_TYPES

    @property
    def _base_type(self) -> str:
        """去掉括号参数后的大写类型名，如 DECIMAL(18,2) → DECIMAL。"""
        return self.dtype.split("(")[0].strip().upper()

    # ── 格式化 ──

    def to_summary(self) -> str:
        """生成单行文本摘要。"""
        if self.is_numeric and self.min_val is not None:
            return (
                f"范围[{self.min_val:.2f}~{self.max_val:.2f}], "
                f"均值{self.mean_val:.2f}, 中位数{self.median_val:.2f}"
            )
        if self.top_values:
            vals = ", ".join(f"'{v}'" for v, _ in self.top_values)
            suffix = f"...共{self.unique_count}种" if self.unique_count > len(self.top_values) else ""
            return f"常见值: {vals}{suffix}"
        if self.unique_count:
            return f"唯一值{self.unique_count}个"
        return ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（兼容旧接口）。"""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.dtype,
            "null_pct": round(self.null_pct, 2),
            "unique_count": self.unique_count,
            "summary": self.to_summary(),
        }
        if self.min_val is not None:
            d["stats"] = {
                "min": self.min_val,
                "max": self.max_val,
                "mean": self.mean_val,
                "median": self.median_val,
            }
        if self.top_values:
            d["top_values"] = self.top_values
        return d

    def format_line(self, show_null_pct: bool = True, max_summary_len: int = 120) -> str:
        """格式化为 prompt 单行：`- "列名" (类型) [缺失:X%] [唯一值:N] -- 摘要`。"""
        line = f'- "{self.name}" ({self.dtype})'

        # 数值字段始终展示缺失率；文本字段缺失率 > 0 时展示
        if show_null_pct and self.null_pct > 0:
            line += f" [缺失:{self.null_pct:.1f}%]"

        if self.unique_count:
            line += f" [唯一值:{self.unique_count}]"

        summary = self.to_summary()
        if summary:
            if len(summary) > max_summary_len:
                summary = summary[:max_summary_len - 3] + "..."
            line += f" -- {summary}"

        return line


# ─────────────────────── Provider 核心类 ───────────────────────


# 执行 SQL 的回调类型：接收 SQL 字符串，返回 list[dict]
SyncExecuteFn = Callable[[str], list[dict[str, Any]]]


class ColumnStatsProvider:
    """实时列统计信息提供者。

    封装了「扫描列 Schema + 统计信息」的完整能力：
    1. 基础统计：缺失率（含伪空值）、唯一值计数
    2. 数值列：MIN / MAX / AVG / MEDIAN
    3. 分类列：高频 top-N 常见值（默认前 10）

    用法::

        provider = ColumnStatsProvider(execute_fn=conn.execute_sync)
        stats = provider.scan_table("my_table")
        prompt_text = ColumnStatsProvider.format_columns(stats, relevant_cols={"年", "月"})
    """

    def __init__(
        self,
        execute_fn: SyncExecuteFn,
        top_k: int = 10,
        max_unique_for_topk: int = 500,
    ):
        """
        Args:
            execute_fn: 同步 SQL 执行函数，签名 (sql: str) -> list[dict]
            top_k: 分类字段展示的高频常见值数量
            max_unique_for_topk: 唯一值超过此数的文本列不扫描常见值
        """
        self._execute = execute_fn
        self._top_k = top_k
        self._max_unique_for_topk = max_unique_for_topk

    # ── 扫描 ──

    def scan_table(self, table_name: str) -> list[ColumnStats]:
        """实时扫描指定表，返回所有列的统计信息。"""
        # 1. 获取列名 + 类型
        desc_rows = self._execute(f'DESCRIBE "{table_name}"')
        if not desc_rows:
            return []

        col_info: list[tuple[str, str]] = []
        for row in desc_rows:
            # DESCRIBE 返回的 dict 可能有不同 key
            name = row.get("column_name") or row.get("Field") or list(row.values())[0]
            dtype = row.get("column_type") or row.get("Type") or list(row.values())[1]
            col_info.append((str(name), str(dtype)))

        if not col_info:
            return []

        col_names = [c for c, _ in col_info]
        col_types = {c: t for c, t in col_info}

        # 2. 基础统计（一条 SQL）
        null_in_clause = ", ".join(f"'{s}'" for s in _NULL_STRINGS)
        parts: list[str] = []
        for i, c in enumerate(col_names):
            parts.append(f'COUNT("{c}") AS "cnt_{i}"')
            parts.append(f'COUNT(DISTINCT "{c}") AS "uq_{i}"')
            base_type = col_types[c].split("(")[0].strip().upper()
            if base_type in _TEXT_TYPES:
                parts.append(
                    f'SUM(CASE WHEN "{c}" IN ({null_in_clause}) THEN 1 ELSE 0 END) AS "str_null_{i}"'
                )
            else:
                parts.append(f'0 AS "str_null_{i}"')

        agg_sql = f'SELECT COUNT(*) AS total, {", ".join(parts)} FROM "{table_name}"'
        agg_rows = self._execute(agg_sql)
        if not agg_rows:
            return []

        agg = agg_rows[0]
        total = agg.get("total") or list(agg.values())[0] or 0

        base_stats: dict[str, dict[str, Any]] = {}
        for i, c in enumerate(col_names):
            non_null = list(agg.values())[1 + i * 3] or 0
            unique = list(agg.values())[1 + i * 3 + 1] or 0
            str_null = list(agg.values())[1 + i * 3 + 2] or 0
            effective_non_null = max(non_null - str_null, 0)
            null_pct = round((1 - effective_non_null / total) * 100, 2) if total > 0 else 0.0
            base_stats[c] = {"null_pct": null_pct, "unique_count": int(unique)}

        # 3. 数值列：MIN / MAX / AVG / MEDIAN
        num_cols = [c for c in col_names if col_types[c].split("(")[0].strip().upper() in _NUMERIC_TYPES]
        num_stats: dict[str, dict[str, float]] = {}
        if num_cols:
            stat_parts = [
                f'MIN("{c}"), MAX("{c}"), AVG("{c}"), MEDIAN("{c}")' for c in num_cols
            ]
            stat_sql = f'SELECT {", ".join(stat_parts)} FROM "{table_name}"'
            stat_rows = self._execute(stat_sql)
            if stat_rows:
                vals = list(stat_rows[0].values())
                for idx, c in enumerate(num_cols):
                    off = idx * 4
                    num_stats[c] = {
                        "min": vals[off], "max": vals[off + 1],
                        "mean": vals[off + 2], "median": vals[off + 3],
                    }

        # 4. 文本列：高频 top-K 常见值
        text_cols = [
            c for c in col_names
            if col_types[c].split("(")[0].strip().upper() in _TEXT_TYPES
            and base_stats[c]["unique_count"] <= self._max_unique_for_topk
        ]
        text_top: dict[str, list[tuple[str, int]]] = {}
        for c in text_cols:
            try:
                tv_sql = (
                    f'SELECT "{c}", COUNT(*) AS cnt FROM "{table_name}" '
                    f'WHERE "{c}" IS NOT NULL '
                    f'GROUP BY "{c}" ORDER BY cnt DESC LIMIT {self._top_k}'
                )
                rows = self._execute(tv_sql)
                text_top[c] = [
                    (str(list(r.values())[0]), int(list(r.values())[1]))
                    for r in rows
                ]
            except Exception:
                pass

        # 5. 组装 ColumnStats 列表
        result: list[ColumnStats] = []
        for c in col_names:
            b = base_stats[c]
            cs = ColumnStats(
                name=c,
                dtype=col_types[c],
                null_pct=b["null_pct"],
                unique_count=b["unique_count"],
            )
            if c in num_stats:
                s = num_stats[c]
                cs.min_val = s["min"]
                cs.max_val = s["max"]
                cs.mean_val = s["mean"]
                cs.median_val = s["median"]
            if c in text_top:
                cs.top_values = text_top[c]
            result.append(cs)

        return result

    # ── 格式化（静态方法，不依赖实例） ──

    @staticmethod
    def format_columns(
        stats_list: list[ColumnStats],
        columns: list[dict[str, Any]] | None = None,
        relevant_cols: set[str] | None = None,
        max_summary_len: int = 120,
    ) -> str:
        """将统计信息格式化为 prompt 文本。

        支持两种模式：
        1. 有 relevant_cols：区分「关键列」和「其他列」
        2. 无 relevant_cols：统一列出所有列

        Args:
            stats_list: scan_table 返回的统计列表
            columns: 基础列信息（可选，用于补充没有统计的列）
            relevant_cols: 与当前查询相关的列名集合
            max_summary_len: 摘要截断长度
        """
        if not stats_list and not columns:
            return "无列信息"

        stats_map = {s.name: s for s in stats_list}

        # 合并列列表：优先用 stats 的列顺序，补充 columns 中额外的列
        all_col_names: list[str] = [s.name for s in stats_list]
        if columns:
            for col in columns:
                cn = col.get("name", col.get("column_name", ""))
                if cn and cn not in stats_map:
                    all_col_names.append(cn)

        if not relevant_cols:
            # 模式 1：不区分，统一展示
            lines: list[str] = []
            for cn in all_col_names:
                st = stats_map.get(cn)
                if st:
                    lines.append(st.format_line(max_summary_len=max_summary_len))
                else:
                    # 回退到基础列信息
                    col_type = _find_col_type(columns, cn)
                    lines.append(f'- "{cn}" ({col_type})')
            return "\n".join(lines)

        # 模式 2：区分关键列和其他列
        key_lines: list[str] = []
        other_lines: list[str] = []
        for cn in all_col_names:
            st = stats_map.get(cn)
            if cn in relevant_cols:
                if st:
                    key_lines.append(st.format_line(max_summary_len=max_summary_len))
                else:
                    col_type = _find_col_type(columns, cn)
                    key_lines.append(f'- "{cn}" ({col_type})')
            else:
                if st:
                    other_lines.append(st.format_line(
                        show_null_pct=False,
                        max_summary_len=80,
                    ))
                else:
                    col_type = _find_col_type(columns, cn)
                    other_lines.append(f'- "{cn}" ({col_type})')

        parts: list[str] = []
        if key_lines:
            parts.append("#### 关键列（与本次查询直接相关）")
            parts.extend(key_lines)
        if other_lines:
            parts.append(f"\n#### 其他可用列（共 {len(other_lines)} 列）")
            parts.extend(other_lines)

        return "\n".join(parts)

    @staticmethod
    def stats_to_dicts(stats_list: list[ColumnStats]) -> list[dict[str, Any]]:
        """将 ColumnStats 列表序列化为 dict 列表（兼容旧接口）。"""
        return [s.to_dict() for s in stats_list]


# ─────────────────────── 辅助函数 ───────────────────────


def _find_col_type(columns: list[dict[str, Any]] | None, col_name: str) -> str:
    """从基础列信息中查找类型。"""
    if not columns:
        return ""
    for col in columns:
        cn = col.get("name", col.get("column_name", ""))
        if cn == col_name:
            return col.get("type", col.get("column_type", ""))
    return ""
