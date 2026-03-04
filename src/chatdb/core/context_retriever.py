"""
上下文检索器（Context Retriever）

对齐 Context Engineering 理念，实现渐进式上下文注入：
1. Schema 层面：BM25 召回高相关表/列，缩小 LLM 搜索空间
2. 值层面：关键词 → 枚举值匹配，降低 WHERE 条件幻觉
3. Few-shot 层面：相似问题 + SQL 示例召回（预留，依赖 ExampleVectorStore）

设计原则（来自 filesystem-context skill）：
- Progressive Disclosure：先加载元数据列表，需要时再加载细节
- Observation Masking：只给 LLM 传摘要 + 引用，不传全量数据
- 零入侵：检索结果作为额外上下文段注入，不修改已有 prompt 结构
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lib.utils.logger import get_component_logger


@dataclass
class RetrievalResult:
    """检索结果（统一格式，供各 Agent prompt 注入）"""

    # Schema 召回
    relevant_tables: list[dict[str, Any]] = field(default_factory=list)
    # [{table_name, score, description}]

    relevant_columns: dict[str, list[str]] = field(default_factory=dict)
    # {table_name: [col1, col2, ...]}

    # 值召回
    value_matches: list[dict[str, Any]] = field(default_factory=list)
    # [{keyword, matched_value, table_name, column_name, score}]

    # Few-shot 召回
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    # [{question, sql, explanation?}]

    # 实时列描述统计（由 DuckDB get_column_stats 注入）
    # {table_name: [{"name", "type", "null_pct", "unique_count", "summary", "stats"}, ...]}
    column_stats_map: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def has_schema_hints(self) -> bool:
        return bool(self.relevant_tables or self.relevant_columns)

    def has_value_hints(self) -> bool:
        return bool(self.value_matches)

    def has_few_shot(self) -> bool:
        return bool(self.few_shot_examples)

    def is_empty(self) -> bool:
        return not (self.has_schema_hints() or self.has_value_hints() or self.has_few_shot())

    # ── Prompt 格式化 ──

    def format_schema_hint(self, max_tables: int = 5, max_cols_per_table: int = 10) -> str:
        """格式化 Schema 召回结果为 prompt 段落（含实时描述统计）"""
        if not self.has_schema_hints():
            return ""

        lines = ["## 检索提示（BM25 召回）"]

        if self.relevant_tables:
            lines.append("### 高相关表")
            for t in self.relevant_tables[:max_tables]:
                desc = t.get("description", "")
                score = t.get("score", 0)
                desc_part = f" — {desc}" if desc else ""
                lines.append(f"- {t['table_name']} (相关度:{score:.2f}){desc_part}")

        if self.relevant_columns:
            lines.append("### 高相关列")
            for tbl, cols in list(self.relevant_columns.items())[:max_tables]:
                stats_list = self.column_stats_map.get(tbl)
                if stats_list:
                    from chatdb.database.column_stats_provider import ColumnStats, ColumnStatsProvider

                    stats_idx = {s["name"]: s for s in stats_list}
                    stats_objs: list[ColumnStats] = []
                    for c in cols[:max_cols_per_table]:
                        st = stats_idx.get(c)
                        if st:
                            cs = ColumnStats(
                                name=st.get("name", ""),
                                dtype=st.get("type", ""),
                                null_pct=st.get("null_pct", 0.0),
                                unique_count=st.get("unique_count", 0),
                            )
                            s = st.get("stats")
                            if s:
                                cs.min_val = s.get("min")
                                cs.max_val = s.get("max")
                                cs.mean_val = s.get("mean")
                                cs.median_val = s.get("median")
                            if st.get("top_values"):
                                cs.top_values = st["top_values"]
                            stats_objs.append(cs)
                        else:
                            stats_objs.append(ColumnStats(name=c, dtype=""))

                    col_lines = [f"  {cs.format_line()}" for cs in stats_objs]
                    lines.append(f"- {tbl}:")
                    lines.extend(col_lines)
                else:
                    cols_str = ", ".join(f'"{c}"' for c in cols[:max_cols_per_table])
                    lines.append(f"- {tbl}: {cols_str}")

        return "\n".join(lines)

    def format_value_hint(self, max_matches: int = 30) -> str:
        """格式化值召回结果为 prompt 段落"""
        if not self.has_value_hints():
            return ""

        lines = ["## 值匹配提示（关键词 → 枚举值）"]
        lines.append("以下是从数据库索引中匹配到的候选值，写 WHERE 条件时优先使用：")

        # 先按 table.column 分组（遍历所有匹配，不截断源数据）
        grouped: dict[str, list[dict[str, Any]]] = {}
        for m in self.value_matches:
            key = f'{m["table_name"]}."{m["column_name"]}"'
            grouped.setdefault(key, []).append(m)

        total = 0
        for col_key, matches in grouped.items():
            if total >= max_matches:
                break
            values = [m["matched_value"] for m in matches]
            # 去重并截取
            unique_values = list(dict.fromkeys(values))[:8]
            lines.append(f"- {col_key}: {', '.join(repr(v) for v in unique_values)}")
            total += len(unique_values)

        return "\n".join(lines)

    def format_few_shot(self, max_examples: int = 3) -> str:
        """格式化 few-shot 示例为 prompt 段落"""
        if not self.has_few_shot():
            return ""

        lines = ["## 参考示例（相似问题 + SQL）"]
        for ex in self.few_shot_examples[:max_examples]:
            lines.append(f"**问题**: {ex.get('question', '')}")
            lines.append(f"```sql\n{ex.get('sql', '').strip()}\n```")
            if ex.get("explanation"):
                lines.append(f"说明: {ex['explanation']}")
            lines.append("")

        return "\n".join(lines)


class ContextRetriever:
    """
    统一检索门面

    封装 TextIndex (BM25) + ExampleVectorStore，提供三级检索：
    1. retrieve_schema_hints()  — 表/列召回
    2. retrieve_value_hints()   — 值召回
    3. retrieve_few_shot()      — 示例召回
    4. retrieve_all()           — 一次性全量召回（用于 process_query 入口）
    """

    def __init__(
        self,
        text_index: Any | None = None,
        example_store: Any | None = None,
    ):
        """
        Args:
            text_index: TextIndex 实例（BM25 检索），None 则跳过关键词检索
            example_store: ExampleVectorStore 实例，None 则跳过 few-shot 检索
        """
        self._text_index = text_index
        self._example_store = example_store
        self._log = get_component_logger("ContextRetriever")

    @property
    def has_text_index(self) -> bool:
        return self._text_index is not None

    @property
    def has_example_store(self) -> bool:
        return self._example_store is not None

    # ── 分层检索 API ──

    def retrieve_schema_hints(
        self,
        query: str,
        table_name: str | None = None,
        top_k_tables: int = 3,
        top_k_columns: int = 10,
    ) -> RetrievalResult:
        """
        Schema 层检索：召回高相关表和列

        用于 SemanticParser / Planner 的 prompt 增强。
        """
        result = RetrievalResult()
        if not self.has_text_index:
            return result

        try:
            # 1. 表级召回
            table_results = self._text_index.search_tables(query, top_k_tables)
            result.relevant_tables = [
                {
                    "table_name": r.table_name,
                    "score": r.score,
                    "description": r.metadata.get("description", ""),
                }
                for r in table_results
            ]

            # 2. 列级召回
            col_results = self._text_index.search_columns(query, table_name, top_k_columns)
            cols_by_table: dict[str, list[str]] = {}
            for r in col_results:
                cols_by_table.setdefault(r.table_name, []).append(r.column_name or "")
            result.relevant_columns = cols_by_table

            self._log.debug(
                f"Schema 召回: {len(result.relevant_tables)} 表, "
                f"{sum(len(v) for v in result.relevant_columns.values())} 列"
            )
        except Exception as e:
            self._log.warn(f"Schema 召回失败: {e}")

        return result

    def retrieve_value_hints(
        self,
        query: str,
        top_k_keywords: int = 5,
        top_k_values: int = 10,
    ) -> RetrievalResult:
        """
        值层检索：关键词 → 枚举值匹配

        用于 SQLTool 的 WHERE 条件增强。
        """
        result = RetrievalResult()
        if not self.has_text_index:
            return result

        try:
            matches = self._text_index.match_keywords_to_values(
                query, top_k_keywords, top_k_values,
            )
            result.value_matches = [
                {
                    "keyword": m.keyword,
                    "matched_value": m.matched_value,
                    "table_name": m.table_name,
                    "column_name": m.column_name,
                    "score": m.score,
                }
                for m in matches
            ]
            self._log.debug(f"值召回: {len(result.value_matches)} 匹配")
        except Exception as e:
            self._log.warn(f"值召回失败: {e}")

        return result

    def retrieve_few_shot(
        self,
        query: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        Few-shot 检索：相似问题 + SQL 示例

        用于 Planner / SQLTool 的 prompt 增强。
        """
        result = RetrievalResult()
        if not self.has_example_store:
            return result

        try:
            examples = self._example_store.get_few_shot_examples(query, top_k)
            result.few_shot_examples = examples
            self._log.debug(f"Few-shot 召回: {len(examples)} 示例")
        except Exception as e:
            self._log.warn(f"Few-shot 召回失败: {e}")

        return result

    def retrieve_all(
        self,
        query: str,
        table_name: str | None = None,
        top_k_tables: int = 3,
        top_k_columns: int = 10,
        top_k_keywords: int = 10,
        top_k_values: int = 10,
        top_k_few_shot: int = 3,
    ) -> RetrievalResult:
        """
        一次性全量检索（合并三级结果）

        在 Orchestrator.process_query() 入口调用一次，
        结果存入 state.retrieval_context，后续各 Agent 按需取用。
        """
        schema = self.retrieve_schema_hints(query, table_name, top_k_tables, top_k_columns)
        values = self.retrieve_value_hints(query, top_k_keywords, top_k_values)
        few_shot = self.retrieve_few_shot(query, top_k_few_shot)

        # 合并
        merged = RetrievalResult(
            relevant_tables=schema.relevant_tables,
            relevant_columns=schema.relevant_columns,
            value_matches=values.value_matches,
            few_shot_examples=few_shot.few_shot_examples,
        )

        if not merged.is_empty():
            self._log.info(
                f"检索完成: {len(merged.relevant_tables)} 表, "
                f"{sum(len(v) for v in merged.relevant_columns.values())} 列, "
                f"{len(merged.value_matches)} 值匹配, "
                f"{len(merged.few_shot_examples)} 示例"
            )

        return merged
