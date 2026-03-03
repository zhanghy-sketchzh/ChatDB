"""
虚拟字段模块（Virtual Field）

整合虚拟字段相关的检索和 Prompt 构建能力：
1. VirtualFieldRetriever: 基于 TF-IDF 的虚拟字段检索器
2. VirtualFieldPromptBuilder: SQL 生成 prompt 中的虚拟字段说明段落生成器
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from chatdb.utils.logger import get_component_logger


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class RetrievedVirtualField:
    """单个召回的虚拟字段"""
    id: str
    description: str = ""
    field_type: str = ""       # condition / column / metric
    scope: str = "optional"    # required / default / optional
    expr: str = ""
    synonyms: list[str] = field(default_factory=list)
    score: float = 0.0
    # 额外信息
    group: str = ""
    unit: str = ""
    column: str = ""           # column 类型的真实列名

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "field_type": self.field_type,
            "scope": self.scope,
            "expr": self.expr,
            "score": round(self.score, 4),
        }
        if self.synonyms:
            d["synonyms"] = self.synonyms
        if self.group:
            d["group"] = self.group
        if self.unit:
            d["unit"] = self.unit
        if self.column:
            d["column"] = self.column
        return d


@dataclass
class VirtualFieldRetrievalResult:
    """虚拟字段检索结果"""
    fields: list[RetrievedVirtualField] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.fields

    def format_prompt_section(self) -> str:
        """格式化为 prompt 注入段落，包含完整信息"""
        if not self.fields:
            return ""

        lines = [
            "## 检索召回的虚拟字段（可能与用户问题相关，按相关度降序）",
            "",
            "以下虚拟字段根据用户问题自动检索召回，**可能会用到**，请参考：",
            "",
        ]

        # 按 field_type 分组展示
        cond_fields = [f for f in self.fields if f.field_type == "condition"]
        metric_fields = [f for f in self.fields if f.field_type == "metric"]
        column_fields = [f for f in self.fields if f.field_type == "column"]

        if cond_fields:
            lines.append("### 条件型（condition — WHERE 筛选）")
            lines.append("")
            lines.append("| ID | 描述 | scope | 同义词 | 展开表达式 | 相关度 |")
            lines.append("|-----|------|-------|--------|-----------|--------|")
            for f in cond_fields:
                syns = ", ".join(f.synonyms) if f.synonyms else ""
                expr_display = _compact_expr(f.expr)
                lines.append(
                    f"| `{f.id}` | {f.description} | {f.scope} "
                    f"| {syns} | `{expr_display}` | {f.score:.2f} |"
                )
            lines.append("")

        if metric_fields:
            lines.append("### 指标型（metric — SELECT 聚合）")
            lines.append("")
            lines.append("| ID | 描述 | scope | 同义词 | 展开表达式 | 相关度 |")
            lines.append("|-----|------|-------|--------|-----------|--------|")
            for f in metric_fields:
                syns = ", ".join(f.synonyms) if f.synonyms else ""
                expr_display = _compact_expr(f.expr)
                unit_str = f" ({f.unit})" if f.unit else ""
                lines.append(
                    f"| `{f.id}` | {f.description}{unit_str} | {f.scope} "
                    f"| {syns} | `{expr_display}` | {f.score:.2f} |"
                )
            lines.append("")

        if column_fields:
            lines.append("### 维度列（column — GROUP BY / ORDER BY）")
            lines.append("")
            lines.append("| ID | 描述 | scope | 同义词 | 真实列名 | 相关度 |")
            lines.append("|-----|------|-------|--------|---------|--------|")
            for f in column_fields:
                syns = ", ".join(f.synonyms) if f.synonyms else ""
                col = f.column or f.expr
                lines.append(
                    f"| `{f.id}` | {f.description} | {f.scope} "
                    f"| {syns} | `{col}` | {f.score:.2f} |"
                )
            lines.append("")

        return "\n".join(lines)


def _compact_expr(expr: str, max_len: int = 80) -> str:
    """将多行 expr 压缩为单行显示"""
    if not expr:
        return ""
    single = " ".join(line.strip() for line in expr.split("\n") if line.strip())
    if len(single) > max_len:
        return single[:max_len - 3] + "..."
    return single


# =============================================================================
# 分词工具
# =============================================================================

# 中文分词简单实现：将字符串拆为连续的中文字符串 + 英文/数字 token
_TOKEN_PATTERN = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf]+'   # 连续中文字符
    r'|[a-zA-Z_][a-zA-Z0-9_]*'         # 英文标识符
    r'|\d+'                              # 数字
)


def _tokenize(text: str) -> list[str]:
    """轻量级分词：中文按字切分（bigram），英文按单词切分"""
    tokens: list[str] = []
    for m in _TOKEN_PATTERN.finditer(text.lower()):
        seg = m.group()
        if '\u4e00' <= seg[0] <= '\u9fff' or '\u3400' <= seg[0] <= '\u4dbf':
            # 中文：unigram + bigram
            for ch in seg:
                tokens.append(ch)
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i+2])
        else:
            tokens.append(seg)
    return tokens


# =============================================================================
# VirtualFieldRetriever — 虚拟字段检索器
# =============================================================================

class VirtualFieldRetriever:
    """
    虚拟字段检索器

    基于 TF-IDF 风格的关键词匹配，将用户查询与虚拟字段的
    description + synonyms + expr 做相似度计算。
    """

    def __init__(self, yml_config: dict[str, Any] | None = None):
        self._log = get_component_logger("VFRetriever")
        self._field_docs: list[dict[str, Any]] = []  # [{id, field_def, tokens, text}]
        self._idf: dict[str, float] = {}
        if yml_config:
            self.build_index(yml_config)

    def build_index(self, yml_config: dict[str, Any]) -> None:
        """从 yml_config 构建虚拟字段索引"""
        virtual_fields = yml_config.get("virtual_fields", {})
        if not virtual_fields:
            self._log.debug("YAML 配置中无 virtual_fields，跳过索引构建")
            return

        self._field_docs = []
        doc_freq: dict[str, int] = {}

        for fid, fdef in virtual_fields.items():
            if not isinstance(fdef, dict):
                continue

            # 构建检索文本：description + synonyms + expr
            parts = []
            desc = fdef.get("description", "")
            if desc:
                parts.append(desc)
            for syn in fdef.get("synonyms", []):
                parts.append(syn)
            expr = fdef.get("expr", "")
            if expr:
                parts.append(expr)
            # 也加入 id 本身
            parts.append(fid)

            text = " ".join(parts)
            tokens = _tokenize(text)
            token_set = set(tokens)

            self._field_docs.append({
                "id": fid,
                "field_def": fdef,
                "tokens": tokens,
                "token_set": token_set,
                "text": text,
            })

            for tok in token_set:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        # 计算 IDF
        n = len(self._field_docs)
        if n > 0:
            self._idf = {
                tok: math.log((n + 1) / (df + 1)) + 1
                for tok, df in doc_freq.items()
            }

        self._log.info(f"虚拟字段索引构建完成: {n} 个字段, {len(self._idf)} 个 token")

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        min_score: float = 0.1,
    ) -> VirtualFieldRetrievalResult:
        """
        检索与用户查询最相关的虚拟字段

        Args:
            query: 用户查询
            top_k: 最大返回数量
            min_score: 最低相关度阈值

        Returns:
            VirtualFieldRetrievalResult，按得分降序排列
        """
        if not self._field_docs:
            return VirtualFieldRetrievalResult()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return VirtualFieldRetrievalResult()

        # 计算 query TF
        query_tf: dict[str, float] = {}
        for tok in query_tokens:
            query_tf[tok] = query_tf.get(tok, 0) + 1
        # 归一化
        max_tf = max(query_tf.values()) if query_tf else 1
        query_tfidf: dict[str, float] = {
            tok: (tf / max_tf) * self._idf.get(tok, 1.0)
            for tok, tf in query_tf.items()
        }

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._field_docs:
            score = self._score_document(query_tfidf, query_tokens, doc)
            if score >= min_score:
                scored.append((score, doc))

        # 按得分降序排列
        scored.sort(key=lambda x: -x[0])

        # 确保 required/default scope 的字段始终被召回
        result_ids = set()
        results: list[RetrievedVirtualField] = []

        for score, doc in scored[:top_k]:
            fid = doc["id"]
            fdef = doc["field_def"]
            results.append(self._build_result_field(fid, fdef, score))
            result_ids.add(fid)

        # 补充 required/default scope 字段（即使相关度不高也必须包含）
        for doc in self._field_docs:
            fid = doc["id"]
            if fid in result_ids:
                continue
            fdef = doc["field_def"]
            scope = fdef.get("scope", "optional")
            if scope in ("required", "default"):
                results.append(self._build_result_field(fid, fdef, 0.0))
                result_ids.add(fid)

        self._log.info(
            f"虚拟字段检索: query='{query[:40]}...' → {len(results)} 个字段召回"
        )
        return VirtualFieldRetrievalResult(fields=results)

    def _score_document(
        self,
        query_tfidf: dict[str, float],
        query_tokens: list[str],
        doc: dict[str, Any],
    ) -> float:
        """计算查询与文档的相似度得分"""
        doc_tokens = doc["tokens"]
        doc_token_set = doc["token_set"]

        if not doc_tokens:
            return 0.0

        # TF-IDF 余弦相似度（简化版）
        doc_tf: dict[str, float] = {}
        for tok in doc_tokens:
            doc_tf[tok] = doc_tf.get(tok, 0) + 1
        max_doc_tf = max(doc_tf.values()) if doc_tf else 1

        dot_product = 0.0
        query_norm = 0.0
        doc_norm = 0.0

        all_tokens = set(query_tfidf.keys()) | doc_token_set
        for tok in all_tokens:
            q_w = query_tfidf.get(tok, 0.0)
            d_tf = doc_tf.get(tok, 0) / max_doc_tf
            d_w = d_tf * self._idf.get(tok, 1.0)

            dot_product += q_w * d_w
            query_norm += q_w * q_w
            doc_norm += d_w * d_w

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        cosine = dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))

        # 额外 boost：同义词精确匹配
        fdef = doc["field_def"]
        synonyms = [s.lower() for s in fdef.get("synonyms", [])]
        query_lower = query_tokens  # 已经是 lower
        synonym_boost = 0.0
        for syn in synonyms:
            syn_tokens = _tokenize(syn)
            if syn_tokens and all(t in query_lower for t in syn_tokens):
                synonym_boost = max(synonym_boost, 0.3)

        # description 关键词匹配 boost
        desc = fdef.get("description", "").lower()
        desc_tokens = _tokenize(desc)
        if desc_tokens:
            overlap = sum(1 for t in desc_tokens if t in set(query_tokens))
            if overlap > 0:
                synonym_boost += 0.1 * min(overlap, 3)

        return cosine + synonym_boost

    @staticmethod
    def _build_result_field(
        fid: str,
        fdef: dict[str, Any],
        score: float,
    ) -> RetrievedVirtualField:
        return RetrievedVirtualField(
            id=fid,
            description=fdef.get("description", ""),
            field_type=fdef.get("field_type", ""),
            scope=fdef.get("scope", "optional"),
            expr=fdef.get("expr", "").strip(),
            synonyms=fdef.get("synonyms", []),
            score=score,
            group=fdef.get("group", ""),
            unit=fdef.get("unit", ""),
            column=fdef.get("column", ""),
        )


# =============================================================================
# VirtualFieldPromptBuilder — 虚拟字段 Prompt 构建器
# =============================================================================

class VirtualFieldPromptBuilder:
    """虚拟字段 prompt 段落生成器"""

    @staticmethod
    def build(
        required_filters: list[dict[str, Any]],
        metric_name: str = "",
    ) -> str:
        """构建完整的虚拟字段说明段落。

        Args:
            required_filters: 虚拟字段列表（包含 condition/column/metric 三种类型）
            metric_name: 当前任务的主指标 ID（用于上下文提示）

        Returns:
            完整的虚拟字段说明 Markdown 文本
        """
        builder = VirtualFieldPromptBuilder()
        return builder._build_full_section(required_filters, metric_name)

    def _build_full_section(
        self,
        required_filters: list[dict[str, Any]],
        metric_name: str = "",
    ) -> str:
        """内部方法：构建完整段落"""
        # ── 1. 分类：条件型 vs 列型 vs 指标型 ──
        condition_fields: list[dict[str, Any]] = []
        column_fields: list[dict[str, Any]] = []
        metric_fields: list[dict[str, Any]] = []
        for f in required_filters:
            ft = f.get('field_type', 'condition')
            if ft == 'column':
                column_fields.append(f)
            elif ft == 'metric':
                metric_fields.append(f)
            else:
                condition_fields.append(f)

        # ── 1.1 检测同组互斥字段 ──
        group_map: dict[str, list[dict[str, Any]]] = {}
        for f in condition_fields:
            grp = f.get("group", "")
            if grp:
                group_map.setdefault(grp, []).append(f)
        # 有多个字段的组才算"互斥组"
        exclusive_groups: dict[str, list[dict[str, Any]]] = {
            g: fs for g, fs in group_map.items() if len(fs) > 1
        }
        # 属于互斥组的字段 ID 集合
        exclusive_ids: set[str] = set()
        for fs in exclusive_groups.values():
            for f in fs:
                exclusive_ids.add(f["id"])
        # 不属于互斥组的普通字段（总是 AND 连接）
        normal_cond_fields = [f for f in condition_fields if f["id"] not in exclusive_ids]

        # ── 2. 构建各部分 ──
        lines = self._build_header()
        lines.extend(self._build_exclusive_groups_warning(exclusive_groups))
        lines.extend(self._build_scope_hints(normal_cond_fields))
        lines.extend(self._build_condition_table(condition_fields, exclusive_ids))
        lines.extend(self._build_column_table(column_fields))
        lines.extend(self._build_metric_table(metric_fields))
        lines.extend(self._build_examples(
            condition_fields, exclusive_groups, normal_cond_fields, exclusive_ids
        ))
        lines.extend(self._build_emphasis(required_filters))

        return "\n".join(lines)

    def _build_header(self) -> list[str]:
        """构建头部说明"""
        return [
            "### 虚拟字段（Virtual Fields）",
            "",
            "虚拟字段是系统预定义的**完整布尔条件占位符**。"
            "每个虚拟字段在执行前会被自动展开为一段完整的 `WHERE` 子表达式。",
            "",
            "**核心规则**：",
            '- 在 SQL 中用双引号引用虚拟字段名（如 `"base_valid_data"`），它等价于一个返回 TRUE/FALSE 的完整条件',
            "- 虚拟字段**已经是完整的条件表达式**，你不能在它后面追加任何操作符（`=`、`IN`、`>`、`AND` 等）",
            "- 如果你需要在虚拟字段覆盖的范围之外增加额外条件，直接用**真实列名**写新条件即可",
            "- ❌ **严禁**自行创造或猜测不存在的虚拟字段名！只能使用下方表格中列出的虚拟字段。"
            "如果需要的筛选条件没有对应的虚拟字段，请直接使用真实列名构造 WHERE 条件",
            "",
        ]

    def _build_exclusive_groups_warning(
        self,
        exclusive_groups: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """构建互斥组特别说明"""
        if not exclusive_groups:
            return []

        lines = [
            "**★ 同组互斥字段（critical）**：",
            "",
        ]
        for grp, fs in exclusive_groups.items():
            ids_str = "、".join(f'`"{f["id"]}"`' for f in fs)
            labels_str = " vs ".join(f.get("label", f["id"]) for f in fs)
            lines.append(f"- **{grp} 组**：{ids_str}（{labels_str}）是**互斥条件**，操作同一个列的不同枚举值")
        lines.extend([
            "",
            "**互斥字段使用规则**：",
            "- ❌ **禁止**将同组互斥字段放在同一个 WHERE 中用 AND 连接（永远返回 0 行）",
            "- ✅ **对比场景**：分别放在不同子查询的 WHERE 中，用 FULL OUTER JOIN 合并",
            "- ✅ **单选场景**：只使用其中一个",
            "",
        ])
        return lines

    def _build_scope_hints(
        self,
        normal_cond_fields: list[dict[str, Any]],
    ) -> list[str]:
        """构建 scope 提示（必选 vs 可选）"""
        lines = []
        required_cond_fields = [f for f in normal_cond_fields if f.get("scope") == "required"]
        optional_cond_fields = [f for f in normal_cond_fields if f.get("scope") != "required"]

        if required_cond_fields:
            req_ids_str = "、".join(f'`"{f["id"]}"`' for f in required_cond_fields)
            lines.append(f"**必选条件**：{req_ids_str} 是数据口径必需的筛选条件，每个 SQL 都必须包含。")
            lines.append("")
        if optional_cond_fields:
            opt_ids_str = "、".join(f'`"{f["id"]}"`' for f in optional_cond_fields)
            lines.append(f"**当前任务条件**：{opt_ids_str} 是本次子任务需要的筛选条件，用 AND 连接。")
            lines.append("")
        return lines

    def _build_condition_table(
        self,
        condition_fields: list[dict[str, Any]],
        exclusive_ids: set[str],
    ) -> list[str]:
        """构建条件型虚拟字段表格"""
        if not condition_fields:
            return []

        lines = [
            "#### 条件型（用于 WHERE / HAVING）",
            "",
            "| 列名 | 含义 | 说明 | 互斥组 | 展开为 |",
            "|------|------|------|--------|--------|",
        ]
        for f in condition_fields:
            fid = f['id']
            label = f.get('label', fid)
            desc = f.get('description', '')
            grp = f.get('group', '')
            expr = f.get('expr', '')
            expr_display = " ".join(line.strip() for line in expr.split("\n") if line.strip())
            if len(expr_display) > 80:
                expr_display = expr_display[:77] + "..."
            grp_display = f"**{grp}**" if fid in exclusive_ids else (grp or "-")
            lines.append(f'| `"{fid}"` | {label} | {desc} | {grp_display} | `{expr_display}` |')
        lines.append("")
        return lines

    def _build_column_table(
        self,
        column_fields: list[dict[str, Any]],
    ) -> list[str]:
        """构建列型虚拟字段表格"""
        if not column_fields:
            return []

        lines = [
            "#### 列型（用于 SELECT / WHERE / GROUP BY / ORDER BY）",
            "",
            "| 列名 | 含义 | 说明 | 展开为 |",
            "|------|------|------|--------|",
        ]
        for f in column_fields:
            fid = f['id']
            label = f.get('label', fid)
            desc = f.get('description', '')
            expr = f.get('expr', '')
            expr_display = " ".join(line.strip() for line in expr.split("\n") if line.strip())
            if len(expr_display) > 80:
                expr_display = expr_display[:77] + "..."
            lines.append(f'| `"{fid}"` | {label} | {desc} | `{expr_display}` |')
        lines.append("")
        return lines

    def _build_metric_table(
        self,
        metric_fields: list[dict[str, Any]],
    ) -> list[str]:
        """构建指标型虚拟字段表格"""
        if not metric_fields:
            return []

        lines = [
            "#### 指标型（metric — 用于 SELECT 聚合计算）",
            "",
            "指标型虚拟字段是系统预定义的**聚合表达式占位符**。"
            "LLM 可以在 SQL 中直接用**裸名称**（不加引号）引用它们，"
            "系统会在执行前自动包裹聚合函数并替换为完整的聚合表达式。",
            "",
            "| 名称 | 含义 | 聚合方式 | 展开为 |",
            "|------|------|----------|--------|",
        ]
        for f in metric_fields:
            fid = f['id']
            label = f.get('label', fid)
            expr = f.get('expr', '')
            agg_type = f.get('agg_type', '').upper().strip()
            expr_display = " ".join(line.strip() for line in expr.split("\n") if line.strip())

            # 显示最终展开的表达式（包含聚合函数包裹）
            if agg_type and agg_type != "EXPR":
                final_display = f"{agg_type}({expr_display})"
            else:
                final_display = expr_display
            if len(final_display) > 80:
                final_display = final_display[:77] + "..."

            agg_label = f"`{agg_type}` 自动包裹" if agg_type and agg_type != "EXPR" else "自包含表达式"
            lines.append(f'| `{fid}` | {label} | {agg_label} | `{final_display}` |')
        lines.extend([
            "",
            "**用法**：直接用裸名称引用（如 `SELECT total_flow AS \"总流水\"`），"
            "系统自动包裹聚合函数并展开为完整表达式。"
            "即使写了 `SUM(total_flow)`，系统也会安全跳过重复包裹。",
            "",
        ])
        return lines

    def _build_examples(
        self,
        condition_fields: list[dict[str, Any]],
        exclusive_groups: dict[str, list[dict[str, Any]]],
        normal_cond_fields: list[dict[str, Any]],
        exclusive_ids: set[str],
    ) -> list[str]:
        """构建 SQL 使用示例"""
        all_cond_ids = [f['id'] for f in condition_fields]
        if not all_cond_ids:
            return []

        if exclusive_groups:
            return self._build_exclusive_examples(
                exclusive_groups, normal_cond_fields, condition_fields, exclusive_ids, all_cond_ids
            )
        else:
            return self._build_normal_examples(all_cond_ids)

    def _build_exclusive_examples(
        self,
        exclusive_groups: dict[str, list[dict[str, Any]]],
        normal_cond_fields: list[dict[str, Any]],
        condition_fields: list[dict[str, Any]],
        exclusive_ids: set[str],
        all_cond_ids: list[str],
    ) -> list[str]:
        """构建有互斥组时的示例"""
        lines = ["#### 正确用法", ""]

        # 示例 1：对比场景（互斥字段分别在子查询中）
        common_where = " AND ".join(f'"{fid}"' for fid in [f["id"] for f in normal_cond_fields]) if normal_cond_fields else ""
        for grp, fs in exclusive_groups.items():
            if len(fs) == 2:
                f_a, f_b = fs[0], fs[1]
                where_a = f'"{f_a["id"]}"'
                where_b = f'"{f_b["id"]}"'
                if common_where:
                    where_a = f"{common_where} AND {where_a}"
                    where_b = f"{common_where} AND {where_b}"
                lines.extend([
                    "```sql",
                    f"-- ✅ 对比场景：{f_a.get('label', f_a['id'])} vs {f_b.get('label', f_b['id'])}",
                    "-- 互斥字段分别在各自子查询的 WHERE 中，不会冲突",
                    'SELECT COALESCE(a."维度", b."维度") AS "维度",',
                    f'       COALESCE(a.val, 0) AS "{f_a.get("label", "A值")}",',
                    f'       COALESCE(b.val, 0) AS "{f_b.get("label", "B值")}"',
                    f'FROM (SELECT "维度", SUM("值") AS val FROM "表" WHERE {where_a} GROUP BY "维度") a',
                    'FULL OUTER JOIN',
                    f'     (SELECT "维度", SUM("值") AS val FROM "表" WHERE {where_b} GROUP BY "维度") b',
                    '     ON a."维度" IS NOT DISTINCT FROM b."维度"',
                    "```",
                    "",
                ])

        # 示例 2：单选场景
        if normal_cond_fields:
            single_where = common_where + f' AND "{condition_fields[0]["id"]}"' if condition_fields[0]["id"] not in exclusive_ids else common_where + f' AND "{list(exclusive_groups.values())[0][0]["id"]}"'
            lines.extend([
                "```sql",
                "-- ✅ 单选场景：只用互斥组中的一个",
                'SELECT "维度", SUM("值") AS val FROM "表"',
                f"WHERE {single_where}",
                'GROUP BY "维度"',
                "```",
                "",
            ])

        # 错误示例
        lines.extend(["#### 禁止用法", "", "```sql"])
        for grp, fs in exclusive_groups.items():
            bad_where_parts = [f'"{f["id"]}"' for f in fs]
            if common_where:
                bad_where_parts = [common_where] + bad_where_parts
            bad_where = " AND ".join(bad_where_parts)
            ids_display = " + ".join(f["id"] for f in fs)
            lines.append(f'-- ❌ 同组互斥字段 AND 连接，结果永远为空！({ids_display} 操作同一个列的不同值)')
            lines.append(f'WHERE {bad_where}')
        lines.extend(["```", ""])

        # 通用禁止
        sample_id = all_cond_ids[0]
        lines.extend([
            "```sql",
            '-- ❌ 虚拟字段后面追加操作符会导致语法错误或语义错误',
            f'WHERE "{sample_id}" = \'某值\'     -- 错误：虚拟字段不是列名',
            f'WHERE "{sample_id}" IN (\'A\',\'B\') -- 错误：虚拟字段已是完整条件',
            "```",
            "",
        ])
        return lines

    def _build_normal_examples(self, all_cond_ids: list[str]) -> list[str]:
        """构建无互斥组时的示例"""
        where_clause = " AND ".join(f'"{fid}"' for fid in all_cond_ids)
        sample_id = all_cond_ids[0]

        return [
            "#### 正确用法",
            "",
            "```sql",
            "-- 每个虚拟字段独立作为一个 AND 条件，不追加任何操作符",
            'SELECT "维度列", SUM("数值列") AS 值',
            'FROM "表名"',
            f"WHERE {where_clause}",
            'GROUP BY "维度列"',
            "```",
            "",
            "#### 禁止用法",
            "",
            "```sql",
            '-- ❌ 虚拟字段后面追加操作符会导致语法错误或语义错误',
            f'WHERE "{sample_id}" = \'某值\'     -- 错误：虚拟字段不是列名',
            f'WHERE "{sample_id}" IN (\'A\',\'B\') -- 错误：虚拟字段已是完整条件',
            "```",
            "",
        ]

    def _build_emphasis(self, required_filters: list[dict[str, Any]]) -> list[str]:
        """构建强调部分"""
        all_ids = [f['id'] for f in required_filters]
        required_ids = [
            f['id'] for f in required_filters
            if f.get('field_type') == 'condition' and f.get('scope') == 'required'
        ]
        task_ids = [fid for fid in all_ids if fid not in required_ids]

        lines = []
        if task_ids:
            lines.append(
                f"**当前任务字段**：`{'`, `'.join(task_ids)}` 是本次子任务的筛选/指标/维度字段，"
                f"按需在 SQL 中使用。"
            )
            # 检查是否有互斥组
            has_exclusive = any(
                f.get('group') and any(
                    f2.get('group') == f.get('group') and f2['id'] != f['id']
                    for f2 in required_filters
                )
                for f in required_filters if f.get('field_type') == 'condition'
            )
            if has_exclusive:
                lines.append(
                    "但同组互斥字段**不能放在同一个 WHERE 中 AND 连接**，"
                    "应根据分析场景分配到不同的子查询中。"
                )

        return lines
