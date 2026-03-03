"""
表级语义配置加载器

加载并管理表级语义配置 (metrics_config.yml)，提供：
1. business_terms 业务术语匹配
2. filters 筛选条件检索
3. dimensions 维度信息
4. metrics 指标定义
5. examples few-shot 示例
6. rules 规则验证
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BusinessTerm:
    """业务术语"""
    id: str
    term: str
    synonyms: list[str] = field(default_factory=list)
    description: str = ""
    related_dimensions: list[str] = field(default_factory=list)
    related_filters: list[str] = field(default_factory=list)
    related_metrics: list[str] = field(default_factory=list)
    example_nl: str = ""


@dataclass
class Filter:
    """筛选条件"""
    id: str
    label: str
    description: str = ""
    expr: str = ""
    is_default: bool = False
    related_term: str = ""
    explanation: str = ""


@dataclass
class Dimension:
    """维度"""
    id: str
    label: str
    column: str = ""
    expr: str = ""
    description: str = ""


@dataclass
class Metric:
    """指标"""
    id: str
    label: str
    description: str = ""
    base_table: str = ""
    filter_refs: list[str] = field(default_factory=list)
    agg: str = ""
    formula: str = ""
    unit: str = ""
    display_unit: str = ""
    display_divisor: float = 1.0
    example_nl: str = ""


@dataclass
class Example:
    """示例"""
    id: str
    query: str
    business_terms: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    sql: str = ""
    explanation: str = ""


@dataclass 
class Rule:
    """规则"""
    id: str
    type: str
    description: str = ""
    match_pattern: str = ""
    default_value: str = ""
    field: str = ""
    default_filter: str = ""
    must_include: str = ""


class TableConfigLoader:
    """表级语义配置加载器"""
    
    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "yml" / "metrics_config.yml"
        
        self.config_path = Path(config_path)
        self._raw: dict = {}
        self._meta: dict = {}
        self._business_terms: dict[str, BusinessTerm] = {}
        self._filters: dict[str, Filter] = {}
        self._dimensions: dict[str, Dimension] = {}
        self._metrics: dict[str, Metric] = {}
        self._examples: list[Example] = []
        self._rules: list[Rule] = []
        
        self._load()
    
    def _load(self) -> None:
        """加载配置"""
        if not self.config_path.exists():
            return
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}
        
        # 预处理：展开模板、表驱动生成 metric_* 筛选器
        from chatdb.config.metrics_loader import preprocess_yaml_config
        self._raw = preprocess_yaml_config(self._raw)
        
        self._meta = self._raw.get("meta", {})
        self._parse_business_terms()
        self._parse_filters()
        self._parse_dimensions()
        self._parse_metrics()
        self._parse_examples()
        self._parse_rules()
    
    def _parse_business_terms(self) -> None:
        """解析业务术语"""
        for item in self._raw.get("business_terms", []):
            term = BusinessTerm(
                id=item.get("id", ""),
                term=item.get("term", ""),
                synonyms=item.get("synonyms", []),
                description=item.get("description", ""),
                related_dimensions=item.get("related_dimensions", []),
                # 兼容 v1.2 的 "filters" 和旧版 "related_filters"
                related_filters=item.get("filters", []) or item.get("related_filters", []),
                related_metrics=item.get("metrics", []) or item.get("related_metrics", []),
                example_nl=item.get("example_nl", ""),
            )
            self._business_terms[term.id] = term
    
    def _parse_filters(self) -> None:
        """解析筛选条件"""
        for fid, fconfig in self._raw.get("filters", {}).items():
            f = Filter(
                id=fid,
                label=fconfig.get("label", ""),
                description=fconfig.get("description", ""),
                expr=fconfig.get("expr", ""),
                is_default=fconfig.get("is_default", False),
                related_term=fconfig.get("related_term", ""),
                explanation=fconfig.get("explanation", ""),
            )
            self._filters[fid] = f
    
    def _parse_dimensions(self) -> None:
        """解析维度"""
        for did, dconfig in self._raw.get("dimensions", {}).items():
            d = Dimension(
                id=did,
                label=dconfig.get("label", ""),
                column=dconfig.get("column", ""),
                expr=dconfig.get("expr", ""),
                description=dconfig.get("description", ""),
            )
            self._dimensions[did] = d
    
    def _parse_metrics(self) -> None:
        """解析指标"""
        for mid, mconfig in self._raw.get("metrics", {}).items():
            m = Metric(
                id=mid,
                label=mconfig.get("label", ""),
                description=mconfig.get("description", ""),
                base_table=mconfig.get("base_table", ""),
                filter_refs=mconfig.get("filter_refs", []),
                agg=mconfig.get("agg", ""),
                formula=mconfig.get("formula", ""),
                unit=mconfig.get("unit", ""),
                display_unit=mconfig.get("display_unit", ""),
                display_divisor=mconfig.get("display_divisor", 1.0),
                example_nl=mconfig.get("example_nl", ""),
            )
            self._metrics[mid] = m
    
    def _parse_examples(self) -> None:
        """解析示例"""
        for item in self._raw.get("examples", []):
            ex = Example(
                id=item.get("id", ""),
                query=item.get("query", ""),
                # 兼容 v1.2 的 "terms" 和旧版 "business_terms"
                business_terms=item.get("terms", []) or item.get("business_terms", []),
                dimensions=item.get("dimensions", []),
                filters=item.get("filters", []),
                sql=item.get("sql", item.get("sql_hint", "")),
                explanation=item.get("explanation", item.get("note", "")),
            )
            self._examples.append(ex)
    
    def _parse_rules(self) -> None:
        """解析规则（兼容 v1 list 和 v2 dict 格式）"""
        raw_rules = self._raw.get("rules", [])
        # v2 格式是 dict，跳过旧版解析
        if isinstance(raw_rules, dict):
            return
        for item in raw_rules:
            # match_pattern 可能是字符串或列表，统一转为字符串供旧 API 兼容
            mp = item.get("match_pattern", "")
            if isinstance(mp, list):
                mp = ", ".join(mp)
            r = Rule(
                id=item.get("id", ""),
                type=item.get("type", ""),
                description=item.get("description", ""),
                match_pattern=mp,
                default_value=item.get("default_filter", item.get("default_metric", item.get("default_value", ""))),
                field=item.get("field", ""),
                default_filter=item.get("default_filter", ""),
                must_include=item.get("must_include", ""),
            )
            self._rules.append(r)
    
    # ==================== 业务术语匹配 ====================
    
    def match_term(self, text: str) -> list[BusinessTerm]:
        """匹配业务术语"""
        matched = []
        text_lower = text.lower()
        
        for term in self._business_terms.values():
            # 匹配主术语
            if term.term.lower() in text_lower:
                matched.append(term)
                continue
            # 匹配同义词
            for syn in term.synonyms:
                if syn.lower() in text_lower:
                    matched.append(term)
                    break
        
        return matched
    
    def get_filters_for_terms(self, terms: list[BusinessTerm]) -> list[Filter]:
        """获取术语关联的筛选条件"""
        filter_ids = set()
        for term in terms:
            filter_ids.update(term.related_filters)
        
        return [self._filters[fid] for fid in filter_ids if fid in self._filters]
    
    # ==================== 筛选条件 ====================
    
    def get_default_filters(self) -> list[Filter]:
        """获取默认筛选条件"""
        return [f for f in self._filters.values() if f.is_default]
    
    def get_filter(self, filter_id: str) -> Filter | None:
        """获取筛选条件"""
        return self._filters.get(filter_id)
    
    def get_filter_expr(self, filter_id: str) -> str:
        """获取筛选表达式"""
        f = self._filters.get(filter_id)
        return f.expr if f else ""
    
    # ==================== 示例检索 ====================
    
    def search_examples(self, query: str, top_k: int = 3) -> list[Example]:
        """搜索相关示例"""
        # 简单的关键词匹配
        query_lower = query.lower()
        scored = []
        
        for ex in self._examples:
            score = 0
            # 匹配查询文本
            ex_lower = ex.query.lower()
            for word in query_lower.split():
                if word in ex_lower:
                    score += 1
            
            # 匹配业务术语
            for term_id in ex.business_terms:
                term = self._business_terms.get(term_id)
                if term and (term.term.lower() in query_lower or 
                            any(s.lower() in query_lower for s in term.synonyms)):
                    score += 2
            
            if score > 0:
                scored.append((score, ex))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]
    
    # ==================== Prompt 生成 ====================
    
    def get_evidence_prompt(self, query: str) -> str:
        """生成 Evidence Prompt"""
        lines = ["## 业务口径配置", ""]
        
        # 1. 匹配的业务术语
        terms = self.match_term(query)
        if terms:
            lines.append("### 匹配的业务术语")
            for term in terms:
                lines.append(f"- **{term.term}**: {term.description}")
                if term.synonyms:
                    lines.append(f"  同义词: {', '.join(term.synonyms[:5])}")
            lines.append("")
        
        # 2. 相关筛选条件
        filters = self.get_filters_for_terms(terms)
        default_filters = self.get_default_filters()
        all_filters = list({f.id: f for f in default_filters + filters}.values())
        
        if all_filters:
            lines.append("### 筛选条件")
            for f in all_filters:
                lines.append(f"- **{f.label}** ({f.id})")
                lines.append(f"  ```sql")
                lines.append(f"  {f.expr}")
                lines.append(f"  ```")
                if f.explanation:
                    lines.append(f"  说明: {f.explanation}")
            lines.append("")
        
        # 3. 相关示例
        examples = self.search_examples(query, top_k=2)
        if examples:
            lines.append("### 参考示例")
            for ex in examples:
                lines.append(f"**问题**: {ex.query}")
                lines.append(f"```sql")
                lines.append(ex.sql.strip())
                lines.append(f"```")
                if ex.explanation:
                    lines.append(f"说明: {ex.explanation}")
                lines.append("")
        
        return "\n".join(lines)
    
    def get_full_context(self) -> str:
        """获取完整上下文（用于 LLM）"""
        lines = [f"# 表: {self._meta.get('display_name', '')} ({self._meta.get('table_name', '')})", ""]
        lines.append(f"描述: {self._meta.get('description', '')}")
        lines.append(f"粒度: {self._meta.get('grain', '')}")
        lines.append(f"值字段: {self._meta.get('value_column', '')}")
        lines.append("")
        
        # 维度
        lines.append("## 维度")
        for d in self._dimensions.values():
            lines.append(f"- {d.label} (`{d.column}`): {d.description}")
        lines.append("")
        
        # 指标
        lines.append("## 指标")
        for m in self._metrics.values():
            lines.append(f"- **{m.label}**: {m.description}")
            lines.append(f"  聚合: {m.agg}, 单位: {m.display_unit}")
        lines.append("")
        
        # 默认筛选
        lines.append("## 默认筛选条件")
        for f in self.get_default_filters():
            lines.append(f"```sql")
            lines.append(f.expr)
            lines.append(f"```")
        
        return "\n".join(lines)
    
    # ==================== 属性访问 ====================
    
    @property
    def meta(self) -> dict:
        return self._meta
    
    @property
    def table_name(self) -> str:
        return self._meta.get("table_name", "")
    
    @property
    def business_terms(self) -> dict[str, BusinessTerm]:
        return self._business_terms
    
    @property
    def filters(self) -> dict[str, Filter]:
        return self._filters
    
    @property
    def dimensions(self) -> dict[str, Dimension]:
        return self._dimensions
    
    @property
    def metrics(self) -> dict[str, Metric]:
        return self._metrics
    
    @property
    def examples(self) -> list[Example]:
        return self._examples
    
    @property
    def rules(self) -> list[Rule]:
        return self._rules


def load_table_config(config_path: str | Path | None = None) -> TableConfigLoader:
    """加载表级语义配置"""
    return TableConfigLoader(config_path)


# ============================================================
# YML meta 信息格式化（供各 Agent Prompt 注入）
# ============================================================

def format_yml_meta_for_prompt(yml_config: dict, level: str = "full") -> str:
    """
    从 YML 配置的 meta 块中提取关键业务信息，格式化为 Prompt 注入文本。

    Args:
        yml_config: 完整的 YML 配置字典
        level: 输出详细程度
            - "full": 完整信息（用于 table_understanding 生成、SQL 生成）
            - "summary": 精简摘要（用于 SemanticParser / Planner）

    Returns:
        格式化后的 meta 信息文本，为空时返回 ""
    """
    meta = yml_config.get("meta", {})
    if not meta:
        return ""

    lines: list[str] = []

    # ── 1. 表基本信息 ──
    display_name = meta.get("display_name", "")
    description = meta.get("description", "")
    grain = meta.get("grain", "")
    table_name = meta.get("table_name", "")

    if display_name or description:
        header = display_name or table_name
        if description:
            header = f"{header}：{description}" if header else description
        lines.append(f"- 表定位: {header}")

    if grain:
        lines.append(f"- 数据粒度: {grain}")

    value_column = meta.get("value_column", "")
    if value_column:
        lines.append(f"- 度量列: {value_column}")

    business_key = meta.get("business_key", [])
    if business_key:
        lines.append(f"- 业务主键: {', '.join(business_key)}")

    # ── 2. 报表项层级结构（对理解指标体系至关重要）──
    hierarchy = meta.get("report_item_hierarchy", {})
    if hierarchy and level == "full":
        lines.append("")
        lines.append("- 报表项层级结构:")
        _format_hierarchy(hierarchy, lines, indent=2)

    # ── 3. 组织范围说明（从 meta.org_scope 或 rules.canonical_scope 读取）──
    org_scope = meta.get("org_scope") or meta.get("organization_scope", {})
    if org_scope:
        lines.append("")
        lines.append("- 组织范围:")
        for scope_name, scope_info in org_scope.items():
            if isinstance(scope_info, dict):
                filter_expr = scope_info.get("filter", "")
                available = scope_info.get("metrics", scope_info.get("available_metrics", ""))
                detail = f"（筛选: {filter_expr}）" if filter_expr else ""
                avail_str = f"，可用指标: {available}" if available else ""
                lines.append(f"    {scope_name}{detail}{avail_str}")
            else:
                lines.append(f"    {scope_name}: {scope_info}")
    else:
        # v1.2: 从 rules 中提取 canonical_scope
        rules = yml_config.get("rules", [])
        scope_rules = [r for r in rules if r.get("type") == "canonical_scope"]
        if scope_rules:
            lines.append("")
            lines.append("- 组织范围:")
            for r in scope_rules:
                scope_name = r.get("scope_name", "")
                filt = r.get("filter", "")
                avail = r.get("available_metrics", "")
                detail = f"（筛选: {filt}）" if filt else ""
                avail_str = f"，可用指标: {avail}" if avail else ""
                lines.append(f"    {scope_name}{detail}{avail_str}")

    # ── 4. 数据预处理说明（极重要：null 处理等）──
    data_prep = meta.get("data_preprocessing", "")
    if data_prep:
        lines.append("")
        lines.append(f"- 数据预处理说明: {data_prep.strip()}")

    # ── 5. 注意事项 ──
    notes = meta.get("notes", "")
    if notes:
        lines.append("")
        lines.append(f"- 重要注意事项:\n{_indent_text(notes.strip(), 4)}")

    if not lines:
        return ""

    return "\n".join(lines)


def _format_hierarchy(data: dict | list | str, lines: list[str], indent: int = 2) -> None:
    """递归格式化报表项层级结构"""
    prefix = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                _format_hierarchy(value, lines, indent + 2)
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _format_hierarchy(item, lines, indent)
            else:
                lines.append(f"{prefix}• {item}")


def _indent_text(text: str, spaces: int) -> str:
    """给多行文本加缩进"""
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))
