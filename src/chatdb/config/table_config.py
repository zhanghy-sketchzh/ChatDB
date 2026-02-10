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
                related_filters=item.get("related_filters", []),
                related_metrics=item.get("related_metrics", []),
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
                business_terms=item.get("business_terms", []),
                dimensions=item.get("dimensions", []),
                filters=item.get("filters", []),
                sql=item.get("sql", ""),
                explanation=item.get("explanation", ""),
            )
            self._examples.append(ex)
    
    def _parse_rules(self) -> None:
        """解析规则"""
        for item in self._raw.get("rules", []):
            r = Rule(
                id=item.get("id", ""),
                type=item.get("type", ""),
                description=item.get("description", ""),
                match_pattern=item.get("match_pattern", ""),
                default_value=item.get("default_value", ""),
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
