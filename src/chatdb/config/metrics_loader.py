"""
业务口径配置加载器 v1.1

读取 YAML 配置文件，支持：
1. 全局表达式模板展开（{{ global.xxx }}）
2. 表驱动生成 metric_* 筛选器
3. YAML 锚点/别名（&, *）
"""

import re
from pathlib import Path
from typing import Any

import yaml


def preprocess_yaml_config(config: dict) -> dict:
    """
    预处理 YAML 配置：
    1. 展开 {{ global.xxx }} 模板
    2. 从 metric_items 表驱动生成 filters
    """
    # 1. 获取全局变量
    global_vars = config.get("global", {})
    
    # 2. 展开模板引用
    def expand_template(obj: Any) -> Any:
        if isinstance(obj, str):
            # 匹配 {{ global.xxx }}
            pattern = r'\{\{\s*global\.(\w+)\s*\}\}'
            match = re.search(pattern, obj)
            if match:
                var_name = match.group(1)
                if var_name in global_vars:
                    return global_vars[var_name]
            return obj
        elif isinstance(obj, dict):
            return {k: expand_template(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_template(item) for item in obj]
        return obj
    
    config = expand_template(config)
    
    # 3. 从 metric_items 表驱动生成 filters
    metric_items = config.get("metric_items", {})
    if metric_items:
        filters = config.setdefault("filters", {})
        for label, filter_id in metric_items.items():
            if filter_id not in filters:
                filters[filter_id] = {
                    "label": label,
                    "expr": f'"大盘报表项" = \'{label}\'',
                    "group": "metric",
                    "merge_mode": "exclusive",
                }
    
    return config


class MetricsConfigLoader:
    """业务口径配置加载器 v1.1"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为 data/yml/metrics_config.yml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "yml" / "metrics_config.yml"
        
        self.config_path = Path(config_path)
        self._config: dict = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """加载并预处理配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        
        # 预处理：展开模板、表驱动生成
        self._config = preprocess_yaml_config(raw_config)
    
    @property
    def config(self) -> dict:
        """获取完整配置"""
        return self._config
    
    # ==================== v1.1 新增方法 ====================
    
    def get_global_expr(self, name: str) -> str:
        """获取全局表达式"""
        return self._config.get("global", {}).get(name, "")
    
    def get_base_valid_expr(self) -> str:
        """获取基础有效数据筛选表达式"""
        return self.get_global_expr("base_valid_expr")
    
    def get_calibration(self) -> dict:
        """获取口径映射配置（兼容 v1.1 calibration 段和 v1.2 rules 段）"""
        # v1.2: 从 rules 中提取 disambiguation 和 canonical_scope
        cal = self._config.get("calibration", {})
        if cal:
            return cal
        # v1.2: 从统一 rules 合成
        rules = self._config.get("rules", [])
        disamb = [r for r in rules if r.get("type") == "disambiguation"]
        scopes = {
            r.get("scope_name", ""): r.get("filter", "")
            for r in rules if r.get("type") == "canonical_scope"
        }
        return {"disambiguation": disamb, "scopes": scopes}
    
    def get_canonical_filter(self, metric_id: str) -> str | None:
        """获取指标的规范口径筛选器"""
        # 从 metrics 定义中查找 canonical_filters
        metrics = self._config.get("metrics", {})
        if metric_id in metrics:
            canonical = metrics[metric_id].get("canonical_filters", [])
            if canonical:
                return canonical[0] if isinstance(canonical, list) else canonical
        return None
    
    def get_disambiguation_rules(self) -> list:
        """获取消歧义规则（兼容 v1.1 calibration 和 v1.2 rules）"""
        # v1.1: calibration.disambiguation
        cal = self._config.get("calibration", {})
        if cal and "disambiguation" in cal:
            return cal["disambiguation"]
        # v1.2: 从 rules 中提取 type=disambiguation
        rules = self._config.get("rules", [])
        return [
            {
                "pattern": r.get("match_pattern", []),
                "default": r.get("default_filter") or r.get("default_metric", ""),
                "note": r.get("description", ""),
            }
            for r in rules if r.get("type") == "disambiguation"
        ]
    
    def resolve_filter_by_term(self, term: str) -> list[str]:
        """根据业务术语查找对应的 filters"""
        business_terms = self._config.get("business_terms", [])
        for bt in business_terms:
            if bt.get("term") == term:
                return bt.get("filters", [])
            if term in bt.get("synonyms", []):
                return bt.get("filters", [])
        return []
    
    def get_filter_expr(self, filter_id: str) -> str:
        """获取筛选器的 SQL 表达式"""
        filters = self._config.get("filters", {})
        if filter_id in filters:
            return filters[filter_id].get("expr", "")
        return ""
    
    def get_metric_definition(self, metric_id: str) -> dict:
        """获取指标定义"""
        return self._config.get("metrics", {}).get(metric_id, {})
    
    # ==================== 原有方法（兼容）====================
    
    def get_base_filters(self) -> dict:
        """获取基础筛选条件（兼容旧版）"""
        return self._config.get("base_filters", {})
    
    def get_base_filters_sql(self, table_alias: str = "") -> str:
        """获取基础筛选 SQL（优先使用 v1.1 全局表达式）"""
        # v1.1: 直接返回全局表达式
        base_expr = self.get_base_valid_expr()
        if base_expr:
            return base_expr.strip()
        # 兼容旧版
        return self._filters_to_sql(self.get_base_filters(), table_alias)
    
    def get_source_mode(self, mode_name: str) -> dict | None:
        """获取数据来源模式配置"""
        modes = self._config.get("data_source_modes", {})
        if mode_name in modes:
            return modes[mode_name]
        for name, config in modes.items():
            if mode_name in config.get("aliases", []):
                return config
        return None
    
    def get_source_mode_sql(self, mode_name: str, table_alias: str = "") -> str:
        """获取数据来源模式的 SQL 条件"""
        mode = self.get_source_mode(mode_name)
        if mode:
            return self._filters_to_sql(mode.get("filters", {}), table_alias)
        return ""
    
    def get_organization(self, org_name: str) -> dict | None:
        """获取组织维度配置"""
        orgs = self._config.get("organization_dimensions", {})
        if org_name in orgs:
            return orgs[org_name]
        for name, config in orgs.items():
            if org_name in config.get("aliases", []):
                return config
        return None
    
    def get_organization_sql(self, org_name: str, table_alias: str = "") -> str:
        """获取组织维度的 SQL 条件"""
        org = self.get_organization(org_name)
        if org:
            return self._filters_to_sql(org.get("filters", {}), table_alias)
        return ""
    
    def get_metric(self, metric_name: str) -> dict | None:
        """获取财务指标配置"""
        metrics = self._config.get("financial_metrics", {})
        if metric_name in metrics:
            return metrics[metric_name]
        for name, config in metrics.items():
            if metric_name in config.get("aliases", []):
                return config
        return None
    
    def get_metric_sql(self, metric_name: str, table_alias: str = "") -> str:
        """获取财务指标的 SQL 条件"""
        metric = self.get_metric(metric_name)
        if metric:
            col = metric.get("column", "大盘报表项")
            val = metric.get("value", metric_name)
            prefix = f'"{table_alias}".' if table_alias else ""
            return f'{prefix}"{col}" = \'{val}\''
        return ""
    
    def get_metric_output_column(self, metric_name: str) -> str:
        """获取指标的输出列名"""
        metric = self.get_metric(metric_name)
        if metric:
            return metric.get("output_column", "ieg口径金额-人民币")
        return "ieg口径金额-人民币"
    
    def get_time_preset(self, preset_name: str) -> dict | None:
        """获取时间预设配置"""
        time_config = self._config.get("time_dimensions", {})
        presets = time_config.get("presets", {})
        if preset_name in presets:
            return presets[preset_name]
        for name, config in presets.items():
            if preset_name in config.get("aliases", []):
                return config
        return None
    
    def get_time_sql(self, preset_name: str, table_alias: str = "") -> str:
        """获取时间预设的 SQL 条件"""
        preset = self.get_time_preset(preset_name)
        if preset:
            months = preset.get("months", [])
            if months:
                time_col = self._config.get("time_dimensions", {}).get("column", "月份")
                prefix = f'"{table_alias}".' if table_alias else ""
                months_str = ", ".join(str(m) for m in months)
                return f'{prefix}"{time_col}" IN ({months_str})'
        return ""
    
    def get_product_match(self, product_name: str) -> dict | None:
        """获取产品匹配配置"""
        products = self._config.get("product_dimensions", {}).get("popular_products", {})
        if product_name in products:
            return {"name": product_name, **products[product_name]}
        for name, config in products.items():
            if product_name in config.get("aliases", []):
                return {"name": name, **config}
        return None
    
    def get_product_sql(self, product_name: str, table_alias: str = "") -> str:
        """获取产品筛选的 SQL 条件"""
        product_col = self._config.get("product_dimensions", {}).get("column", "考核产品")
        prefix = f'"{table_alias}".' if table_alias else ""
        product = self.get_product_match(product_name)
        if product:
            real_name = product["name"]
            if product.get("fuzzy_match"):
                return f'{prefix}"{product_col}" LIKE \'%{real_name}%\''
            else:
                return f'{prefix}"{product_col}" = \'{real_name}\''
        return f'{prefix}"{product_col}" = \'{product_name}\''
    
    def get_llm_hints(self) -> dict:
        """获取 LLM 提示信息"""
        return self._config.get("llm_hints", {})
    
    # ==================== v1.2 分层配置 ====================
    
    def get_hard_config(self) -> dict:
        """
        硬配置（执行层依赖）
        
        这些是 SQL 生成/执行的必要信息，不能随意改动：
        - table_name: 表名
        - value_column: 金额列
        - base_valid_expr: 基础数据筛选表达式
        - metrics: 指标定义（含 agg 表达式和 default_filters）
        - filters: 筛选器定义（含 SQL expr）
        """
        meta = self._config.get("meta", {})
        return {
            "table_name": meta.get("table_name", ""),
            "value_column": meta.get("value_column", ""),
            "base_valid_expr": self.get_base_valid_expr(),
            "metrics": self._config.get("metrics", {}),
            "filters": self._config.get("filters", {}),
        }
    
    def get_semantic_config(self) -> dict:
        """
        参考配置（给 LLM 做语义提示）
        
        这些是帮助 LLM 理解业务语义的信息，可以随时增删：
        - business_terms: 业务术语词典
        - dimensions: 维度定义（含 terms）
        - examples: 示例查询
        - rules: 业务规则（消歧义等）
        """
        return {
            "business_terms": self._config.get("business_terms", []),
            "dimensions": self._config.get("dimensions", {}),
            "examples": self._config.get("examples", []),
            "rules": self._config.get("rules", []),
        }
    
    def get_default_assumptions(self) -> dict:
        """获取默认假设"""
        return self.get_llm_hints().get("defaults", {})
    
    def get_terminology(self) -> dict:
        """获取术语解释"""
        return self.get_llm_hints().get("terminology", {})
    
    def resolve_ambiguity(self, term: str) -> str:
        """解决术语歧义"""
        # v1.1: 优先使用 calibration.disambiguation
        for rule in self.get_disambiguation_rules():
            patterns = rule.get("pattern", [])
            if isinstance(patterns, str):
                patterns = [patterns]
            for p in patterns:
                if p in term:
                    return rule.get("default", term)
        
        # 兼容旧版
        disambiguation = self.get_llm_hints().get("disambiguation", [])
        for rule in disambiguation:
            if rule.get("pattern", "") in term:
                return rule.get("default", term)
        return term
    
    def build_query_sql(
        self,
        table_name: str,
        metric: str,
        organization: str = None,
        time_range: str = None,
        source_mode: str = None,
        product: str = None,
        group_by: str = None,
    ) -> str:
        """根据口径配置构建完整的 SQL 查询"""
        defaults = self.get_default_assumptions()
        organization = organization or defaults.get("organization", "IEG本部")
        time_range = time_range or defaults.get("time_range", "今年")
        source_mode = source_mode or defaults.get("data_source_mode", "实际")
        
        metric = self.resolve_ambiguity(metric)
        
        conditions = []
        
        base_sql = self.get_base_filters_sql()
        if base_sql:
            conditions.append(f"({base_sql})")
        
        org_sql = self.get_organization_sql(organization)
        if org_sql:
            conditions.append(org_sql)
        
        source_sql = self.get_source_mode_sql(source_mode)
        if source_sql:
            conditions.append(source_sql)
        
        metric_sql = self.get_metric_sql(metric)
        if metric_sql:
            conditions.append(metric_sql)
        
        time_sql = self.get_time_sql(time_range)
        if time_sql:
            conditions.append(time_sql)
        
        if product:
            product_sql = self.get_product_sql(product)
            if product_sql:
                conditions.append(product_sql)
        
        output_col = self.get_metric_output_column(metric)
        where_clause = " AND ".join(conditions)
        
        if group_by:
            sql = f'''SELECT "{group_by}", SUM("{output_col}") as "{metric}"
FROM "{table_name}"
WHERE {where_clause}
GROUP BY "{group_by}"
ORDER BY "{metric}" DESC'''
        else:
            sql = f'''SELECT SUM("{output_col}") as "{metric}"
FROM "{table_name}"
WHERE {where_clause}'''
        
        return sql
    
    def generate_prompt_context(self) -> str:
        """生成供 LLM 使用的口径说明文本"""
        lines = ["## 业务口径配置说明\n"]
        
        # v1.1: 基础筛选
        base_expr = self.get_base_valid_expr()
        if base_expr:
            lines.append("### 基础筛选条件（所有查询必须应用）")
            lines.append(f"```sql\n{base_expr.strip()}\n```\n")
        
        # 口径映射
        calibration = self.get_calibration()
        if calibration:
            lines.append("### 口径映射")
            for metric_id, filter_id in calibration.get("metrics_default_scope", {}).items():
                lines.append(f"- {metric_id} → {filter_id}")
            lines.append("")
        
        # 消歧义
        disambiguation = self.get_disambiguation_rules()
        if disambiguation:
            lines.append("### 消歧义规则")
            for rule in disambiguation:
                patterns = rule.get("pattern", [])
                if isinstance(patterns, list):
                    patterns = ", ".join(patterns)
                use_filter = rule.get("use_filter", rule.get("default", ""))
                lines.append(f"- 「{patterns}」→ {use_filter}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _filters_to_sql(self, filters: dict, table_alias: str = "") -> str:
        """将筛选条件字典转换为 SQL WHERE 子句"""
        conditions = []
        prefix = f'"{table_alias}".' if table_alias else ""
        
        for col, rule in filters.items():
            if "eq" in rule:
                val = rule["eq"]
                if val == "null":
                    conditions.append(f'({prefix}"{col}" IS NULL OR {prefix}"{col}" = \'null\')')
                else:
                    conditions.append(f'{prefix}"{col}" = \'{val}\'')
            
            if "in" in rule:
                vals = rule["in"]
                null_check = ""
                non_null_vals = [v for v in vals if v not in ("null", "为空")]
                if "null" in vals or "为空" in vals:
                    null_check = f'{prefix}"{col}" IS NULL OR {prefix}"{col}" = \'null\''
                
                if non_null_vals:
                    vals_str = ", ".join(f"'{v}'" for v in non_null_vals)
                    in_check = f'{prefix}"{col}" IN ({vals_str})'
                    if null_check:
                        conditions.append(f'({null_check} OR {in_check})')
                    else:
                        conditions.append(in_check)
                elif null_check:
                    conditions.append(f'({null_check})')
            
            if "not_in" in rule:
                vals = rule["not_in"]
                vals_str = ", ".join(f"'{v}'" for v in vals)
                conditions.append(f'{prefix}"{col}" NOT IN ({vals_str})')
        
        return " AND ".join(conditions)


# 便捷函数
def load_metrics_config(config_path: str = None) -> MetricsConfigLoader:
    """加载口径配置"""
    return MetricsConfigLoader(config_path)


if __name__ == "__main__":
    # 测试
    loader = MetricsConfigLoader()
    
    print("=== v1.1 全局表达式 ===")
    print(loader.get_base_valid_expr()[:200] + "...")
    
    print("\n=== 口径映射 ===")
    print(loader.get_calibration())
    
    print("\n=== 消歧义规则 ===")
    for rule in loader.get_disambiguation_rules():
        print(f"  {rule}")
    
    print("\n=== metric_items 展开后的 filters ===")
    filters = loader.config.get("filters", {})
    metric_filters = [k for k in filters if k.startswith("metric_")]
    print(f"  共 {len(metric_filters)} 个: {metric_filters[:5]}...")
    
    print("\n=== canonical_filter 查询 ===")
    print(f"  total_flow → {loader.get_canonical_filter('total_flow')}")
    print(f"  total_hc → {loader.get_canonical_filter('total_hc')}")
