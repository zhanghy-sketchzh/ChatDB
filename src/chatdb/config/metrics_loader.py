"""
业务口径配置加载器

读取 YAML 配置文件，将业务术语映射为 SQL 筛选条件。
帮助 LLM 理解业务口径，生成准确的 SQL。
"""

import re
from pathlib import Path
from typing import Any

import yaml


class MetricsConfigLoader:
    """业务口径配置加载器"""
    
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
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
    
    @property
    def config(self) -> dict:
        """获取完整配置"""
        return self._config
    
    # ==================== 基础筛选条件 ====================
    
    def get_base_filters(self) -> dict:
        """获取基础筛选条件（所有查询默认应用）"""
        return self._config.get("base_filters", {})
    
    def get_base_filters_sql(self, table_alias: str = "") -> str:
        """将基础筛选条件转换为 SQL WHERE 子句"""
        return self._filters_to_sql(self.get_base_filters(), table_alias)
    
    # ==================== 数据来源模式 ====================
    
    def get_source_mode(self, mode_name: str) -> dict | None:
        """
        获取数据来源模式配置
        
        Args:
            mode_name: 模式名称或别名（如"实际"、"预测"、"预算"）
        """
        modes = self._config.get("data_source_modes", {})
        
        # 直接匹配
        if mode_name in modes:
            return modes[mode_name]
        
        # 别名匹配
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
    
    # ==================== 组织维度 ====================
    
    def get_organization(self, org_name: str) -> dict | None:
        """
        获取组织维度配置
        
        Args:
            org_name: 组织名称或别名（如"IEG本部"、"投资公司"）
        """
        orgs = self._config.get("organization_dimensions", {})
        
        # 直接匹配
        if org_name in orgs:
            return orgs[org_name]
        
        # 别名匹配
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
    
    # ==================== 财务指标 ====================
    
    def get_metric(self, metric_name: str) -> dict | None:
        """
        获取财务指标配置
        
        Args:
            metric_name: 指标名称或别名（如"递延后利润"、"利润"）
        """
        metrics = self._config.get("financial_metrics", {})
        
        # 直接匹配
        if metric_name in metrics:
            return metrics[metric_name]
        
        # 别名匹配
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
    
    # ==================== 时间维度 ====================
    
    def get_time_preset(self, preset_name: str) -> dict | None:
        """
        获取时间预设配置
        
        Args:
            preset_name: 预设名称或别名（如"今年"、"Q1"）
        """
        time_config = self._config.get("time_dimensions", {})
        presets = time_config.get("presets", {})
        
        # 直接匹配
        if preset_name in presets:
            return presets[preset_name]
        
        # 别名匹配
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
    
    # ==================== 产品维度 ====================
    
    def get_product_match(self, product_name: str) -> dict | None:
        """
        获取产品匹配配置
        
        Args:
            product_name: 产品名称或别名
        """
        products = self._config.get("product_dimensions", {}).get("popular_products", {})
        
        # 直接匹配
        if product_name in products:
            return {"name": product_name, **products[product_name]}
        
        # 别名匹配
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
        
        # 未找到配置，使用原名
        return f'{prefix}"{product_col}" = \'{product_name}\''
    
    # ==================== LLM 辅助 ====================
    
    def get_llm_hints(self) -> dict:
        """获取 LLM 提示信息"""
        return self._config.get("llm_hints", {})
    
    def get_default_assumptions(self) -> dict:
        """获取默认假设"""
        return self.get_llm_hints().get("defaults", {})
    
    def get_terminology(self) -> dict:
        """获取术语解释"""
        return self.get_llm_hints().get("terminology", {})
    
    def resolve_ambiguity(self, term: str) -> str:
        """
        解决术语歧义
        
        Args:
            term: 可能有歧义的术语
            
        Returns:
            解析后的标准术语
        """
        disambiguation = self.get_llm_hints().get("disambiguation", [])
        for rule in disambiguation:
            if rule["pattern"] in term:
                return rule["default"]
        return term
    
    # ==================== 生成 SQL 辅助 ====================
    
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
        """
        根据口径配置构建完整的 SQL 查询
        
        Args:
            table_name: 表名
            metric: 指标名称
            organization: 组织（如"IEG本部"）
            time_range: 时间范围（如"今年"、"Q1"）
            source_mode: 数据来源模式（如"实际"、"预测"）
            product: 产品名称
            group_by: 分组列名
        
        Returns:
            完整的 SQL 查询语句
        """
        # 获取默认值
        defaults = self.get_default_assumptions()
        organization = organization or defaults.get("organization", "IEG本部")
        time_range = time_range or defaults.get("time_range", "今年")
        source_mode = source_mode or defaults.get("data_source_mode", "实际")
        
        # 解决歧义
        metric = self.resolve_ambiguity(metric)
        
        # 构建 WHERE 条件
        conditions = []
        
        # 基础筛选
        base_sql = self.get_base_filters_sql()
        if base_sql:
            conditions.append(base_sql)
        
        # 组织筛选
        org_sql = self.get_organization_sql(organization)
        if org_sql:
            conditions.append(org_sql)
        
        # 数据来源模式
        source_sql = self.get_source_mode_sql(source_mode)
        if source_sql:
            conditions.append(source_sql)
        
        # 指标筛选
        metric_sql = self.get_metric_sql(metric)
        if metric_sql:
            conditions.append(metric_sql)
        
        # 时间筛选
        time_sql = self.get_time_sql(time_range)
        if time_sql:
            conditions.append(time_sql)
        
        # 产品筛选
        if product:
            product_sql = self.get_product_sql(product)
            if product_sql:
                conditions.append(product_sql)
        
        # 输出列
        output_col = self.get_metric_output_column(metric)
        
        # 构建 SQL
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
    
    # ==================== 生成 Prompt 上下文 ====================
    
    def generate_prompt_context(self) -> str:
        """
        生成供 LLM 使用的口径说明文本
        
        Returns:
            格式化的口径说明，用于注入 prompt
        """
        lines = ["## 业务口径配置说明\n"]
        
        # 1. 基础筛选说明
        lines.append("### 基础筛选条件（所有查询必须应用）")
        for col, rule in self.get_base_filters().items():
            if "eq" in rule:
                lines.append(f"- `{col}` = '{rule['eq']}'")
            elif "in" in rule:
                lines.append(f"- `{col}` IN {rule['in']}")
        lines.append("")
        
        # 2. 组织维度
        lines.append("### 组织维度")
        for name, config in self._config.get("organization_dimensions", {}).items():
            aliases = config.get("aliases", [])
            desc = config.get("description", "")
            lines.append(f"- **{name}**: {desc}（别名: {', '.join(aliases)}）")
        lines.append("")
        
        # 3. 财务指标
        lines.append("### 财务指标")
        for name, config in self._config.get("financial_metrics", {}).items():
            aliases = config.get("aliases", [])
            desc = config.get("description", "")
            lines.append(f"- **{name}**: {desc}（别名: {', '.join(aliases)}）")
        lines.append("")
        
        # 4. 时间维度
        lines.append("### 时间维度")
        for name, config in self._config.get("time_dimensions", {}).get("presets", {}).items():
            aliases = config.get("aliases", [])
            months = config.get("months", [])
            lines.append(f"- **{name}**: 月份 {months}（别名: {', '.join(aliases)}）")
        lines.append("")
        
        # 5. 数据来源模式
        lines.append("### 数据来源模式")
        for name, config in self._config.get("data_source_modes", {}).items():
            aliases = config.get("aliases", [])
            desc = config.get("description", "")
            lines.append(f"- **{name}**: {desc}（别名: {', '.join(aliases)}）")
        lines.append("")
        
        # 6. 默认假设
        lines.append("### 默认假设（用户未明确时）")
        defaults = self.get_default_assumptions()
        for key, val in defaults.items():
            lines.append(f"- {key}: {val}")
        lines.append("")
        
        # 7. 歧义处理
        lines.append("### 歧义处理规则")
        for rule in self.get_llm_hints().get("disambiguation", []):
            lines.append(f"- 当用户提到「{rule['pattern']}」时，默认使用「{rule['default']}」")
        
        return "\n".join(lines)
    
    # ==================== 内部方法 ====================
    
    def _filters_to_sql(self, filters: dict, table_alias: str = "") -> str:
        """将筛选条件字典转换为 SQL WHERE 子句"""
        conditions = []
        prefix = f'"{table_alias}".' if table_alias else ""
        
        for col, rule in filters.items():
            if "eq" in rule:
                val = rule["eq"]
                if val == "null":
                    conditions.append(f'({prefix}"{col}" IS NULL OR {prefix}"{col}" = \'null\' OR {prefix}"{col}" = \'为空\')')
                else:
                    conditions.append(f'{prefix}"{col}" = \'{val}\'')
            
            if "in" in rule:
                vals = rule["in"]
                # 处理 null 值
                null_check = ""
                non_null_vals = [v for v in vals if v not in ("null", "为空")]
                if "null" in vals or "为空" in vals:
                    null_check = f'{prefix}"{col}" IS NULL OR {prefix}"{col}" = \'null\' OR {prefix}"{col}" = \'为空\''
                
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
            
            if "gt" in rule:
                conditions.append(f'{prefix}"{col}" > {rule["gt"]}')
            
            if "lt" in rule:
                conditions.append(f'{prefix}"{col}" < {rule["lt"]}')
            
            if "between" in rule:
                left, right = rule["between"]
                conditions.append(f'{prefix}"{col}" BETWEEN {left} AND {right}')
        
        return " AND ".join(conditions)


# 便捷函数
def load_metrics_config(config_path: str = None) -> MetricsConfigLoader:
    """加载口径配置"""
    return MetricsConfigLoader(config_path)


if __name__ == "__main__":
    # 测试
    loader = MetricsConfigLoader()
    
    print("=== 基础筛选条件 SQL ===")
    print(loader.get_base_filters_sql())
    
    print("\n=== IEG本部递延后利润查询 ===")
    sql = loader.build_query_sql(
        table_name="脚本测试数据",
        metric="递延后利润",
        organization="IEG本部",
        time_range="今年",
        source_mode="实际",
    )
    print(sql)
    
    print("\n=== 口径配置 Prompt 上下文 ===")
    print(loader.generate_prompt_context()[:2000] + "...")
