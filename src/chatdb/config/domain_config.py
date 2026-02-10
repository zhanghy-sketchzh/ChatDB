"""
渐进式领域配置加载器

实现两阶段配置加载策略：
1. 第一阶段：加载所有领域的概述（~200 tokens），供 LLM 决策关联领域
2. 第二阶段：按需加载选中领域的详细配置（~1000 tokens/领域）

优势：
- 第一步轻量级：只看领域概述，Token 消耗极少
- 第二步精准化：只加载相关领域的详细信息
- 避免信息污染：不相关领域的复杂信息完全不会干扰模型
- 便于扩展：扩展领域只需要按照设定格式添加对应的文档即可
"""

from pathlib import Path
from typing import Any

import yaml

from chatdb.utils.logger import logger


class DomainConfigLoader:
    """领域配置渐进式加载器"""
    
    def __init__(self, config_path: str | Path | None = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为 data/yml/config.yml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "yml" / "config.yml"
        
        self.config_path = Path(config_path)
        self._config: dict = {}
        self._domain_summaries: dict[str, dict] = {}  # 领域概述缓存
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}")
            return
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}
        
        # 预处理领域概述
        self._build_domain_summaries()
    
    def _build_domain_summaries(self) -> None:
        """构建所有领域的概述信息"""
        self._domain_summaries = {
            "base_filters": {
                "name": "基础筛选条件",
                "description": "所有查询默认应用的筛选条件（统一剔除标签、考核口径、特殊口径）",
                "keywords": ["剔除", "考核口径", "特殊口径", "默认筛选"],
            },
            "data_source_modes": {
                "name": "数据来源模式",
                "description": "数据类型筛选（实际数据/预测数据/预算数据）",
                "keywords": ["实际", "预测", "预算", "forecast", "budget", "actual"],
            },
            "organization_dimensions": {
                "name": "组织维度",
                "description": "组织范围筛选（IEG本部/控股投资公司抵消前/抵消后）",
                "keywords": ["IEG", "本部", "投资公司", "抵消前", "抵消后", "控股", "子公司"],
            },
            "financial_metrics": {
                "name": "财务指标",
                "description": "财务指标定义（流水/收入/成本/利润/HC等）",
                "keywords": ["流水", "收入", "成本", "利润", "递延", "毛收入", "Gross", "HC", "人数", "费用"],
            },
            "composite_metrics": {
                "name": "复合指标",
                "description": "需要计算的派生指标（利润率/成本率）",
                "keywords": ["利润率", "成本率", "比率", "率"],
            },
            "time_dimensions": {
                "name": "时间维度",
                "description": "时间范围筛选（今年/Q1-Q4/上下半年）",
                "keywords": ["今年", "本年", "Q1", "Q2", "Q3", "Q4", "季度", "上半年", "下半年", "月份"],
            },
            "product_dimensions": {
                "name": "产品维度",
                "description": "产品筛选（产品大类/热门产品别名映射）",
                "keywords": ["产品", "手游", "端游", "王者", "和平", "LOL", "CF", "游戏"],
            },
            "llm_hints": {
                "name": "LLM 提示增强",
                "description": "术语解释、默认假设和歧义处理规则",
                "keywords": ["默认", "歧义", "术语"],
            },
        }
    
    # ==================== 第一阶段：领域概述 ====================
    
    def get_all_domain_summaries(self) -> str:
        """
        获取所有领域的概述（第一阶段）
        
        返回轻量级的领域描述，供 LLM 决策关联领域
        Token 消耗约 200 tokens
        
        Returns:
            格式化的领域概述文本
        """
        lines = [
            "## 可用业务领域配置",
            "",
            "以下是所有可用的业务口径领域，请根据用户查询选择需要加载的领域：",
            "",
        ]
        
        for domain_id, summary in self._domain_summaries.items():
            if domain_id not in self._config:
                continue
            
            name = summary["name"]
            desc = summary["description"]
            keywords = summary.get("keywords", [])
            
            lines.append(f"- **{domain_id}** ({name})")
            lines.append(f"  描述: {desc}")
            if keywords:
                lines.append(f"  关键词: {', '.join(keywords[:5])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_domain_keywords_map(self) -> dict[str, list[str]]:
        """
        获取领域关键词映射表
        
        Returns:
            {domain_id: [keywords]}
        """
        return {
            domain_id: summary.get("keywords", [])
            for domain_id, summary in self._domain_summaries.items()
            if domain_id in self._config
        }
    
    # ==================== 第二阶段：领域详情 ====================
    
    def get_domain_details(self, domain_ids: list[str]) -> str:
        """
        获取指定领域的详细配置（第二阶段）
        
        Args:
            domain_ids: 要加载的领域 ID 列表
            
        Returns:
            格式化的领域详细配置文本
        """
        if not domain_ids:
            return ""
        
        lines = ["## 业务口径详细配置", ""]
        
        for domain_id in domain_ids:
            if domain_id not in self._config:
                continue
            
            domain_config = self._config[domain_id]
            domain_name = self._domain_summaries.get(domain_id, {}).get("name", domain_id)
            
            lines.append(f"### {domain_name} ({domain_id})")
            lines.append("")
            
            # 根据领域类型格式化详情
            if domain_id == "base_filters":
                lines.append(self._format_base_filters(domain_config))
            elif domain_id == "data_source_modes":
                lines.append(self._format_data_source_modes(domain_config))
            elif domain_id == "organization_dimensions":
                lines.append(self._format_organization_dimensions(domain_config))
            elif domain_id == "financial_metrics":
                lines.append(self._format_financial_metrics(domain_config))
            elif domain_id == "composite_metrics":
                lines.append(self._format_composite_metrics(domain_config))
            elif domain_id == "time_dimensions":
                lines.append(self._format_time_dimensions(domain_config))
            elif domain_id == "product_dimensions":
                lines.append(self._format_product_dimensions(domain_config))
            elif domain_id == "llm_hints":
                lines.append(self._format_llm_hints(domain_config))
            else:
                # 通用格式化
                lines.append(yaml.dump(domain_config, allow_unicode=True, default_flow_style=False))
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_domain_raw(self, domain_id: str) -> dict:
        """获取领域的原始配置（用于 SQL 生成）"""
        return self._config.get(domain_id, {})
    
    # ==================== 智能领域识别 ====================
    
    def identify_relevant_domains(self, user_query: str) -> list[str]:
        """
        基于关键词识别用户查询可能相关的领域
        
        这是一个简单的基于规则的预筛选，最终决策由 LLM 完成
        
        Args:
            user_query: 用户查询
            
        Returns:
            可能相关的领域 ID 列表
        """
        query_lower = user_query.lower()
        relevant_domains = []
        
        # 基础筛选条件始终相关
        relevant_domains.append("base_filters")
        
        # 关键词匹配
        for domain_id, summary in self._domain_summaries.items():
            if domain_id == "base_filters":
                continue
            
            keywords = summary.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    if domain_id not in relevant_domains:
                        relevant_domains.append(domain_id)
                    break
        
        # 如果识别不到具体领域，默认添加财务指标和组织维度
        if len(relevant_domains) <= 1:
            if "financial_metrics" not in relevant_domains:
                relevant_domains.append("financial_metrics")
            if "organization_dimensions" not in relevant_domains:
                relevant_domains.append("organization_dimensions")
        
        return relevant_domains
    
    # ==================== 领域格式化方法 ====================
    
    def _format_base_filters(self, config: dict) -> str:
        """格式化基础筛选条件"""
        lines = ["所有查询必须应用以下筛选条件：", ""]
        
        for col, rule in config.items():
            if "eq" in rule:
                val = rule["eq"]
                if val == "null":
                    lines.append(f"- `{col}` 为空（IS NULL 或 = 'null' 或 = '为空'）")
                else:
                    lines.append(f"- `{col}` = '{val}'")
            elif "in" in rule:
                vals = rule["in"]
                if len(vals) <= 5:
                    lines.append(f"- `{col}` IN {vals}")
                else:
                    lines.append(f"- `{col}` IN {vals[:5]}... (共{len(vals)}个值)")
        
        return "\n".join(lines)
    
    def _format_data_source_modes(self, config: dict) -> str:
        """格式化数据来源模式"""
        lines = ["数据来源模式决定查询实际/预测/预算数据：", ""]
        
        for mode_name, mode_config in config.items():
            desc = mode_config.get("description", "")
            aliases = mode_config.get("aliases", [])
            filters = mode_config.get("filters", {})
            
            lines.append(f"**{mode_name}**: {desc}")
            if aliases:
                lines.append(f"  别名: {', '.join(aliases)}")
            
            for col, rule in filters.items():
                if "in" in rule:
                    lines.append(f"  筛选: `{col}` IN {rule['in']}")
                elif "eq" in rule:
                    lines.append(f"  筛选: `{col}` = '{rule['eq']}'")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_organization_dimensions(self, config: dict) -> str:
        """格式化组织维度"""
        lines = ["组织维度决定查询范围（本部/投资公司）：", ""]
        
        for org_name, org_config in config.items():
            desc = org_config.get("description", "")
            aliases = org_config.get("aliases", [])
            filters = org_config.get("filters", {})
            
            lines.append(f"**{org_name}**: {desc}")
            if aliases:
                lines.append(f"  别名: {', '.join(aliases)}")
            
            for col, rule in filters.items():
                if "eq" in rule:
                    lines.append(f"  筛选: `{col}` = '{rule['eq']}'")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_financial_metrics(self, config: dict) -> str:
        """格式化财务指标"""
        lines = ["财务指标定义（通过 `大盘报表项` 字段筛选）：", ""]
        
        # 按类型分组
        revenue_metrics = []
        cost_metrics = []
        profit_metrics = []
        other_metrics = []
        
        for metric_name, metric_config in config.items():
            desc = metric_config.get("description", "")
            aliases = metric_config.get("aliases", [])
            value = metric_config.get("value", metric_name)
            
            metric_info = {
                "name": metric_name,
                "desc": desc,
                "aliases": aliases,
                "value": value,
            }
            
            # 分类
            if any(kw in metric_name for kw in ["流水", "收入", "税", "Gross"]):
                revenue_metrics.append(metric_info)
            elif any(kw in metric_name for kw in ["成本", "费用", "费", "投入", "支持"]):
                cost_metrics.append(metric_info)
            elif "利润" in metric_name:
                profit_metrics.append(metric_info)
            else:
                other_metrics.append(metric_info)
        
        # 输出分组
        if revenue_metrics:
            lines.append("**收入类指标**:")
            for m in revenue_metrics:
                lines.append(f"  - {m['name']}: {m['desc']}")
                if m['aliases']:
                    lines.append(f"    别名: {', '.join(m['aliases'][:3])}")
            lines.append("")
        
        if cost_metrics:
            lines.append("**成本类指标**:")
            for m in cost_metrics:
                lines.append(f"  - {m['name']}: {m['desc']}")
                if m['aliases']:
                    lines.append(f"    别名: {', '.join(m['aliases'][:3])}")
            lines.append("")
        
        if profit_metrics:
            lines.append("**利润类指标**:")
            for m in profit_metrics:
                lines.append(f"  - {m['name']}: {m['desc']}")
                if m['aliases']:
                    lines.append(f"    别名: {', '.join(m['aliases'][:3])}")
            lines.append("")
        
        if other_metrics:
            lines.append("**其他指标**:")
            for m in other_metrics:
                lines.append(f"  - {m['name']}: {m['desc']}")
                if m['aliases']:
                    lines.append(f"    别名: {', '.join(m['aliases'][:3])}")
            lines.append("")
        
        lines.append("**注意**: 使用 `大盘报表项` 字段筛选，输出列为 `ieg口径金额-人民币`")
        
        return "\n".join(lines)
    
    def _format_composite_metrics(self, config: dict) -> str:
        """格式化复合指标"""
        lines = ["复合指标需要通过公式计算：", ""]
        
        for metric_name, metric_config in config.items():
            desc = metric_config.get("description", "")
            aliases = metric_config.get("aliases", [])
            formula = metric_config.get("formula", "")
            unit = metric_config.get("unit", "")
            
            lines.append(f"**{metric_name}**: {desc}")
            if aliases:
                lines.append(f"  别名: {', '.join(aliases)}")
            if formula:
                lines.append(f"  公式: {formula}")
            if unit:
                lines.append(f"  单位: {unit}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_time_dimensions(self, config: dict) -> str:
        """格式化时间维度"""
        lines = ["时间维度配置：", ""]
        
        column = config.get("column", "月份")
        format_str = config.get("format", "YYYYMM")
        presets = config.get("presets", {})
        
        lines.append(f"时间列: `{column}`，格式: {format_str}")
        lines.append("")
        lines.append("**时间预设**:")
        
        for preset_name, preset_config in presets.items():
            desc = preset_config.get("description", "")
            aliases = preset_config.get("aliases", [])
            months = preset_config.get("months", [])
            
            lines.append(f"  - {preset_name}: {desc}")
            if aliases:
                lines.append(f"    别名: {', '.join(aliases)}")
            if months:
                if len(months) <= 6:
                    lines.append(f"    月份: {months}")
                else:
                    lines.append(f"    月份: {months[:3]}...{months[-3:]} (共{len(months)}个月)")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_product_dimensions(self, config: dict) -> str:
        """格式化产品维度"""
        lines = ["产品维度配置：", ""]
        
        product_column = config.get("column", "考核产品")
        categories = config.get("categories", {})
        popular_products = config.get("popular_products", {})
        
        lines.append(f"产品列: `{product_column}`")
        lines.append("")
        
        if categories:
            category_column = categories.get("column", "产品大类")
            values = categories.get("values", [])
            lines.append(f"**产品大类** (`{category_column}`): {values}")
            lines.append("")
        
        if popular_products:
            lines.append("**热门产品别名映射**:")
            for product_name, product_config in popular_products.items():
                aliases = product_config.get("aliases", [])
                lines.append(f"  - {product_name}: {', '.join(aliases)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_llm_hints(self, config: dict) -> str:
        """格式化 LLM 提示"""
        lines = []
        
        terminology = config.get("terminology", {})
        defaults = config.get("defaults", {})
        disambiguation = config.get("disambiguation", [])
        
        if terminology:
            lines.append("**术语解释**:")
            for term, explanation in terminology.items():
                lines.append(f"  - {term}: {explanation}")
            lines.append("")
        
        if defaults:
            lines.append("**默认假设**（用户未明确时）:")
            for key, val in defaults.items():
                lines.append(f"  - {key}: {val}")
            lines.append("")
        
        if disambiguation:
            lines.append("**歧义处理规则**:")
            for rule in disambiguation:
                pattern = rule.get("pattern", "")
                default = rule.get("default", "")
                note = rule.get("note", "")
                lines.append(f"  - 当用户提到「{pattern}」时 → 默认「{default}」")
                if note:
                    lines.append(f"    说明: {note}")
            lines.append("")
        
        return "\n".join(lines)
    
    # ==================== 便捷方法 ====================
    
    @property
    def config(self) -> dict:
        """获取完整配置"""
        return self._config
    
    def has_domain(self, domain_id: str) -> bool:
        """检查领域是否存在"""
        return domain_id in self._config


# ==================== 两阶段 Prompt 构建器 ====================

class TwoStagePromptBuilder:
    """
    两阶段 Prompt 构建器
    
    实现渐进式披露策略：
    1. 第一阶段 Prompt：领域概述 + 领域选择指令
    2. 第二阶段 Prompt：选中领域详情 + SQL 生成指令
    """
    
    def __init__(self, config_loader: DomainConfigLoader | None = None):
        self.loader = config_loader or DomainConfigLoader()
    
    def build_stage1_prompt(self, user_query: str, schema_text: str) -> str:
        """
        构建第一阶段 Prompt（领域选择）
        
        Args:
            user_query: 用户查询
            schema_text: 数据表 Schema
            
        Returns:
            第一阶段 Prompt
        """
        domain_summaries = self.loader.get_all_domain_summaries()
        
        return f"""你是一个业务口径识别助手。根据用户的查询，从可用领域中选择需要加载的配置。

{domain_summaries}

## 用户查询
{user_query}

## 数据表 Schema（参考）
{schema_text[:1000]}...

## 任务
请分析用户查询，返回需要加载的领域 ID 列表。

**输出格式**（JSON）：
```json
{{
  "selected_domains": ["base_filters", "financial_metrics", ...],
  "reasoning": "简要说明选择理由"
}}
```

**注意**：
1. `base_filters` 是必选的（所有查询都需要应用基础筛选）
2. 只选择与查询直接相关的领域，避免加载不必要的配置
3. 如果用户查询涉及财务指标（如利润、收入、成本），必须选择 `financial_metrics`
4. 如果用户查询涉及组织范围（如 IEG、本部、投资公司），必须选择 `organization_dimensions`
"""
    
    def build_stage2_prompt(
        self, 
        user_query: str, 
        schema_text: str,
        selected_domains: list[str],
        rewritten_query: str | None = None,
    ) -> str:
        """
        构建第二阶段 Prompt（SQL 生成）
        
        Args:
            user_query: 用户原始查询
            schema_text: 数据表 Schema
            selected_domains: 选中的领域 ID 列表
            rewritten_query: 改写后的查询（可选）
            
        Returns:
            第二阶段 Prompt（包含领域详细配置）
        """
        domain_details = self.loader.get_domain_details(selected_domains)
        
        effective_query = rewritten_query or user_query
        
        return f"""你是一个 SQL 生成助手。根据业务口径配置，将用户查询转换为精确的 SQL。

{domain_details}

## 数据表 Schema
{schema_text}

## 用户查询
原始查询: {user_query}
{"改写后查询: " + rewritten_query if rewritten_query else ""}

## 任务
根据上述业务口径配置，生成查询 SQL。

**要求**：
1. 必须应用 `base_filters` 中的所有筛选条件
2. 根据用户查询应用相应的组织维度、指标、时间等筛选
3. 指标筛选使用 `大盘报表项` 字段
4. 金额输出使用 `ieg口径金额-人民币` 字段
5. 如果用户未明确组织范围，默认使用 IEG本部（数据集来源 = '一方报表'）
6. 如果用户未明确时间范围，默认使用今年全年

**输出格式**（JSON）：
```json
{{
  "sql": "生成的 SQL 语句",
  "explanation": "SQL 解释",
  "filters_applied": {{
    "base": true,
    "organization": "IEG本部",
    "metric": "递延后利润",
    "time": "今年"
  }}
}}
```
"""
    
    def get_preidentified_domains(self, user_query: str) -> list[str]:
        """
        预识别可能相关的领域（基于关键词）
        
        用于优化：如果关键词匹配明确，可以跳过第一阶段
        
        Args:
            user_query: 用户查询
            
        Returns:
            预识别的领域 ID 列表
        """
        return self.loader.identify_relevant_domains(user_query)


# 便捷函数
def load_domain_config(config_path: str | Path | None = None) -> DomainConfigLoader:
    """加载领域配置"""
    return DomainConfigLoader(config_path)


def create_prompt_builder(config_path: str | Path | None = None) -> TwoStagePromptBuilder:
    """创建两阶段 Prompt 构建器"""
    loader = DomainConfigLoader(config_path)
    return TwoStagePromptBuilder(loader)


if __name__ == "__main__":
    # 测试
    loader = DomainConfigLoader()
    builder = TwoStagePromptBuilder(loader)
    
    test_queries = [
        "IEG本部的递延后利润是多少",
        "Q1的流水",
        "王者荣耀的Gross收入",
        "投资公司抵消后的成本",
    ]
    
    print("=" * 60)
    print("第一阶段：领域概述")
    print("=" * 60)
    print(loader.get_all_domain_summaries())
    
    print("\n" + "=" * 60)
    print("测试关键词识别")
    print("=" * 60)
    for query in test_queries:
        domains = loader.identify_relevant_domains(query)
        print(f"\n查询: {query}")
        print(f"识别领域: {domains}")
    
    print("\n" + "=" * 60)
    print("第二阶段：领域详情示例")
    print("=" * 60)
    detail_domains = ["base_filters", "organization_dimensions", "financial_metrics"]
    print(loader.get_domain_details(detail_domains)[:2000] + "...")
