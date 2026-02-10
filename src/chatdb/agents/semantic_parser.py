"""
SemanticParser - 语义解析能力

职责：从自然语言查询提取结构化意图 JSON

支持多轮补完：
- 第一轮：提取业务术语 + 时间语义
- 第二轮：匹配 YAML 中的指标/维度/筛选器
- 如果有缺口，标记 need_schema_resolve

输出格式：
{
  "intent_type": "basic" | "trend" | "complex",
  "metrics": ["metric_id"],
  "time": {
    "granularity": "year" | "quarter" | "month",
    "year": 2024,
    "quarter": 1,
    "comparison": "yoy" | "qoq" | "ytd" | null
  },
  "dimensions": ["dimension_id"],
  "filters": [
    {"dimension": "region", "value": "overseas", "operator": "="}
  ],
  "exclusions": ["exclusion_expr"],
  "order_by": {"column": "metric_id", "direction": "DESC"},
  "limit": 10
}
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re
import yaml

from chatdb.agents.base import AgentContext, AgentResult, AgentStatus
from chatdb.llm.base import BaseLLM
from chatdb.utils.logger import get_component_logger


@dataclass
class StructuredIntent:
    """
    结构化查询意图（简化版 - 只描述"要什么"，不决定"怎么分析"）
    
    设计理念：
    - SemanticParser 只负责提取"用户想查什么对象，在什么约束下"
    - 不负责判断 basic/trend/comparison/source 等分析类型
    - 分析类型由 Planner 根据意图 + 上下文动态决策
    
    核心字段：
    - mode: analysis（数据分析）/ meta（元指令）
    - tables: 涉及的表名列表
    - table_relation: 表关系（single/join/union）
    - metrics/dimensions/filters: 具体的查询约束
    """
    raw_query: str
    table_name: str = ""
    
    # === 核心字段 ===
    mode: str = "analysis"  # analysis（数据分析）/ meta（元指令：生成问题/解释结构等）
    tables: list[str] = field(default_factory=list)  # 涉及的表名列表
    table_relation: str = "single"  # single / join / union
    
    # === 数据分析相关字段（约束信息）===
    metrics: list[str] = field(default_factory=list)  # metric IDs from YAML
    dimensions: list[str] = field(default_factory=list)  # dimension IDs from YAML
    filter_refs: list[str] = field(default_factory=list)  # 预定义筛选器 ID（如 metric_flow, time_q1）
    
    time: dict[str, Any] = field(default_factory=lambda: {
        "granularity": "year",
        "year": None,
        "quarter": None,
        "month": None,
        "comparison": None,  # yoy / qoq / ytd（这是时间约束，不是分析类型）
    })
    
    filters: list[dict[str, Any]] = field(default_factory=list)  # 自定义筛选条件
    exclusions: list[str] = field(default_factory=list)
    
    order_by: dict[str, str] | None = None  # {"column": "...", "direction": "DESC"}
    limit: int | None = None
    meta_request: str | None = None  # 元查询描述（如：模拟测试问题、解释数据结构等）
    
    # === 向后兼容（intent_type/qa_type 由 Planner 设置）===
    _analysis_type: str = field(default="", repr=False)  # 由 Planner 决定，不在解析阶段设置
    
    @property
    def intent_type(self) -> str:
        """向后兼容：返回分析类型（由 Planner 设置）"""
        return self._analysis_type or "basic"
    
    @intent_type.setter
    def intent_type(self, value: str) -> None:
        self._analysis_type = value
    
    @property
    def qa_type(self) -> str:
        """向后兼容：qa_type 等同于 intent_type"""
        return self.intent_type
    
    @qa_type.setter
    def qa_type(self, value: str) -> None:
        self._analysis_type = value
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "table_name": self.table_name,
            "mode": self.mode,
            "tables": self.tables,
            "table_relation": self.table_relation,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "filter_refs": self.filter_refs,
            "time": self.time,
            "filters": self.filters,
            "exclusions": self.exclusions,
            "order_by": self.order_by,
            "limit": self.limit,
            "meta_request": self.meta_request,
            # 向后兼容
            "intent_type": self.intent_type,
            "qa_type": self.qa_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], raw_query: str = "") -> "StructuredIntent":
        intent = cls(
            raw_query=raw_query,
            table_name=data.get("table_name", ""),
            mode=data.get("mode", "analysis"),
            tables=data.get("tables", []),
            table_relation=data.get("table_relation", "single"),
            metrics=data.get("metrics", []),
            dimensions=data.get("dimensions", []),
            filter_refs=data.get("filter_refs", []),
            time=data.get("time", {}),
            filters=data.get("filters", []),
            exclusions=data.get("exclusions", []),
            order_by=data.get("order_by"),
            limit=data.get("limit"),
            meta_request=data.get("meta_request"),
        )
        # 兼容旧字段
        if data.get("qa_type") or data.get("intent_type"):
            intent._analysis_type = data.get("qa_type") or data.get("intent_type", "")
        return intent
    
    def is_meta_query(self) -> bool:
        """是否为元查询（非数据分析）"""
        return self.mode == "meta"
    
    def is_multi_table(self) -> bool:
        """是否为多表查询"""
        return self.table_relation in ("join", "union") or len(self.tables) > 1
    
    def has_time_comparison(self) -> bool:
        """是否有时间对比需求（同比/环比）"""
        return self.time.get("comparison") is not None


class SemanticParser:
    """
    语义解析能力：从自然语言提取结构化意图 JSON。
    
    支持多轮补完、YAML 匹配、严格从 YAML 选 ID，不发明新字段。
    """
    
    def __init__(self, llm: BaseLLM, yml_config: str | Path | None = None):
        self.llm = llm
        self._log = get_component_logger("SemanticParser")
        self.yml_config_path: Path | None = None
        self.yml_config_dir: Path | None = None
        
        if yml_config:
            path = Path(yml_config)
            if path.is_file():
                self.yml_config_path = path  # 直接指定文件
            elif path.is_dir():
                self.yml_config_dir = path  # 指定目录
            elif path.suffix in ('.yml', '.yaml'):
                # 文件不存在但看起来像 YAML 文件路径
                self.yml_config_path = path
            else:
                self.yml_config_dir = path
        
        self._yml_cache: dict[str, dict[str, Any]] = {}
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """执行语义解析，输出结构化意图"""
        self._log.info(f"解析: {context.user_query[:50]}...")
        
        try:
            # 1. 选择表
            table_name = await self._select_table(context)
            
            # 2. 加载 YAML 配置
            yml_config = self._load_yml_config(table_name)
            
            # 3. 使用 LLM 提取结构化意图（传入 schema_text）
            intent = await self._extract_intent(
                context.user_query, 
                table_name, 
                yml_config,
                context.schema_text,  # 传入表结构
                context.available_tables,  # 传入表元数据
            )
            
            # 4. 更新上下文
            context.selected_tables = [table_name] if table_name else []
            context.query_intent = intent
            context.yml_config = yml_config
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                message="语义解析成功",
                data={
                    "intent": intent.to_dict(),
                    "table_name": table_name,
                    "yml_config": yml_config,
                },
            )
        except Exception as e:
            self._log.error(f"解析失败: {e}")
            return AgentResult(
                status=AgentStatus.FAILED,
                message="语义解析失败",
                error=str(e),
            )
    
    async def _select_table(self, context: AgentContext) -> str:
        """选择表"""
        if context.selected_tables:
            return context.selected_tables[0]
        
        if context.available_tables and len(context.available_tables) == 1:
            return context.available_tables[0].get("table_name", "")
        
        if context.available_tables and len(context.available_tables) > 1:
            tables_desc = "\n".join([
                f"- {t.get('table_name')}: {t.get('table_description', '')}"
                for t in context.available_tables
            ])
            
            response = await self.llm.chat(
                prompt=f"用户查询：{context.user_query}\n\n可用表：\n{tables_desc}\n\n只输出最相关的表名：",
                system_prompt="选择最相关的表，只输出表名。",
                caller_name="select_table",
            )
            
            for table in context.available_tables:
                if table.get("table_name") in response:
                    return table.get("table_name", "")
        
        if context.available_tables:
            return context.available_tables[0].get("table_name", "")
        return ""
    
    def _load_yml_config(self, table_name: str) -> dict[str, Any]:
        """加载 YAML 配置"""
        if table_name in self._yml_cache:
            return self._yml_cache[table_name]
        
        # 1. 直接指定的 YAML 文件
        if self.yml_config_path and self.yml_config_path.exists():
            try:
                with open(self.yml_config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    self._yml_cache[table_name] = config
                    self._log.observe(f"加载 YAML: {self.yml_config_path}")
                    return config
            except Exception as e:
                self._log.warn(f"加载 YAML 失败: {e}")
        
        # 2. 从目录中查找
        if self.yml_config_dir:
            for name in [f"{table_name}.yml", f"{table_name}.yaml", "metrics_config.yml"]:
                yml_path = self.yml_config_dir / name
                if yml_path.exists():
                    try:
                        with open(yml_path, "r", encoding="utf-8") as f:
                            config = yaml.safe_load(f)
                            self._yml_cache[table_name] = config
                            self._log.observe(f"加载 YAML: {yml_path}")
                            return config
                    except Exception as e:
                        self._log.warn(f"加载 YAML 失败: {e}")
        
        return {}
    
    async def _extract_intent(
        self,
        query: str,
        table_name: str,
        yml_config: dict[str, Any],
        schema_text: str = "",
        tables_meta: list[dict[str, Any]] | None = None,
    ) -> StructuredIntent:
        """使用 LLM 提取结构化意图"""
        
        # 判断是否有 YAML 配置
        has_yml = bool(yml_config.get("metrics") or yml_config.get("dimensions") or yml_config.get("filters"))
        
        if has_yml:
            # 有 YAML 配置：基于配置提取
            return await self._extract_intent_with_yml(query, table_name, yml_config)
        else:
            # 无 YAML 配置：基于 schema 提取
            return await self._extract_intent_from_schema(query, table_name, schema_text, tables_meta)
    
    async def _extract_intent_with_yml(
        self,
        query: str,
        table_name: str,
        yml_config: dict[str, Any],
    ) -> StructuredIntent:
        """
        基于 YAML 配置提取意图（简化版 - 只提取约束，不判断分析类型）
        
        设计理念：
        - 只负责提取"用户想查什么对象，在什么约束下"
        - 不判断 basic/trend/comparison/source 等分析类型（由 Planner 决定）
        - 只判断 mode: analysis vs meta（这是职责边界，不是分析类型）
        """
        
        # 构建可用选项列表
        available_business_terms = self._format_business_terms(yml_config.get("business_terms", []))
        available_metrics = self._format_metrics(yml_config.get("metrics", {}))
        available_dimensions = self._format_dimensions(yml_config.get("dimensions", {}))
        available_filters = self._format_filters(yml_config.get("filters", {}))
        rules = self._format_rules(yml_config.get("rules", []))
        
        prompt = f"""请从以下查询中提取结构化意图（只提取"要查什么"，不负责判断"怎么分析"）。

## 用户查询
{query}

## 可用表
- {table_name}

## 业务术语词典（根据同义词匹配，识别用户意图）
{available_business_terms}

## 可用指标（只能从这里选择 ID）
{available_metrics}

## 可用维度（只能从这里选择 ID）
{available_dimensions}

## 可用筛选器（filter_id 对应具体 SQL 表达式）
{available_filters}

## 业务规则
{rules}

## 输出 JSON（只描述约束条件，不判断分析类型）：
{{
  "mode": "analysis|meta",
  "tables": ["{table_name}"],
  "table_relation": "single|join|union",
  "metrics": ["从指标列表选择ID"],
  "dimensions": ["从维度列表选择ID"],
  "filter_refs": ["从筛选器列表选择ID"],
  "time": {{
    "granularity": "year|quarter|month|null",
    "year": null或具体年份,
    "quarter": null或1-4,
    "month": null或1-12,
    "comparison": null|"yoy"|"qoq"|"ytd"
  }},
  "filters": [
    {{"dimension": "维度ID", "value": "值", "operator": "=|!=|>|<|IN|LIKE"}}
  ],
  "exclusions": ["排除表达式"],
  "order_by": null或{{"column": "维度或指标ID", "direction": "DESC|ASC"}},
  "limit": null或数字,
  "meta_request": null或"如果是meta模式，描述用户想要什么"
}}

### 关键规则：

**mode 判断**（这是唯一的分类决策）：
- 用户在问数据相关问题（查指标/趋势/对比等）→ mode=analysis
- 用户在请求元指令（与sql生成无关的问题）→ mode=meta

**约束提取规则**：
- time.comparison: 只是标记时间约束（yoy=同比, qoq=环比, ytd=年累计），不代表分析类型
- 如果查询提到"剔除"/"去掉"/"不含"，放入 exclusions
- 优先使用 filter_refs 引用预定义筛选器

只输出 JSON，不要其他文字。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是意图提取器。只提取约束条件，不判断分析类型。严格按 JSON 格式输出。",
                caller_name="extract_intent",
            )
            
            intent_dict = self._parse_json_response(response)
            intent = StructuredIntent.from_dict(intent_dict, raw_query=query)
            
            # === 仅做校验和补默认值 ===
            
            # 补充表名
            if not intent.tables and table_name:
                intent.tables = [table_name]
            if not intent.table_name and table_name:
                intent.table_name = table_name
            
            # 校验 metrics ID 是否存在于 YAML
            valid_metrics = set(yml_config.get("metrics", {}).keys())
            if valid_metrics and intent.metrics:
                intent.metrics = [m for m in intent.metrics if m in valid_metrics]
            
            # 校验 dimensions ID 是否存在于 YAML
            valid_dims = set(yml_config.get("dimensions", {}).keys())
            if valid_dims and intent.dimensions:
                intent.dimensions = [d for d in intent.dimensions if d in valid_dims]
            
            # 校验 filter_refs ID 是否存在于 YAML
            valid_filters = set(yml_config.get("filters", {}).keys())
            if valid_filters and intent.filter_refs:
                intent.filter_refs = [f for f in intent.filter_refs if f in valid_filters]
            
            self._log.observe(
                f"提取意图(YAML): mode={intent.mode}, metrics={intent.metrics}, "
                f"dimensions={intent.dimensions}, filter_refs={intent.filter_refs}"
            )
            return intent
            
        except Exception as e:
            self._log.error(f"意图提取失败: {e}")
            return StructuredIntent(raw_query=query, table_name=table_name)
    
    async def _extract_intent_from_schema(
        self,
        query: str,
        table_name: str,
        schema_text: str,
        tables_meta: list[dict[str, Any]] | None = None,
    ) -> StructuredIntent:
        """
        基于 Schema 提取意图（无 YAML 配置时，简化版）
        
        只提取约束条件，不判断分析类型。
        """
        
        # 构建列信息
        columns_info = self._extract_columns_from_meta(tables_meta, table_name) if tables_meta else schema_text
        
        # 如果仍然没有列信息，尝试从 schema_text 提取
        if columns_info == "无列信息" and schema_text:
            columns_info = schema_text
        
        # 构建可用表列表
        available_tables = ""
        if tables_meta:
            available_tables = "\n".join([
                f"- {t.get('table_name', '')}: {t.get('table_description', '')}"
                for t in tables_meta
            ])
        
        self._log.think(f"提取列信息: tables_meta={len(tables_meta) if tables_meta else 0}个表, table_name={table_name}")
        
        prompt = f"""请从以下查询中提取结构化意图（只提取约束条件，不判断分析类型）。

## 用户查询
{query}

## 当前表名
{table_name}

## 可用表（如果涉及多表）
{available_tables or "仅当前表"}

## 表结构（实际列名）
{columns_info}

## 输出 JSON（只描述约束，不判断分析类型）：
{{
  "mode": "analysis|meta",
  "tables": ["主表名"],
  "table_relation": "single|join|union",
  "agg_column": "要聚合的数值列名（从上面列中选择）",
  "agg_func": "SUM|COUNT|AVG|MAX|MIN",
  "group_by_columns": ["分组列名"],
  "time_column": "时间筛选用的列名（如 年、月份 等）",
  "time_value": "时间值（如 2025）",
  "time_comparison": null|"yoy"|"qoq"|"ytd",
  "filter_columns": [
    {{"column": "实际列名", "value": "筛选值", "operator": "=|!=|>|<|IN|LIKE"}}
  ],
  "order_by": null或{{"column": "列名", "direction": "DESC|ASC"}},
  "limit": null或数字,
  "meta_request": null或"如果是meta模式，描述用户想要什么"
}}

### 关键规则：

**mode 判断**（唯一的分类决策）：
- 数据查询问题 → mode=analysis
- 元指令（与sql分析无关的问题）→ mode=meta

**重要**：
1. 所有列名必须从"表结构"中选择
2. 时间筛选：查询提到"2025年"时，找表中类似"年"的列
3. 数值聚合：找金额/数量类的数值列

只输出 JSON，不要其他文字。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是意图提取器。只提取约束条件，不判断分析类型。",
                caller_name="extract_intent_schema",
            )
            
            intent_dict = self._parse_json_response(response)
            
            # 转换为 StructuredIntent
            intent = StructuredIntent(
                raw_query=query,
                table_name=table_name,
                mode=intent_dict.get("mode", "analysis"),
                tables=intent_dict.get("tables", [table_name] if table_name else []),
                table_relation=intent_dict.get("table_relation", "single"),
            )
            
            # 存储 schema 模式特有的字段到 filters 中
            if intent_dict.get("agg_column"):
                intent.filters.append({
                    "_agg_column": intent_dict.get("agg_column"),
                    "_agg_func": intent_dict.get("agg_func", "SUM"),
                })
            
            if intent_dict.get("group_by_columns"):
                intent.dimensions = intent_dict.get("group_by_columns", [])
            
            if intent_dict.get("time_column") and intent_dict.get("time_value"):
                intent.filters.append({
                    "column": intent_dict.get("time_column"),
                    "value": intent_dict.get("time_value"),
                    "operator": "=",
                })
                intent.time["year"] = intent_dict.get("time_value") if isinstance(intent_dict.get("time_value"), int) else None
            
            # 时间对比约束（不是分析类型）
            if intent_dict.get("time_comparison"):
                intent.time["comparison"] = intent_dict.get("time_comparison")
            
            for f in intent_dict.get("filter_columns", []):
                intent.filters.append(f)
            
            intent.order_by = intent_dict.get("order_by")
            intent.limit = intent_dict.get("limit")
            intent.meta_request = intent_dict.get("meta_request")
            
            self._log.observe(
                f"提取意图(Schema): mode={intent.mode}, "
                f"dims={intent.dimensions}, filters={len(intent.filters)}"
            )
            return intent
            
        except Exception as e:
            self._log.error(f"Schema意图提取失败: {e}")
            return StructuredIntent(raw_query=query, table_name=table_name)
    
    def _extract_columns_from_meta(self, tables_meta: list[dict[str, Any]], table_name: str) -> str:
        """从 tables_meta 提取列信息"""
        for table in tables_meta:
            if table.get("table_name") == table_name:
                # 兼容不同的键名：columns_info 或 columns
                columns = table.get("columns_info") or table.get("columns", [])
                if columns:
                    lines = []
                    for col in columns:
                        col_name = col.get("name", col.get("column_name", ""))
                        col_type = col.get("type", col.get("column_type", ""))
                        lines.append(f"- {col_name} ({col_type})")
                    return "\n".join(lines)
        return "无列信息"
    
    def _format_business_terms(self, business_terms: list[dict[str, Any]]) -> str:
        """格式化业务术语列表供 LLM 召回"""
        if not business_terms:
            return "无业务术语定义"
        
        lines = []
        for term in business_terms:
            term_id = term.get("id", "")
            term_name = term.get("term", "")
            synonyms = term.get("synonyms", [])
            related_filters = term.get("related_filters", [])
            
            syn_str = f" (同义词: {', '.join(synonyms)})" if synonyms else ""
            filter_str = f" → 关联筛选器: {', '.join(related_filters)}" if related_filters else ""
            lines.append(f"- {term_id}: {term_name}{syn_str}{filter_str}")
        return "\n".join(lines)
    
    def _format_metrics(self, metrics: dict[str, Any]) -> str:
        """格式化指标列表供 LLM 选择"""
        if not metrics:
            return "无可用指标"
        
        lines = []
        for metric_id, metric_def in metrics.items():
            label = metric_def.get("label", "")
            synonyms = metric_def.get("synonyms", [])
            syn_str = f" (别名: {', '.join(synonyms)})" if synonyms else ""
            lines.append(f"- {metric_id}: {label}{syn_str}")
        return "\n".join(lines)
    
    def _format_dimensions(self, dimensions: dict[str, Any]) -> str:
        """格式化维度列表供 LLM 选择"""
        if not dimensions:
            return "无可用维度"
        
        lines = []
        for dim_id, dim_def in dimensions.items():
            label = dim_def.get("label", "")
            terms = dim_def.get("terms", {})
            
            term_str = ""
            if terms:
                term_labels = [t.get("term", k) for k, t in terms.items()]
                term_str = f" (含术语: {', '.join(term_labels[:5])})"
            
            lines.append(f"- {dim_id}: {label}{term_str}")
        return "\n".join(lines)
    
    def _format_filters(self, filters: dict[str, Any]) -> str:
        """格式化筛选器列表"""
        if not filters:
            return "无预定义筛选器"
        
        lines = []
        for filter_id, filter_def in filters.items():
            label = filter_def.get("label", "")
            description = filter_def.get("description", "")
            lines.append(f"- {filter_id}: {label} - {description}")
        return "\n".join(lines)
    
    def _format_rules(self, rules: list[dict[str, Any]]) -> str:
        """格式化业务规则"""
        if not rules:
            return "无特殊规则"
        
        lines = []
        for rule in rules:
            rule_type = rule.get("type", "")
            description = rule.get("description", "")
            if rule_type == "disambiguation":
                match_pattern = rule.get("match_pattern", "")
                default_value = rule.get("default_value", "")
                lines.append(f"- 歧义消解: 提到'{match_pattern}'时，默认指'{default_value}' ({description})")
            elif rule_type == "default_value":
                field = rule.get("field", "")
                default_filter = rule.get("default_filter", "")
                lines.append(f"- 默认值: {field} 未指定时使用 {default_filter} ({description})")
        return "\n".join(lines) if lines else "无特殊规则"
    
    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        self._log.warn(f"JSON 解析失败: {response[:200]}")
        return {}
    
    def get_system_prompt(self) -> str:
        return "你是意图提取器，将自然语言查询转换为结构化 JSON。"
