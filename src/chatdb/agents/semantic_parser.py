"""
SemanticParser - 语义解析能力

职责：从自然语言查询提取结构化意图 JSON

v4 设计：
- SemanticParser 提取：task_type, metrics, dimensions, conditions
- task_type 决定分析类型（basic/ratio/comparison/ranking/trend/source）
- 具体参数（order_by/limit等）由 SQLAgent 根据 task_type 自行处理

输出格式：
{
  "mode": "analysis" | "other",
  "task_type": "basic" | "ratio" | "comparison" | "ranking" | "trend" | "source",
  "metrics": ["metric_id"],
  "dimensions": ["dimension_id"],
  "conditions": [
    {"type": "ref", "id": "filter_id"},
    {"type": "custom", "column": "列名", "op": "=", "value": "值"}
  ]
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
    结构化查询意图（v4 精简版）
    
    核心字段：
    - task_type: 任务类型（basic/ratio/comparison/ranking/trend/source）
    - metrics: 要查的指标
    - dimensions: 要分组的维度
    - conditions: 筛选条件
    
    mode 只有两种：
    - analysis: 与数据分析有关的问题（需要生成 SQL）
    - other: 与数据分析无关的问题（闲聊、解释概念等）
    
    注意：order_by/limit 等具体参数由 SQLAgent 根据 task_type 自行处理
    """
    raw_query: str
    table_name: str = ""
    
    # === 核心字段 ===
    mode: str = "analysis"  # analysis / other
    task_type: str = "basic"  # basic/ratio/comparison/ranking/trend/source
    tables: list[str] = field(default_factory=list)
    table_relation: str = "single"  # single / join / union
    
    # === 数据分析相关字段 ===
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    
    # === 向后兼容字段 ===
    _filter_refs: list[str] = field(default_factory=list)
    _filters: list[dict[str, Any]] = field(default_factory=list)
    time: dict[str, Any] = field(default_factory=dict)
    _analysis_type: str = field(default="", repr=False)
    
    # === 其他字段 ===
    other_request: str | None = None  # mode=other 时的请求描述
    
    def __post_init__(self):
        """初始化后处理"""
        self._migrate_legacy_fields()
        self._sync_task_type()
    
    def _migrate_legacy_fields(self):
        """将旧版 filter_refs 和 filters 迁移到 conditions"""
        for ref_id in self._filter_refs:
            if not any(c.get("type") == "ref" and c.get("id") == ref_id for c in self.conditions):
                self.conditions.append({"type": "ref", "id": ref_id})
        
        for f in self._filters:
            if any(k.startswith("_") for k in f.keys()):
                self.conditions.append(f)
                continue
            col = f.get("column") or f.get("dimension", "")
            val = f.get("value")
            op = f.get("operator", "=")
            if col and val is not None:
                self.conditions.append({"type": "custom", "column": col, "op": op, "value": val})
    
    def _sync_task_type(self):
        """同步 task_type 和 _analysis_type（向后兼容）"""
        # 优先使用 task_type，同步到 _analysis_type
        if self.task_type and self.task_type != "basic":
            self._analysis_type = self.task_type
        # 如果 _analysis_type 有值但 task_type 没有，则反向同步
        elif self._analysis_type and not self.task_type:
            self.task_type = self._analysis_type
    
    @property
    def filter_refs(self) -> list[str]:
        """向后兼容：从 conditions 动态获取所有预定义筛选器引用"""
        return [c["id"] for c in self.conditions if c.get("type") == "ref"]
    
    @filter_refs.setter
    def filter_refs(self, value: list[str]) -> None:
        """设置 filter_refs 时，自动更新 conditions"""
        # 移除旧的 ref 类型 conditions
        self.conditions = [c for c in self.conditions if c.get("type") != "ref"]
        # 添加新的
        for ref_id in value:
            self.conditions.append({"type": "ref", "id": ref_id})
    
    @property
    def filters(self) -> list[dict[str, Any]]:
        """向后兼容：从 conditions 动态获取所有自定义筛选条件"""
        result = []
        for c in self.conditions:
            if c.get("type") == "custom":
                result.append({
                    "column": c.get("column"),
                    "value": c.get("value"),
                    "operator": c.get("op", "=")
                })
            elif any(k.startswith("_") for k in c.keys()):
                # schema 模式的特殊字段
                result.append(c)
        return result
    
    @filters.setter
    def filters(self, value: list[dict[str, Any]]) -> None:
        """设置 filters 时，自动更新 conditions"""
        # 移除旧的 custom 类型和特殊字段 conditions
        self.conditions = [c for c in self.conditions if c.get("type") == "ref"]
        # 添加新的
        for f in value:
            if any(k.startswith("_") for k in f.keys()):
                self.conditions.append(f)
            else:
                col = f.get("column") or f.get("dimension", "")
                val = f.get("value")
                op = f.get("operator", "=")
                if col and val is not None:
                    self.conditions.append({"type": "custom", "column": col, "op": op, "value": val})
    
    @property
    def intent_type(self) -> str:
        """向后兼容：返回分析类型（等同于 task_type）"""
        return self.task_type or "basic"
    
    @intent_type.setter
    def intent_type(self, value: str) -> None:
        self.task_type = value
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
            "task_type": self.task_type,
            "tables": self.tables,
            "table_relation": self.table_relation,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "conditions": self.conditions,
            # 向后兼容
            "filter_refs": self.filter_refs,
            "time": self.time,
            "filters": self.filters,
            "other_request": self.other_request,
            "intent_type": self.intent_type,
            "qa_type": self.qa_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], raw_query: str = "") -> "StructuredIntent":
        intent = cls(
            raw_query=raw_query,
            table_name=data.get("table_name", ""),
            mode=data.get("mode", "analysis"),
            task_type=data.get("task_type", "basic"),
            tables=data.get("tables", []),
            table_relation=data.get("table_relation", "single"),
            metrics=data.get("metrics", []),
            dimensions=data.get("dimensions", []),
            conditions=data.get("conditions", []),
            _filter_refs=data.get("filter_refs", []),
            time=data.get("time", {}),
            _filters=data.get("filters", []),
            other_request=data.get("other_request"),
        )
        # 兼容旧字段
        if data.get("qa_type") or data.get("intent_type"):
            intent._analysis_type = data.get("qa_type") or data.get("intent_type", "")
            if not intent.task_type or intent.task_type == "basic":
                intent.task_type = intent._analysis_type
        return intent
    
    def is_other_query(self) -> bool:
        """是否为非数据分析问题"""
        return self.mode == "other"
    
    def is_analysis_query(self) -> bool:
        """是否为数据分析问题"""
        return self.mode == "analysis"
    
    def is_multi_table(self) -> bool:
        """是否为多表查询"""
        return self.table_relation in ("join", "union") or len(self.tables) > 1
    
    def has_time_comparison(self) -> bool:
        """是否有时间对比需求（同比/环比）"""
        return self.task_type == "comparison"
    
    def get_filter_refs(self) -> list[str]:
        """获取所有预定义筛选器引用"""
        return [c["id"] for c in self.conditions if c.get("type") == "ref"]
    
    def get_custom_filters(self) -> list[dict[str, Any]]:
        """获取所有自定义筛选条件"""
        return [c for c in self.conditions if c.get("type") == "custom"]


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
        基于 YAML 配置提取意图（v4 精简版）
        
        一次 LLM 调用提取：task_type + metrics + dimensions + conditions
        """
        
        # 构建可用选项列表
        available_business_terms = self._format_business_terms(yml_config.get("business_terms", []))
        available_metrics = self._format_metrics(yml_config.get("metrics", {}))
        available_dimensions = self._format_dimensions(yml_config.get("dimensions", {}))
        available_filters = self._format_filters(yml_config.get("filters", {}))
        rules = self._format_rules(yml_config.get("rules", []))
        
        prompt = f"""请从以下查询中提取结构化意图。

## 用户查询
{query}

## 可用表
- {table_name}

## 业务术语词典（参考，帮助理解查询）
{available_business_terms}

## 可用指标（优先从这里选择 ID，如需自定义可用 metric_name 字段补充）
{available_metrics}

## 可用维度（优先从这里选择 ID，用于 GROUP BY）
{available_dimensions}

## 可用筛选器（优先从这里选择 ID；若找不到合适的，可用 type=custom 补充）
{available_filters}

## 业务规则
{rules}

## 输出 JSON 格式
```json
{{
  "mode": "analysis|other",
  "task_type": "basic|ratio|comparison|ranking|trend|source",
  "metrics": ["指标ID或自定义描述"],
  "dimensions": ["维度ID或列名"],
  "conditions": [
    {{"type": "ref", "id": "筛选器ID"}},
    {{"type": "custom", "column": "列名", "op": "=|!=|>|<|>=|<=|IN|LIKE", "value": "值"}}
  ]
}}
```

---

## 字段说明

### mode
- `analysis`: 数据分析问题（需要生成 SQL）
- `other`: 非数据分析问题（闲聊、问概念、解释术语等）

### task_type（分析类型）

| 类型 | 含义 | 典型关键词 |
|------|------|------------|
| **ratio** | 占比/比例分析 | "占比"、"比例"、"占多少"、"百分比" |
| **comparison** | 对比分析 | "同比"、"环比"、"增长"、"对比"、"变化" |
| **trend** | 趋势分析 | "趋势"、"走势"、"逐月"、"逐年"、"随时间" |
| **ranking** | 排名分析 | "Top"、"前N"、"排名"、"最高"、"最低" |
| **source** | 来源/构成分析 | "按XX分"、"各个"、"构成"、"分布"、"拆解" |
| **basic** | 基础查询 | "是多少"、"有多少"、"总计"、"合计" |

### metrics / dimensions / conditions
- **metrics**: 要计算的指标（SUM/COUNT/AVG 等聚合对象）
- **dimensions**: 要分组的维度（GROUP BY 的列）
- **conditions**: 筛选条件
  - 优先用预定义筛选器（type=ref）
  - 兜底用自定义（type=custom）

---

## 示例

**占比**: "A类产品在总销售额中的占比"
```json
{{"mode": "analysis", "task_type": "ratio", "metrics": ["sales"], "dimensions": ["product_category"], "conditions": []}}
```

**同比**: "今年销售额同比增长多少"
```json
{{"mode": "analysis", "task_type": "comparison", "metrics": ["sales"], "dimensions": [], "conditions": [{{"type": "custom", "column": "year", "op": "=", "value": "2025"}}]}}
```

**趋势**: "最近12个月的订单量趋势"
```json
{{"mode": "analysis", "task_type": "trend", "metrics": ["order_count"], "dimensions": ["month"], "conditions": []}}
```

**排名**: "销售额最高的10个客户"
```json
{{"mode": "analysis", "task_type": "ranking", "metrics": ["sales"], "dimensions": ["customer"], "conditions": []}}
```

**来源分析**: "各渠道的用户数量"
```json
{{"mode": "analysis", "task_type": "source", "metrics": ["user_count"], "dimensions": ["channel"], "conditions": []}}
```

**基础查询**: "上个月的总订单数"
```json
{{"mode": "analysis", "task_type": "basic", "metrics": ["order_count"], "dimensions": [], "conditions": [{{"type": "custom", "column": "month", "op": "=", "value": "上月"}}]}}
```

**自定义筛选**: "北京地区VIP客户的消费金额"
```json
{{"mode": "analysis", "task_type": "basic", "metrics": ["amount"], "dimensions": [], "conditions": [{{"type": "custom", "column": "region", "op": "=", "value": "北京"}}, {{"type": "custom", "column": "customer_level", "op": "=", "value": "VIP"}}]}}
```

**非分析**: "什么是环比"
```json
{{"mode": "other", "task_type": "basic", "metrics": [], "dimensions": [], "conditions": []}}
```

只输出 JSON，不要其他文字。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是意图提取器。同时提取查询对象、约束条件和分析类型。严格按 JSON 格式输出。",
                caller_name="extract_intent",
            )
            
            intent_dict = self._parse_json_response(response)
            intent = StructuredIntent.from_dict(intent_dict, raw_query=query)
            
            # === 校验和转换 ===
            
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
            
            # 构建业务术语→筛选器的映射表
            term_to_filters = self._build_term_filter_map(yml_config.get("business_terms", []))
            
            # 校验并转换 conditions 中的 ref 类型 ID
            valid_filters = set(yml_config.get("filters", {}).keys())
            if intent.conditions:
                intent.conditions = self._resolve_conditions(
                    intent.conditions, valid_filters, term_to_filters
                )
            
            self._log.observe(
                f"提取意图(YAML): mode={intent.mode}, task_type={intent.task_type}, "
                f"metrics={intent.metrics}, dimensions={intent.dimensions}, conditions={len(intent.conditions)}"
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
        基于 Schema 提取意图（无 YAML 配置时）
        """
        
        # 构建列信息
        columns_info = self._extract_columns_from_meta(tables_meta, table_name) if tables_meta else schema_text
        
        if columns_info == "无列信息" and schema_text:
            columns_info = schema_text
        
        available_tables = ""
        if tables_meta:
            available_tables = "\n".join([
                f"- {t.get('table_name', '')}: {t.get('table_description', '')}"
                for t in tables_meta
            ])
        
        self._log.think(f"提取列信息: tables_meta={len(tables_meta) if tables_meta else 0}个表, table_name={table_name}")
        
        prompt = f"""请从以下查询中提取结构化意图。

## 用户查询
{query}

## 当前表名
{table_name}

## 可用表
{available_tables or "仅当前表"}

## 表结构（列名和类型）
{columns_info}

## 输出 JSON
```json
{{
  "mode": "analysis|other",
  "task_type": "basic|ratio|comparison|ranking|trend|source",
  "tables": ["表名"],
  "agg_column": "要聚合的数值列名（如 amount, price, count 等）",
  "agg_func": "SUM|COUNT|AVG|MAX|MIN",
  "group_by_columns": ["分组列名"],
  "filter_columns": [
    {{"column": "列名", "value": "值", "operator": "=|!=|>|<|>=|<=|IN|LIKE"}}
  ]
}}
```

## task_type 识别规则

| 类型 | 关键词 | 示例问题 |
|------|--------|----------|
| **ratio** | 占比、比例、百分比 | "A在B中占多少" |
| **comparison** | 同比、环比、增长、对比 | "今年比去年增长多少" |
| **ranking** | Top、前N、排名、最高/低 | "销量最高的10个产品" |
| **trend** | 趋势、走势、逐月/年 | "最近一年的变化趋势" |
| **source** | 按XX分、各个、构成、分布 | "各地区的销售额" |
| **basic** | 是多少、总计、合计 | "总销售额是多少" |

## 列选择建议
- **agg_column**: 选择数值类型列（INT, FLOAT, DECIMAL 等）
- **group_by_columns**: 选择分类/维度列（VARCHAR, DATE 等）
- **filter_columns**: 根据用户提到的条件选择

只输出 JSON，不要其他文字。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是意图提取器。提取查询对象、约束条件和分析类型。",
                caller_name="extract_intent_schema",
            )
            
            intent_dict = self._parse_json_response(response)
            
            intent = StructuredIntent(
                raw_query=query,
                table_name=table_name,
                mode=intent_dict.get("mode", "analysis"),
                task_type=intent_dict.get("task_type", "basic"),
                tables=intent_dict.get("tables", [table_name] if table_name else []),
                table_relation=intent_dict.get("table_relation", "single"),
            )
            
            # 存储 schema 模式特有的字段
            if intent_dict.get("agg_column"):
                intent.filters.append({
                    "_agg_column": intent_dict.get("agg_column"),
                    "_agg_func": intent_dict.get("agg_func", "SUM"),
                })
            
            if intent_dict.get("group_by_columns"):
                intent.dimensions = intent_dict.get("group_by_columns", [])
            
            for f in intent_dict.get("filter_columns", []):
                intent.filters.append(f)
            
            self._log.observe(
                f"提取意图(Schema): mode={intent.mode}, task_type={intent.task_type}, dims={intent.dimensions}"
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
        """格式化业务术语列表供 LLM 使用（术语ID可直接作为筛选器引用）"""
        if not business_terms:
            return "无业务术语定义"
        
        lines = []
        for term in business_terms:
            term_id = term.get("id", "")
            term_name = term.get("term", "")
            synonyms = term.get("synonyms", [])
            
            syn_str = f" (同义词: {', '.join(synonyms)})" if synonyms else ""
            # 显示术语ID，让LLM可以直接使用
            lines.append(f"- {term_id}: {term_name}{syn_str}")
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
    
    def _build_term_filter_map(self, business_terms: list[dict[str, Any]]) -> dict[str, list[str]]:
        """
        构建业务术语ID → 筛选器ID列表的映射表
        
        例如: {"ieg_total": ["source_actual"], "mobile_game": ["category_mobile"]}
        """
        term_map = {}
        for term in business_terms:
            term_id = term.get("id", "")
            # 兼容 filters 和 related_filters 两种写法
            filters = term.get("filters", []) or term.get("related_filters", [])
            if term_id and filters:
                term_map[term_id] = filters
        return term_map
    
    def _resolve_conditions(
        self,
        conditions: list[dict[str, Any]],
        valid_filters: set[str],
        term_to_filters: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """
        解析并转换 conditions 中的筛选器引用
        
        - 如果 ref ID 是有效筛选器，直接保留
        - 如果 ref ID 是业务术语，展开为对应的筛选器列表
        - 如果都不是，记录警告并跳过
        - 自定义条件（type=custom）直接保留
        """
        resolved = []
        seen_filter_ids = set()  # 避免重复
        
        for cond in conditions:
            if cond.get("type") != "ref":
                # 自定义条件，直接保留
                resolved.append(cond)
                continue
            
            ref_id = cond.get("id", "")
            
            # 1. 检查是否为有效筛选器
            if ref_id in valid_filters:
                if ref_id not in seen_filter_ids:
                    resolved.append(cond)
                    seen_filter_ids.add(ref_id)
                continue
            
            # 2. 检查是否为业务术语，展开为筛选器
            if ref_id in term_to_filters:
                expanded_filters = term_to_filters[ref_id]
                self._log.observe(f"术语 '{ref_id}' → 筛选器 {expanded_filters}")
                for filter_id in expanded_filters:
                    if filter_id in valid_filters and filter_id not in seen_filter_ids:
                        resolved.append({"type": "ref", "id": filter_id})
                        seen_filter_ids.add(filter_id)
                    elif filter_id not in valid_filters:
                        self._log.warn(f"术语 '{ref_id}' 关联的筛选器 '{filter_id}' 不存在")
                continue
            
            # 3. 都不匹配，记录警告
            self._log.warn(f"筛选器引用不存在且无法解析为术语: {ref_id}")
        
        return resolved
    
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
