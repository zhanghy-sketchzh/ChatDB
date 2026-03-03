"""
SemanticParser - 语义解析能力

职责：从自然语言查询提取结构化意图 JSON

v5 设计：
- 只有两个核心概念：virtual_fields（可能用到的虚拟字段 ID 列表）+ examples（匹配的示例索引）
- SemanticParser 提取：task_type, virtual_fields
- task_type 决定分析类型（basic/ratio/comparison/ranking/trend/source）
- metrics/dimensions/conditions 由 virtual_fields 自动派生（向后兼容）

输出格式：
{
  "mode": "analysis" | "other",
  "task_type": "basic | ratio | comparison | ranking | trend | source",
  "virtual_fields": ["field_id_1", "field_id_2", ...],
  "examples": [0, 2]
}
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re
import yaml

from chatdb.agents.base import AgentContext, AgentResult, AgentStatus
from chatdb.config.metrics_loader import preprocess_yaml_config
from chatdb.llm.base import BaseLLM
from chatdb.utils.logger import get_component_logger


@dataclass
class StructuredIntent:
    """
    结构化查询意图（v5）
    
    核心字段：
    - task_type: 任务类型（basic/ratio/comparison/ranking/trend/source）
    - virtual_fields: 可能用到的虚拟字段 ID 列表（统一概念，不再区分 metrics/dimensions/conditions）
    
    mode 只有两种：
    - analysis: 与数据分析有关的问题（需要生成 SQL）
    - other: 与数据分析无关的问题（闲聊、解释概念等）
    
    向后兼容：
    - metrics/dimensions/conditions 从 virtual_fields + yml_config 自动派生
    """
    raw_query: str
    rewritten_query: str = ""
    table_name: str = ""
    
    # === 核心字段 ===
    mode: str = "analysis"  # analysis / other
    task_types: list[str] = field(default_factory=lambda: ["basic"])
    tables: list[str] = field(default_factory=list)
    table_relation: str = "single"
    virtual_fields: list[str] = field(default_factory=list)  # v5 核心输出
    
    # === 向后兼容字段（从 virtual_fields 派生） ===
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    _filter_refs: list[str] = field(default_factory=list)
    _filters: list[dict[str, Any]] = field(default_factory=list)
    time: dict[str, Any] = field(default_factory=dict)
    _analysis_type: str = field(default="", repr=False)
    
    # === 其他字段 ===
    other_request: str | None = None
    
    def __post_init__(self):
        self._migrate_legacy_fields()
        self._sync_task_types()
    
    def _migrate_legacy_fields(self):
        """将旧版 filter_refs 迁移到 conditions（去重）"""
        for ref_id in self._filter_refs:
            if not any(c.get("type") == "ref" and c.get("id") == ref_id for c in self.conditions):
                self.conditions.append({"type": "ref", "id": ref_id})
    
    def _sync_task_types(self):
        """同步 task_types 和 _analysis_type（向后兼容）"""
        primary = self.task_types[0] if self.task_types else "basic"
        if primary and primary != "basic":
            self._analysis_type = primary
        elif self._analysis_type and primary == "basic":
            self.task_types = [self._analysis_type]
    
    @property
    def task_type(self) -> str:
        """主任务类型（兼容下游 .task_type 访问）"""
        return self.task_types[0] if self.task_types else "basic"
    
    @task_type.setter
    def task_type(self, value: str) -> None:
        """设置主任务类型时，替换 task_types 第一个元素"""
        if self.task_types:
            self.task_types[0] = value
        else:
            self.task_types = [value]
        self._analysis_type = value
    
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
        """向后兼容：返回空列表（不再支持 custom 条件）"""
        return []
    
    @filters.setter
    def filters(self, value: list[dict[str, Any]]) -> None:
        """向后兼容：忽略设置（不再支持 custom 条件）"""
        pass
    
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
            "rewritten_query": self.rewritten_query,
            "table_name": self.table_name,
            "mode": self.mode,
            "task_types": self.task_types,
            "tables": self.tables,
            "table_relation": self.table_relation,
            "virtual_fields": self.virtual_fields,
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
        # task_type(s) 归一化：LLM 可能输出字符串或数组
        # 兼容 to_dict() 输出的 "task_types" 和 LLM 输出的 "task_type" 两种键名
        raw_tt = data.get("task_type") or data.get("task_types") or "basic"
        if isinstance(raw_tt, list):
            task_types = [str(t) for t in raw_tt] if raw_tt else ["basic"]
        else:
            task_types = [str(raw_tt)] if raw_tt else ["basic"]

        intent = cls(
            raw_query=raw_query,
            rewritten_query=data.get("rewritten_query", ""),
            table_name=data.get("table_name", ""),
            mode=data.get("mode", "analysis"),
            task_types=task_types,
            tables=data.get("tables", []),
            table_relation=data.get("table_relation", "single"),
            virtual_fields=data.get("virtual_fields", []),
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
            if intent.task_type == "basic":
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
    
    def populate_from_virtual_fields(self, yml_config: dict[str, Any]) -> None:
        """从 virtual_fields 列表 + YML 配置反推 metrics/dimensions/conditions（向后兼容）
        
        根据每个 virtual_field 的 field_type 分派：
        - metric  → self.metrics
        - column  → self.dimensions
        - condition → self.conditions (ref)
        
        同时自动补全 required/default scope 的字段。
        """
        vf_config = yml_config.get("virtual_fields", {})
        if not vf_config or not self.virtual_fields:
            return
        
        metrics, dimensions, condition_ids = [], [], set()
        for c in self.conditions:
            if c.get("type") == "ref":
                condition_ids.add(c["id"])
        
        for fid in self.virtual_fields:
            fdef = vf_config.get(fid, {})
            if not isinstance(fdef, dict):
                continue
            ft = fdef.get("field_type", "")
            if ft == "metric" and fid not in self.metrics:
                metrics.append(fid)
            elif ft == "column" and fid not in self.dimensions:
                dimensions.append(fid)
            elif ft == "condition" and fid not in condition_ids:
                self.conditions.append({"type": "ref", "id": fid})
                condition_ids.add(fid)
        
        if metrics:
            self.metrics = metrics
        if dimensions:
            self.dimensions = dimensions
        
        # 自动补全 required/default scope 字段
        for fid, fdef in vf_config.items():
            if not isinstance(fdef, dict) or fdef.get("field_type") != "condition":
                continue
            scope = fdef.get("scope", "optional")
            if scope == "required" and fid not in condition_ids:
                self.conditions.append({"type": "ref", "id": fid})
                condition_ids.add(fid)
            elif scope == "default" and fid not in condition_ids:
                group = fdef.get("group", "")
                group_selected = any(
                    vf_config.get(cid, {}).get("group") == group
                    for cid in condition_ids if group
                )
                if not group_selected:
                    self.conditions.append({"type": "ref", "id": fid})
                    condition_ids.add(fid)
    
    def get_filter_refs(self) -> list[str]:
        """获取所有预定义筛选器引用"""
        return [c["id"] for c in self.conditions if c.get("type") == "ref"]
    


class SemanticParser:
    """
    语义解析能力：从自然语言提取结构化意图 JSON。
    
    支持多轮补完、YAML 匹配、严格从 YAML 选 ID，不发明新字段。
    """
    
    def __init__(self, llm: BaseLLM, yml_config: str | Path | None = None,
                 skill_registry: Any = None):
        self.llm = llm
        self._log = get_component_logger("SemanticParser")
        self.yml_config_path: Path | None = None
        self.yml_config_dir: Path | None = None
        self._skill_registry = skill_registry  # SkillRegistry 实例（可选）
        
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
            
            # 3. 提取实时列统计和检索相关列（由 SemanticParseTool 注入）
            column_stats_map = getattr(context, "column_stats_map", {}) or {}
            relevant_columns = getattr(context, "relevant_columns", {}) or {}
            
            # 4. 使用 LLM 提取结构化意图（传入 schema_text + chat_history）
            intent = await self._extract_intent(
                context.user_query, 
                table_name, 
                yml_config,
                context.schema_text,  # 传入表结构
                getattr(context, "available_tables", []),  # 兼容精简后的 AgentContext
                context.chat_history,  # 传入历史对话
                column_stats_map=column_stats_map,
                relevant_columns=relevant_columns,
            )
            
            # 4. 返回结果（不再写入 context，ReAct 流程由 SemanticParseTool 处理）
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
        selected = getattr(context, "selected_tables", [])
        if selected:
            return selected[0]
        
        available = getattr(context, "available_tables", [])
        if available and len(available) == 1:
            return available[0].get("table_name", "")
        
        if available and len(available) > 1:
            tables_desc = "\n".join([
                f"- {t.get('table_name')}: {t.get('table_description', '')}"
                for t in available
            ])
            
            response = await self.llm.chat(
                prompt=f"用户查询：{context.user_query}\n\n可用表：\n{tables_desc}\n\n只输出最相关的表名：",
                system_prompt="选择最相关的表，只输出表名。",
                caller_name="select_table",
            )
            
            for table in available:
                if table.get("table_name") in response:
                    return table.get("table_name", "")
        
        if available:
            return available[0].get("table_name", "")
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
                    config = preprocess_yaml_config(config)
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
                            config = preprocess_yaml_config(config)
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
        chat_history: list[dict[str, str]] | None = None,
        column_stats_map: dict[str, list[dict[str, Any]]] | None = None,
        relevant_columns: dict[str, list[str]] | None = None,
    ) -> StructuredIntent:
        """使用 LLM 提取结构化意图"""
        
        # 判断是否有 YAML 配置（v2: virtual_fields）
        has_yml = bool(yml_config.get("virtual_fields"))
        
        if has_yml:
            # 有 YAML 配置：基于配置提取
            return await self._extract_intent_with_yml(
                query, table_name, yml_config, chat_history,
                column_stats_map=column_stats_map,
                relevant_columns=relevant_columns,
                schema_text=schema_text,
            )
        else:
            # 无 YAML 配置：基于 schema 提取
            return await self._extract_intent_from_schema(
                query, table_name, schema_text, tables_meta, chat_history,
                column_stats_map=column_stats_map,
                relevant_columns=relevant_columns,
            )
    
    async def _extract_intent_with_yml(
        self,
        query: str,
        table_name: str,
        yml_config: dict[str, Any],
        chat_history: list[dict[str, str]] | None = None,
        column_stats_map: dict[str, list[dict[str, Any]]] | None = None,
        relevant_columns: dict[str, list[str]] | None = None,
        schema_text: str = "",
    ) -> StructuredIntent:
        """
        基于 YAML 配置提取意图（v5 统一 virtual_fields）
        
        一次 LLM 调用提取：task_type + virtual_fields + examples
        """
        virtual_fields = yml_config.get("virtual_fields", {})
        
        history_section = self._format_chat_history(chat_history)
        column_stats_section = self._build_column_stats_section(
            table_name, column_stats_map, relevant_columns,
        )
        
        # 提取表业务说明
        table_understanding_section = ""
        if schema_text and "## 表业务说明" in schema_text:
            m = re.search(r"(## 表业务说明.*?)(?=\n## |\Z)", schema_text, re.DOTALL)
            if m:
                table_understanding_section = f"\n{m.group(1).strip()}\n"
        
        # v2: 统一 virtual_fields prompt（YAML 模式不注入列统计信息，虚拟字段已包含完整业务语义）
        vf_section = self._format_virtual_fields_full(virtual_fields)
        examples_section = self._format_examples(yml_config.get("examples", []))
        
        prompt = f"""你同时承担两个任务：**查询改写** 和 **意图提取**。
{history_section}
## 用户查询
{query}

## 数据 Schema
{table_understanding_section}
### 可用表
- {table_name}

## 虚拟字段定义

虚拟字段是预定义的业务概念，分为三种类型：
- **condition** (布尔条件)：用于 WHERE 筛选，展开为布尔表达式
- **column** (维度列)：用于 SELECT/GROUP BY/ORDER BY，映射到真实列名
- **metric** (聚合指标)：用于 SELECT 聚合计算，展开为聚合表达式

作用域 (scope)：
- **required**：所有查询必须包含（如数据有效性筛选）
- **default**：未指定时自动补充（如默认使用实际数据）
- **optional**：用户明确提及时才加入

{vf_section}

## 参考示例
{examples_section}

## 你的任务

### 1. 查询改写（rewritten_query）
将用户问题改写为**完整、独立、无歧义**的查询。解析指代词，补全上下文中的省略信息。

**消歧规则**：
- 当用户说"占总XX的多少"或"XX占比"时，"总XX"默认指**全部数据的汇总**（即不带额外筛选的全量聚合），而非仅限上文提到的若干对象的小计
  - 例：上文提到"Top5产品"，用户问"各自占总流水的比例" → 分母应为**全部产品的总流水**，而非仅 Top5 产品的流水之和
- 改写时应保留用户的原始语义范围，不要擅自缩小分母/基准的统计口径

### 2. 意图提取

{self._get_task_type_table()}

**复合分析**：多步分析时 task_type 用数组，如 `["trend", "comparison"]`。

### 3. virtual_fields 选取规则
- 从上方虚拟字段定义中选取**所有可能用到的字段 ID**
- 包含 condition（筛选条件）、column（维度列）、metric（聚合指标）
- scope=required 的字段**必须**包含
- 根据用户意图选择 optional 字段（参考 synonyms 匹配用户措辞）
- 指标专属的 condition（如 metric_flow）**不需要**手动选，系统会根据 metric 自动注入

### 4. examples 选取规则
- 从参考示例中选出与用户问题**最相似的**示例索引（0-based）
- 如无匹配可留空

## 输出 JSON
```json
{{
  "rewritten_query": "改写后的完整问题",
  "mode": "analysis|other",
  "task_type": "单类型字符串或数组",
  "virtual_fields": ["field_id_1", "field_id_2", "..."],
  "examples": [0]
}}
```

只输出 JSON，不要其他文字。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是意图提取器。从用户查询提取结构化意图，选出相关的虚拟字段。严格按 JSON 格式输出。",
                caller_name="extract_intent",
            )
            
            intent_dict = self._parse_json_response(response)
            intent = StructuredIntent.from_dict(intent_dict, raw_query=query)
            
            rq = intent_dict.get("rewritten_query", "")
            if rq and rq != query:
                intent.rewritten_query = rq
                self._log.info(f"查询改写: {query[:40]} → {rq[:60]}")
            
            # 补充表名
            if not intent.tables and table_name:
                intent.tables = [table_name]
            if not intent.table_name and table_name:
                intent.table_name = table_name
            
            # 校验 virtual_fields ID 并派生 metrics/dimensions/conditions
            valid_vf_ids = set(virtual_fields.keys())
            intent.virtual_fields = [fid for fid in intent.virtual_fields if fid in valid_vf_ids]
            intent.populate_from_virtual_fields(yml_config)
            
            # 校验 dimensions（column 类型字段也是有效维度）
            valid_dims = {
                fid for fid, fdef in virtual_fields.items()
                if isinstance(fdef, dict) and fdef.get("field_type") == "column"
            }
            if intent.dimensions:
                intent.dimensions = [d for d in intent.dimensions if d in valid_dims]
            
            self._log.observe(
                f"提取意图: mode={intent.mode}, task_type={intent.task_type}, "
                f"virtual_fields={intent.virtual_fields}, metrics={intent.metrics}, "
                f"dimensions={intent.dimensions}, conditions={len(intent.conditions)}"
            )
            return intent
            
        except Exception as e:
            self._log.error(f"意图提取失败: {e}")
            raise

    async def _extract_intent_from_schema(
        self,
        query: str,
        table_name: str,
        schema_text: str,
        tables_meta: list[dict[str, Any]] | None = None,
        chat_history: list[dict[str, str]] | None = None,
        column_stats_map: dict[str, list[dict[str, Any]]] | None = None,
        relevant_columns: dict[str, list[str]] | None = None,
    ) -> StructuredIntent:
        """
        基于 Schema 提取意图（无 YAML 配置时）
        """
        
        # 构建列信息（带实时描述统计）
        columns_info = self._format_columns_with_stats(
            tables_meta, table_name,
            column_stats_map=column_stats_map,
            relevant_columns=relevant_columns,
        ) if tables_meta else schema_text
        
        if columns_info == "无列信息" and schema_text:
            columns_info = schema_text
        
        available_tables = ""
        if tables_meta:
            available_tables = "\n".join([
                f"- {t.get('table_name', '')}: {t.get('table_description', '')}"
                for t in tables_meta
            ])
        
        history_section = self._format_chat_history(chat_history)
        
        self._log.think(f"提取列信息: tables_meta={len(tables_meta) if tables_meta else 0}个表, table_name={table_name}")
        
        prompt = f"""请从以下查询中提取结构化意图。
{history_section}
## 用户查询
{query}

## 当前表名
{table_name}

## 可用表
{available_tables or "仅当前表"}

## 表结构（列名和类型）
{columns_info}

## 你的任务
1. **查询改写**：将用户问题改写为完整、独立、无歧义的自然语言查询（解析指代词，补全省略信息）
2. **意图提取**：提取结构化字段

## 输出 JSON
```json
{{
  "rewritten_query": "改写后的完整问题",
  "mode": "analysis|other",
  "task_type": "basic|ratio|comparison|ranking|trend|source（单类型用字符串；复合分析用数组 [\"trend\", \"comparison\"]）",
  "tables": ["表名"],
  "metrics": ["要聚合的数值列名（如 amount, price, count 等）"],
  "dimensions": ["分组列名"],
  "conditions": [
    {{"type": "ref", "id": "筛选器ID"}}
  ]
}}
```

## task_type 识别规则

{self._get_task_type_table_for_schema()}

**复合分析**：问题涉及多步骤时输出数组，如 `["trend", "comparison"]`。

## 列选择建议
- **metrics**: 选择数值类型列（INT, FLOAT, DECIMAL 等）作为要聚合的指标列
- **dimensions**: 选择分类/维度列（VARCHAR, DATE 等）作为分组列
- **conditions**: 只填用户**明确提到**的通用筛选条件，使用 `{{"type": "ref", "id": "筛选器ID"}}` 格式引用已知筛选器

只输出 JSON，不要其他文字。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是意图提取器。提取查询对象、约束条件和分析类型。",
                caller_name="extract_intent_schema",
            )
            
            intent_dict = self._parse_json_response(response)
            
            # task_type(s) 归一化（同 from_dict）
            raw_tt = intent_dict.get("task_type", "basic")
            if isinstance(raw_tt, list):
                tt_list = [str(t) for t in raw_tt] if raw_tt else ["basic"]
            else:
                tt_list = [str(raw_tt)] if raw_tt else ["basic"]

            intent = StructuredIntent(
                raw_query=query,
                rewritten_query=intent_dict.get("rewritten_query", ""),
                table_name=table_name,
                mode=intent_dict.get("mode", "analysis"),
                task_types=tt_list,
                tables=intent_dict.get("tables", [table_name] if table_name else []),
                table_relation=intent_dict.get("table_relation", "single"),
                metrics=intent_dict.get("metrics", []),
                dimensions=intent_dict.get("dimensions", []),
                conditions=intent_dict.get("conditions", []),
            )
            
            if intent.rewritten_query and intent.rewritten_query != query:
                self._log.info(f"查询改写: {query[:40]} → {intent.rewritten_query[:60]}")
            
            # 兼容旧格式字段：如果 LLM 仍返回旧字段，迁移到新格式
            if not intent.metrics and intent_dict.get("agg_column"):
                intent.metrics = [intent_dict["agg_column"]]
            if not intent.dimensions and intent_dict.get("group_by_columns"):
                intent.dimensions = intent_dict["group_by_columns"]
            if not intent.conditions and intent_dict.get("filter_columns"):
                for f in intent_dict["filter_columns"]:
                    col = f.get("column", "")
                    val = f.get("value")
                    if col and val is not None:
                        # 旧格式只能尝试作为 ref 引用
                        intent.conditions.append({"type": "ref", "id": col})
            
            self._log.observe(
                f"提取意图(Schema): mode={intent.mode}, task_type={intent.task_type}, dims={intent.dimensions}"
            )
            return intent
            
        except Exception as e:
            self._log.error(f"Schema意图提取失败: {e}")
            raise
    
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
    
    def _format_columns_with_stats(
        self,
        tables_meta: list[dict[str, Any]] | None,
        table_name: str,
        column_stats_map: dict[str, list[dict[str, Any]]] | None = None,
        relevant_columns: dict[str, list[str]] | None = None,
    ) -> str:
        """增强版列信息：委托 ColumnStatsProvider 统一格式化。"""
        from chatdb.database.column_stats_provider import ColumnStatsProvider, ColumnStats

        if not tables_meta:
            return "无列信息"

        # 找到目标表的列元数据
        all_columns: list[dict[str, Any]] = []
        for table in tables_meta:
            if table.get("table_name") == table_name:
                all_columns = table.get("columns_info") or table.get("columns", [])
                break
        if not all_columns:
            return "无列信息"

        # dict → ColumnStats
        stats_objs: list[ColumnStats] = []
        if column_stats_map:
            for d in column_stats_map.get(table_name, []):
                cs = ColumnStats(
                    name=d.get("name", ""),
                    dtype=d.get("type", ""),
                    null_pct=d.get("null_pct", 0.0),
                    unique_count=d.get("unique_count", 0),
                )
                stats = d.get("stats")
                if stats:
                    cs.min_val = stats.get("min")
                    cs.max_val = stats.get("max")
                    cs.mean_val = stats.get("mean")
                    cs.median_val = stats.get("median")
                if d.get("top_values"):
                    cs.top_values = d["top_values"]
                stats_objs.append(cs)

        recalled = set(relevant_columns.get(table_name, [])) if relevant_columns else None
        return ColumnStatsProvider.format_columns(
            stats_objs, columns=all_columns, relevant_cols=recalled,
        )

    def _build_column_stats_section(
        self,
        table_name: str,
        column_stats_map: dict[str, list[dict[str, Any]]] | None = None,
        relevant_columns: dict[str, list[str]] | None = None,
    ) -> str:
        """构建列统计信息段落（YAML 模式 prompt 补充），委托 ColumnStatsProvider。"""
        from chatdb.database.column_stats_provider import ColumnStatsProvider, ColumnStats

        if not column_stats_map:
            return "\n"
        stats_dicts = column_stats_map.get(table_name, [])
        if not stats_dicts:
            return "\n"

        stats_objs: list[ColumnStats] = []
        for d in stats_dicts:
            cs = ColumnStats(
                name=d.get("name", ""),
                dtype=d.get("type", ""),
                null_pct=d.get("null_pct", 0.0),
                unique_count=d.get("unique_count", 0),
            )
            stats = d.get("stats")
            if stats:
                cs.min_val = stats.get("min")
                cs.max_val = stats.get("max")
                cs.mean_val = stats.get("mean")
                cs.median_val = stats.get("median")
            if d.get("top_values"):
                cs.top_values = d["top_values"]
            stats_objs.append(cs)

        recalled = set(relevant_columns.get(table_name, [])) if relevant_columns else None
        formatted = ColumnStatsProvider.format_columns(
            stats_objs, relevant_cols=recalled,
        )
        return f"\n## 列统计信息\n{formatted}\n"
    
    def _format_virtual_fields_full(self, virtual_fields: dict[str, Any]) -> str:
        """格式化完整的虚拟字段定义（v2 核心方法）
        
        按 field_type 分组展示，包含 scope、synonyms、description。
        """
        conditions = []
        columns = []
        metrics = []
        for fid, fdef in virtual_fields.items():
            if not isinstance(fdef, dict):
                continue
            ft = fdef.get("field_type", "condition")
            entry = {"id": fid, **fdef}
            if ft == "condition":
                conditions.append(entry)
            elif ft == "column":
                columns.append(entry)
            elif ft == "metric":
                metrics.append(entry)
        
        lines = []
        
        # 条件型
        if conditions:
            lines.append("### condition (布尔条件 — 用于 WHERE)")
            lines.append("")
            lines.append("| ID | 描述 | scope | synonyms | group |")
            lines.append("|-----|------|-------|----------|-------|")
            for c in conditions:
                syns = ", ".join(c.get("synonyms", []))
                lines.append(
                    f"| {c['id']} | {c.get('description', '')} "
                    f"| {c.get('scope', 'optional')} | {syns} | {c.get('group', '')} |"
                )
            lines.append("")
        
        # 维度列
        if columns:
            lines.append("### column (维度列 — 用于 GROUP BY / ORDER BY)")
            lines.append("")
            lines.append("| ID | 描述 | 真实列名 | synonyms | sql_type |")
            lines.append("|-----|------|----------|----------|----------|")
            for c in columns:
                col = c.get("column", c["id"])
                syns = ", ".join(c.get("synonyms", []))
                lines.append(
                    f"| {c['id']} | {c.get('description', '')} "
                    f"| `{col}` | {syns} | {c.get('sql_type', '')} |"
                )
            lines.append("")
        
        # 聚合指标
        if metrics:
            lines.append("### metric (聚合指标 — 用于 SELECT)")
            lines.append("")
            lines.append("| ID | 描述 | synonyms | unit |")
            lines.append("|-----|------|----------|------|")
            for m in metrics:
                syns = ", ".join(m.get("synonyms", []))
                unit = m.get("unit", "")
                lines.append(
                    f"| {m['id']} | {m.get('description', '')} "
                    f"| {syns} | {unit} |"
                )
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_examples(self, examples: list[dict[str, Any]]) -> str:
        """格式化示例列表，供 LLM 参考匹配"""
        if not examples:
            return "无参考示例"
        lines = []
        for i, ex in enumerate(examples):
            query = ex.get("query", "")
            vf = ex.get("virtual_fields", [])
            note = ex.get("note", "")
            lines.append(f"[{i}] \"{query}\"")
            lines.append(f"    virtual_fields: {vf}")
            if note:
                lines.append(f"    note: {note}")
        return "\n".join(lines)
    
    def _format_business_terms(self, business_terms: list[dict[str, Any]]) -> str:
        """格式化业务术语列表供 LLM 使用。
        
        结构: [{term, synonyms, virtual_fields}]
        """
        if not business_terms:
            return "无业务术语定义"
        
        lines = []
        for term in business_terms:
            if not isinstance(term, dict):
                continue
            term_name = term.get("term", "")
            synonyms = term.get("synonyms", [])
            vfields = term.get("virtual_fields", [])
            syn_str = f" (同义词: {', '.join(synonyms)})" if synonyms else ""
            vf_str = f" → {', '.join(vfields)}" if vfields else ""
            lines.append(f"- {term_name}{syn_str}{vf_str}")
        return "\n".join(lines) if lines else "无业务术语定义"
    
    def _format_rules(self, rules: dict[str, Any]) -> str:
        """格式化 v2 格式的规则。
        
        v2 rules 结构:
          defaults: {data_source: ..., time: ...}
          disambiguation: [{match, default_field, note}, ...]
          display: [{field, divisor, suffix}, ...]
        """
        lines = []
        
        # defaults
        defaults = rules.get("defaults", {})
        if isinstance(defaults, dict):
            for key, val in defaults.items():
                lines.append(f"- 默认值: 未指定 {key} 时使用 `{val}`")
        
        # disambiguation
        disamb = rules.get("disambiguation", [])
        if isinstance(disamb, list):
            for item in disamb:
                if not isinstance(item, dict):
                    continue
                matches = item.get("match", [])
                field = item.get("default_field", "")
                note = item.get("note", "")
                match_str = "、".join(f"「{m}」" for m in matches)
                note_str = f" ({note})" if note else ""
                lines.append(f"- 消歧义: 提到 {match_str} → 使用 `{field}`{note_str}")
        
        # display
        display = rules.get("display", [])
        if isinstance(display, list):
            for item in display:
                if not isinstance(item, dict):
                    continue
                field = item.get("field", "")
                divisor = item.get("divisor", "")
                suffix = item.get("suffix", "")
                fmt_parts = []
                if divisor:
                    fmt_parts.append(f"÷{divisor}")
                if suffix:
                    fmt_parts.append(f"单位「{suffix}」")
                fmt_str = "，".join(fmt_parts) if fmt_parts else ""
                lines.append(f"- 展示格式: `{field}` → {fmt_str}")
        
        return "\n".join(lines) if lines else "无特殊规则"
    
    def _build_term_filter_map(self, business_terms: list[dict[str, Any]]) -> dict[str, list[str]]:
        """构建业务术语 → 虚拟字段ID列表的映射表
        
        term_name → virtual_field_ids  (e.g. {"IEG整体": ["source_actual"]})
        """
        term_map = {}
        for term in business_terms:
            if not isinstance(term, dict):
                continue
            term_name = term.get("term", "")
            vfields = term.get("virtual_fields", [])
            if term_name and vfields:
                term_map[term_name] = vfields
                for syn in term.get("synonyms", []):
                    term_map[syn] = vfields
        return term_map
    
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
    
    @staticmethod
    def _format_chat_history(chat_history: list[dict[str, str]] | None) -> str:
        """格式化历史对话，直接拼接 user/assistant 多轮文本供 LLM 理解上下文。"""
        if not chat_history:
            return ""
        
        lines = ["\n## 历史对话"]
        for msg in chat_history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        lines.append("")
        return "\n".join(lines)
    
    def _get_task_type_table(self) -> str:
        """
        生成 task_type 判断表（渐进披露：识别层）
        
        优先从 SkillRegistry 动态生成，降级到硬编码。
        """
        if self._skill_registry:
            table = self._skill_registry.get_semantic_table()
            if table:
                return table

        # ── fallback: 硬编码默认表 ──
        return """| task_type | 含义 | 判断标准 | 关键词 |
|-----------|------|----------|--------|
| basic | 基础查询 | 简单聚合，获取单一数值 | 总共, 多少, 合计, 是多少 |
| trend | 趋势分析 | 按时间聚合，观察走势 | 趋势, 走势, 逐年, 逐月, 历年 |
| comparison | 对比分析 | 计算两期差值或增长率 | 同比, 环比, 涨幅, 跌幅, 变化, 下降最多 |
| source | 来源分析 | 按维度拆解单期构成 | 各个, 构成, 拆解, 分布, 按XX分 |
| ranking | 排名分析 | 按指标排序取 TopN | 排名, 最高, 最低, 前N, TopN |
| ratio | 占比分析 | 计算部分在整体中的百分比 | 占比, 比例, 百分比, 比重 |
| drilldown | 下钻分析 | 对上游结果进一步细分 | 下钻, 细分, 进一步, 展开 |
| validation | 数据验证 | 诊断 SQL 报错或空结果 | 验证, 诊断, 检查, 数据对不上 |"""

    def _get_task_type_table_for_schema(self) -> str:
        """
        生成 Schema 模式下的 task_type 识别规则
        
        与 _get_task_type_table 类似但格式适配 Schema 模式的 prompt。
        """
        if self._skill_registry:
            table = self._skill_registry.get_semantic_table()
            if table:
                return table

        # ── fallback: Schema 模式默认表 ──
        return """| task_type | 含义 | 判断标准 | 关键词 |
|-----------|------|----------|--------|
| basic | 基础查询 | 简单聚合，获取单一数值 | 总共, 多少, 合计, 是多少 |
| trend | 趋势分析 | 按时间聚合，观察走势 | 趋势, 走势, 逐年, 逐月, 历年 |
| comparison | 对比分析 | 计算两期差值或增长率 | 同比, 环比, 涨幅, 跌幅, 变化, 下降最多 |
| source | 来源分析 | 按维度拆解单期构成 | 各个, 构成, 拆解, 分布, 按XX分 |
| ranking | 排名分析 | 按指标排序取 TopN | 排名, 最高, 最低, 前N, TopN |
| ratio | 占比分析 | 计算部分在整体中的百分比 | 占比, 比例, 百分比, 比重 |
| drilldown | 下钻分析 | 对上游结果进一步细分 | 下钻, 细分, 进一步, 展开 |
| validation | 数据验证 | 诊断 SQL 报错或空结果 | 验证, 诊断, 检查, 数据对不上 |"""

    def get_system_prompt(self) -> str:
        return "你是意图提取器，将自然语言查询转换为结构化 JSON。"
