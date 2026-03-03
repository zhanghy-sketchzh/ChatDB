"""
SemanticParseTool - 语义解析工具

将用户自然语言查询解析为结构化意图（StructuredIntent）。

适用场景：
- 任务开始时，理解用户意图
- 用户意图不明确，需要重新解析
- 业务术语需要匹配到 YAML 配置

输入：
- user_query: 用户原始查询
- schema_text: 表结构摘要
- yml_config: YAML 业务配置（可选）

输出：
- intent: StructuredIntent 对象
- table_name: 选择的表名
"""

from typing import Any, TYPE_CHECKING
from pathlib import Path

from chatdb.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from chatdb.llm.base import BaseLLM
from chatdb.utils.logger import get_component_logger

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext


class SemanticParseTool(BaseTool):
    """
    语义解析工具
    
    核心能力：
    - 从自然语言提取结构化意图
    - 支持 YAML 配置的指标/维度/筛选器匹配
    - 支持无 YAML 时的 Schema 模式解析
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        yml_config: str | Path | None = None,
        skill_registry: Any = None,
    ):
        metadata = ToolMetadata(
            name="semantic_parse",
            description="将自然语言查询解析为结构化意图，识别指标、维度、筛选条件等",
            category="analysis",
            inputs={
                "user_query": {"type": "str", "description": "用户的原始问题"},
                "schema_text": {"type": "str", "description": "表结构摘要"},
                "table_name": {"type": "str", "description": "目标表名"},
            },
            outputs={
                "intent": {"type": "StructuredIntent", "description": "解析后的结构化意图"},
                "yml_config": {"type": "dict", "description": "加载的 YAML 配置"},
            },
            subtools=[
                "semantic_parse.extract_terms",      # 提取业务术语
                "semantic_parse.match_to_config",    # 匹配到 YAML 配置
                "semantic_parse.select_table",       # 选择最相关的表
            ],
            is_core=True,
        )
        super().__init__(metadata)
        
        self.llm = llm
        self.yml_config_path = Path(yml_config) if yml_config else None
        self._log = get_component_logger("SemanticParseTool")
        self._skill_registry = skill_registry  # SkillRegistry 实例（可选）
        
        # 延迟导入避免循环依赖
        self._parser_agent = None
    
    @property
    def name(self) -> str:
        return "semantic_parse"
    
    @property
    def description(self) -> str:
        return """语义解析工具：将自然语言查询解析为结构化意图。

使用场景：
- 任务开始时，理解用户意图
- 用户意图不明确，需要重新解析
- 业务术语需要匹配到 YAML 配置

输出：
- StructuredIntent 对象，包含 metrics、dimensions、filters 等"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="user_query", type="string", description="用户原始查询"),
            ToolParameter(name="schema_text", type="string", description="表结构摘要", required=False),
            ToolParameter(name="table_name", type="string", description="目标表名", required=False),
        ]
    
    def _get_parser(self):
        """延迟初始化 SemanticParser"""
        if self._parser_agent is None:
            from chatdb.agents.semantic_parser import SemanticParser
            self._parser_agent = SemanticParser(
                self.llm, self.yml_config_path,
                skill_registry=self._skill_registry,
            )
        return self._parser_agent
    
    async def execute(
        self,
        user_query: str,
        schema_text: str = "",
        table_name: str = "",
        available_tables: list[dict] | None = None,
        chat_history: list[dict[str, str]] | None = None,
        column_stats_map: dict[str, list[dict]] | None = None,
        relevant_columns: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行语义解析"""
        self._log.info(f"解析: {user_query[:50]}...")
        
        try:
            from chatdb.agents.base import AgentContext
            
            # 构建 context（AgentContext 精简后不含 available_tables/selected_tables，
            # 通过动态属性传递给 SemanticParser，其内部用 getattr 安全读取）
            context = AgentContext(
                user_query=user_query,
                schema_text=schema_text,
                chat_history=chat_history or [],
            )
            context.available_tables = available_tables or []  # type: ignore[attr-defined]
            context.selected_tables = [table_name] if table_name else []  # type: ignore[attr-defined]
            context.column_stats_map = column_stats_map or {}  # type: ignore[attr-defined]
            context.relevant_columns = relevant_columns or {}  # type: ignore[attr-defined]
            
            # 调用 parser agent
            parser = self._get_parser()
            result = await parser.execute(context)
            
            if result.status.value == "success":
                return ToolResult.ok(
                    data={
                        "intent": result.data.get("intent"),
                        "table_name": result.data.get("table_name", ""),
                        "yml_config": result.data.get("yml_config", {}),
                    },
                    message="语义解析成功",
                )
            else:
                return ToolResult.fail(result.error or "语义解析失败")
        
        except Exception as e:
            self._log.error(f"解析失败: {e}")
            return ToolResult.fail(str(e))
    
    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        """ReAct 模式执行：直接修改 state"""
        from chatdb.core.react_state import ReActPhase, ErrorType
        
        state.phase = ReActPhase.SEMANTIC_PARSE

        # ★ schema_text 不再携带原始表列结构（避免 temp 表、源表列信息污染 prompt）
        #   v2（有 YAML）：prompt 中已有 虚拟字段定义 + column_stats，不需要原始列信息
        #   v1（无 YAML）：_extract_intent_from_schema 通过 tables_meta 获取列信息
        schema_text = ""

        # 表理解文本：注入 LLM 生成的表业务说明
        table_understanding = getattr(state, "table_understanding", "")
        if table_understanding:
            schema_text = f"## 表业务说明\n{table_understanding}"

        rc = getattr(state, "retrieval_context", None)
        column_stats_map: dict[str, list[dict]] = {}
        relevant_columns: dict[str, list[str]] = {}
        if rc is not None:
            if hasattr(rc, "format_schema_hint"):
                schema_hint = rc.format_schema_hint()
                if schema_hint:
                    schema_text = f"{schema_text}\n\n{schema_hint}"
            column_stats_map = getattr(rc, "column_stats_map", {}) or {}
            relevant_columns = getattr(rc, "relevant_columns", {}) or {}
        
        # 虚拟字段检索结果：将召回的虚拟字段信息注入 schema_text
        vf_retrieval = getattr(state, "retrieved_virtual_fields", None)
        if vf_retrieval is not None and hasattr(vf_retrieval, "format_prompt_section"):
            vf_section = vf_retrieval.format_prompt_section()
            if vf_section:
                schema_text = f"{schema_text}\n\n{vf_section}"
        
        result = await self.execute(
            user_query=state.user_query,
            schema_text=schema_text,
            table_name=state.table_name or "",
            available_tables=state.available_tables,
            chat_history=context.chat_history or None,
            column_stats_map=column_stats_map,
            relevant_columns=relevant_columns,
        )
        
        if result.success:
            # 更新 state（唯一权威源，不再同步到 context）
            from chatdb.agents.semantic_parser import StructuredIntent
            intent_dict = result.data.get("intent", {})
            if isinstance(intent_dict, dict):
                state.intent = StructuredIntent.from_dict(intent_dict, state.user_query)
            else:
                state.intent = intent_dict
            
            state.yml_config = result.data.get("yml_config", {})
            state.mark_need(need_intent=False, need_sql=True)
        else:
            state.set_error(result.error or "语义解析失败", ErrorType.AMBIGUOUS_INTENT)
