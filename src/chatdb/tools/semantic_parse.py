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
            self._parser_agent = SemanticParser(self.llm, self.yml_config_path)
        return self._parser_agent
    
    async def execute(
        self,
        user_query: str,
        schema_text: str = "",
        table_name: str = "",
        available_tables: list[dict] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行语义解析"""
        self._log.info(f"解析: {user_query[:50]}...")
        
        try:
            from chatdb.agents.base import AgentContext
            
            # 构建 context
            context = AgentContext(
                user_query=user_query,
                schema_text=schema_text,
                available_tables=available_tables or [],
                selected_tables=[table_name] if table_name else [],
            )
            
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
        
        result = await self.execute(
            user_query=context.user_query,
            schema_text=context.schema_text,
            table_name=state.table_name or "",
            available_tables=context.available_tables,
        )
        
        if result.success:
            # 更新 state
            from chatdb.agents.semantic_parser import StructuredIntent
            intent_dict = result.data.get("intent", {})
            if isinstance(intent_dict, dict):
                state.intent = StructuredIntent.from_dict(intent_dict, context.user_query)
            else:
                state.intent = intent_dict
            
            state.yml_config = result.data.get("yml_config", {})
            state.observe(f"意图: {state.intent.intent_type if state.intent else 'unknown'}")
            state.mark_need(need_intent=False, need_sql=True)
            
            # 同步到 context
            context.query_intent = state.intent
            context.yml_config = state.yml_config
        else:
            state.set_error(result.error or "语义解析失败", ErrorType.AMBIGUOUS_INTENT)
