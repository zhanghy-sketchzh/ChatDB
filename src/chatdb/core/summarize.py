"""
SummarizeAnswerTool - 总结回答工具

根据查询结果生成自然语言总结。

适用场景：
- 查询已完成，需要生成用户可读的总结
- 需要整合多个分析结果

输入：
- user_query: 用户原始问题
- rows: 查询结果
- analysis_results: 分析结果（可选）

输出：
- summary: 自然语言总结
"""

from typing import Any, TYPE_CHECKING

from chatdb.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from chatdb.llm.base import BaseLLM
from chatdb.utils.logger import get_component_logger
from chatdb.utils.common import format_rows

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext


class SummarizeAnswerTool(BaseTool):
    """
    总结回答工具
    
    核心能力：
    - 根据查询结果生成自然语言总结
    - 整合多个分析结果
    - 突出关键数据
    """
    
    def __init__(self, llm: BaseLLM):
        metadata = ToolMetadata(
            name="summarize_answer",
            description="根据查询结果生成自然语言总结，回答用户问题",
            category="analysis",
            inputs={
                "user_query": {"type": "str", "description": "用户原始问题"},
                "rows": {"type": "list", "description": "查询结果"},
                "analysis_results": {"type": "list", "description": "分析结果（可选）"},
            },
            outputs={
                "summary": {"type": "str", "description": "自然语言总结"},
            },
            is_core=False,
        )
        super().__init__(metadata)
        
        self.llm = llm
        self._log = get_component_logger("SummarizeTool")
    
    @property
    def name(self) -> str:
        return "summarize_answer"
    
    @property
    def description(self) -> str:
        return """总结回答工具：根据查询结果生成自然语言总结。

使用场景：
- 查询已完成，需要生成用户可读的总结
- 需要整合多个分析结果

输出：
- 简洁的自然语言总结，突出关键数据"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="user_query", type="string", description="用户问题"),
            ToolParameter(name="rows", type="array", description="查询结果"),
            ToolParameter(name="analysis_results", type="array", description="分析结果", required=False),
        ]
    
    async def execute(
        self,
        user_query: str,
        rows: list[dict],
        analysis_results: list[dict] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """生成总结"""
        self._log.info("生成总结...")
        
        # 空结果处理
        if not rows or len(rows) == 0:
            return ToolResult.ok(
                data={"summary": "查询未返回结果。"},
                message="空结果",
            )
        
        try:
            # 构建 prompt
            prompt = f"""用户问题: {user_query}

查询结果（共 {len(rows)} 行，前 {min(10, len(rows))} 行）:
{format_rows(rows[:10])}"""
            
            # 添加分析结果
            if analysis_results:
                for ar in analysis_results:
                    dim = ar.get("dimension", "")
                    ar_rows = ar.get("rows", [])
                    if ar_rows:
                        sub = format_rows(ar_rows[:5])
                        prompt += f"\n\n按维度「{dim}」拆解（共 {ar.get('row_count', len(ar_rows))} 类）:\n{sub}"
            
            prompt += "\n\n请用简洁的语言总结，突出关键数据；若有拆解结果请结合说明。"
            
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是数据分析专家，请简洁回答。",
                caller_name="summarize_answer",
            )
            
            return ToolResult.ok(
                data={"summary": response.strip()},
                message="总结完成",
            )
        
        except Exception as e:
            self._log.error(f"总结失败: {e}")
            summary = f"查询返回 {len(rows)} 行结果。"
            return ToolResult.ok(
                data={"summary": summary},
                message="使用默认总结",
            )
    
    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        """ReAct 模式执行：直接修改 state"""
        rows = state.execute_result.get("rows", []) if state.execute_result else []
        
        # 获取分析结果
        analysis_results = None
        if state.analysis_slices:
            analysis_results = [s.to_dict() for s in state.analysis_slices]
        
        result = await self.execute(
            user_query=state.user_query,
            rows=rows,
            analysis_results=analysis_results,
        )
        
        if result.success:
            state.summary = result.data.get("summary", "")
