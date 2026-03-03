"""
SummarizeAnswerTool - 总结回答工具

根据多阶段分析结果生成自然语言总结。

适用场景：
- 多阶段分析计划执行完毕后，整合所有任务结果生成最终回答
- 单任务场景视为"一个阶段"的多阶段汇总，统一走同一条路径

输入：
- user_query: 用户原始问题
- summary_context: 多阶段分析结果汇总文本

输出：
- summary: 自然语言总结
"""

from typing import Any, TYPE_CHECKING

from chatdb.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from chatdb.llm.base import BaseLLM
from chatdb.utils.logger import get_component_logger

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext


class SummarizeAnswerTool(BaseTool):
    """
    总结回答工具
    
    核心能力：
    - 根据多阶段分析结果生成自然语言总结
    - 整合多个任务的 SQL 和查询结果
    - 突出关键数据，结构化呈现
    
    设计：单任务 = 一个阶段的多阶段汇总，统一入口。
    """
    
    def __init__(self, llm: BaseLLM):
        metadata = ToolMetadata(
            name="summarize_answer",
            description="根据多阶段分析结果生成自然语言总结，回答用户问题",
            category="analysis",
            inputs={
                "user_query": {"type": "str", "description": "用户原始问题"},
                "summary_context": {"type": "str", "description": "多阶段分析结果汇总文本"},
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
        return """总结回答工具：根据多阶段分析结果生成自然语言总结。

使用场景：
- 分析计划执行完毕后，整合所有任务结果生成最终回答

输出：
- 简洁的自然语言总结，突出关键数据"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="user_query", type="string", description="用户问题"),
            ToolParameter(name="summary_context", type="string", description="多阶段分析结果汇总文本"),
        ]
    
    async def summarize(
        self,
        user_query: str,
        summary_context: str,
        research_mode: bool = False,
    ) -> ToolResult:
        """生成多阶段分析结果汇总
        
        将多个任务阶段的描述、SQL、查询结果整合后，
        让 LLM 综合所有分支的数据生成最终回答。
        
        单任务场景也走此路径（一个阶段 = 一个任务的结果）。
        
        Args:
            research_mode: 深度分析模式下追加证据链要求
        """
        self._log.info(f"生成分析结果汇总...{' (research_mode)' if research_mode else ''}")
        
        try:
            prompt = f"""{summary_context}

---

请根据以上所有阶段的查询结果，综合回答用户的问题。

## 输出要求

### 内容准确性
- **只使用查询结果中实际存在的数据**，严禁编造、推测或补充查询结果中没有的数字
- 数据要具体到数字，不要笼统概括
- 如果不同阶段的数据有关联关系（如先查 Top3 产品，再查各产品明细），要把它们关联起来呈现

### 数据质量
- 查询结果中维度列值为 None、NULL、null、空字符串的行可能是无效/未分类数据，在回答中应单独标注或排除
- 不要将无效数据作为正常的分类维度值参与分析和排名

### 呈现格式
- 用结构化的方式呈现（如表格、列表），突出关键数据
- 一段话给出完整、准确、精简的回答
- 如果有多个阶段，先总述结论，再分阶段展开关键细节

### 单位换算
- 如果上文提供了「指标单位说明」（含 display_divisor 和 unit），**所有阶段的同一指标必须统一使用相同的换算规则**
- display_divisor 的含义：将查询结果的原始数值**除以** display_divisor 后，以 unit 为单位展示
  - 例：display_divisor=100000000, unit=亿元 → 原始值 52613134466 应展示为 526.13 亿元
- 严禁对已经换算过的数值重复换算，也严禁遗漏换算。同一指标在不同阶段中的数值量级应一致"""

            if research_mode:
                prompt += """

## 深度分析报告要求（research_mode）

### 结论层
- 给出明确的核心结论（一句话回答用户问题）
- 标注置信度：高（多个数据源交叉验证）/ 中（数据支撑但未验证）/ 低（样本不足或有矛盾）

### 证据层
- 每个关键陈述后标注 [来源: task_id]，说明数据来自哪个分析步骤
- 如果存在数据矛盾，明确说明矛盾点和可能原因

### 不确定性
- 列出分析过程中的限制（数据时间范围、缺失维度等）
- 如果某些结论需要更多数据验证，指出具体需要什么

### 额外输出
在自然语言回答末尾追加一个 JSON 块（用 ```json 包裹）：
```json
{
  "confidence": "high/medium/low",
  "key_findings": [{"finding": "...", "evidence_task": "task_id", "data_point": "..."}],
  "limitations": ["..."],
  "suggested_follow_ups": ["..."]
}
```"""

            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是数据分析专家。请基于查询结果综合回答用户问题。回答要准确、完整、结构清晰。严格基于数据回答，不得编造数据。",
                caller_name="summarize_answer",
            )
            
            return ToolResult.ok(
                data={"summary": response.strip()},
                message="汇总完成",
            )
        
        except Exception as e:
            self._log.error(f"汇总失败: {e}")
            return ToolResult.ok(
                data={"summary": f"分析已完成，但汇总生成失败: {e}"},
                message="汇总失败",
            )

    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        """ReAct 模式执行：直接修改 state
        
        统一入口：通过 state.extra_context 传入多阶段分析结果汇总文本。
        """
        summary_context = getattr(state, "extra_context", "")
        if not summary_context:
            self._log.warn("无 extra_context，跳过汇总")
            return
        
        result = await self.summarize(
            user_query=state.user_query,
            summary_context=summary_context,
            research_mode=getattr(state, "research_mode", False),
        )
        if result.success:
            state.summary = result.data.get("summary", "")
