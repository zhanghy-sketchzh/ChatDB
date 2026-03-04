"""
lib/agents/base_writer.py — 内容生成 Agent 基类

从 chatdb.core.summarize.SummarizeAnswerTool 和
chatreport.agents.writer.ReportWriter 提取的通用内容生成模式。

通用流程：
1. build_write_prompt() — 构建写作 prompt（注入数据 + 指令）
2. 调用 LLM 生成内容
3. verify() — 可选自检验证
4. 返回最终内容

子类实现各阶段的业务逻辑。
"""

from abc import abstractmethod
from typing import Any

from lib.agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from lib.llm import BaseLLM


class BaseWriter(BaseAgent):
    """
    内容生成 Agent 基类 — LLM 驱动的文本输出

    通用流程：
    1. 构建内容生成 prompt
    2. 调用 LLM 生成内容
    3. 可选自检验证
    4. 返回最终内容

    子类实现：
    - build_write_prompt(): 构建写作提示词
    - verify(): 可选覆写自定义验证逻辑
    - get_system_prompt(): 系统提示词
    """

    def __init__(self, name: str, llm: BaseLLM, description: str = ""):
        super().__init__(name, llm, description)

    async def generate(
        self,
        context: dict[str, Any],
        verify: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        通用内容生成流程

        Args:
            context: 写作上下文（数据、指令等）
            verify: 是否执行自检验证
            **kwargs: 额外参数

        Returns:
            生成的内容文本
        """
        prompt = self.build_write_prompt(context, **kwargs)
        content = await self.llm.chat(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            caller_name=f"{self.name}_write",
        )
        content = content.strip()

        if verify:
            content = await self.verify(content, context, **kwargs)

        return content

    @abstractmethod
    def build_write_prompt(self, context: dict[str, Any], **kwargs: Any) -> str:
        """构建写作提示词"""
        ...

    async def verify(
        self,
        content: str,
        context: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """
        自检验证（默认不做验证，子类可覆写）

        Args:
            content: 初始生成的内容
            context: 原始写作上下文

        Returns:
            验证/改进后的内容
        """
        return content

    async def execute(self, context: AgentContext) -> AgentResult:
        """默认 execute 实现 — 子类通常直接使用 generate()"""
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message="Writer 应通过 generate() 调用",
        )
