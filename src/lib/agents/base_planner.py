"""
lib/agents/base_planner.py — 规划 Agent 基类

从 chatdb.agents.planner.PlannerAgent 和
chatreport.agents.planner.ReportPlanner 提取的通用规划模式。

通用流程：
1. build_plan_prompt() — 构建规划 prompt（注入上下文 + 约束）
2. 调用 LLM 生成 JSON 计划
3. parse_plan_response() — 解析 LLM 输出为结构化计划
4. validate_plan() — 计划校验（可覆写）

子类实现各阶段的业务逻辑。
"""

from abc import abstractmethod
from typing import Any

from lib.agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from lib.llm import BaseLLM


class BasePlanner(BaseAgent):
    """
    规划 Agent 基类 — LLM 驱动的任务分解

    通用流程：
    1. 构建规划 prompt
    2. 调用 LLM 生成计划
    3. 解析 + 校验计划
    4. 返回结构化计划

    子类实现：
    - build_plan_prompt(): 构建规划提示词
    - parse_plan_response(): 解析 LLM 输出
    - validate_plan(): 计划校验（可选覆写）
    - get_system_prompt(): 系统提示词
    """

    def __init__(self, name: str, llm: BaseLLM, description: str = ""):
        super().__init__(name, llm, description)
        self._plan_history: list[dict[str, Any]] = []

    async def generate_plan(
        self,
        state: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        通用规划流程

        Args:
            state: 当前状态
            context: 上下文信息
            **kwargs: 额外参数

        Returns:
            结构化计划（由 parse_plan_response 定义类型）
        """
        prompt = self.build_plan_prompt(state, context, **kwargs)
        response = await self.llm.chat(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            caller_name=f"{self.name}_plan",
        )
        plan = self.parse_plan_response(response)
        plan = self.validate_plan(plan, state)
        return plan

    @abstractmethod
    def build_plan_prompt(self, state: Any, context: Any = None, **kwargs: Any) -> str:
        """构建规划提示词"""
        ...

    @abstractmethod
    def parse_plan_response(self, response: str) -> Any:
        """解析 LLM 输出为结构化计划"""
        ...

    def validate_plan(self, plan: Any, state: Any) -> Any:
        """计划校验（默认不做额外校验，子类可覆写）"""
        return plan

    async def execute(self, context: AgentContext) -> AgentResult:
        """默认 execute 实现 — 子类通常直接使用 generate_plan"""
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message="Planner 应通过 generate_plan() 调用",
        )

    def clear_history(self) -> None:
        """清除规划历史"""
        self._plan_history.clear()
