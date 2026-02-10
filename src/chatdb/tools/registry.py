"""
ToolRegistry - 工具注册中心（Agent + Tool 架构）

设计理念（参考 Agno）：
- 统一管理所有工具
- 维护 "Agent → 可用工具列表" 的映射
- 给 LLM/Planner 提供可用工具的描述清单
- 支持按分类、按 Agent 过滤工具

核心功能：
1. 工具注册/注销
2. Agent-Tool 绑定
3. 工具描述生成（供 LLM prompt）
4. 工具调用
"""

from typing import Any

from chatdb.tools.base import BaseTool, ToolResult, ToolMetadata
from chatdb.utils.logger import logger


class ToolRegistry:
    """
    工具注册中心
    
    管理所有可用工具，支持：
    - 注册/注销工具
    - 绑定 Agent 与 Tool 的关系
    - 按 Agent 获取可用工具
    - 按分类获取工具
    - 生成 LLM prompt 用的工具描述
    """
    
    def __init__(self):
        # 所有已注册工具 {tool_name: BaseTool}
        self._tools: dict[str, BaseTool] = {}
        
        # Agent → Tool 映射 {agent_name: [tool_name, ...]}
        self._agent_tools: dict[str, list[str]] = {}
        
        # Tool → Agent 反向映射（方便查询）{tool_name: [agent_name, ...]}
        self._tool_agents: dict[str, list[str]] = {}
    
    # ============================================================
    # 工具注册
    # ============================================================
    
    def register(
        self,
        tool: BaseTool,
        agent_name: str | None = None,
    ) -> None:
        """
        注册工具
        
        Args:
            tool: 工具实例
            agent_name: 绑定的 Agent 名称（可选）
        """
        if tool.name in self._tools:
            logger.warning(f"工具已存在，将被覆盖: {tool.name}")
        
        self._tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")
        
        # 如果指定了 Agent，建立绑定关系
        if agent_name:
            self.bind_to_agent(tool.name, agent_name)
    
    def register_for_agent(
        self,
        tool: BaseTool,
        agent_name: str,
    ) -> None:
        """
        注册工具并绑定到指定 Agent
        
        这是 Agent 注册自己专属工具的推荐方式
        """
        self.register(tool, agent_name)
    
    def unregister(self, name: str) -> bool:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            
            # 清理绑定关系
            if name in self._tool_agents:
                for agent in self._tool_agents[name]:
                    if agent in self._agent_tools:
                        self._agent_tools[agent] = [
                            t for t in self._agent_tools[agent] if t != name
                        ]
                del self._tool_agents[name]
            
            return True
        return False
    
    # ============================================================
    # Agent-Tool 绑定
    # ============================================================
    
    def bind_to_agent(self, tool_name: str, agent_name: str) -> None:
        """
        将工具绑定到 Agent
        
        绑定后，该 Agent 才能使用此工具
        """
        if tool_name not in self._tools:
            logger.warning(f"工具不存在: {tool_name}")
            return
        
        # Agent → Tools
        if agent_name not in self._agent_tools:
            self._agent_tools[agent_name] = []
        if tool_name not in self._agent_tools[agent_name]:
            self._agent_tools[agent_name].append(tool_name)
        
        # Tool → Agents
        if tool_name not in self._tool_agents:
            self._tool_agents[tool_name] = []
        if agent_name not in self._tool_agents[tool_name]:
            self._tool_agents[tool_name].append(agent_name)
        
        logger.debug(f"绑定 {tool_name} → {agent_name}")
    
    def unbind_from_agent(self, tool_name: str, agent_name: str) -> None:
        """解除工具与 Agent 的绑定"""
        if agent_name in self._agent_tools:
            self._agent_tools[agent_name] = [
                t for t in self._agent_tools[agent_name] if t != tool_name
            ]
        
        if tool_name in self._tool_agents:
            self._tool_agents[tool_name] = [
                a for a in self._tool_agents[tool_name] if a != agent_name
            ]
    
    # ============================================================
    # 工具获取
    # ============================================================
    
    def get(self, name: str) -> BaseTool | None:
        """获取工具"""
        return self._tools.get(name)
    
    def get_for_agent(self, agent_name: str) -> list[BaseTool]:
        """
        获取指定 Agent 可用的工具列表
        
        这是 Agent 获取自己可用工具的标准方式
        """
        tool_names = self._agent_tools.get(agent_name, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_by_category(self, category: str) -> list[BaseTool]:
        """按分类获取工具"""
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]
    
    def list_tools(self) -> list[BaseTool]:
        """列出所有工具"""
        return list(self._tools.values())
    
    def list_names(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())
    
    def list_agent_tools(self, agent_name: str) -> list[str]:
        """列出指定 Agent 的工具名称"""
        return self._agent_tools.get(agent_name, [])
    
    def list_tool_agents(self, tool_name: str) -> list[str]:
        """列出使用指定工具的 Agent"""
        return self._tool_agents.get(tool_name, [])
    
    # ============================================================
    # LLM Prompt 生成
    # ============================================================
    
    def get_tools_description(
        self,
        agent_name: str | None = None,
        category: str | None = None,
        include_subtools: bool = False,
    ) -> str:
        """
        获取工具描述文本（供 LLM prompt 使用）
        
        Args:
            agent_name: 只获取该 Agent 可用的工具
            category: 只获取该分类的工具
            include_subtools: 是否包含子工具详情（渐进式披露）
        
        Returns:
            格式化的工具描述文本
        """
        # 筛选工具
        if agent_name:
            tools = self.get_for_agent(agent_name)
        elif category:
            tools = self.get_by_category(category)
        else:
            tools = self.list_tools()
        
        if not tools:
            return "无可用工具"
        
        lines = ["## 可用工具\n"]
        for tool in tools:
            lines.append(tool.get_prompt_description(include_subtools))
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def get_tools_brief(self, agent_name: str | None = None) -> str:
        """
        获取工具简介（一行一个，用于快速展示）
        
        格式：- tool_name: 简短描述
        """
        if agent_name:
            tools = self.get_for_agent(agent_name)
        else:
            tools = self.list_tools()
        
        if not tools:
            return "无可用工具"
        
        lines = ["可用工具："]
        for tool in tools:
            # 取描述的第一行
            desc_first_line = tool.description.split("\n")[0]
            lines.append(f"- {tool.name}: {desc_first_line}")
        
        return "\n".join(lines)
    
    def get_function_schemas(
        self,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取工具的 Function Schema（用于 OpenAI Function Calling）
        
        Args:
            agent_name: 只获取该 Agent 可用的工具
        """
        if agent_name:
            tools = self.get_for_agent(agent_name)
        else:
            tools = self.list_tools()
        
        return [tool.to_function_schema() for tool in tools]
    
    # ============================================================
    # 工具调用
    # ============================================================
    
    async def call(
        self,
        name: str,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        调用工具
        
        Args:
            name: 工具名称
            agent_name: 调用的 Agent（如指定，会检查权限）
            **kwargs: 工具参数
        
        Returns:
            ToolResult: 执行结果
        """
        tool = self.get(name)
        if not tool:
            return ToolResult.fail(f"未知工具: {name}")
        
        # 权限检查
        if agent_name:
            allowed_tools = self._agent_tools.get(agent_name, [])
            if name not in allowed_tools:
                return ToolResult.fail(
                    f"Agent '{agent_name}' 无权使用工具 '{name}'。"
                    f"可用工具: {allowed_tools}"
                )
        
        # 验证参数
        is_valid, error = tool.validate_params(**kwargs)
        if not is_valid:
            return ToolResult.fail(error)
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"工具 {name} 执行失败: {e}")
            return ToolResult.fail(str(e))
    
    # ============================================================
    # 魔术方法
    # ============================================================
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"<ToolRegistry: {len(self._tools)} tools, {len(self._agent_tools)} agents>"


# ============================================================
# 全局注册表
# ============================================================

_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """获取默认注册表"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def register_tool(tool: BaseTool, agent_name: str | None = None) -> None:
    """注册工具到默认注册表"""
    get_default_registry().register(tool, agent_name)
