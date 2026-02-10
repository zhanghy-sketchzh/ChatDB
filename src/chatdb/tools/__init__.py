"""
Tool 模块 - 结构化工具定义（Agent + Tool 架构）

设计理念（参考 Agno）：
- Tool 是独立的"能力单元"
- Agent 只能调用自己注册的 Tool
- 通过 ToolRegistry 管理 Agent→Tool 映射

Tool 分类：
1. 核心 Tool（分析型）：
   - SemanticParseTool: 语义解析
   - SQLTool: SQL 流程（验证/执行/生成/执行与评估，单一流程类，由 Orchestrator 内部调用）
   - SummarizeAnswerTool: 总结回答

2. 分析 Tool：
   - ExploreByDimensionTool: 维度探索

3. 原子 Tool：
   - GetSchemaTool: 获取 Schema

使用示例：
```python
from chatdb.tools import ToolRegistry, SemanticParseTool, SQLTool

# 创建注册中心
registry = ToolRegistry()

# 注册工具并绑定到 Agent
registry.register_for_agent(SemanticParseTool(llm), "planner")
# SQL 流程由 Orchestrator 通过 SQLTool(llm, db_connector).run_workflow / run_generate / run_execute_and_evaluate 调用

# 获取 Agent 可用的工具
tools = registry.get_for_agent("planner")

# 生成 LLM prompt
prompt = registry.get_tools_description(agent_name="planner")
```
"""

# 基类和元数据
from chatdb.tools.base import (
    BaseTool,
    ToolParameter,
    ToolResult,
    ToolMetadata,
)

# 注册中心
from chatdb.tools.registry import (
    ToolRegistry,
    get_default_registry,
    register_tool,
)

# 原子工具
from chatdb.tools.get_schema import GetSchemaTool
from chatdb.tools.sql import (
    SQLTool,
    SQLCandidate,
    EvaluationResult,
    ExecuteSQLTool,
    ExecuteAndEvaluateTool,
    GenerateSQLTool,
    SQLWorkflowTool,
    ValidateSQLTool,
)

# 高级工具（Agent 能力封装）
from chatdb.tools.semantic_parse import SemanticParseTool
from chatdb.tools.explore_dimension import ExploreByDimensionTool
from chatdb.tools.summarize import SummarizeAnswerTool

# 简单 SQL 执行工具
from chatdb.tools.run_sql import RunSQLTool, create_run_sql_tool

__all__ = [
    # 基类
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "ToolMetadata",
    # 注册中心
    "ToolRegistry",
    "get_default_registry",
    "register_tool",
    # 原子工具
    "ExecuteSQLTool",
    "GetSchemaTool",
    "ValidateSQLTool",
    # SQL 工具（单一 SQLTool 类 + 薄包装）
    "SQLTool",
    "SQLCandidate",
    "EvaluationResult",
    "SQLWorkflowTool",
    "GenerateSQLTool",
    "ExecuteAndEvaluateTool",
    # 高级工具
    "SemanticParseTool",
    "ExploreByDimensionTool",
    "SummarizeAnswerTool",
    # 简单 SQL 执行
    "RunSQLTool",
    "create_run_sql_tool",
]
