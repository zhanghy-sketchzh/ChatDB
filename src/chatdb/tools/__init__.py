"""
Tool 模块 - Agent 可动态选择的工具

设计理念：
- Tool 是 Agent 可选择调用的"能力单元"
- 流程步骤（SemanticParseTool、SummarizeAnswerTool）已移至 core/
- 通过 ToolRegistry 管理 Agent→Tool 映射

可用工具：
1. SQLTool:  SQL 生成、验证、执行与评估
2. UnixTool: 文件读写、目录遍历、内容搜索

使用示例：
```python
from chatdb.tools import ToolRegistry, SQLTool, UnixTool

registry = ToolRegistry()
registry.register_for_agent(SQLTool(llm, db), "sql_agent")
registry.register_for_agent(UnixTool(workspace="."), "planner")
```
"""

from chatdb.tools.base import (
    BaseTool,
    SubToolDef,
    ToolParameter,
    ToolResult,
    ToolMetadata,
    AgentBackedTool,
)

from chatdb.tools.registry import (
    ToolRegistry,
    get_default_registry,
    register_tool,
)

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

from chatdb.tools.unix import UnixTool

__all__ = [
    # 基类
    "BaseTool",
    "SubToolDef",
    "ToolParameter",
    "ToolResult",
    "ToolMetadata",
    "AgentBackedTool",
    # 注册中心
    "ToolRegistry",
    "get_default_registry",
    "register_tool",
    # SQL 工具
    "SQLTool",
    "SQLCandidate",
    "EvaluationResult",
    "ExecuteSQLTool",
    "ExecuteAndEvaluateTool",
    "GenerateSQLTool",
    "SQLWorkflowTool",
    "ValidateSQLTool",
    # 文件系统工具
    "UnixTool",
]
