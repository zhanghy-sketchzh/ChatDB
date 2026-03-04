"""
lib — 公共模块（chatdb / chatreport / chathtml / chatpython 共用）

子模块：
  - lib.agents    : Agent 框架基类（BaseAgent, BasePlanner, BaseWriter）
  - lib.classify  : 统一查询分类器（QueryRouter）
  - lib.core      : 核心框架（BaseState, BaseOrchestrator, BaseAGUIAdapter, BaseDAG, ResultCache, BaseScratchPadManager）
  - lib.llm       : LLM 基类和工厂
  - lib.storage   : 存储层（TaskHistory, ChatHistory）
  - lib.tools     : 工具基类（BaseTool, ToolResult, ...）
  - lib.utils     : 通用工具（config, logger, common, exceptions, json_utils）
"""
