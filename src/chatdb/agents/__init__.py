"""智能体模块 - Agent + Tool 架构

设计理念（v2 架构）：
1. semantic_parser 是**前置 workflow**，不属于 Planner 调度范围
2. Planner 只输出"分析任务"（type + description + notes + depends_on），支持 DAG 依赖
3. SQLAgent 接收 TaskRequest 消息，返回 TaskResponse 消息
4. Orchestrator 通过消息对象协调 Planner ↔ SQLAgent 通信
5. Planner 通过 collected_results 参数查看数据（不直接读写共享 state）
6. apply_adjustment 执行调整（跳过任务等），由 LLM 动态决策

Agent 通信协议：
- TaskRequest:  Orchestrator → SQLAgent（任务描述 + 上游结果快照）
- TaskResponse: SQLAgent → Orchestrator（执行结果列表）
- PlanDecision: Planner → Orchestrator（调度决策）

核心架构：
- SQLAgent: SQL 分析 Agent（核心）
  - run_task(state, context, request: TaskRequest) -> TaskResponse
  - 内部 ReAct 拆解：理解任务 → 生成 SQL → 执行 → 评估
  
- PlannerAgent: 规划决策（v2 增强）
  - generate_analysis_plan(): 生成分析任务列表（支持 DAG 依赖）
  - inspect_temp_results(): 查看 SQL Agent 执行结果
  - summarize_results_for_planner(): 数据摘要（供 LLM 分析）
  - apply_adjustment(): 执行计划调整（由 LLM 动态决定跳过哪些任务）
  - decide_next_action(): 根据结果决定下一步
  
- SemanticParser: 语义解析（前置 workflow）

使用方式：
```python
from chatdb.core.orchestrator import AgentOrchestrator

orch = AgentOrchestrator(llm, db, yml_config="data/yml/ieg.yml")
result = await orch.process_query("王者荣耀流水增长来自哪里")
```
"""

from chatdb.agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from chatdb.agents.semantic_parser import SemanticParser, StructuredIntent
from chatdb.agents.sql_agent import SQLAgent, DomainConfig, TaskResult, create_sql_agent
from chatdb.agents.planner import (
    PlannerAgent, 
    AnalysisPlan, 
    AnalysisTask,
    DEFAULT_TASK_TYPES,
    get_task_types,
    load_task_types_from_config,
)
from chatdb.tools.sql import SQLCandidate, EvaluationResult
from chatdb.core.react_state import ReActState, ReActPhase, ErrorType
from chatdb.core.messages import TaskRequest, TaskResponse, TaskResultEntry, PlanDecision

# 别名兼容
SemanticParserAgent = SemanticParser
AnalysisStep = AnalysisTask  # 旧名称兼容

__all__ = [
    # 基类
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    # SQL Agent（核心）
    "SQLAgent",
    "DomainConfig",
    "TaskResult",
    "create_sql_agent",
    # Planner（分析型规划 v2）
    "PlannerAgent",
    "AnalysisPlan",
    "AnalysisTask",
    "AnalysisStep",  # 兼容
    "DEFAULT_TASK_TYPES",
    "get_task_types",
    "load_task_types_from_config",
    # 语义解析（前置 workflow，含查询改写）
    "SemanticParser",
    "SemanticParserAgent",
    "StructuredIntent",
    # SQL 数据类
    "SQLCandidate",
    "EvaluationResult",
    # 状态管理
    "ReActState",
    "ReActPhase",
    "ErrorType",
    # Agent 通信消息
    "TaskRequest",
    "TaskResponse",
    "TaskResultEntry",
    "PlanDecision",
]
