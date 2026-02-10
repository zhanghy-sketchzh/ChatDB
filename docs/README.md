# ChatDB 文档

> NL2SQL 系统 - 4 Agent + 3 Tool 架构

## 文档目录

| 文档 | 说明 |
|------|------|
| [架构概述](./architecture.md) | 系统架构、目录结构、Agent/Tool 关系 |
| [YAML 配置](./yaml-config.md) | 业务口径配置规则 |
| [Agent 与 Tool 详解](./agents.md) | 4 Agent + 3 Tool 的设计与实现 |
| [数据预处理](./preprocessing.md) | 离线处理、Schema 生成、BM25 索引 |

## 快速开始

```python
from chatdb.core import AgentOrchestrator
from chatdb.llm import LLMFactory
from chatdb.database.duckdb import DuckDBConnector

# 初始化
llm = LLMFactory.create()
db = DuckDBConnector("data/database/test.db")

# 使用 Orchestrator
orchestrator = AgentOrchestrator(
    llm=llm,
    db_connector=db,
    yml_config="data/yml/metrics_config.yml",
)
result = await orchestrator.process_query("IEG本部今年的流水是多少")

print(result["sql"])      # 生成的 SQL
print(result["summary"])  # 自然语言总结
```

## 核心流程

```
用户查询
    │
    ▼
┌─────────────────────────────────────────┐
│           AgentOrchestrator             │
│                                         │
│  PLAN → PARSE → GENERATE → EVALUATE     │
│                                         │
│  ┌──────────┐ ┌────────────┐            │
│  │ Planner  │→│ Semantic   │            │
│  │          │ │ Parser     │            │
│  └──────────┘ └────────────┘            │
│       │              │                  │
│       ▼              ▼                  │
│  ┌────────────┐ ┌──────────────────┐   │
│  │    SQL     │→│ Result Evaluator │   │
│  │  Generator │ │                  │   │
│  └────────────┘ │ Tools:           │   │
│                 │ • execute_sql    │   │
│                 │ • get_schema     │   │
│                 │ • validate_sql   │   │
│                 └──────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
SQL + 结果 + 总结
```

## 架构概览

### 4 Agent

| Agent | 职责 |
|-------|------|
| **PlannerAgent** | 任务规划：决定下一步调用哪个 Agent |
| **SemanticParserAgent** | 语义解析：理解查询意图，输出 `StructuredIntent` |
| **SQLGeneratorAgent** | SQL 生成：从 Intent 生成多候选 SQL |
| **ResultEvaluatorAgent** | 结果评估：执行 + 验证 + 修正 + 总结 |

### 3 Tool

| Tool | 职责 |
|------|------|
| **execute_sql** | 执行 SQL 语句，返回结果或错误 |
| **get_schema** | 获取表 Schema（DDL 或 Markdown 格式） |
| **validate_sql** | SQL 语法验证（基于 DuckDB 语法规则） |

### Agent 与 Tool 的关系

- **Agent** 是高级智能体，负责决策和 LLM 交互
- **Tool** 是原子操作，负责具体执行
- Agent 可配置使用哪些 Tool（`tools` 参数）
- 配置后 Agent 只能使用这些 Tool（`use_tool` 方法）

```python
# 示例：SQL 流程（单一 SQLTool）
from chatdb.tools import SQLTool, GetSchemaTool

sql_tool = SQLTool(llm=llm, db_connector=db)
result = await sql_tool.execute_sql(sql="SELECT ...")
```

## 配置文件

| 配置 | 路径 |
|------|------|
| 应用配置 | `config.toml` |
| YAML 业务配置 | `data/yml/*.yml` |
| 元数据库 | `data/pilot/meta_data.db` |
| BM25 索引 | `data/pilot/text_index.db` |
