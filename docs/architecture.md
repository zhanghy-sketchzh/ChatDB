# 架构概述

## 系统定位

ChatDB 是基于 **4 Agent + 3 Tool** 架构的 NL2SQL 系统，通过自然语言与数据库交互。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | OpenAI / Anthropic / 混元 |
| 数据库 | DuckDB（主）、PostgreSQL、MySQL、SQLite |
| 检索 | BM25 + jieba 分词 |
| 框架 | Python 3.11+、FastAPI、asyncio |

## 目录结构

```
ChatDB/
├── src/chatdb/
│   ├── agents/                 # 智能体模块（4 Agent）
│   │   ├── base.py             # 基类（含工具配置）
│   │   ├── planner.py          # 计划 Agent
│   │   ├── semantic_parser.py  # 语义解析 Agent
│   │   ├── sql_generator.py    # SQL 生成 Agent
│   │   └── result_evaluator.py # 结果评估 Agent
│   ├── tools/                  # 工具模块（3 Tool）
│   │   ├── base.py             # BaseTool, ToolParameter, ToolResult
│   │   ├── registry.py         # ToolRegistry 工具注册表
│   │   ├── execute_sql.py      # 执行 SQL
│   │   ├── get_schema.py       # 获取 Schema
│   │   └── validate_sql.py     # SQL 语法验证
│   ├── core/                   # 核心模块
│   │   ├── orchestrator.py     # ReAct 协调器
│   │   ├── agui_adapter.py     # AG-UI 协议适配器
│   │   ├── react_state.py      # 状态管理
│   │   ├── summarize.py        # 总结生成（含 research_mode 证据链）
│   │   └── tracer.py           # 任务追踪
│   ├── config/                 # 配置加载
│   │   ├── table_config.py     # YAML 配置加载器
│   │   ├── domain_config.py    # 领域配置
│   │   └── metrics_loader.py   # 指标加载
│   ├── preprocessing/          # 离线预处理
│   │   ├── preprocessor.py     # 预处理器
│   │   ├── column_profiler.py  # 列摘要
│   │   ├── schema_builder.py   # Schema 构建
│   │   └── text_index.py       # BM25 索引
│   ├── database/               # 数据库连接器
│   │   ├── base.py             # 基类
│   │   ├── schema.py           # Schema 检查
│   │   ├── duckdb/             # DuckDB（含 syntax_rules.py）
│   │   ├── mysql/              # MySQL
│   │   ├── postgresql/         # PostgreSQL
│   │   └── sqlite/             # SQLite
│   ├── llm/                    # LLM 适配层
│   │   ├── base.py             # 基类
│   │   ├── factory.py          # 工厂
│   │   ├── openai_llm.py       # OpenAI
│   │   ├── anthropic_llm.py    # Anthropic
│   │   └── hunyuan_llm.py      # 混元
│   ├── storage/                # 持久化存储
│   │   ├── chat_history.py     # 聊天历史
│   │   ├── meta_data.py        # 元数据
│   │   └── task_history.py     # 任务历史
│   ├── api/                    # FastAPI 接口
│   └── utils/                  # 工具函数
├── data/
│   ├── yml/                    # YAML 配置文件
│   └── pilot/                  # 本地数据库
└── docs/                       # 文档
```

## Agent 与 Tool 关系

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Planner   │  │  Semantic   │  │    SQL      │         │
│  │   Agent     │  │   Parser    │  │  Generator  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │              ResultEvaluatorAgent                │       │
│  │  (执行 + 验证 + 修正 + 总结)                      │       │
│  │                                                  │       │
│  │  配置的 Tools:                                   │       │
│  │  ┌──────────┐ ┌───────────┐ ┌─────────────┐    │       │
│  │  │execute   │ │get_schema │ │validate_sql │    │       │
│  │  │_sql      │ │           │ │             │    │       │
│  │  └──────────┘ └───────────┘ └─────────────┘    │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**设计原则：**
- **Agent** 是高级智能体，负责决策和 LLM 交互
- **Tool** 是原子操作，负责具体执行（数据库操作、语法检查）
- Agent 可配置使用哪些 Tool，配置后只能用这些 Tool
- Tool 不依赖 Agent 内部状态，可被多个 Agent 复用

## 核心流程

```
用户查询
    │
    ├── _quick_classify → "chat"
    │   └── _handle_chat_query（极简 LLM 回复，跳过全部分析流程）
    │
    ├── _quick_classify → "analysis" / "ambiguous"
    │   └── PARSE → PLAN → EXECUTE → SUMMARIZE → DONE
    │                  │
    │                  └── Planner decide 循环:
    │                       A(继续) / B(插入·重试) / C(跳过) / D(结束)
    │                       E(人类介入, 仅 research_mode)
    │
    └── SemanticParseTool → mode=other（兜底非数据问题）
```

| 阶段 | 说明 |
|------|------|
| CLASSIFY | 前置轻量分类（~50 token），分流 chat/analysis/ambiguous |
| PARSE | SemanticParserAgent 解析意图 → `StructuredIntent`（含 `research_mode` 自动判断） |
| PLAN | Planner 生成多步分析计划 |
| EXECUTE | SQLAgent 按计划执行，Planner 每批任务后决策（含深度分析引导 + 人类介入判断） |
| SUMMARIZE | 生成自然语言总结（research_mode 下包含证据链 + 结构化结论） |

### Deep Research 模式

LLM 在语义解析阶段自动判断是否开启 `research_mode`（归因分析、异常诊断等复杂问题）。开启后：

- **Planner E 选项**：LLM 可请求人类介入（语义不清、数据质量问题、能力边界）
- **SummarizeAnswer** 输出证据链报告（置信度、来源标注、局限性、建议后续）

### 人类介入（Human-in-the-Loop）

当 Planner 选择 E 时，查询暂停并返回 `status: need_clarification`，前端展示交互 UI。用户提交选择后通过 `/query/continue` 或 `/agui/continue` 恢复执行，用户选择作为自然语言上下文注入，由 LLM 自主调整分析策略。

### 统一响应结构

所有查询返回统一 status 字段：

| status | 含义 |
|--------|------|
| `completed` | 分析正常完成 |
| `need_clarification` | Planner 请求用户澄清 |

## ReActState 数据结构

```python
@dataclass
class ReActState:
    user_query: str
    table_name: str | None = None
    skill: str | None = None          # basic / trend / complex
    intent: StructuredIntent | None = None
    yml_config: dict = field(default_factory=dict)
    sql_candidates: list[dict] = field(default_factory=list)
    final_sql: str = ""
    execute_result: dict | None = None
    summary: str = ""
    thoughts: list[str] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)

    # Deep Research 相关
    research_mode: bool = False              # 由 LLM 自动判断
    intervention: dict | None = None         # 人类介入信息（Planner E 决策）
    intervention_step_id: int = 0            # 介入时的计划步骤号
```

## 查询分类（intent_type）

SemanticParserAgent 解析用户查询后会输出 `intent_type`，用于指导 SQL 生成策略：

| 类型 | 说明 | 关键词示例 |
|------|------|-----------|
| basic | 简单聚合查询 | 多少、总计、合计、是什么 |
| trend | 同比/环比计算 | 同比、环比、YTD、趋势、增长 |
| dimension | 分组/排序查询 | 分组、按、各、TOP、排名 |
| complex | 特殊业务逻辑 | 剔除、去掉、不含、递延前/后 |

## 错误类型（ErrorType）

Planner 根据错误类型决定回退策略：

| 错误类型 | 说明 | 回退策略 |
|----------|------|----------|
| UNKNOWN_COLUMN | 列不存在 | Critic 修正 → SemanticParser |
| TYPE_MISMATCH | 类型不匹配 | Critic 局部修正 |
| SYNTAX_ERROR | SQL 语法错误 | Critic 修正语法 |
| SEMANTIC_GAP | 业务口径缺失 | SemanticParser 重新匹配 |
| AMBIGUOUS_INTENT | 意图不明确 | SemanticParser 澄清 |
| NO_DATA | 无数据返回 | 放宽条件或接受结果 |
| TIMEOUT | 执行超时 | 放弃 |

## ReAct 阶段（ReActPhase）

```
INIT → SEMANTIC_PARSE → SCHEMA_RESOLVE → SQL_BUILD → EXECUTE → CRITIQUE → REFINE → DONE
                ↑                                       ↓         ↓
                └───────────────────────────────────────┴─────────┘
                              (错误回退)
```

| 阶段 | 说明 |
|------|------|
| INIT | 初始化，选择表 |
| SEMANTIC_PARSE | 语义解析，提取 StructuredIntent |
| SCHEMA_RESOLVE | Schema 补全（术语 → 列名映射） |
| SQL_BUILD | 生成/修正 SQL |
| EXECUTE | 执行 SQL |
| CRITIQUE | 评估结果/诊断错误 |
| REFINE | 修正 SQL |
| DONE | 完成 |
| GIVE_UP | 放弃（超过最大步数或无法修正） |
