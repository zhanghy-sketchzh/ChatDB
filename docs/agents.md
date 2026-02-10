# Agent 与 Tool 详解

## 概述

**4 Agent + 3 Tool** 架构：
- **Agent**：高级智能体，负责 LLM 交互和决策
- **Tool**：原子操作，Agent 可配置使用

| Agent | 职责 | LLM 输入 | LLM 输出 |
|-------|------|----------|----------|
| Planner | 任务规划 | 查询 + 可用表 | 下一步行动 |
| SemanticParser | 语义解析 | 查询 + YAML 选项 | 结构化 Intent JSON |
| SQLGenerator | SQL 生成 | Intent + Schema | 多候选 SQL + reason |
| ResultEvaluator | 评估修正 | 报错 + YAML + SQL | 诊断 + 最小修正 |

| Tool | 职责 | 输入 | 输出 |
|------|------|------|------|
| execute_sql | 执行 SQL | SQL 语句 | 查询结果或错误 |
| get_schema | 获取 Schema | 表名 | DDL/Markdown Schema |
| validate_sql | 语法验证 | SQL 语句 | 验证结果 |

---

## Tool 系统

### 设计原则

1. **原子性**：每个 Tool 只做一件事
2. **独立性**：Tool 不依赖 Agent 的内部状态
3. **可复用**：多个 Agent 可以共享同一个 Tool

### Tool 基类

```python
class BaseTool(ABC):
    @property
    def name(self) -> str: ...       # 工具名称
    @property
    def description(self) -> str: ... # 工具描述（供 LLM 理解）
    @property
    def parameters(self) -> list[ToolParameter]: ...  # 参数定义
    
    async def execute(self, **kwargs) -> ToolResult: ...  # 执行方法
    def to_function_schema(self) -> dict: ...  # 转 OpenAI Function Calling 格式
```

### 现有 Tool

```
tools/
├── base.py           # BaseTool, ToolParameter, ToolResult
├── registry.py       # ToolRegistry 管理工具
├── execute_sql.py    # 执行 SQL（数据库操作）
├── get_schema.py     # 获取 Schema（数据库操作）
└── validate_sql.py   # SQL 语法验证（DuckDB 语法规则）
```

### Agent 使用 Tool

```python
# SQL 流程（单一 SQLTool）
from chatdb.tools import SQLTool, GetSchemaTool

sql_tool = SQLTool(llm=llm, db_connector=db)

# 执行 SQL
result = await sql_tool.execute_sql(sql="SELECT ...")
if result.success:
    rows = result.data["rows"]
else:
    error = result.error
```

---

## 1. PlannerAgent（Team Captain）

**职责**：作为团队队长，负责任务分配和策略调整，不亲自执行。

### 设计理念（参考 Agno/MiroThinker）
- Planner 是"队长"，不亲自下场写 SQL
- 维护团队执行历史，了解每个队友做了什么
- 遇到重复错误时，主动改变策略（回退到上游）

### 团队成员

| 队友 | 代号 | 技能 |
|------|------|------|
| 语义专家 | semantic_parser | 理解用户意图、提取业务术语、匹配 YAML 配置 |
| SQL 专家 | sql_generator | 根据结构化意图生成 SQL、提供多候选方案 |
| 执行专家 | result_evaluator | 执行 SQL、诊断错误、做最小修正、生成结果总结 |

### 决策模式

**规则决策** (`decide`)：基于 `state.need_*` 标记的快速决策

**LLM 决策** (`decide_with_llm`)：Team Captain 模式，综合考虑：
- 当前任务状态
- 团队执行历史（每个 Agent 的输入输出）
- 错误模式分析（重复错误检测）
- 策略调整建议

### 执行历史追踪

```python
# 记录 Agent 执行结果
planner.record_execution(
    agent="result_evaluator",
    step=3,
    input_summary="执行 SQL: SELECT ...",
    output_summary="错误: 列不存在",
    success=False,
    error="Column 'xxx' not found"
)

# 分析错误模式
error_pattern = planner._get_error_pattern()
# {'Column not found': 2}  # 同类错误出现2次

# 检测是否需要改变策略
if planner._should_change_strategy(state):
    # 回退到语义解析，而不是继续修正
    return "semantic_parser"
```

### 策略调整原则

| 情况 | 策略 |
|------|------|
| 首次错误 | 让 result_evaluator 尝试修正 |
| 同类错误重复出现 | 改变策略！回退到 semantic_parser |
| 修正超过2次仍失败 | 考虑是否理解错了用户意图 |
| 同一 Agent 连续失败 | 换个 Agent 试试 |
| 多次回退仍失败 | give_up 放弃任务 |

### LLM Prompt 示例

```
你是 NL2SQL 团队的队长（Planner）。你的职责是分配任务给队友，而不是亲自执行。

## 当前任务
用户问题: IEG本部今年的流水是多少

## 你的团队
- **语义专家** (semantic_parser)
  技能: 理解用户意图、提取业务术语、匹配 YAML 配置
  
- **SQL 专家** (sql_generator)
  技能: 根据结构化意图生成 SQL、提供多候选方案
  
- **执行专家** (result_evaluator)
  技能: 执行 SQL、诊断错误、做最小修正

## 团队执行历史
1. [✓] semantic_parser (Step 1)
   输入: 解析用户查询...
   输出: 提取意图: flow, IEG本部...
2. [✓] sql_generator (Step 2)
   输入: 生成 SQL...
   输出: SELECT SUM(...) FROM ...
3. [✗] result_evaluator (Step 3)
   输入: 执行 SQL...
   错误: Column '数据来源' not found...

## 策略建议
⚠️ 检测到执行错误，建议让 result_evaluator 修正
   如果再次失败，考虑回退到 semantic_parser

你的选择:
```

---

## 2. SemanticParserAgent

**职责**：从自然语言提取结构化意图，不是"改写问题"。

### 输入
- 用户查询
- YAML 中可选的 metrics / dimensions / filters ID

### 输出格式（StructuredIntent）
```json
{
  "intent_type": "basic|trend|dimension|complex",
  "metrics": ["metric_id_from_yaml"],
  "dimensions": ["dimension_id_from_yaml"],
  "time": {
    "granularity": "year|quarter|month",
    "year": 2024,
    "quarter": 1,
    "comparison": "yoy|qoq|ytd|null"
  },
  "filters": [
    {"dimension": "region", "value": "overseas", "operator": "="}
  ],
  "exclusions": ["exclusion_expr"],
  "order_by": {"column": "metric_id", "direction": "DESC"},
  "limit": 10
}
```

### intent_type 分类

| 类型 | 说明 | 示例查询 |
|------|------|----------|
| basic | 简单聚合 | "今年流水是多少" |
| trend | 同比/环比 | "Q1流水同比增长多少" |
| dimension | 分组/排序 | "各部门流水TOP10" |
| complex | 特殊逻辑 | "递延后利润（剔除XX）" |

### 核心规则
1. `metrics`/`dimensions` **必须从 YAML 选择**，不能发明
2. `comparison`: yoy=同比, qoq=环比, ytd=年累计
3. 只输出 JSON，不要自然语言解释

---

## 3. SQLGeneratorAgent

**职责**：从 StructuredIntent 生成多候选 SQL，每条带业务解释。

### 输入
- StructuredIntent（上一步输出）
- YAML 中的指标/维度定义（含 expr、default_filters）
- Schema

### 输出格式
```json
{
  "candidates": [
    {
      "sql": "SELECT 投资公司, 季度, SUM(流水) ... GROUP BY ...",
      "reason": "使用流水口径 + 海外过滤 + 按季度汇总 + 计算同比"
    },
    {
      "sql": "SELECT ... WITH prev_year AS ...",
      "reason": "使用 CTE 方式计算同比，更精确"
    }
  ]
}
```

### 核心规则
1. 生成 1-3 个候选 SQL
2. `reason` 说明使用了哪些口径、筛选、聚合方式
3. 复杂问题（trend/complex）应提供多种可行方案
4. **使用 DuckDB 语法**（从 `database/duckdb/syntax_rules.py` 导入）

### DuckDB 语法规则
```python
from chatdb.database.duckdb.syntax_rules import get_duckdb_syntax_rules

# 获取语法规则文本（用于 Prompt）
rules = get_duckdb_syntax_rules()
```

---

## 4. ResultEvaluatorAgent

**职责**：执行 SQL，失败时让 LLM "看"报错做最小修正。

### 配置的 Tool
- `execute_sql`: 执行 SQL 语句
- `get_schema`: 获取表 Schema（可选）
- `validate_sql`: SQL 语法验证（可选）

### 流程
1. 执行 SQL（使用 `execute_sql` Tool）
2. 如果失败，调用 LLM 诊断 + 修正
3. 重试（最多 2 次）
4. 成功后生成总结

### 诊断输入
- 当前 SQL
- DB 报错信息（语法错误 / 列不存在 / 类型不匹配）
- YAML 业务规则
- 原始查询 + Intent

### 诊断输出格式
```json
{
  "diagnosis": "列名错误：使用了 '月份' 作为字符串，但它是 BIGINT 类型",
  "refined_sql": "SELECT ... WHERE \"月份\" = 1 ..."
}
```

### 核心规则
1. **最小修改**：只改出错的部分，不重写整个 SQL
2. **对照 Schema**：列名、类型要与 Schema 一致
3. **对照 YAML**：业务筛选条件要完整

---

## 数据流

```
用户查询
    ↓
┌─────────────────┐
│    Planner      │  → action: semantic_parser
└─────────────────┘
    ↓
┌─────────────────┐
│ SemanticParser  │  → StructuredIntent (JSON)
└─────────────────┘
    ↓
┌─────────────────┐
│ SQLGenerator    │  → candidates: [{sql, reason}]
└─────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│          ResultEvaluator                     │
│                                              │
│  use_tool("execute_sql") → (diagnose → refine)* → summary
│                                              │
└─────────────────────────────────────────────┘
    ↓
最终结果
```

---

## 关键设计点

### 1. Agent 与 Tool 分离
- **旧方案**：Agent 直接执行所有操作
- **新方案**：Agent 通过配置的 Tool 执行原子操作
- **好处**：职责清晰、可复用、便于测试

### 2. 结构化意图而非问题改写
- **旧方案**：LLM 把"海外流水"改写成"流水且地区=海外"
- **新方案**：LLM 输出 `{"filters": [{"dimension": "region", "value": "overseas"}]}`
- **好处**：可控、可验证、便于 SQL 映射

### 3. 多候选 + reason
- 复杂问题可能有多种正确解法
- reason 用于自我诊断和错误修正
- 执行失败可以切换候选

### 4. 最小修正而非重新生成
- 报错时只改出错的部分
- 保持原 SQL 的整体结构
- 减少 LLM 幻觉和偏离
