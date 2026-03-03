# AG-UI 协议集成文档

## 概述

ChatDB 实现了 [AG-UI（Agent User Interaction Protocol）](https://docs.ag-ui.com) 协议，将多 Agent 执行流程通过 **SSE（Server-Sent Events）** 实时推送给前端。前端可据此展示分析步骤进度、中间结果、人类介入交互等。

核心组件：

| 组件 | 文件 | 职责 |
|------|------|------|
| AGUIAdapter | `core/agui_adapter.py` | Orchestrator 内部事件 → AG-UI 标准事件的转换层 |
| AG-UI 路由 | `api/routes/agui.py` | 3 个 SSE 流式端点 |
| EventEncoder | `ag_ui.encoder` | AG-UI 事件序列化为 SSE 帧 |

## 架构

```
前端 (CopilotKit / 自定义)
    │
    │  POST /agui/         (简化版请求)
    │  POST /agui/run      (标准 RunAgentInput)
    │  POST /agui/continue (人类介入恢复)
    │
    ▼
┌─ FastAPI SSE 端点 ─────────────────────────────┐
│  创建 Orchestrator + AGUIAdapter               │
│  返回 StreamingResponse(adapter.stream(...))   │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ AGUIAdapter.stream() ─────────────────────────┐
│  1. yield RUN_STARTED                          │
│  2. 启动 Orchestrator 后台任务                   │
│     Orchestrator 通过 asyncio.Queue 发射事件    │
│  3. _consume() 循环读取 Queue                   │
│     内部事件 → _map_event() → AG-UI 事件        │
│  4. 等待 Orchestrator 完成                      │
│  5. yield TEXT_MESSAGE (summary)               │
│  6. yield STATE_SNAPSHOT (完整结果快照)          │
│  7. yield RUN_FINISHED / RUN_ERROR             │
└────────────────────────────────────────────────┘
    │                          ▲
    │  asyncio.Queue           │ _emit(event_type, data)
    ▼                          │
┌─ AgentOrchestrator ────────────────────────────┐
│  语义解析 → Planner → SQL 执行 → 总结           │
│  每个阶段调用 _emit() 写入事件队列              │
│  完成后 put(None) 作为结束信号                  │
└────────────────────────────────────────────────┘
```

**解耦设计**：Orchestrator 与 Adapter 通过 `asyncio.Queue` 解耦。Orchestrator 只负责 `_emit(event_type, data)` 写入队列，不感知 AG-UI 协议细节；Adapter 消费队列并负责事件格式转换。

## API 端点

### POST `/agui/` — 简化版查询

接收 ChatDB 风格的请求，返回 AG-UI SSE 事件流。

**请求体** (`ChatDBRunRequest`)：

```json
{
  "query": "上个月各渠道的流水对比",
  "db_path": "/path/to/data.duckdb",
  "session_id": "optional-session-id",
  "thread_id": "optional-thread-id",
  "run_id": "optional-run-id"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 自然语言查询 |
| `db_path` | string | ✅ | DuckDB 或 CSV 文件路径 |
| `session_id` | string | - | 会话 ID（多轮对话记忆） |
| `thread_id` | string | - | AG-UI 线程 ID，未传则自动生成 |
| `run_id` | string | - | AG-UI 运行 ID，未传则自动生成 |

### POST `/agui/run` — 标准 AG-UI 端点

兼容 [AG-UI RunAgentInput](https://docs.ag-ui.com/concepts/events) 格式，适配 CopilotKit 等标准客户端。

- 从 `messages` 中提取最后一条 `role=user` 消息作为 query
- 从 `state.db_path` 或 `forwarded_props.db_path` 读取数据库路径

**请求体**：标准 `RunAgentInput` JSON。

### POST `/agui/continue` — 人类介入恢复

当查询返回 `need_clarification` 状态后，用户提交选择，此端点以 SSE 事件流返回恢复执行的全过程。

**请求体** (`AGUIContinueRequest`)：

```json
{
  "db_path": "/path/to/data.duckdb",
  "session_id": "original-session-id",
  "chosen_option": "option_a",
  "extra_input": "我指的是自然月口径",
  "original_query": "为什么上个月流水下降了",
  "thread_id": "optional-thread-id",
  "run_id": "optional-run-id"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `db_path` | string | ✅ | DuckDB 或 CSV 文件路径 |
| `session_id` | string | ✅ | 原始查询的 session_id |
| `chosen_option` | string | ✅ | 用户选择的 option id |
| `extra_input` | string | - | 用户自由输入的补充信息 |
| `original_query` | string | - | 原始用户查询（用于恢复上下文） |
| `thread_id` | string | - | AG-UI 线程 ID |
| `run_id` | string | - | AG-UI 运行 ID |

**恢复机制**：用户的选择被拼接为自然语言上下文注入到查询中（如 `"为什么上个月流水下降了（用户澄清：选择了「自然月口径」，补充说明：我指的是自然月口径）"`），LLM 自主决定如何调整分析策略，而非硬编码选项→动作映射。

## 事件流

所有端点返回的 SSE 流遵循相同的事件生命周期。

### 事件类型映射

Orchestrator 内部事件到 AG-UI 标准事件的完整映射：

| Orchestrator 内部事件 | AG-UI 事件 | 触发时机 |
|----------------------|------------|---------|
| `step_start` | `STEP_STARTED` | 进入新的处理阶段 |
| `step_end` | `STEP_FINISHED` | 阶段处理完成 |
| `tool_start` | `TOOL_CALL_START` | Agent/Tool 开始执行 |
| `tool_args` | `TOOL_CALL_ARGS` | Tool 参数传入 |
| `tool_end` | `TOOL_CALL_END` | Tool 执行完成 |
| `tool_result` | `TOOL_CALL_RESULT` | Tool 返回结果 |
| `text_chunk` | `TEXT_MESSAGE_*` | 流式文本片段 |
| `custom` | `CUSTOM` | 自定义事件（如 Planner 决策、人类介入） |
| *(适配层生成)* | `RUN_STARTED` | 流开始 |
| *(适配层生成)* | `TEXT_MESSAGE_*` | 最终 summary |
| *(适配层生成)* | `STATE_SNAPSHOT` | 完整结果快照 |
| *(适配层生成)* | `RUN_FINISHED` | 正常结束 |
| *(适配层生成)* | `RUN_ERROR` | 异常/超时 |

### 典型事件序列

**正常查询**：
```
RUN_STARTED
├── STEP_STARTED (semantic_parse)
│   ├── TOOL_CALL_START (semantic_parse)
│   ├── TOOL_CALL_ARGS
│   ├── TOOL_CALL_RESULT
│   └── TOOL_CALL_END
├── STEP_FINISHED (semantic_parse)
├── STEP_STARTED (planner)
│   ├── TOOL_CALL_START (planner)
│   ├── TOOL_CALL_ARGS
│   ├── TOOL_CALL_RESULT
│   └── TOOL_CALL_END
├── STEP_FINISHED (planner)
├── STEP_STARTED (sql_execute)
│   ├── TOOL_CALL_START (sql_task_1)
│   ├── TOOL_CALL_ARGS
│   ├── TOOL_CALL_RESULT
│   ├── TOOL_CALL_END
│   ├── CUSTOM (planner_decision: continue)
│   ├── TOOL_CALL_START (sql_task_2)
│   ├── ...
│   └── CUSTOM (planner_decision: done)
├── STEP_FINISHED (sql_execute)
├── STEP_STARTED (summarize)
├── STEP_FINISHED (summarize)
├── TEXT_MESSAGE_START
├── TEXT_MESSAGE_CONTENT (summary 文本)
├── TEXT_MESSAGE_END
├── STATE_SNAPSHOT
RUN_FINISHED
```

**人类介入场景**（Planner 判断需要用户澄清）：
```
RUN_STARTED
├── STEP_STARTED (semantic_parse) ... STEP_FINISHED
├── STEP_STARTED (planner) ... STEP_FINISHED
├── STEP_STARTED (sql_execute)
│   ├── TOOL_CALL_START (sql_task_1) ... TOOL_CALL_END
│   ├── CUSTOM (planner_decision: intervention)    ← Planner 请求人类介入
│   └── CUSTOM (human_intervention)                ← 介入详情推送给前端
├── STEP_FINISHED (sql_execute)
├── STATE_SNAPSHOT (status: need_clarification)
RUN_FINISHED
```

**闲聊快速响应**（`_quick_classify` 判断为非数据分析）：
```
RUN_STARTED
├── STEP_STARTED (semantic_parse) ... STEP_FINISHED
├── TEXT_MESSAGE_START
├── TEXT_MESSAGE_CONTENT (LLM 直接回答)
├── TEXT_MESSAGE_END
├── STATE_SNAPSHOT (status: completed)
RUN_FINISHED
```

### 自定义事件详情

#### `planner_decision`

每批任务执行完成后，Planner 的决策结果：

```json
{
  "name": "planner_decision",
  "value": {
    "action": "continue | done | adjust | retry | intervention",
    "reason": "决策原因描述"
  }
}
```

| action | 含义 |
|--------|------|
| `continue` | 继续执行下一批任务 |
| `done` | 分析完成，进入总结 |
| `adjust` | 调整当前分析方向（修改后续任务） |
| `retry` | 重试当前任务 |
| `intervention` | 请求人类介入 |

#### `human_intervention`

Planner 判断当前分析遇到无法自主解决的问题，请求人类介入：

```json
{
  "name": "human_intervention",
  "value": {
    "question": "您所说的\"月活\"是指以下哪种统计口径？",
    "options": [
      {"id": "a", "label": "月登录用户数"},
      {"id": "b", "label": "月付费用户数"},
      {"id": "c", "label": "月活跃天数>=3 的用户数"}
    ],
    "free_input_allowed": true
  }
}
```

## STATE_SNAPSHOT 结构

每次查询结束后通过 `STATE_SNAPSHOT` 事件推送完整结果快照，前端据此渲染最终 UI。

### 基础字段

```json
{
  "success": true,
  "status": "completed",
  "query": "各渠道收入对比",
  "rewritten_query": null,
  "sql": "SELECT channel, SUM(revenue) ...",
  "row_count": 5,
  "table_name": "sales",
  "result": [{"channel": "直销", "revenue": 1000000}, ...],
  "intent": { "mode": "basic", "metrics": [...], ... }
}
```

### 人类介入相关字段

当 `status` 为 `need_clarification` 时，额外包含：

```json
{
  "status": "need_clarification",
  "success": false,
  "clarification_request": {
    "reason": "...",
    "question": "...",
    "options": [...],
    "free_input_allowed": true
  },
  "run_context": {
    "step_id": 2,
    "plan_state": "paused"
  }
}
```

### research_mode 结构化结论

当 `research_mode=true` 且分析完成时，额外包含：

```json
{
  "confidence": 0.85,
  "key_findings": [
    "渠道A流水下降主要受新用户减少驱动 [来源: task_1, task_3]",
    "渠道B逆势增长与促销活动相关 [来源: task_2]"
  ],
  "limitations": [
    "2024年1月数据有30%空值，可能影响趋势判断"
  ],
  "suggested_follow_ups": [
    "进一步分析渠道A新用户获取成本变化",
    "对比促销前后渠道B的留存率"
  ]
}
```

### status 值说明

| status | 含义 | 前端行为 |
|--------|------|---------|
| `completed` | 分析正常完成 | 渲染结果表格 + summary |
| `need_clarification` | 需要用户澄清 | 弹出选择/输入交互，提交到 `/agui/continue` |

## 前端集成指南

### SSE 连接

```javascript
const response = await fetch('/agui/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
  },
  body: JSON.stringify({
    query: '上个月各渠道的流水对比',
    db_path: '/path/to/data.duckdb',
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  // 按 SSE 规范解析 "data: {...}\n\n" 帧
  for (const line of text.split('\n')) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      handleEvent(event);
    }
  }
}
```

### 事件处理

```javascript
function handleEvent(event) {
  switch (event.type) {
    case 'RUN_STARTED':
      showLoadingState();
      break;

    case 'STEP_STARTED':
      updateProgress(event.step_name);  // "semantic_parse" | "planner" | "sql_execute" | "summarize"
      break;

    case 'TOOL_CALL_START':
      showToolExecution(event.tool_call_name);  // "semantic_parse" | "planner" | "sql_task_1" | ...
      break;

    case 'TOOL_CALL_RESULT':
      showIntermediateResult(JSON.parse(event.content));
      break;

    case 'CUSTOM':
      if (event.name === 'planner_decision') {
        showPlannerDecision(event.value);
      }
      if (event.name === 'human_intervention') {
        showInterventionDialog(event.value);  // 弹出交互 UI
      }
      break;

    case 'TEXT_MESSAGE_CONTENT':
      appendSummaryText(event.delta);
      break;

    case 'STATE_SNAPSHOT':
      renderFinalResult(event.snapshot);
      break;

    case 'RUN_FINISHED':
      hideLoadingState();
      break;

    case 'RUN_ERROR':
      showError(event.message);
      break;
  }
}
```

### 人类介入交互流程

```javascript
function showInterventionDialog(intervention) {
  // 渲染选项列表 + 可选自由输入框
  const dialog = renderClarificationDialog({
    question: intervention.question,
    options: intervention.options,
    allowFreeInput: intervention.free_input_allowed,
  });

  dialog.onSubmit(async ({ chosenOption, extraInput }) => {
    // 提交到 /agui/continue，获取新的 SSE 事件流
    const response = await fetch('/agui/continue', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        db_path: currentDbPath,
        session_id: currentSessionId,
        chosen_option: chosenOption,
        extra_input: extraInput,
        original_query: originalQuery,
      }),
    });
    // 以同样方式消费 SSE 事件流...
  });
}
```

### CopilotKit 集成

使用标准 `/agui/run` 端点，兼容 CopilotKit 的 `RunAgentInput` 格式：

```typescript
import { CopilotKit } from "@copilotkit/react-core";

<CopilotKit
  runtimeUrl="/agui/run"
  // 通过 state 传递 db_path
  initialState={{ db_path: "/path/to/data.duckdb" }}
/>
```

`db_path` 可通过 `state.db_path` 或 `forwarded_props.db_path` 传入。

## Step 名称参考

事件流中 `step_name` 的可能值及含义：

| step_name | 阶段 | 说明 |
|-----------|------|------|
| `semantic_parse` | 语义解析 | 意图提取、指代消解、问题改写、research_mode 判断 |
| `planner` | 计划生成 | 根据意图生成多步分析计划 |
| `sql_execute` | SQL 执行 | 包含多个 `sql_task_*` 的 tool_call |
| `summarize` | 结果总结 | LLM 生成自然语言回答 |

## 超时与错误处理

- **队列消费超时**：单次 `queue.get()` 等待上限 **600 秒**，超时则发送 `RUN_ERROR` 并终止流
- **Orchestrator 异常**：捕获后发送 `RUN_ERROR(message=str(exc))`，取消后台任务
- **队列结束信号**：Orchestrator 执行完毕（成功或失败）均 `put(None)` 通知 Adapter
