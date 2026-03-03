# ChatDB Deep Research 改造方案

## 一、设计哲学

**核心原则：让 LLM 做决策，而不是写 if-else。**

ChatDB 现有架构的精髓是"把上下文喂给 LLM，让 LLM 做判断"：
- Planner `decide_next_action` → LLM 看数据摘要，决定 A/B/C/D
- `apply_adjustment` → LLM 说要 insert/retry/skip 哪个任务
- `transition_context` → LLM 总结承上启下分析

改造同样遵循这个哲学：**不新建 Agent 类，不写硬编码检查逻辑，只扩展 Prompt 让 LLM 看得更多、想得更深、做得更多。** 同时在 LLM 判断"我搞不定"时，优雅地将控制权交给人类。

---

## 二、整体架构改动概览

```
改动范围：
├── planner.py          # Planner decide prompt 扩展（深度分析 + 人类介入判断）
├── summarize.py        # SummarizeAnswer prompt 扩展（证据链 + 结构化输出）
├── orchestrator.py     # process_query 增加 research_mode 透传 + 人类介入响应处理
├── react_state.py      # ReActState 增加 research_mode / intervention 相关字段
├── api/schemas.py      # 增加 ClarificationResponse 模型
├── api/routes/query.py # /query 增加 research_mode 参数 + /continue 端点
└── agui_adapter.py     # AG-UI 增加 HUMAN_CLARIFICATION 事件类型

不新建任何业务逻辑类，不新建 Agent。
总改动量：~200 行（Prompt 工程 + 参数透传 + 响应结构）
```

---

## 三、改造 1：research_mode 由 LLM 自动判断

**不让用户手动指定 `research_mode`**，而是在语义解析阶段让 LLM 自己判断这个问题是否需要深度分析。

### 实现方式

在 `SemanticParseTool` 的 prompt 输出结构中增加一个字段 `research_mode`：

```
在 SemanticParseTool 的 prompt 中增加：

## research_mode 判断
判断此问题是否需要深度研究模式（research_mode）。以下情况应设为 true：
- 问"为什么"类的归因分析（如"为什么流水下降"）
- 需要多维度交叉验证的复杂问题
- 涉及趋势变化原因的探索性分析
- 需要假设-验证循环的问题
以下情况设为 false：
- 简单的数据查询（如"上个月总流水多少"）
- 明确的排名/占比/趋势展示
- 单维度单指标的直接查询

输出字段：
"research_mode": true/false
```

解析后写入 `state.intent.research_mode`，后续 Planner 和 Summarize 据此切换 prompt 档位。

### 为什么这样做

- 用户不需要知道什么是 research_mode，LLM 根据问题语义自动判断
- 随着 LLM 能力提升，判断准确度自动提升
- 如果 LLM 判断错误（本该深度但没触发），Planner 在执行过程中发现数据矛盾时仍可通过 B(insert_task) 动态追加验证任务

---

## 四、改造 2：Planner decide prompt 增加"深度分析 + 人类介入"能力

### 现状

Planner 决策选项是 A(继续)/B(插入·重试)/C(跳过)/D(结束)。B 选项已支持 `insert_task`，LLM 已经可以动态追加任务。但 prompt 没有引导 LLM 做深度分析和人类介入判断。

### 改造

在 `_llm_decide` 中，当 `research_mode=True` 时替换 system_prompt：

```python
# planner.py _llm_decide 中
if research_mode:
    system_prompt = """你是数据分析决策专家。你不仅要判断任务是否成功，更要像分析师一样思考数据背后的故事。

关键原则：
1. 看到数据后先判断：这个结果是否回答了用户的问题？是否有异常模式？
2. 如果发现值得深挖的模式（某个产品突然下滑、某个区域异常增长），考虑追加验证任务
3. 如果多个任务的结果之间存在矛盾（如总量增长但所有分项都在下降），必须追加验证
4. 结论的每一个关键陈述都应有对应的数据支撑

人类介入判断（重要）：
5. 如果用户问题语义不清、存在多种合理解释，选 E 请求用户澄清
6. 如果数据质量严重影响结论可靠性（关键列大面积缺失、结果明显不合理），选 E 告知用户
7. 如果当前数据/表结构根本无法支撑用户要求的分析粒度，选 E 说明限制

8. decision 字段只能是单个字母：A、B、C、D、E
9. 输出 JSON"""
```

### 决策选项扩展

在决策选项中增加 B 的深度分析引导和 E 选项：

```
**B. 插入/重试任务**：
  - SQL 错误 → 重试并给出 fix_hint
  - 数据异常（全0/全NULL/明显矛盾）→ 插入 validation 任务诊断
  - ★ 深度分析：发现值得深挖的模式 → 插入新的分析任务
    例：source 结果显示"手游"大类下滑最多 → 插入 drilldown 任务细查手游子品类
    例：两个任务的合计不一致 → 插入 validation 任务交叉验证

**E. 请求人类协助**（仅 research_mode）：需要用户澄清或确认才能继续
  - 输出 intervention_type: "need_clarification" | "blocked"
  - 输出结构化的 question + options 供用户选择
```

### E 选项的输出格式

```
选 E（澄清）: {
  "decision": "E",
  "reason": "用户问题存在多种可能解释",
  "intervention": {
    "type": "need_clarification",
    "question": "请确认您更想要哪种分析？",
    "options": [
      {"id": "A", "label": "2024国内流水同比2023变化原因", "hint": "归因分析"},
      {"id": "B", "label": "2024国内各产品流水分布", "hint": "结构分析"}
    ],
    "free_input_allowed": true
  }
}

选 E（数据问题）: {
  "decision": "E",
  "reason": "数据质量严重影响结论可靠性",
  "intervention": {
    "type": "need_clarification",
    "question": "当前数据质量会影响结论可信度，您希望如何处理？",
    "details": ["'国内/海外'字段约82%缺失", "汇总结果与大盘总表差异35%"],
    "options": [
      {"id": "continue_with_warning", "label": "继续分析，但标注'数据质量较差，仅供参考'"},
      {"id": "change_scope", "label": "调整分析范围", "need_extra_input": true, "input_hint": "如：改为分析2023年"},
      {"id": "stop", "label": "终止本次分析"}
    ]
  }
}

选 E（能力边界）: {
  "decision": "E",
  "reason": "当前数据无法支撑要求的分析粒度",
  "intervention": {
    "type": "blocked",
    "message": "现有数据表只到'产品-月份-部门'粒度，无法按用户路径拆解流水",
    "details": ["表《脚本测试数据》不包含用户级明细"],
    "suggestions": ["改为按产品维度分析流水变化", "接入用户级流水明细表后重试"]
  }
}
```

### 关键：复用已有机制

- **不需要新的决策处理分支**：E 选项由 `_handle_planner_decision` 新增一个 `elif action == "intervention"` 分支处理
- **LLM 的 insert_task（B 选项）已足够做假设生成+验证**，prompt 只是引导它这么做
- **现有的 validation 上限、死循环检测完全复用**

---

## 五、改造 3：Orchestrator 处理人类介入

### _handle_planner_decision 增加 intervention 处理

```python
# orchestrator.py _handle_planner_decision 中增加
if action == "intervention":
    intervention = decision.get("intervention", {})
    intervention_type = intervention.get("type", "need_clarification")
    orch_log.info(f"Planner 请求人类介入: {intervention_type}")

    # 将介入信息存入 state，中断执行循环
    state.intervention = intervention
    state.intervention_step_id = state.plan_step

    # AG-UI: 推送人类介入事件
    await self._emit("custom", {
        "name": "human_intervention",
        "value": intervention,
    })
    return True  # 中断 _execute_plan
```

### process_query 响应结构统一

```python
# _build_result 中增加
if state.intervention:
    intervention = state.intervention
    itype = intervention.get("type", "need_clarification")

    if itype == "blocked":
        result["status"] = "blocked"
        result["block_reason"] = {
            "message": intervention.get("message", ""),
            "details": intervention.get("details", []),
        }
        result["suggestions"] = intervention.get("suggestions", [])
    else:
        result["status"] = "need_clarification"
        result["clarification_request"] = {
            "reason": intervention.get("reason", ""),
            "question": intervention.get("question", ""),
            "options": intervention.get("options", []),
            "free_input_allowed": intervention.get("free_input_allowed", True),
        }
    result["success"] = False  # 未完成
    result["run_context"] = {
        "session_id": scratch_session_id,
        "step_id": state.intervention_step_id,
        "plan_state": "paused",
    }
else:
    result["status"] = "completed"
```

### 统一响应结构

所有 `/query` 返回统一格式：

```json
{
  "status": "completed | need_clarification | blocked",
  "success": true/false,
  "query": "...",
  "summary": "...",

  // status=completed 时
  "sql": "...",
  "result": [...],
  "row_count": 0,
  "temp_results": {...},

  // status=need_clarification 时
  "clarification_request": {
    "reason": "...",
    "question": "...",
    "options": [...],
    "free_input_allowed": true
  },
  "run_context": {"session_id": "...", "step_id": 0, "plan_state": "paused"},

  // status=blocked 时
  "block_reason": {"message": "...", "details": [...]},
  "suggestions": [...]
}
```

---

## 六、改造 4：/continue 端点（恢复暂停的分析）

### API 层

```python
# api/schemas.py
class ClarificationResponse(BaseModel):
    session_id: str           # run_context.session_id
    step_id: int              # run_context.step_id
    chosen_option: str        # 用户选择的 option id
    extra_input: str | None = None  # 自由输入

# api/routes/query.py
@router.post("/query/continue")
async def continue_query(req: ClarificationResponse):
    orchestrator = get_orchestrator()
    result = await orchestrator.resume_query(
        session_id=req.session_id,
        step_id=req.step_id,
        user_decision={
            "chosen_option": req.chosen_option,
            "extra_input": req.extra_input,
        }
    )
    return result
```

### Orchestrator 层

`resume_query` 利用现有的 **Plan Persistence** 机制恢复执行：

```python
async def resume_query(self, session_id, step_id, user_decision):
    # 1. 从持久化计划恢复 state（已有 plan_resumed 机制）
    state = await self._restore_state(session_id)

    # 2. 将用户决定注入 state，作为 Planner 下一轮决策的上下文
    state.extra_context = f"用户选择: {user_decision['chosen_option']}"
    if user_decision.get("extra_input"):
        state.extra_context += f"\n用户补充: {user_decision['extra_input']}"
    state.intervention = None  # 清除介入状态

    # 3. 根据用户选择调整计划
    #    - 不写硬编码逻辑，而是让 Planner 根据用户选择重新规划
    #    - 用户选择作为 context 传给 Planner decide，由 LLM 决定如何调整
    await self._execute_plan(state, context, orch_log, session_id)
    state = await self._generate_summary(state, state.user_query)
    return self._build_result(state, state.user_query, time.time())
```

关键点：**用户的选择不是硬编码映射到某个操作，而是作为上下文喂给 LLM，由 LLM 决定如何调整后续计划。**

---

## 七、改造 5：SummarizeAnswer prompt 增加证据链思维

### 实现方式

在 `summarize.py` 的 `summarize` 方法中，当 `research_mode=True` 时追加 prompt：

```python
if research_mode:
    prompt += """

## 深度分析报告要求（research_mode）

### 结论层
- 给出明确的核心结论（一句话回答用户问题）
- 标注置信度：高（多个数据源交叉验证）/ 中（数据支撑但未验证）/ 低（样本不足或有矛盾）

### 证据层
- 每个关键陈述后标注 [来源: task_id]，说明数据来自哪个分析步骤
- 如果存在数据矛盾，明确说明矛盾点和可能原因

### 不确定性
- 列出分析过程中的限制（数据时间范围、缺失维度等）
- 如果某些结论需要更多数据验证，指出具体需要什么

### 额外输出（在自然语言回答之后）
在回答末尾追加一个 JSON 块：
```json
{
  "confidence": "high/medium/low",
  "key_findings": [
    {"finding": "...", "evidence_task": "task_id", "data_point": "..."}
  ],
  "limitations": ["..."],
  "suggested_follow_ups": ["..."]
}
```"""
```

### _build_result 中解析

```python
if research_mode and state.summary:
    parsed = self._try_parse_research_json(state.summary)
    if parsed:
        result["confidence"] = parsed.get("confidence")
        result["key_findings"] = parsed.get("key_findings", [])
        result["limitations"] = parsed.get("limitations", [])
        result["suggested_follow_ups"] = parsed.get("suggested_follow_ups", [])
```

---

## 八、改造 6：ReActState 增加字段

```python
# react_state.py ReActState 中增加
research_mode: bool = False              # 是否深度分析模式（由 LLM 自动判断）
intervention: dict | None = None         # 人类介入信息（Planner E 决策产生）
intervention_step_id: int = 0            # 介入时的计划步骤号
```

字段通过 Plan Persistence 自动持久化/恢复，支持 `/continue` 恢复执行。

---

## 九、改造 7：AG-UI 事件扩展

在 `agui_adapter.py` 中增加人类介入事件映射：

```python
# 当 Orchestrator emit("custom", {"name": "human_intervention", ...}) 时
# AGUIAdapter 转换为：
{
    "type": "custom",
    "name": "human_intervention_request",
    "value": {
        "intervention_type": "need_clarification | blocked",
        "question": "...",
        "options": [...],
        ...
    }
}
```

前端收到后弹出澄清对话框，用户提交后调用 `/query/continue`。

---

## 十、方案对比

| 维度 | 硬编码方案 | LLM-Native 方案（本方案） |
|---|---|---|
| 数据质量检查 | if NULL > 70% then... | LLM 看数据摘要自己判断"数据质量是否影响结论" |
| 语义澄清 | 规则打分 confidence < 0.6 | LLM 在 Planner decide 时判断"用户意图是否有多种解释" |
| 假设生成 | 新建 HypothesisGeneratorAgent | Planner B(insert_task) + prompt 引导 |
| 假设验证 | 新建 HypothesisTesterAgent | Planner 插入 validation 任务，SQLAgent 照常执行 |
| 证据链 | 硬编码遍历结果提取 | LLM 在 summary 时自己标注 [来源: task_id] |
| 人类介入触发 | HumanInterventionGate 硬编码规则 | **LLM 在 Planner decide 时自主判断选 E** |
| 介入后恢复 | 硬编码映射 option → action | **用户选择作为 context 喂给 LLM，LLM 决定如何调整** |
| 新增 Agent | 3-4 个 | **0 个** |
| 新增类 | HumanInterventionGate 等 | **0 个业务类**（只加数据模型） |
| 新增代码量 | ~400 行 + 多个新文件 | **~200 行改 prompt + 参数透传** |

---

## 十一、实施步骤

| 步骤 | 内容 | 改动文件 | 估计行数 |
|---|---|---|---|
| 1 | SemanticParseTool prompt 增加 `research_mode` 判断输出 | `semantic_parser.py` | ~15 行 |
| 2 | ReActState 增加 `research_mode` / `intervention` 字段 | `react_state.py` | ~5 行 |
| 3 | `process_query` 读取 `intent.research_mode` 写入 state，透传到 `_execute_plan` 和 `_generate_summary` | `orchestrator.py` | ~20 行 |
| 4 | Planner `_llm_decide` 根据 `research_mode` 切换 system_prompt，增加 E 选项 | `planner.py` | ~50 行 |
| 5 | `_handle_planner_decision` 增加 `intervention` 分支 | `orchestrator.py` | ~15 行 |
| 6 | `_build_result` 增加 status/clarification_request/block_reason 输出 | `orchestrator.py` | ~25 行 |
| 7 | SummarizeAnswer 根据 `research_mode` 追加证据链 prompt | `summarize.py` | ~30 行 |
| 8 | API 增加 `/query/continue` 端点 + ClarificationResponse 模型 | `schemas.py` + `routes/query.py` | ~25 行 |
| 9 | `resume_query` 方法：恢复 state + 注入用户决定 + 继续执行 | `orchestrator.py` | ~20 行 |
| 10 | AG-UI 增加 human_intervention 事件类型 | `agui_adapter.py` | ~10 行 |

**总计：~215 行，0 个新业务类，0 个新 Agent。**

---

## 十二、用户体验示例

### 场景 1：正常深度分析（LLM 自动进入 research_mode）

```
用户: "今年国内流水为什么下降？"

→ SemanticParseTool: research_mode=true（归因分析）
→ Planner: 生成 [source → drilldown → comparison] 计划
→ 执行 source 任务后，Planner decide:
   "发现手游大类下滑15%，插入 drilldown 任务细查手游子品类"
   → decision=B, insert_task={type: "drilldown", description: "手游子品类流水明细"}
→ 执行完所有任务后，SummarizeAnswer 输出带证据链的结论
→ status=completed, confidence=high
```

### 场景 2：语义不清，LLM 请求澄清

```
用户: "流水情况怎么样？"

→ SemanticParseTool: research_mode=true
→ Planner: 生成初步计划
→ Planner decide 时发现问题可以多种理解:
   → decision=E, intervention={
       type: "need_clarification",
       question: "请确认您想了解的流水分析维度",
       options: [
         {id: "A", label: "按产品维度看流水分布"},
         {id: "B", label: "按时间维度看流水趋势"},
         {id: "C", label: "国内vs海外流水对比"}
       ]
     }
→ 返回 status=need_clarification

用户选择 B → POST /query/continue
→ 用户选择作为 context 注入 → Planner 调整后续计划 → 继续执行
→ 最终返回 status=completed
```

### 场景 3：数据质量问题

```
用户: "2024年国内流水按产品拆解"

→ 执行第一个 source 任务
→ Planner decide 看到结果: "国内/海外"字段 82% 为 NULL
→ decision=E, intervention={
     type: "need_clarification",
     question: "数据质量会影响结论可信度",
     details: ["'国内/海外'字段约82%缺失"],
     options: [
       {id: "continue_with_warning", label: "继续分析，标注数据质量警告"},
       {id: "change_scope", label: "调整分析范围", need_extra_input: true},
       {id: "stop", label: "终止"}
     ]
   }
→ 返回 status=need_clarification
```

### 场景 4：能力边界

```
用户: "按用户留存路径拆解流水"

→ Planner decide 审视 schema 后判断:
   → decision=E, intervention={
       type: "blocked",
       message: "现有数据表只到产品-月份粒度，无法按用户路径分析",
       suggestions: ["改为按产品维度分析流水变化", "接入用户级明细表后重试"]
     }
→ 返回 status=blocked
```

---

## 十三、风险与缓解

| 风险 | 缓解措施 |
|---|---|
| LLM 滥用 E 选项，频繁打断用户 | Prompt 中明确"只在确实无法自主判断时才选 E"；增加 E 选项使用次数上限（类似 validation_count） |
| LLM 输出的 intervention JSON 格式不对 | 复用现有的 JSON 鲁棒解析逻辑（extract_json + 尾部修复）；解析失败降级为 A(继续) |
| research_mode 误判 | 影响有限：false→true 只是多一些 prompt 引导，性能略降；true→false 仍有 B(insert_task) 兜底 |
| /continue 恢复失败 | 利用现有 Plan Persistence 机制，state 已持久化到 plan.json；恢复失败时返回明确错误 |

---

## 十四、设计决策说明

### 为什么不用 HumanInterventionGate（硬编码检查类）

HumanInterventionGate 用 `if null_ratio > 0.7` 等硬编码规则判断是否需要人类介入。问题：
1. NULL 70% 不一定是问题（可能就是真实数据分布）
2. "所有子项增长但总量下降"这种逻辑矛盾，硬编码检测不了
3. 语义歧义判断更不可能用规则做

**让 LLM 在 Planner decide 时统一判断**，它能看到完整上下文（用户问题 + schema + 数据结果 + 计划状态），判断质量远超规则。

### 为什么用 Planner E 选项而不是独立的检查点

在 Orchestrator 的固定位置插入检查点（如"语义解析后检查"、"SQL 执行后检查"）是静态的、不灵活的。
Planner 在每批任务后都会 decide，**这本身就是一个动态检查点**。LLM 在任何时刻发现问题都可以选 E，不限于预设位置。

### 为什么 /continue 不做硬编码映射

用户选了 `change_scope`，我们不需要写 `if option == "change_scope": modify_where_clause(...)`。
把用户选择作为自然语言 context 喂给 Planner，Planner 自己决定怎么调整计划。这样任何新的选项类型都不需要改代码。

---

## 十五、改造 8：非数据分析问题的轻量快速响应

### 现状与问题

当前已有 `mode=other` 机制：SemanticParseTool 判断 `mode=other` → `_handle_other_query` 直接用 LLM 回复。

但存在两个问题：

1. **仍经过完整的语义解析流程** — 对"广东的省会城市是谁"这种问题，先做虚拟字段检索、context retrieval、再调用 SemanticParseTool 的大 prompt 解析 intent，浪费 token 和时间
2. **判断时机太晚** — 在语义解析完成后才判断 `is_other_query()`，此时已经付出了检索和解析的开销

### 设计原则

和整个方案一致：**让 LLM 做决策，但要在最早的时机、用最轻的方式做。**

### 实现方式：前置轻量分类器

在 `process_query` 最前面（缓存检查之后、虚拟字段检索之前），增加一次**极轻量的 LLM 调用**，只做一件事：判断这个问题是否和数据分析有关。

```python
# orchestrator.py process_query 中，在虚拟字段检索之前
async def _quick_classify(self, query: str, chat_history: list | None) -> str:
    """轻量分类：判断问题类型，一次极短的 LLM 调用"""
    history_hint = ""
    if chat_history:
        # 只取最近 1 轮，给 LLM 判断是否是追问
        last = chat_history[-1] if chat_history else None
        if last:
            history_hint = f"\n上一轮对话: Q={last.get('query','')[:50]} A={last.get('answer','')[:50]}"

    prompt = f"""判断以下用户问题的类型，只输出一个单词：
- analysis: 需要查询数据库、做数据分析的问题（如流水、收入、趋势、排名、对比等）
- chat: 闲聊、常识问答、与数据无关的问题（如天气、地理、历史等）
- ambiguous: 不确定是否需要数据分析

用户问题: {query}{history_hint}

类型:"""

    response = await self.llm.chat(
        prompt=prompt,
        system_prompt="你是问题分类器。只输出 analysis/chat/ambiguous 其中一个单词。",
        caller_name="quick_classify",
    )
    return response.strip().lower().split()[0] if response else "analysis"
```

### 在 process_query 中的位置

```python
async def process_query(self, query, session_id=None, ...):
    # ... 缓存检查、历史加载 ...

    # ★ 前置轻量分类（在所有重操作之前）
    query_type = await self._quick_classify(query, chat_history)

    if query_type == "chat":
        # 直接用 LLM 回复，跳过整个分析流程
        state = await self._handle_chat_query(state, context, orch_log)
        result = self._build_result(state, query, start_time)
        # 仍写入历史（支持追问场景判断上下文）
        self._history.save_to_history(session_id, query, state.summary or "", state)
        return result

    # query_type == "analysis" 或 "ambiguous" → 进入正常流程
    # ambiguous 的好处：不确定时走分析流程，SemanticParseTool 会二次判断
    # （现有 mode=other 机制作为兜底）

    # 0.5 虚拟字段检索...
    # 1. 语义解析...
    # 后续正常流程不变
```

### _handle_chat_query 实现

```python
async def _handle_chat_query(self, state, context, orch_log):
    """处理纯闲聊/常识问题，极简路径"""
    orch_log.info(f"闲聊问题，直接回复: {state.user_query[:50]}...")

    # 提供最少的上下文：只告诉 LLM 自己是谁、能做什么
    prompt = f"用户问题: {state.user_query}"
    response = await self.llm.chat(
        prompt=prompt,
        system_prompt="你是 ChatDB 数据分析助手。友好地回答用户问题。"
                      "如果问题和数据分析无关，正常回答即可，"
                      "可以自然地提及你的数据分析能力，但不要强行引导。",
        caller_name="chat_response",
    )
    state.summary = response
    state.phase = ReActPhase.DONE
    return state
```

### 和现有 mode=other 的关系

```
用户问题
  │
  ├─ _quick_classify → "chat"
  │   └─ _handle_chat_query（极简路径，~1次LLM调用，无检索无解析）
  │
  ├─ _quick_classify → "analysis"
  │   └─ 正常分析流程
  │
  └─ _quick_classify → "ambiguous"
      └─ 正常分析流程
          └─ SemanticParseTool → mode=other?
              ├─ yes → _handle_other_query（现有逻辑，含数据上下文引导）
              └─ no → 继续分析
```

**两层过滤，各有分工：**

| 层 | 角色 | 上下文开销 | 适用场景 |
|---|---|---|---|
| `_quick_classify` | 门卫 | 极低（~50 token prompt） | "广东省会"、"你好"、"今天天气" |
| `SemanticParseTool mode=other` | 兜底 | 中等（完整语义解析） | 看起来像分析但其实不是的问题（如"数据库有哪些表"） |

### 为什么不直接复用 SemanticParseTool 的 mode=other

1. **开销差距巨大** — SemanticParseTool 的 prompt 包含完整的 schema、虚拟字段定义、few-shot examples，动辄数千 token。前置分类器只需 ~50 token
2. **前置分类器在检索之前** — 省掉了虚拟字段检索、context retrieval 的 I/O 开销
3. **SemanticParseTool 保留作为兜底** — ambiguous 类型仍走完整流程，mode=other 机制不删除，确保不漏判

### 追问场景的处理

关键边界情况：用户先问了数据问题，然后追问一个和数据无关的问题。

```
用户: "上个月总流水多少？"  → analysis → 正常分析
用户: "这个数据准确吗？"    → ambiguous（涉及"数据"关键词）→ 进分析流程 → mode=other → 带上下文回复
用户: "谢谢"              → chat → 直接回复
```

`_quick_classify` 传入了最近一轮对话历史，LLM 能判断"这个数据准确吗"是对上一轮结果的追问（ambiguous），不会错误地走 chat 路径。

### 改动量

| 文件 | 内容 | 行数 |
|---|---|---|
| `orchestrator.py` | `_quick_classify` 方法 | ~20 行 |
| `orchestrator.py` | `_handle_chat_query` 方法 | ~15 行 |
| `orchestrator.py` | `process_query` 中插入分类判断 | ~10 行 |

**总计 ~45 行，不改动任何现有逻辑，纯新增前置路径。**
