# ChatReport — 独立报告引擎方案与实施计划

## 一、核心理念

> **Report = Plan → Ask → Explore → Write**
>
> 借鉴 deep research 领域的共性范式（GPT Researcher / open_deep_research / Egnyte），
> 将报告生成抽象为「规划 → 提问 → 检索 → 写作」四阶段流水线。
> ChatReport 是独立的报告引擎，拥有自己的 API 层、路由分类和 AG-UI 适配，ChatDB 仅作为数据查询工具被调用。

### 设计原则

1. **完全独立**：ChatReport 有独立的 API 入口、`quick_classify`、AG-UI 适配器，不经过 ChatDB 的 `process_query`
2. **结构对齐**：chatreport 的子包（agents / api / config / core / tools）与 chatdb 一一对应
3. **两级 DAG**：章节级 DAG（SectionDAG）管并行/串行，章节内子问题按序查数
4. **证据链**：每个结论可追溯到具体 SQL、数据、虚拟字段
5. **承上启下**：章节间通过结构化 transition 保持逻辑连贯
6. **可审计**：章节数据和内容独立存储到 scratch，支持回溯和验证

---

## 二、整体架构：统一网关 + 双引擎

### 请求路由架构

```
前端请求
  │
  ▼
统一 API 网关（FastAPI app — src/chatdb/api/app.py 升级为顶层网关）
  │
  ├─ /chat     → ChatDB 路由（数据查询）
  ├─ /query    → ChatDB 路由（数据查询）
  ├─ /agui     → ChatDB AG-UI SSE（数据查询流式）
  │
  ├─ /report          → ChatReport 路由（报告生成）
  ├─ /report/agui     → ChatReport AG-UI SSE（报告流式）
  │
  ├─ /gateway         → 统一入口（自动分类到 chatdb 或 chatreport）
  ├─ /gateway/agui    → 统一 AG-UI 入口（自动分类 + 流式）
  │
  └─ /health   → 健康检查
      /database → 数据库管理
```

### 双引擎分流设计

```
/gateway 统一入口:
  │
  ├─ quick_classify(query)
  │     │
  │     ├─ "query" / "basic" / "comparison" / ...
  │     │     → ChatDB.process_query()               # 数据查询
  │     │
  │     └─ "report" / "analysis_report" / "deep_research"
  │           → ChatReport.generate_report()          # 报告生成
  │
  └─ 返回统一响应格式

/gateway/agui 统一 AG-UI 入口:
  │
  ├─ quick_classify(query)
  │     │
  │     ├─ "query" 类
  │     │     → ChatDB AGUIAdapter.stream()           # 数据查询 SSE
  │     │
  │     └─ "report" 类
  │           → ChatReport ReportAGUIAdapter.stream()  # 报告 SSE
  │
  └─ StreamingResponse
```

---

## 三、目录结构

```
src/
├── lib/                            # 公共基础设施（从 chatdb 逐步抽离）
│   ├── __init__.py
│   ├── llm/                        # LLM 调用框架
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseLLM, Message, LLMResponse
│   │   └── factory.py              # LLMFactory + 各实现注册
│   ├── storage/                    # 通用存储
│   │   ├── __init__.py
│   │   ├── task_history.py         # TaskHistoryDB, TaskTracker
│   │   └── chat_history.py         # ChatHistoryManager, HistoryConfig
│   ├── context/                    # 上下文管理
│   │   ├── __init__.py
│   │   ├── scratch_pad.py          # 通用文件暂存（不含 DB 临时表）
│   │   └── result_cache.py         # LRU + TTL 缓存
│   ├── classify/                   # 统一分类器
│   │   ├── __init__.py
│   │   └── router.py               # QueryRouter — 统一 quick_classify，分流到 chatdb / chatreport
│   ├── skills/                     # 技能框架
│   │   ├── __init__.py
│   │   ├── skill_registry.py       # SkillRegistry 基类
│   │   └── skills_data/            # 技能定义文件（原 skills/ 目录）
│   │       ├── basic/
│   │       ├── comparison/
│   │       ├── drilldown/
│   │       ├── ranking/
│   │       ├── ratio/
│   │       ├── source/
│   │       ├── trend/
│   │       └── validation/
│   └── utils/                      # 通用工具
│       ├── __init__.py
│       ├── config.py               # Settings 基类
│       ├── logger.py               # 三层日志
│       ├── exceptions.py           # 异常基类
│       ├── json_utils.py           # JSON 序列化
│       └── common.py               # parse_json, extract_json 等
│
├── chatdb/                         # SQL 查询引擎（现有）
│   ├── api/                        # API 层（保留，同时服务于统一网关）
│   │   ├── app.py                  # → 升级为顶层网关，注册 chatreport 路由
│   │   ├── dependencies.py         # → 新增 get_report_orchestrator
│   │   ├── schemas.py              # → 新增 ReportRequest / ReportResponse
│   │   └── routes/
│   │       ├── __init__.py         # → 新增 report_router, gateway_router
│   │       ├── chat.py             # 现有
│   │       ├── query.py            # 现有
│   │       ├── agui.py             # 现有（chatdb 的 AG-UI）
│   │       ├── report.py           # 新增：报告专用路由
│   │       ├── report_agui.py      # 新增：报告 AG-UI SSE
│   │       ├── gateway.py          # 新增：统一入口（自动分类分流）
│   │       ├── database.py         # 现有
│   │       └── health.py           # 现有
│   ├── agents/                     # semantic_parser / planner / sql_agent
│   ├── config/                     # domain_config / table_config / metrics / virtual_field
│   ├── core/                       # orchestrator / react_state / summarize / scratch_pad ...
│   │   └── orchestrator.py         # 不再包含 report 分支，纯粹做数据查询
│   ├── tools/                      # sql / unix / virtual_field_converter / registry
│   ├── llm/                        # → 后续迁到 lib/llm/
│   ├── storage/                    # → 后续迁到 lib/storage/
│   ├── skills/                     # → 后续迁到 lib/skills/
│   ├── utils/                      # → 后续迁到 lib/utils/
│   └── ...
│
└── chatreport/                     # 报告引擎（新增）
    ├── __init__.py
    │
    ├── agents/                     # 智能体（对标 chatdb/agents/）
    │   ├── __init__.py
    │   ├── planner.py              # ReportPlanner — 大纲生成 + 章节子问题规划
    │   │                             对标 chatdb/agents/planner.py
    │   └── writer.py               # ReportWriter — 章节撰写 + 全局拼装 + 质量审计
    │                                 对标 chatdb/agents/sql_agent.py
    │
    ├── config/                     # 配置（对标 chatdb/config/）
    │   ├── __init__.py
    │   └── report_config.py        # ReportConfig — 报告模板、章节上限、质量阈值等
    │                                 对标 chatdb/config/domain_config.py
    │
    ├── core/                       # 核心协调层（对标 chatdb/core/）
    │   ├── __init__.py
    │   ├── orchestrator.py         # ReportOrchestrator — 报告主流程编排
    │   │                             对标 chatdb/core/orchestrator.py
    │   ├── agui_adapter.py         # ReportAGUIAdapter — 报告专用 AG-UI 适配器
    │   │                             对标 chatdb/core/agui_adapter.py
    │   ├── react_state.py          # ReportState — 报告级状态管理
    │   │                             对标 chatdb/core/react_state.py
    │   ├── section_dag.py          # SectionDAG — 章节级 DAG 拓扑管理
    │   │                             对标 chatdb/agents/planner.py 的 AnalysisPlan
    │   ├── summarize.py            # TransitionPlanner — 章节承上启下 + 上下文串联
    │   │                             对标 chatdb/core/summarize.py
    │   └── evidence.py             # EvidenceCollector — 证据链管理
    │                                 (新概念，无 chatdb 对标)
    │
    └── tools/                      # 工具（对标 chatdb/tools/）
        ├── __init__.py
        └── data_service.py         # DataService — ChatDB 适配层
                                      对标 chatdb/tools/sql.py
```

### chatreport 与 chatdb 的模块对齐关系

| chatreport | chatdb | 角色 |
|------------|--------|------|
| **agents/** | **agents/** | 智能体层 |
| `agents/planner.py` | `agents/planner.py` + `agents/semantic_parser.py` | 规划（大纲 + 子问题） |
| `agents/writer.py` | `agents/sql_agent.py` | 执行（撰写 + 拼装 + 审计） |
| **config/** | **config/** | 配置层 |
| `config/report_config.py` | `config/domain_config.py` | 业务配置 |
| **core/** | **core/** | 核心协调层 |
| `core/orchestrator.py` | `core/orchestrator.py` | 主流程编排 |
| `core/agui_adapter.py` | `core/agui_adapter.py` | AG-UI 事件流适配 |
| `core/react_state.py` | `core/react_state.py` | 状态管理 |
| `core/section_dag.py` | planner 内 AnalysisPlan | DAG 管理 |
| `core/summarize.py` | `core/summarize.py` | 上下文总结/承上启下 |
| `core/evidence.py` | （新增） | 证据链追踪 |
| **tools/** | **tools/** | 工具层 |
| `tools/data_service.py` | `tools/sql.py` | 数据获取 |

---

## 四、API 层详细设计

### 4.1 统一分类器 — `lib/classify/router.py`

```python
class QueryRouter:
    """统一查询分类器 — 决定请求走 ChatDB 还是 ChatReport"""
    
    REPORT_KEYWORDS = ["报告", "report", "分析报告", "研究报告", "深度分析", "总结报告"]
    
    async def classify(self, query: str, llm: BaseLLM | None = None) -> RouteDecision:
        """
        分类逻辑（由简到繁）：
        1. 关键词快速匹配 → 命中即返回
        2. 如有 LLM 则做精确分类（类似 chatdb 的 quick_classify）
        
        Returns:
            RouteDecision(
                target="chatdb" | "chatreport",
                query_type="basic" | "comparison" | ... | "report",
                confidence=0.95,
            )
        """
        ...

# RouteDecision
@dataclass
class RouteDecision:
    target: Literal["chatdb", "chatreport"]   # 目标引擎
    query_type: str                            # 细分类型
    confidence: float                          # 置信度
```

### 4.2 新增 Schemas — `chatdb/api/schemas.py` 扩展

```python
# ==================== 报告相关 ====================

class ReportRequest(BaseModel):
    """报告生成请求"""
    query: str = Field(..., description="报告主题描述", min_length=1, max_length=5000)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")
    # 报告专属参数
    max_sections: int = Field(default=7, description="最大章节数", ge=2, le=15)
    include_evidence: bool = Field(default=True, description="是否包含证据链")
    verify: bool = Field(default=True, description="是否启用质量审计")

class ReportResponse(BaseModel):
    """报告生成响应"""
    success: bool
    query: str
    title: str = ""
    markdown: str = ""                          # 完整报告 Markdown
    outline: dict | None = None                 # 大纲 JSON
    section_count: int = 0
    evidence_count: int = 0                     # 证据条数
    verification: dict | None = None            # 质量审计结果
    error: str | None = None
    elapsed_seconds: float = 0.0

# ==================== 网关相关 ====================

class GatewayRequest(BaseModel):
    """统一网关请求 — 自动分流到 chatdb 或 chatreport"""
    query: str = Field(..., description="自然语言输入", min_length=1)
    db_path: str = Field(..., description="DuckDB 或 CSV 文件路径")
    session_id: str | None = Field(default=None, description="会话 ID")

class GatewayResponse(BaseModel):
    """统一网关响应"""
    routed_to: Literal["chatdb", "chatreport"]  # 路由去向
    query_type: str                              # 分类结果
    # chatdb 结果字段
    sql: str | None = None
    result: list[dict] | None = None
    row_count: int = 0
    summary: str = ""
    # chatreport 结果字段
    report_markdown: str | None = None
    report_outline: dict | None = None
    # 通用字段
    success: bool = True
    error: str | None = None
```

### 4.3 报告路由 — `chatdb/api/routes/report.py`

```python
router = APIRouter(prefix="/report", tags=["Report"])

@router.post("/", response_model=ReportResponse, summary="生成数据分析报告")
async def generate_report(request: ReportRequest) -> ReportResponse:
    """
    独立的报告生成端点。
    不经过 ChatDB 的 process_query，直接调用 ReportOrchestrator。
    """
    orchestrator = await _create_chatdb_orchestrator(request.db_path)
    
    from chatreport.core.orchestrator import ReportOrchestrator
    from chatreport.tools.data_service import DataService
    
    report_orch = ReportOrchestrator(
        llm=orchestrator.llm,
        data_service=DataService(orchestrator),
    )
    report_result = await report_orch.generate_report(
        query=request.query,
        session_id=request.session_id,
        max_sections=request.max_sections,
        verify=request.verify,
    )
    return ReportResponse(
        success=True,
        query=request.query,
        title=report_result.title,
        markdown=report_result.markdown,
        outline=report_result.outline,
        section_count=report_result.section_count,
        evidence_count=report_result.evidence_count,
        verification=report_result.verification,
        elapsed_seconds=report_result.elapsed_seconds,
    )
```

### 4.4 报告 AG-UI SSE — `chatdb/api/routes/report_agui.py`

```python
router = APIRouter(prefix="/report/agui", tags=["Report AG-UI"])

@router.post("/", summary="报告生成 AG-UI SSE 流式端点")
async def report_agui_endpoint(request_body: ReportAGUIRequest, request: Request):
    """
    报告生成的 AG-UI 流式端点。
    
    事件映射（与 chatdb 的 agui 对齐但事件含义不同）：
      ReportOrchestrator 阶段     →  AG-UI 事件
      ─────────────────────────────────────────
      generate_report 开始         →  RUN_STARTED
      大纲生成                     →  STEP(outline) + TOOL_CALL
      章节规划                     →  STEP(section_plan) + TOOL_CALL
      数据查询（调 ChatDB）        →  STEP(data_query) + TOOL_CALL
      章节撰写                     →  STEP(section_write) + TEXT_MESSAGE(章节内容流式)
      质量审计                     →  STEP(verify) + CUSTOM(verification_result)
      承上启下                     →  CUSTOM(transition)
      全局拼装                     →  TEXT_MESSAGE(最终报告流式)
      报告完成                     →  STATE_SNAPSHOT + RUN_FINISHED
    """
    orchestrator = await _create_chatdb_orchestrator(request_body.db_path)
    
    from chatreport.core.agui_adapter import ReportAGUIAdapter
    from chatreport.core.orchestrator import ReportOrchestrator
    from chatreport.tools.data_service import DataService
    
    report_orch = ReportOrchestrator(
        llm=orchestrator.llm,
        data_service=DataService(orchestrator),
    )
    
    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)
    adapter = ReportAGUIAdapter(encoder=encoder)
    
    return StreamingResponse(
        adapter.stream(
            report_orchestrator=report_orch,
            query=request_body.query,
            thread_id=request_body.thread_id or uuid.uuid4().hex,
            run_id=request_body.run_id or uuid.uuid4().hex,
            session_id=request_body.session_id,
        ),
        media_type=encoder.get_content_type(),
    )
```

### 4.5 统一网关 — `chatdb/api/routes/gateway.py`

```python
router = APIRouter(prefix="/gateway", tags=["Gateway"])

@router.post("/", summary="统一入口 — 自动分流到数据查询或报告生成")
async def gateway_endpoint(request: GatewayRequest) -> GatewayResponse:
    """
    自动判断用户意图：
    - 数据查询类 → ChatDB.process_query()
    - 报告生成类 → ChatReport.generate_report()
    """
    from lib.classify.router import QueryRouter
    
    router = QueryRouter()
    decision = await router.classify(request.query)
    
    if decision.target == "chatdb":
        # 走 ChatDB 数据查询
        result = await chatdb_orchestrator.process_query(request.query, session_id=request.session_id)
        return GatewayResponse(
            routed_to="chatdb", query_type=decision.query_type,
            sql=result.get("sql"), result=result.get("result"),
            row_count=result.get("row_count", 0), summary=result.get("summary", ""),
            success=result.get("success", True),
        )
    else:
        # 走 ChatReport 报告生成
        report_result = await report_orchestrator.generate_report(
            query=request.query, session_id=request.session_id,
        )
        return GatewayResponse(
            routed_to="chatreport", query_type=decision.query_type,
            report_markdown=report_result.markdown, report_outline=report_result.outline,
            summary=report_result.markdown[:500], success=True,
        )

@router.post("/agui", summary="统一 AG-UI 入口 — 自动分流 + SSE 流式")
async def gateway_agui_endpoint(request_body: GatewayAGUIRequest, request: Request):
    """
    统一 AG-UI 流式入口。
    先 quick_classify，再根据结果选用 ChatDB 或 ChatReport 的 AGUIAdapter。
    
    前端只需对接一个端点，无需关心走的是查询还是报告。
    首个 CUSTOM 事件会携带 route_decision 告知前端当前模式。
    """
    from lib.classify.router import QueryRouter
    
    router_cls = QueryRouter()
    decision = await router_cls.classify(request_body.query)
    
    # 先发一个 route_decision 自定义事件，让前端知道走哪条路
    # 然后根据 target 选择不同的 adapter.stream()
    ...
```

### 4.6 ChatDB `app.py` 升级 — 注册新路由

```python
# chatdb/api/app.py 新增路由注册
from chatdb.api.routes import (
    database_router, health_router, query_router,
    chat_router, agui_router,
    report_router,        # 新增
    report_agui_router,   # 新增
    gateway_router,       # 新增
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(query_router)
app.include_router(database_router)
app.include_router(agui_router)
app.include_router(report_router)          # 新增
app.include_router(report_agui_router)     # 新增
app.include_router(gateway_router)         # 新增
```

---

## 五、ReportAGUIAdapter — 报告专用 AG-UI 适配

### 对标 chatdb 的 `AGUIAdapter`，但事件语义不同

```python
# chatreport/core/agui_adapter.py

class ReportAGUIAdapter:
    """
    将 ReportOrchestrator 的执行流程转换为 AG-UI SSE 事件流。
    
    与 ChatDB 的 AGUIAdapter 对标：
    - 相同：都用 asyncio.Queue 与 Orchestrator 解耦，都遵循 AG-UI 协议
    - 不同：事件粒度是"章节级"而非"SQL 任务级"
    
    事件流示意：
    
    RUN_STARTED
    │
    ├─ STEP_STARTED(name="outline_generation")
    │   ├─ TOOL_CALL(generate_outline) → 大纲 JSON
    │   └─ CUSTOM(outline_result, value={title, sections})
    ├─ STEP_FINISHED(name="outline_generation")
    │
    ├─ STEP_STARTED(name="section_s1")
    │   ├─ TOOL_CALL(plan_sub_questions) → 子问题列表
    │   ├─ TOOL_CALL(data_query_q1) → DataResult
    │   ├─ TOOL_CALL(data_query_q2) → DataResult
    │   ├─ TEXT_MESSAGE(section_content 流式输出)
    │   ├─ CUSTOM(evidence, value=[...])
    │   ├─ CUSTOM(transition, value={key_findings, ...})
    │   └─ CUSTOM(verification, value={...})
    ├─ STEP_FINISHED(name="section_s1")
    │
    ├─ STEP_STARTED(name="section_s2")
    │   └─ ...（同上）
    ├─ STEP_FINISHED(name="section_s2")
    │
    ├─ STEP_STARTED(name="assembly")
    │   ├─ CUSTOM(global_verification, value={...})
    │   └─ TEXT_MESSAGE(final_report 流式输出)
    ├─ STEP_FINISHED(name="assembly")
    │
    ├─ STATE_SNAPSHOT(snapshot={
    │     success, title, section_count, evidence_count,
    │     outline, report_markdown, verification
    │   })
    │
    └─ RUN_FINISHED
    """
    
    def __init__(self, encoder: EventEncoder | None = None):
        self._encoder = encoder or EventEncoder()
    
    async def stream(
        self,
        report_orchestrator: "ReportOrchestrator",
        query: str,
        thread_id: str,
        run_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """驱动 ReportOrchestrator 执行并逐步 yield AG-UI SSE 帧。"""
        ...
    
    def _map_event(self, event_type: str, data: dict, thread_id: str, run_id: str) -> list:
        """
        内部事件映射。
        
        ReportOrchestrator 内部事件类型：
        - outline_start / outline_end
        - section_start / section_end
        - sub_question_plan
        - data_query_start / data_query_end
        - section_write_chunk（流式章节内容）
        - evidence_recorded
        - transition_generated
        - verification_result
        - assembly_start / assembly_end
        - report_chunk（流式最终报告）
        """
        ...
```

### AG-UI 事件对比

| 维度 | ChatDB AGUIAdapter | ChatReport ReportAGUIAdapter |
|------|-------------------|-------------------------------|
| **STEP 粒度** | semantic_parse / planner / sql_task / summarize | outline / section_s1 / section_s2 / assembly |
| **TOOL_CALL 含义** | SQL 生成/执行/验证 | 大纲生成 / 子问题规划 / 数据查询 |
| **TEXT_MESSAGE** | 最终 summary（一次性） | 章节内容（逐章流式）+ 最终报告（流式） |
| **CUSTOM** | planner_decision | outline_result / transition / evidence / verification |
| **STATE_SNAPSHOT** | {sql, result, row_count, ...} | {title, markdown, outline, section_count, ...} |

---

## 六、四阶段 Pipeline 详细设计

### Stage 1: Planning（规划）— `agents/planner.py`

**输入**：用户查询 + 数据上下文（表描述、可用指标/维度）  
**输出**：结构化大纲 JSON

```json
{
  "title": "2024年游戏流水走势及下滑原因分析报告",
  "sections": [
    {
      "id": "s1",
      "title": "整体流水趋势",
      "description": "分析2024年月度流水变化趋势，与2023年同期对比",
      "data_needs": "月度流水汇总、同比变化率",
      "dependencies": [],
      "analysis_hints": ["trend", "comparison"]
    },
    {
      "id": "s2",
      "title": "产品结构分析",
      "description": "分析各产品类型的流水占比和变化",
      "data_needs": "按产品类型的流水明细、占比变化",
      "dependencies": [],
      "analysis_hints": ["structure", "ranking"]
    },
    {
      "id": "s3",
      "title": "下滑原因归因",
      "description": "基于前两章的发现，定位流水下滑的具体原因",
      "data_needs": "需结合s1趋势和s2结构变化做交叉分析",
      "dependencies": ["s1", "s2"],
      "analysis_hints": ["drilldown", "correlation"]
    }
  ]
}
```

**关键规则**：
- 章节数 3-7 个，根据复杂度灵活
- `dependencies` 决定章节间并行/串行

**ReportPlanner 内部两个核心方法**：

```python
class ReportPlanner:
    async def generate_outline(self, query, data_context, chat_history) -> dict:
        """大纲生成 — 对标 chatdb SemanticParser 的意图解析"""
        ...
    
    async def plan_sub_questions(self, section, previous_context, outline) -> list[SubQuestion]:
        """章节子问题规划 — 对标 chatdb Planner 的任务分解"""
        ...
```

### Stage 2: Question Developing（子问题展开）— `agents/planner.py`

每个章节执行前，ReportPlanner 将章节目标分解为可直接查数的自然语言子问题。

**输入**：章节描述 + 前序章节摘要 + 全局大纲  
**输出**：子问题列表 JSON

```json
[
  {
    "question": "2024年1-12月每月流水总额是多少？",
    "purpose": "获取月度流水基础数据，观察趋势",
    "priority": 1
  },
  {
    "question": "2024年各月流水与2023年同期相比变化率是多少？",
    "purpose": "量化同比变化幅度，识别关键下滑月份",
    "priority": 2
  }
]
```

**关键点**：
- 子问题是"人话"，直接丢给 ChatDB 的 `process_query` 处理
- ChatDB 内部自己走 Planner → SQLAgent → Summary 完整流程
- ReportPlanner 不关心 SQL 怎么写、怎么执行

### Stage 3: Data Exploration（数据探索）— `tools/data_service.py`

对每个子问题调用 `DataService.query()`，获取数据 + 摘要。

```python
class DataService:
    """ChatDB 的薄适配层 — 唯一的跨包依赖点
    
    对标 chatdb/tools/sql.py 的角色：前者执行 SQL，后者调用 ChatDB 问数。
    """
    
    async def query(self, question: str, session_id: str | None = None) -> DataResult:
        result = await self._orch.process_query(query=question, session_id=session_id)
        return DataResult(
            success=result.get("success", False),
            summary=result.get("summary", ""),
            data=result.get("result", []),
            sql=result.get("sql", ""),
            error=result.get("error"),
        )
    
    async def batch_query(self, questions: list[str], session_id: str | None = None) -> list[DataResult]:
        """批量查询（串行，共享 session 上下文）"""
        ...
```

### Stage 4: Report Generation（报告生成）— `agents/writer.py`

ReportWriter 是一个多职责 Agent，内部用方法区分三个子能力：

```python
class ReportWriter:
    """报告撰写 Agent — 对标 chatdb 的 SQLAgent（都是"执行+产出"角色）"""
    
    async def write_section(self, section, data_results, previous_context) -> str:
        """撰写单章节 Markdown — 对标 SQLAgent 的单次 SQL 执行"""
        ...
    
    async def assemble_report(self, report_state) -> str:
        """全局拼装 — 摘要 + 口径 + 各章节 + 风险 + 结论"""
        ...
    
    async def verify(self, section_content, data_results, mode="section") -> VerifyResult:
        """质量审计 — 无证据断言检测 + 跨章节一致性检查"""
        ...
```

---

## 七、两级 DAG 结构

```
                    章节级 DAG（SectionDAG — core/section_dag.py）
                    ┌─────────────────────┐
                    │  s1 ──┐             │
                    │  s2 ──┤─→ s3 ──→ s4 │
                    │  (并行)  (串行)      │
                    └─────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     章节内子问题     章节内子问题    章节内子问题
     ┌─────────┐    ┌─────────┐   ┌──────────┐
     │ q1 → DB │    │ q1 → DB │   │ q1 → DB  │
     │ q2 → DB │    │ q2 → DB │   │ q2 → DB  │
     │ q3 → DB │    └─────────┘   │ q3 → DB  │
     └─────────┘                  └──────────┘
         │                             │
    (每个 DB 查询内部)           (每个 DB 查询内部)
    ChatDB 的 Planner                ChatDB 的 Planner
         → SQLAgent                      → SQLAgent
         → Summary                       → Summary
```

- **第一级**：SectionDAG 管理章节间依赖，无依赖章节 `asyncio.gather` 并行
- **第二级**：每个章节内的子问题按序执行（共享 session 上下文，后续问题可引用前序结果）
- **第三级**（ChatDB 内部）：每个子问题触发 ChatDB 完整的 Planner → SQL → Summary 流程

---

## 八、证据链与质量机制

### 证据链（Evidence Chain）— `core/evidence.py`

每个查询的数据结果自动记录为一条证据：

```json
{
  "section_id": "s1",
  "evidences": [
    {
      "id": "e1",
      "question": "2024年各月流水总额",
      "sql": "SELECT 月份, SUM(流水) FROM ... GROUP BY 月份",
      "row_count": 12,
      "data_summary": "12个月数据，最高月份为3月（15.2亿），最低为11月（8.7亿）",
      "source_table": "ieg_flow",
      "virtual_fields_used": ["total_flow"],
      "timestamp": "2026-03-04T10:30:00"
    }
  ]
}
```

### 质量审计 — `agents/writer.py` 的 `verify()` 方法

**章节级检查**（`mode="section"`）：
- 无证据断言：章节内容中出现具体数字但未关联到任何查询结果
- 数据覆盖：章节 `data_needs` 中提到但未被任何子问题覆盖的数据需求
- 输出 `verification_result`，必要时生成补充查询建议

**全局检查**（`mode="global"`）：
- 跨章节数据一致性：不同章节引用同一指标的数字是否一致
- 结论矛盾检测：相邻章节的结论是否存在逻辑冲突
- 输出 `global_issues`，在最终报告中显式标注

```json
{
  "section_issues": [
    {
      "section_id": "s2",
      "type": "ungrounded_claim",
      "detail": "提到'端游增长20%'但无对应查询数据",
      "suggestion": "补充查询：2024年端游流水同比变化"
    }
  ],
  "global_issues": [
    {
      "type": "inconsistency",
      "detail": "s1 称总流水下滑15%，s3 称下滑12%",
      "sections": ["s1", "s3"]
    }
  ]
}
```

---

## 九、章节间承上启下 — `core/summarize.py`

TransitionPlanner 输出结构化的上下文：

```json
{
  "section_summary": "2024年流水整体下滑15%，下半年降幅加速",
  "key_findings": [
    "3月为全年峰值（15.2亿），11月为全年最低（8.7亿）",
    "Q3-Q4 环比下滑幅度达 25%，显著高于 Q1-Q2"
  ],
  "data_warnings": ["12月数据可能不完整（仅有20天记录）"],
  "adjustments": [
    {
      "target_section_id": "s3",
      "hint": "归因分析应重点关注Q3-Q4的加速下滑，而非全年均匀下滑"
    }
  ],
  "coherence_notes": "本章的同比下滑结论将作为s3归因分析的前提"
}
```

---

## 十、完整数据流

```
用户: "帮我生成一份2024年游戏流水分析报告"
  │
  ▼
方式A: POST /report （直接报告端点）
方式B: POST /gateway → QueryRouter.classify() → "report" → 转到 ChatReport
方式C: POST /report/agui （报告 AG-UI 流式）
方式D: POST /gateway/agui → classify → "report" → ReportAGUIAdapter
  │
  ▼  （以下以方式A/B为例）
ReportOrchestrator.generate_report()
  │
  ├─ [Stage 1] ReportPlanner.generate_outline()
  │     → { title, sections: [s1, s2, s3] }
  │     → 写入 scratch/{session}/report_outline.json
  │     → event_sink.put(("outline_end", {outline}))
  │
  ├─ SectionDAG(sections)
  │     s1(无依赖) ──┐
  │     s2(无依赖) ──┤── asyncio.gather 并行
  │     s3(依赖s1,s2)┘── 等待
  │
  ├─ 并行执行 s1, s2:
  │     │
  │     ├─ event_sink.put(("section_start", {id: "s1"}))
  │     │
  │     ├─ [Stage 2] ReportPlanner.plan_sub_questions(s1)
  │     │     → ["月度流水趋势？", "同比变化？"]
  │     │     → event_sink.put(("sub_question_plan", {questions}))
  │     │
  │     ├─ [Stage 3] DataService.query("月度流水趋势？")
  │     │     └─ ChatDB.process_query() → DataResult
  │     │     → event_sink.put(("data_query_end", {result}))
  │     │
  │     ├─ EvidenceCollector.record(...)
  │     │     → event_sink.put(("evidence_recorded", {evidence}))
  │     │
  │     ├─ [Stage 4a] ReportWriter.write_section(s1)
  │     │     → event_sink.put(("section_write_chunk", {delta}))  # 流式
  │     │
  │     ├─ ReportWriter.verify(s1, mode="section")
  │     │     → event_sink.put(("verification_result", {issues}))
  │     │
  │     ├─ TransitionPlanner.plan_transition(s1)
  │     │     → event_sink.put(("transition_generated", {transition}))
  │     │
  │     └─ event_sink.put(("section_end", {id: "s1"}))
  │
  ├─ 串行执行 s3（s1,s2 完成后）:
  │     ReportPlanner 收到 s1,s2 的 transition
  │     → 调整子问题方向 → 查数 → 写章节
  │
  ├─ ReportWriter.verify(mode="global")
  │     → event_sink.put(("global_verification", {issues}))
  │
  └─ ReportWriter.assemble_report()
       → event_sink.put(("report_chunk", {delta}))  # 流式
       → event_sink.put(None)  # 结束信号
       → 返回 ReportResult
```

---

## 十一、Scratch 存储结构

```
data/scratch/{session_id}/
├── report_outline.json             # 大纲
├── s1/
│   ├── sub_questions.json          # 子问题列表
│   ├── query_results/              # 各子问题的查询结果
│   │   ├── q1_result.json
│   │   └── q2_result.json
│   ├── evidence.json               # 证据链
│   ├── content.md                  # 章节正文
│   ├── transition.json             # 承上启下上下文
│   └── verification.json           # 质量检查结果
├── s2/
│   └── ...（同上）
├── s3/
│   └── ...
├── global_verification.json        # 全局一致性检查
└── final_report.md                 # 最终报告
```

---

## 十二、ChatDB 接口改造（去掉 report 分支）

### 原则：ChatDB 的 orchestrator 只做数据查询，不再处理 report

**改动点**：

1. `chatdb/core/orchestrator.py` 的 `process_query` 中，删除 `if query_type == "report"` 分支
2. `chatdb/core/orchestrator.py` 的 `quick_classify` 去掉 "report" 类型，遇到报告类请求返回 "unsupported" 或在网关层提前截断
3. chatdb 的 `AGUIAdapter` 不变，继续只处理数据查询的事件流

```python
# orchestrator.py 改动示意
# 删除:
#   if query_type == "report":
#       ... report 相关逻辑 ...
# 保留: "basic" / "comparison" / "ranking" / "trend" 等数据查询类型
```

---

## 十三、lib/ 公共模块抽离策略

### 原则：先 re-export，后迁移

Phase 1 先在 chatdb 内 import 不变，chatreport 直接 import chatdb 的公共模块。
Phase 2 将公共模块移入 `lib/`，chatdb 改为 `from lib.xxx import ...`。

### 抽离清单

| 模块 | 来源 | 优先级 | 说明 |
|------|------|--------|------|
| `lib/llm/` | `chatdb/llm/` | P0 | BaseLLM + Factory + 各实现 |
| `lib/classify/router.py` | 新建 | P0 | QueryRouter 统一分类器 |
| `lib/utils/logger.py` | `chatdb/utils/logger.py` | P0 | 三层日志 |
| `lib/utils/config.py` | `chatdb/utils/config.py` | P0 | Settings 基类（去掉 DB 特有配置） |
| `lib/utils/exceptions.py` | `chatdb/utils/exceptions.py` | P0 | 基础异常（去掉 SQLError 等） |
| `lib/utils/json_utils.py` | `chatdb/utils/json_utils.py` | P1 | JSON 序列化 |
| `lib/utils/common.py` | `chatdb/utils/common.py` | P1 | parse_json, extract_json |
| `lib/storage/task_history.py` | `chatdb/storage/task_history.py` | P1 | TaskHistoryDB + TaskTracker |
| `lib/storage/chat_history.py` | `chatdb/storage/chat_history.py` | P1 | ChatHistoryManager |
| `lib/context/scratch_pad.py` | `chatdb/core/scratch_pad.py` | P2 | 文件暂存（去掉临时表管理） |
| `lib/context/result_cache.py` | `chatdb/core/result_cache.py` | P2 | LRU + TTL 缓存 |
| `lib/skills/` | `chatdb/skills/` + `skills/` | P2 | SkillRegistry + 技能定义文件 |

---

## 十四、实施计划

### Phase 1: 基础框架（chatreport 骨架 + 数据层）

- [ ] 创建 `src/chatreport/` 四级子包目录结构（agents / config / core / tools）
- [ ] 实现 `core/react_state.py` — ReportState, SectionResult, DataResult, SubQuestion
- [ ] 实现 `core/section_dag.py` — SectionDAG, SectionNode
- [ ] 实现 `tools/data_service.py` — DataService（ChatDB 适配层）
- [ ] 实现 `core/evidence.py` — EvidenceCollector

**验证**：单元测试 SectionDAG 拓扑排序和状态管理

### Phase 2: 规划 Agent + 配置

- [ ] 实现 `agents/planner.py` — ReportPlanner（generate_outline + plan_sub_questions）
- [ ] 实现 `config/report_config.py` — ReportConfig

**验证**：给定查询生成大纲 JSON，大纲拆解为子问题列表

### Phase 3: 撰写 Agent + 承上启下

- [ ] 实现 `agents/writer.py` — ReportWriter（write_section + assemble_report + verify）
- [ ] 实现 `core/summarize.py` — TransitionPlanner

**验证**：单章节端到端——规划子问题 → 调 ChatDB 查数 → 写章节内容

### Phase 4: 主流程编排 + AG-UI 适配

- [ ] 实现 `core/orchestrator.py` — ReportOrchestrator（含 event_sink 事件推送）
- [ ] 实现 `core/agui_adapter.py` — ReportAGUIAdapter

**验证**：多章节端到端——完整报告生成 + AG-UI 事件流

### Phase 5: API 层 + 统一网关

- [ ] 新增 `chatdb/api/schemas.py` 中的 ReportRequest / ReportResponse / GatewayRequest / GatewayResponse
- [ ] 新增 `chatdb/api/routes/report.py` — 报告 REST 端点
- [ ] 新增 `chatdb/api/routes/report_agui.py` — 报告 AG-UI SSE 端点
- [ ] 新增 `chatdb/api/routes/gateway.py` — 统一网关（自动分类分流）
- [ ] 实现 `lib/classify/router.py` — QueryRouter 统一分类器
- [ ] 更新 `chatdb/api/app.py` 注册新路由
- [ ] ChatDB `orchestrator.py` 移除 report 分支

**验证**：通过 /report 端点生成完整报告，通过 /gateway 自动分流

### Phase 6: 质量审计完善

- [ ] 完善 ReportWriter.verify() 的章节级和全局级检查
- [ ] 在 ReportOrchestrator 中集成验证流程

**验证**：检测无证据断言、跨章节数字不一致

### Phase 7: lib/ 公共模块抽离 + skills 迁移（可选，报告功能稳定后）

- [ ] 创建 `src/lib/` 目录
- [ ] 迁移 llm/ 到 lib/llm/
- [ ] 迁移 utils/ 到 lib/utils/
- [ ] 迁移 storage/ 到 lib/storage/
- [ ] 迁移 skills/ 到 lib/skills/
- [ ] chatdb 内部 import 切换为从 lib 导入（re-export 兼容）

---

## 十五、设计优势总结

| 维度 | 设计 | 收益 |
|------|------|------|
| **完全独立** | ChatReport 有独立 API/AG-UI/分类，不经过 ChatDB.process_query | 两者完全解耦，互不影响 |
| **统一网关** | /gateway 自动分流，前端无感切换 | 向后兼容，前端只需一个端点 |
| **结构对齐** | chatreport 的 agents/config/core/tools 与 chatdb 一一对应 | 认知一致，便于后续类提取 |
| **AG-UI 对齐** | ReportAGUIAdapter 遵循相同协议，事件粒度为章节级 | 前端可复用 AG-UI 组件 |
| **Pipeline** | Plan → Ask → Explore → Write 四阶段 | 对齐 deep research 最佳实践 |
| **两级 DAG** | 章节级 DAG + 章节内子问题 | 灵活的并行/串行控制 |
| **证据链** | evidence.json 记录每条数据来源 | 结论可追溯、报告可审计 |
| **质量机制** | Writer.verify() 自检 + 补充查询 | 减少幻觉和数据不一致 |
| **承上启下** | transition JSON 串联章节 | 逻辑递进、表达连贯 |
| **skills 共享** | lib/skills/ 统一管理技能定义 | chatdb 和 chatreport 都可复用 |
| **渐进落地** | 7 个 Phase 逐步推进 | 每步可独立验证，风险可控 |
