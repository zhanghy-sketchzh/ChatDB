# lib/ 公共模块深度重构设计方案

## 一、现状问题

当前 `lib/` 采用 **re-export 桥接**策略——`lib/` 模块只是从 `chatdb/` 导入再导出，没有真正的代码迁移。这意味着：

1. **`lib/` 不是真正独立的**——所有代码仍然在 `chatdb/` 中，`lib/` 只是别名
2. **未来新场景（chathtml、chatpython、chatexcel...）仍然会重复造轮子**——因为 agent、orchestrator、state、agui_adapter 等核心框架没有抽象
3. **chatdb 和 chatreport 之间存在大量结构重复**——orchestrator、agui_adapter、react_state、summarize 的 80%+ 代码结构相同

### 代码重复对比

| 组件 | chatdb | chatreport | 重复模式 |
|------|--------|------------|----------|
| Orchestrator | `AgentOrchestrator` (1925行) | `ReportOrchestrator` (792行) | event_sink/_emit、process入口、阶段编排、结果构建、scratch持久化、错误处理 |
| AGUIAdapter | `AGUIAdapter` (351行) | `ReportAGUIAdapter` (324行) | stream()、_consume()、_map_event()、_map_step_start/end/text_chunk/custom 几乎一模一样 |
| State | `ReActState` (663行) | `ReportState` (291行) | phase枚举、dataclass结构、to_dict/get_debug_info、状态管理方法 |
| Summarize | `SummarizeAnswerTool` (366行) | `TransitionPlanner` (246行) | LLM调用+结构化输出+验证模式 |
| Planner | `PlannerAgent` (1849行) | `ReportPlanner` (462行) | LLM规划+JSON解析+任务分解 |
| Agent基类 | `BaseAgent` (448行) | 无 | chatreport的agent没有基类，直接裸写 |

---

## 二、设计目标

```
lib/                          ← 框架层：通用基类 + 工具函数
  ├── core/                   ← 编排框架基类
  │   ├── base_orchestrator.py
  │   ├── base_state.py
  │   ├── base_agui_adapter.py
  │   └── base_dag.py
  ├── agents/                 ← Agent 框架基类
  │   ├── base_agent.py
  │   ├── base_planner.py
  │   └── base_writer.py
  ├── llm/                    ← LLM 抽象（已存在，需真正迁移）
  ├── storage/                ← 持久化（已存在，需真正迁移）
  ├── utils/                  ← 工具函数（已存在，需真正迁移）
  └── classify/               ← 分类器（已存在）

chatdb/                       ← 数据查询场景
  ├── core/orchestrator.py    ← extends BaseOrchestrator
  ├── core/react_state.py     ← extends BaseState
  ├── core/agui_adapter.py    ← extends BaseAGUIAdapter
  └── agents/                 ← extends BaseAgent/BasePlanner

chatreport/                   ← 报告生成场景
  ├── core/orchestrator.py    ← extends BaseOrchestrator
  ├── core/react_state.py     ← extends BaseState
  ├── core/agui_adapter.py    ← extends BaseAGUIAdapter
  └── agents/                 ← extends BaseAgent/BasePlanner

chathtml/ (未来)              ← HTML生成场景
chatpython/ (未来)            ← Python分析场景
```

**核心原则：**
- `lib/` 是**零业务逻辑**的框架层，定义「怎么编排」而不是「编排什么」
- 场景包（chatdb/chatreport/chathtml/...）只需关注**自己的业务逻辑**，框架行为继承自 `lib/`
- 现有 chatdb 的 `from chatdb.xxx import ...` 路径通过 re-export **完全兼容**

---

## 三、`lib/core/` 核心基类设计

### 3.1 `base_state.py` — 状态机基类

从 `ReActState` 和 `ReportState` 中提取公共部分：

```python
"""lib/core/base_state.py — 通用状态机基类"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BasePhase(str, Enum):
    """基础阶段枚举 — 子类扩展"""
    INIT = "init"
    DONE = "done"
    ERROR = "error"


@dataclass
class BaseState:
    """
    通用状态机基类

    提供：
    - user_query: 用户输入
    - phase: 当前阶段（子类扩展枚举）
    - summary: 最终输出文本
    - error: 错误信息
    - step/max_steps: 步数控制
    - thoughts/actions/observations/reflections: ReAct日志
    - exec_meta: 执行元信息
    - temp_results: 中间结果
    """
    # ===== 输入 =====
    user_query: str = ""
    session_id: str = ""

    # ===== 阶段控制 =====
    phase: str = "init"
    step: int = 0
    max_steps: int = 10

    # ===== 输出 =====
    summary: str = ""
    error: str | None = None

    # ===== ReAct 日志 =====
    thoughts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)

    # ===== 执行元信息 =====
    exec_meta: dict[str, Any] = field(default_factory=lambda: {
        "plan_step": 0,
        "max_plan_steps": 8,
    })

    # ===== 中间结果 =====
    temp_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # ===== 方法 =====

    def think(self, thought: str) -> None: ...
    def act(self, action: str) -> None: ...
    def observe(self, observation: str) -> None: ...
    def reflect(self, reflection: str) -> None: ...
    def next_step(self) -> None: ...

    @property
    def is_done(self) -> bool:
        return self.phase in ("done", "error")

    @property
    def can_continue(self) -> bool:
        return not self.is_done and self.step < self.max_steps

    def to_dict(self) -> dict[str, Any]: ...
    def get_debug_info(self) -> dict[str, Any]: ...
    def get_reasoning_trace(self) -> str: ...
    # exec_meta helpers: inc_plan_step, plan_step, max_plan_steps, ...
```

**继承关系：**
- `chatdb.core.react_state.ReActState(BaseState)` — 增加 SQL/intent/schema/分析切片等字段
- `chatreport.core.react_state.ReportState(BaseState)` — 增加 outline/sections/evidence 等字段
- 未来 `chathtml.core.state.HTMLState(BaseState)` — 增加 template/components 等字段

### 3.2 `base_orchestrator.py` — 编排器基类

从 `AgentOrchestrator` 和 `ReportOrchestrator` 提取公共模式：

```python
"""lib/core/base_orchestrator.py — 编排器抽象基类"""

from abc import ABC, abstractmethod
import asyncio
import time
from typing import Any

from lib.llm import BaseLLM
from lib.utils.logger import get_component_logger


class BaseOrchestrator(ABC):
    """
    多 Agent 编排器抽象基类

    定义标准流程：
    1. 初始化状态
    2. 前置处理（分类、语义解析等）
    3. 计划生成
    4. 计划执行（串行/并行）
    5. 总结/输出
    6. 后置处理（持久化、缓存等）

    子类只需实现各阶段的业务逻辑。
    """

    def __init__(
        self,
        llm: BaseLLM,
        component_name: str = "Orchestrator",
        scratch_base: str = "data/scratch",
    ):
        self.llm = llm
        self._log = get_component_logger(component_name)
        self._event_sink: asyncio.Queue | None = None
        self._scratch_base = scratch_base

    # ================================================================
    # event_sink 机制（chatdb 和 chatreport 完全相同）
    # ================================================================

    async def _emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """向 AG-UI 事件队列推送一条内部事件（无队列时静默忽略）。"""
        if self._event_sink is not None:
            await self._event_sink.put((event_type, data or {}))

    # ================================================================
    # 主入口模板方法（Template Method Pattern）
    # ================================================================

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        event_sink: asyncio.Queue | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        主执行入口 — 模板方法

        定义固定的编排骨架，子类通过覆写各阶段钩子实现业务逻辑。
        """
        self._event_sink = event_sink
        start_time = time.time()

        state = await self.init_state(query, session_id, **kwargs)

        try:
            # 子类实现的各阶段
            state = await self.pre_process(state, **kwargs)
            state = await self.plan(state, **kwargs)
            state = await self.execute(state, **kwargs)
            state = await self.post_process(state, **kwargs)

            return self.build_result(state, query, start_time)

        except Exception as e:
            self._log.error(f"执行失败: {e}")
            await self._emit("error", {"error": str(e)})
            return self.build_error_result(state, query, start_time, e)

        finally:
            if self._event_sink is not None:
                await self._event_sink.put(None)  # 结束信号
                self._event_sink = None

    # ================================================================
    # 子类必须实现的抽象方法
    # ================================================================

    @abstractmethod
    async def init_state(self, query: str, session_id: str | None, **kwargs) -> Any:
        """初始化状态对象"""
        ...

    @abstractmethod
    async def pre_process(self, state: Any, **kwargs) -> Any:
        """前置处理（分类、语义解析、上下文加载等）"""
        ...

    @abstractmethod
    async def plan(self, state: Any, **kwargs) -> Any:
        """生成执行计划"""
        ...

    @abstractmethod
    async def execute(self, state: Any, **kwargs) -> Any:
        """执行计划"""
        ...

    @abstractmethod
    async def post_process(self, state: Any, **kwargs) -> Any:
        """后置处理（总结、持久化等）"""
        ...

    @abstractmethod
    def build_result(self, state: Any, query: str, start_time: float) -> dict[str, Any]:
        """构建返回结果"""
        ...

    # ================================================================
    # 可覆写的默认实现
    # ================================================================

    def build_error_result(self, state: Any, query: str, start_time: float, error: Exception) -> dict[str, Any]:
        """构建错误结果（可覆写）"""
        return {
            "success": False,
            "query": query,
            "error": str(error),
            "elapsed_ms": (time.time() - start_time) * 1000,
        }

    # ================================================================
    # Scratch Pad 公共方法
    # ================================================================

    def save_scratch(self, session_id: str, filename: str, content: str) -> None:
        """保存中间结果到 scratch 目录"""
        from pathlib import Path
        try:
            path = Path(self._scratch_base) / session_id / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            self._log.warn(f"Scratch 写入失败 ({filename}): {e}")
```

**继承关系：**
- `chatdb.core.orchestrator.AgentOrchestrator(BaseOrchestrator)` — 实现 SQL 查询的 pre_process/plan/execute/post_process
- `chatreport.core.orchestrator.ReportOrchestrator(BaseOrchestrator)` — 实现报告生成的各阶段
- 未来 `chathtml.core.orchestrator.HTMLOrchestrator(BaseOrchestrator)` — 实现 HTML 生成的各阶段

### 3.3 `base_agui_adapter.py` — AG-UI 适配器基类

从 `AGUIAdapter` 和 `ReportAGUIAdapter` 提取 **90%+ 相同的代码**：

```python
"""lib/core/base_agui_adapter.py — AG-UI 适配器基类"""

from abc import ABC, abstractmethod
import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator

from ag_ui.core import (
    EventType,
    RunStartedEvent, RunFinishedEvent, RunErrorEvent,
    StepStartedEvent, StepFinishedEvent,
    TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent,
    ToolCallStartEvent, ToolCallArgsEvent, ToolCallEndEvent, ToolCallResultEvent,
    StateSnapshotEvent, CustomEvent,
)
from ag_ui.encoder import EventEncoder
from lib.utils.logger import get_component_logger


class BaseAGUIAdapter(ABC):
    """
    AG-UI SSE 适配器基类

    提供完全通用的：
    - stream() 主循环
    - _consume() 队列消费
    - _map_event() 事件映射（含所有基础映射方法）

    子类只需覆写：
    - _drive_orchestrator(): 启动后台任务
    - _build_final_output(): 处理最终结果（summary/markdown/快照）
    - _extra_event_handlers(): 注册额外的事件映射
    """

    def __init__(self, component_name: str = "AGUIAdapter", encoder: EventEncoder | None = None):
        self._encoder = encoder or EventEncoder()
        self._event_count = 0
        self._start_time = 0.0
        self._log = get_component_logger(component_name)

    async def stream(
        self,
        orchestrator: Any,
        query: str,
        thread_id: str,
        run_id: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """通用 AG-UI 事件流主循环"""
        self._event_count = 0
        self._start_time = time.time()
        queue: asyncio.Queue = asyncio.Queue()
        enc = self._encoder.encode

        # RUN_STARTED
        yield enc(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
        self._event_count += 1

        # 子类启动后台任务
        task = asyncio.create_task(self._drive_orchestrator(orchestrator, query, queue, **kwargs))

        try:
            async for agui_event_str in self._consume(queue, thread_id, run_id):
                yield agui_event_str
            final_result = await task
        except Exception as exc:
            self._log.error(f"执行异常: {exc}")
            yield enc(RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc)))
            if not task.done():
                task.cancel()
            return

        # 子类处理最终输出
        async for event_str in self._build_final_output(final_result, thread_id, run_id):
            yield event_str

        # RUN_FINISHED
        yield enc(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))
        self._event_count += 1

    # ---- 子类必须实现 ----

    @abstractmethod
    async def _drive_orchestrator(self, orchestrator, query, queue, **kwargs) -> dict[str, Any]:
        """启动 Orchestrator 后台任务"""
        ...

    @abstractmethod
    async def _build_final_output(self, result, thread_id, run_id) -> AsyncGenerator[str, None]:
        """处理最终结果（yield SSE 帧）"""
        ...

    # ---- 完全通用的队列消费和事件映射 ----
    # _consume(), _map_event()
    # _map_step_start(), _map_step_end(), _map_text_chunk(), _map_custom()
    # _map_tool_start(), _map_tool_args(), _map_tool_end(), _map_tool_result()
    # 这些方法在 chatdb 和 chatreport 中完全相同，直接放基类
```

**收益：chatdb 和 chatreport 的 agui_adapter 各自减少约 200 行重复代码。**

### 3.4 `base_dag.py` — DAG 执行框架基类

从 `SectionDAG` 和 `AnalysisPlan` 提取 DAG 拓扑管理能力：

```python
"""lib/core/base_dag.py — DAG 拓扑调度框架"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """DAG 节点基类"""
    id: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    meta: dict[str, Any] = field(default_factory=dict)


class BaseDAG:
    """
    DAG 拓扑排序 + 批次执行框架

    提供：
    - add_node() / get_node()
    - get_ready_nodes(): 获取依赖已满足的节点
    - mark_done() / mark_failed() / mark_skipped()
    - iter_batches(): 逐批返回可并行节点
    - cycle_detection: Kahn 算法
    - progress_summary(): 进度摘要

    chatdb 的 AnalysisPlan 和 chatreport 的 SectionDAG 都可以继承此类。
    """
    ...
```

---

## 四、`lib/agents/` Agent 框架基类设计

### 4.1 `base_agent.py` — Agent 抽象基类

从 `chatdb/agents/base.py` 迁移并泛化（去掉 chatdb 的 DB/SQL 特有逻辑）：

```python
"""lib/agents/base_agent.py — Agent 抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lib.llm import BaseLLM
from lib.utils.logger import get_component_logger


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AgentContext:
    """Agent 通用上下文"""
    user_query: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Agent 执行结果"""
    status: AgentStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseAgent(ABC):
    """
    Agent 抽象基类

    提供：
    - LLM 交互
    - 工具注册和调用
    - 组件日志
    - 执行追踪

    子类实现：
    - execute(): 核心逻辑
    - get_system_prompt(): 系统提示词
    """

    def __init__(self, name: str, llm: BaseLLM, description: str = ""):
        self.name = name
        self.llm = llm
        self.description = description
        self._log = get_component_logger(name)
        self._tools: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        ...

    def get_system_prompt(self) -> str:
        return ""

    # 工具管理方法...
```

**注意：** 现有 `chatdb/agents/base.py` 中的 `BaseAgent` 包含大量 ChatDB 特有逻辑（history 管理、tool registry 集成、workflow history）。迁移到 `lib/` 时需要分层：
- `lib/agents/base_agent.py` — 纯粹的 Agent 抽象（LLM + 工具 + 日志）
- `chatdb/agents/base.py` — 继承 lib 基类，增加 ChatDB 特有功能（DB history、task tracking）

### 4.2 `base_planner.py` — 规划 Agent 基类

从 `PlannerAgent` 和 `ReportPlanner` 提取：

```python
"""lib/agents/base_planner.py — 规划 Agent 基类"""

from abc import abstractmethod
from typing import Any

from lib.agents.base_agent import BaseAgent
from lib.llm import BaseLLM
from lib.utils.common import parse_json


class BasePlanner(BaseAgent):
    """
    规划 Agent 基类 — LLM 驱动的任务分解

    通用流程：
    1. 构建规划 prompt（注入上下文 + 约束）
    2. 调用 LLM 生成 JSON 计划
    3. 解析 + 校验计划
    4. 构建 DAG

    子类实现：
    - build_plan_prompt(): 构建规划提示词
    - parse_plan_response(): 解析 LLM 输出为结构化计划
    - validate_plan(): 计划校验
    """

    def __init__(self, name: str, llm: BaseLLM):
        super().__init__(name, llm)
        self._plan_history: list[dict[str, Any]] = []

    async def generate_plan(self, state: Any, context: Any = None, **kwargs) -> Any:
        """通用规划流程"""
        prompt = self.build_plan_prompt(state, context, **kwargs)
        response = await self.llm.chat(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            caller_name=f"{self.name}_plan",
        )
        plan = self.parse_plan_response(response)
        plan = self.validate_plan(plan, state)
        return plan

    @abstractmethod
    def build_plan_prompt(self, state: Any, context: Any = None, **kwargs) -> str: ...

    @abstractmethod
    def parse_plan_response(self, response: str) -> Any: ...

    def validate_plan(self, plan: Any, state: Any) -> Any:
        """默认不做额外校验，子类可覆写"""
        return plan

    def clear_history(self) -> None:
        self._plan_history.clear()
```

### 4.3 `base_writer.py` — 内容生成 Agent 基类

从 `SummarizeAnswerTool` 和 `ReportWriter` 提取：

```python
"""lib/agents/base_writer.py — 内容生成 Agent 基类"""

from abc import abstractmethod
from typing import Any

from lib.agents.base_agent import BaseAgent
from lib.llm import BaseLLM


class BaseWriter(BaseAgent):
    """
    内容生成 Agent 基类 — LLM 驱动的文本输出

    通用流程：
    1. 构建内容生成 prompt（注入数据 + 指令）
    2. 调用 LLM 生成内容
    3. 可选：自检验证
    4. 返回最终内容

    子类实现：
    - build_write_prompt(): 构建写作提示词
    - 可选覆写 verify(): 自定义验证逻辑
    """

    async def generate(self, context: dict[str, Any], **kwargs) -> str:
        """通用内容生成流程"""
        prompt = self.build_write_prompt(context, **kwargs)
        content = await self.llm.chat(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            caller_name=f"{self.name}_write",
        )
        if kwargs.get("verify", False):
            content = await self.verify(content, context)
        return content.strip()

    @abstractmethod
    def build_write_prompt(self, context: dict[str, Any], **kwargs) -> str: ...

    async def verify(self, content: str, context: dict[str, Any]) -> str:
        """自检验证（默认不验证，子类可覆写）"""
        return content
```

---

## 五、`lib/utils/` 等公共模块真正迁移

### 5.1 迁移策略：剪切而非复制

不再 re-export。而是：

1. **剪切**代码到 `lib/`
2. `chatdb/` 原位置改为 `from lib.xxx import *`（re-export 兼容层，保持旧路径可用）
3. 所有新代码（chatreport/chathtml/chatpython）统一从 `lib/` 导入

### 5.2 迁移清单

| 模块 | 当前位置 | 迁移目标 | 处理方式 |
|------|---------|----------|----------|
| `config.py` | `chatdb/utils/config.py` | `lib/utils/config.py` | 剪切通用部分（Settings 基类、LLM/API/Log 配置），DB 配置留 chatdb |
| `logger.py` | `chatdb/utils/logger.py` | `lib/utils/logger.py` | 完整剪切（710行全是通用日志能力） |
| `exceptions.py` | `chatdb/utils/exceptions.py` | `lib/utils/exceptions.py` | 剪切通用异常，DB/SQL 异常留 chatdb |
| `common.py` | `chatdb/utils/common.py` | `lib/utils/common.py` | 剪切 parse_json/parse_json_array/format_rows 等通用函数 |
| `json_utils.py` | `chatdb/utils/json_utils.py` | `lib/utils/json_utils.py` | 完整剪切 |
| `base.py` (LLM) | `chatdb/llm/base.py` | `lib/llm/base.py` | 完整剪切 BaseLLM + Message + LLMResponse |
| `factory.py` (LLM) | `chatdb/llm/factory.py` | `lib/llm/factory.py` | 完整剪切 LLMFactory |
| LLM 实现 | `chatdb/llm/hunyuan.py` 等 | `lib/llm/` | 完整剪切所有 LLM provider 实现 |
| `task_history.py` | `chatdb/storage/task_history.py` | `lib/storage/task_history.py` | 完整剪切 |
| `chat_history.py` | `chatdb/storage/chat_history.py` | `lib/storage/chat_history.py` | 完整剪切 |
| `result_cache.py` | `chatdb/core/result_cache.py` | `lib/core/result_cache.py` | 完整剪切 |
| `scratch_pad.py` | `chatdb/core/scratch_pad.py` | `lib/core/scratch_pad.py` | 剪切通用文件暂存能力，临时表管理留 chatdb |
| `base.py` (Agent) | `chatdb/agents/base.py` | `lib/agents/base_agent.py` | 重构：通用部分到 lib，DB 特有留 chatdb |
| `base.py` (Tool) | `chatdb/tools/base.py` | `lib/tools/base.py` | 完整剪切（ToolMetadata, BaseTool, ToolResult 全是通用的） |
| `messages.py` | `chatdb/core/messages.py` | 不迁移 | TaskRequest/TaskResponse 是 chatdb 特有的 SQL 任务消息 |

### 5.3 chatdb re-export 兼容层示例

迁移后 `chatdb/utils/logger.py` 变为：

```python
"""chatdb/utils/logger.py — 兼容层（真正代码已迁移到 lib/）"""
from lib.utils.logger import *  # noqa: F401,F403
```

这样 **所有 `from chatdb.utils.logger import xxx` 的旧代码无需修改**。

---

## 六、`lib/core/scratch_pad.py` 分层

现有 `chatdb/core/scratch_pad.py` (710行) 包含两类功能：

1. **通用文件暂存**（save_task_result, read_task_result, load_plan, save_plan, expand_results, cleanup）—— **迁到 lib/**
2. **DB 临时表管理**（save_result_to_temp_table, cleanup_temp_tables）—— **留在 chatdb/**

```python
# lib/core/scratch_pad.py — 通用文件暂存
class ScratchPadManager:
    """文件暂存管理器（纯文件操作，无DB依赖）"""
    def save_task_result(self, session_id, task_id, result_entry) -> dict: ...
    def read_task_result(self, file_path) -> dict: ...
    def load_plan(self, session_id) -> dict | None: ...
    def save_plan(self, session_id, plan_data) -> None: ...
    def expand_results(self, temp_results) -> dict: ...
    def cleanup(self, session_id) -> None: ...

# chatdb/core/scratch_pad.py — 扩展：增加临时表能力
from lib.core.scratch_pad import ScratchPadManager as _BaseScratchPad

class ScratchPadManager(_BaseScratchPad):
    """带 DB 临时表功能的 ScratchPad"""
    def __init__(self, base_path, db_connector=None):
        super().__init__(base_path)
        self.db_connector = db_connector

    async def save_result_to_temp_table(self, session_id, task_id, rows) -> str: ...
    async def cleanup_temp_tables(self, session_id) -> None: ...
```

---

## 七、完整的 `lib/` 目录结构

```
src/lib/
├── __init__.py
│
├── core/                          # 编排框架
│   ├── __init__.py
│   ├── base_orchestrator.py       # 编排器基类（Template Method）
│   ├── base_state.py              # 状态机基类
│   ├── base_agui_adapter.py       # AG-UI 适配器基类
│   ├── base_dag.py                # DAG 拓扑调度基类
│   ├── result_cache.py            # LRU+TTL 缓存（从 chatdb 迁移）
│   └── scratch_pad.py             # 文件暂存（从 chatdb 迁移，去掉 DB 依赖）
│
├── agents/                        # Agent 框架
│   ├── __init__.py
│   ├── base_agent.py              # Agent 基类（从 chatdb 迁移+泛化）
│   ├── base_planner.py            # 规划 Agent 基类
│   └── base_writer.py             # 内容生成 Agent 基类
│
├── tools/                         # 工具框架
│   ├── __init__.py
│   └── base.py                    # 工具基类（从 chatdb 迁移）
│
├── llm/                           # LLM 层（完整迁移）
│   ├── __init__.py
│   ├── base.py                    # BaseLLM, Message, LLMResponse
│   ├── factory.py                 # LLMFactory
│   ├── hunyuan.py                 # 混元实现
│   ├── openai_compat.py           # OpenAI 兼容实现
│   └── ...                        # 其他 provider
│
├── storage/                       # 持久化（完整迁移）
│   ├── __init__.py
│   ├── task_history.py            # TaskHistoryDB + TaskTracker
│   └── chat_history.py            # ChatHistoryManager
│
├── utils/                         # 工具函数（完整迁移）
│   ├── __init__.py
│   ├── config.py                  # 通用 Settings（去掉 DB 配置）
│   ├── logger.py                  # 三层日志系统
│   ├── exceptions.py              # 通用异常
│   ├── common.py                  # parse_json, format_rows 等
│   └── json_utils.py              # JSON 序列化
│
└── classify/                      # 分类器（已存在）
    ├── __init__.py
    └── router.py
```

---

## 八、继承体系全景图

```
lib.core.BaseState
  ├── chatdb.core.ReActState         (+ SQL/intent/schema/分析切片)
  ├── chatreport.core.ReportState    (+ outline/sections/evidence)
  ├── chathtml.core.HTMLState        (未来)
  └── chatpython.core.PythonState    (未来)

lib.core.BaseOrchestrator
  ├── chatdb.core.AgentOrchestrator  (+ DB/SQL 编排逻辑)
  ├── chatreport.core.ReportOrchestrator (+ 章节 DAG 编排)
  ├── chathtml.core.HTMLOrchestrator (未来)
  └── chatpython.core.PythonOrchestrator (未来)

lib.core.BaseAGUIAdapter
  ├── chatdb.core.AGUIAdapter        (+ tool_start/tool_end 映射)
  ├── chatreport.core.ReportAGUIAdapter (+ 章节进度映射)
  └── ...

lib.core.BaseDAG
  ├── chatdb.agents.planner.AnalysisPlan (+ 分析任务节点)
  ├── chatreport.core.SectionDAG     (+ 章节节点)
  └── ...

lib.agents.BaseAgent
  ├── chatdb.agents.BaseAgent        (+ DB history, tool registry)
  │   ├── chatdb.agents.PlannerAgent
  │   ├── chatdb.agents.SQLAgent
  │   └── chatdb.agents.SemanticParser
  ├── chatreport.agents.*            (未来改为继承)
  │   ├── chatreport.agents.ReportPlanner
  │   └── chatreport.agents.ReportWriter
  └── ...

lib.agents.BasePlanner
  ├── chatdb.agents.PlannerAgent
  ├── chatreport.agents.ReportPlanner
  └── ...

lib.agents.BaseWriter
  ├── chatdb.core.SummarizeAnswerTool
  ├── chatreport.agents.ReportWriter
  └── ...

lib.tools.BaseTool
  ├── chatdb.tools.BaseTool
  ├── chatdb.core.SemanticParseTool
  └── chatdb.core.SummarizeAnswerTool
```

---

## 九、实施计划

### Phase A: 基础工具模块真正迁移（~2天）

1. 剪切 `utils/` 5个文件到 `lib/utils/`，chatdb 原位置改为 re-export 兼容层
2. 剪切 `llm/` 全部文件到 `lib/llm/`，chatdb 原位置改为 re-export
3. 剪切 `storage/` 2个文件到 `lib/storage/`，chatdb 原位置改为 re-export
4. 剪切 `tools/base.py` 到 `lib/tools/base.py`
5. 验证：chatdb 所有功能正常（re-export 兼容）、chatreport 所有导入正常

### Phase B: 核心基类提取（~3天）

1. 实现 `lib/core/base_state.py` — 从 ReActState 和 ReportState 提取公共字段和方法
2. 实现 `lib/core/base_orchestrator.py` — Template Method 模式的编排器基类
3. 实现 `lib/core/base_agui_adapter.py` — 提取完全相同的 stream/consume/map 代码
4. 实现 `lib/core/base_dag.py` — 从 AnalysisPlan 和 SectionDAG 提取拓扑管理

### Phase C: Agent 基类提取（~2天）

1. 实现 `lib/agents/base_agent.py` — 从 chatdb BaseAgent 提取通用部分
2. 实现 `lib/agents/base_planner.py` — 从 PlannerAgent 和 ReportPlanner 提取
3. 实现 `lib/agents/base_writer.py` — 从 SummarizeAnswerTool 和 ReportWriter 提取

### Phase D: chatdb 继承改造（~3天）

1. `chatdb/core/react_state.py` — `ReActState` 继承 `BaseState`，去掉重复字段和方法
2. `chatdb/core/orchestrator.py` — `AgentOrchestrator` 继承 `BaseOrchestrator`，只保留 SQL 查询的业务逻辑
3. `chatdb/core/agui_adapter.py` — `AGUIAdapter` 继承 `BaseAGUIAdapter`
4. `chatdb/agents/base.py` — 继承 `lib.agents.BaseAgent`
5. `chatdb/core/scratch_pad.py` — 拆分：通用文件暂存到 lib，DB 临时表留 chatdb
6. 全面回归测试

### Phase E: chatreport 继承改造（~2天）

1. `chatreport/core/react_state.py` — `ReportState` 继承 `BaseState`
2. `chatreport/core/orchestrator.py` — `ReportOrchestrator` 继承 `BaseOrchestrator`
3. `chatreport/core/agui_adapter.py` — `ReportAGUIAdapter` 继承 `BaseAGUIAdapter`
4. `chatreport/core/section_dag.py` — `SectionDAG` 继承 `BaseDAG`
5. `chatreport/agents/planner.py` — `ReportPlanner` 继承 `BasePlanner`
6. `chatreport/agents/writer.py` — `ReportWriter` 继承 `BaseWriter`

### Phase F: 验证 + 清理（~1天）

1. 完整的端到端测试（chatdb 查询 + chatreport 报告生成）
2. 导入路径一致性验证
3. 清理 lib/ 中多余的 re-export 文件
4. 更新 pyproject.toml packages 配置

---

## 十、预期收益量化

| 指标 | 重构前 | 重构后 | 减少 |
|------|--------|--------|------|
| chatdb AGUIAdapter | 351行 | ~80行（只剩业务映射） | -270行 |
| chatreport AGUIAdapter | 324行 | ~60行 | -264行 |
| chatdb ReActState 重复方法 | ~200行 | 0（继承） | -200行 |
| chatreport ReportState 重复方法 | ~100行 | 0（继承） | -100行 |
| chatreport 无 Agent 基类 | N/A | 有标准基类 | 结构规范化 |
| 新场景开发成本 | 从零开始 ~3000行 | 继承基类 ~500行 | **-83%** |

### 新场景开发模板

当开发 chathtml/chatpython/chatexcel 时，只需：

```python
# chathtml/core/orchestrator.py
from lib.core.base_orchestrator import BaseOrchestrator

class HTMLOrchestrator(BaseOrchestrator):
    async def init_state(self, query, session_id, **kwargs):
        return HTMLState(user_query=query, ...)

    async def pre_process(self, state, **kwargs):
        # HTML 特有的预处理
        ...

    async def plan(self, state, **kwargs):
        # HTML 生成计划
        ...

    async def execute(self, state, **kwargs):
        # 执行 HTML 生成
        ...

    async def post_process(self, state, **kwargs):
        # 输出 HTML
        ...

    def build_result(self, state, query, start_time):
        return {"success": True, "html": state.html_content, ...}
```

**从 3000+ 行降到 ~500 行——这就是框架基类的价值。**

---

## 十一、风险和注意事项

1. **向后兼容**：chatdb 的 re-export 兼容层确保所有 `from chatdb.xxx import yyy` 的旧代码继续工作，外部依赖方（如 scripts/、examples/）无需修改
2. **循环导入**：`lib/` 不能导入 `chatdb/` 或 `chatreport/`（严格单向依赖）。BaseState/BaseOrchestrator 的方法签名中使用 `Any` 类型注解避免循环导入
3. **渐进式迁移**：每个 Phase 独立可测试，不需要一次性改完所有代码
4. **config 分层**：`lib/utils/config.py` 只保留 LLM/API/Log 的通用配置，`chatdb/utils/config.py` 继承并增加 DatabaseSettings
5. **logger 依赖链**：logger.py 依赖 config.py → 两者必须同步迁移到 lib/
