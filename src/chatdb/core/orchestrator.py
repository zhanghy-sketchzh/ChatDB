"""
AgentOrchestrator - 多 Agent 协调器

设计理念：
1. semantic_parser 是**前置 workflow**，不属于 Planner 调度范围
2. Planner 只输出"分析任务"（type + description + notes），不给具体参数
3. SQLAgent 接收任务后自己做 ReAct loop，结果写入 temp_results
4. Planner 通过 inspect_temp_results 查看数据，动态决策下一步

核心流程：
```
1. [前置] 语义解析（SemanticParseTool）
2. Planner 生成分析任务列表
3. 循环执行：
   - SQLAgent.run_task() 执行当前任务，结果写入 temp_results
   - Planner.inspect_temp_results() 查看结果
   - Planner.decide_next_action() 决定：继续/调整/结束
4. 生成总结
```

使用方式：
```python
orch = AgentOrchestrator(llm, db, yml_config="data/yml/ieg.yml")
result = await orch.process_query("王者荣耀流水增长来自哪里")
```
"""

from pathlib import Path
from typing import Any, Optional, Union
import asyncio
import copy
import json
import time
import uuid

from chatdb.agents.base import AgentContext, AgentStatus
from chatdb.core.messages import TaskRequest, TaskResponse
from chatdb.core.react_state import ReActState, ReActPhase, ErrorType
from chatdb.core.scratch_pad import ScratchPadManager
from chatdb.core.result_cache import ResultCache
from chatdb.core.history_mixin import HistoryHelper
from chatdb.agents.planner import AnalysisPlan, PlannerAgent
from chatdb.agents.sql_agent import SQLAgent
from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.schema import SchemaInspector
from chatdb.llm.base import BaseLLM
from chatdb.storage.chat_history import ChatHistoryManager, HistoryConfig
from chatdb.storage.task_history import TaskHistoryDB, TaskTracker, TaskStatus
from chatdb.tools import ToolRegistry, UnixTool
from chatdb.core.semantic_parse import SemanticParseTool
from chatdb.core.summarize import SummarizeAnswerTool
from chatdb.core.context_retriever import ContextRetriever, RetrievalResult
from chatdb.skills import SkillRegistry as SkillReg, init_skill_registry, get_skill_registry
from chatdb.utils.logger import logger, set_log_level_to_debug, enable_llm_debug, task_log, get_component_logger
from chatdb.utils.common import select_best_table, build_schema_text, get_tables_info, format_rows


class AgentOrchestrator:
    """
    多 Agent 协调器
    
    核心组件：
    - SemanticParseTool: 语义解析（**前置 workflow**）
    - PlannerAgent: 生成分析型 ToDo
    - SQLAgent: SQL 分析 Agent（接收指令执行）
    - SummarizeAnswerTool: 总结回答
    
    流程设计：
    1. semantic_parser 是必要的前置步骤，不在 Planner 调度范围内
    2. Planner 生成"分析任务"而非"技术步骤"
    3. SQLAgent 接收 Planner 指令，内部 ReAct 拆解执行
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        db_connector: BaseDatabaseConnector,
        yml_config: Optional[Union[str, Path]] = None,
        tables_meta: Optional[list[dict[str, Any]]] = None,
        debug: bool = False,
        max_steps: int = 10,
        # 会话记忆配置
        history_db_path: Optional[Union[str, Path]] = None,
        history_config: Optional[HistoryConfig] = None,
        # 检索增强配置（Context Engineering）
        text_index: Any = None,
        example_store: Any = None,
    ):
        self.llm = llm
        self.db_connector = db_connector
        self.schema_inspector = SchemaInspector(db_connector)
        self.yml_config = yml_config
        self.tables_meta = tables_meta
        self.debug = debug
        self.max_steps = max_steps
        
        self._log = get_component_logger("Orchestrator")
        self._event_sink: asyncio.Queue | None = None
        
        if debug:
            set_log_level_to_debug()
            enable_llm_debug(True, show_input=True)  # debug 显示完整输入输出
        
        # 初始化 ToolRegistry
        self.registry = ToolRegistry()
        
        # ===== SkillRegistry（分析技能知识源）=====
        self._skill_registry = init_skill_registry("skills")
        
        # SQLAgent（核心 Agent，接受 yml_config + skill_registry）
        self._sql_agent = SQLAgent(self.llm, self.db_connector, yml_config,
                                   skill_registry=self._skill_registry)
        
        # 语义解析 Tool（前置 workflow）
        self._semantic_parse_tool = SemanticParseTool(llm, yml_config,
                                                      skill_registry=self._skill_registry)
        
        # 总结 Tool
        self._summarize_tool = SummarizeAnswerTool(llm)
        
        # 文件系统工具（供 Planner 观察 scratch 数据）
        self._unix_tool = UnixTool(workspace=".", readonly=True)

        # Planner（生成分析型 ToDo，数据直接内联到上下文）
        self.planner = PlannerAgent(llm, skill_registry=self._skill_registry,
                                    db_connector=db_connector)
        
        # ===== 统一历史 DB（tasks + agent_steps + plan_nodes + llm_calls）=====
        _history_db_path = history_db_path or "data/pilot/history.db"
        _history_db = TaskHistoryDB(_history_db_path) if _history_db_path else None
        
        # 会话记忆（基于同一 TaskHistoryDB）
        _history_manager: ChatHistoryManager | None = None
        if _history_db:
            _history_manager = ChatHistoryManager(
                _history_db, history_config or HistoryConfig()
            )
            _history_manager.set_agent("orchestrator")
        
        # 任务追踪（共用同一 DB 实例）
        self._task_tracker = TaskTracker(_history_db)
        
        # 注入 LLM 调用回调，自动记录每次 LLM 请求
        self.llm._on_llm_call = self._task_tracker.log_llm_call
        
        # ===== 会话历史辅助 =====
        self._history = HistoryHelper(_history_manager, self._task_tracker)
        
        # ===== 查询结果缓存 =====
        self._result_cache = ResultCache(max_size=50, ttl=300.0)
        
        # ===== Scratch Pad（文件暂存）=====
        self._scratch_pad = ScratchPadManager(base_path="data/scratch", db_connector=db_connector)

        # ===== Context Retriever（检索增强）=====
        self._context_retriever = ContextRetriever(
            text_index=text_index,
            example_store=example_store,
        )

    async def _emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """向 AG-UI 事件队列推送一条内部事件（无队列时静默忽略）。"""
        if self._event_sink is not None:
            await self._event_sink.put((event_type, data or {}))
    
    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        use_cache: bool = True,
        event_sink: asyncio.Queue | None = None,
    ) -> dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            session_id: 会话 ID。传入后启用多轮对话记忆，
                        相同 session_id 的查询共享历史上下文。
                        不传则为无状态单轮模式（向后兼容）。
            use_cache: 是否启用结果缓存（默认 True）。
                       完全相同的查询在缓存有效期内直接返回上次结果。
            event_sink: AG-UI 事件队列（可选）。传入后 Orchestrator 会在关键节点
                        推送内部事件，由 AGUIAdapter 转换为标准 AG-UI SSE 帧。
        
        流程：
        1. [前置] 检查结果缓存
        2. [前置] 加载会话历史
        3. [前置] 语义解析（不属于 Planner 调度）
        4. Planner 生成分析计划
        5. SQLAgent 按计划执行
        6. 生成总结
        7. [后置] 保存本轮结果到历史 + 写入缓存
        """
        self._event_sink = event_sink
        start_time = time.time()
        
        # Task View: 开始任务
        task_log.start(query)
        orch_log = get_component_logger("Orchestrator")
        
        # ===== 结果缓存：命中检查 =====
        if use_cache:
            cached = self._result_cache.get(session_id, query)
            if cached is not None:
                orch_log.info(f"结果缓存命中 (query={query[:30]}...)")
                cached["cached"] = True
                task_log.done(cached.get("summary", ""))
                return cached
        
        # 初始化
        self.planner.clear_history()
        state = await self._init_state(query)
        context = await self._init_context(query, state)
        
        # scratch session ID（用于文件暂存目录）
        scratch_session_id = session_id or uuid.uuid4().hex[:12]
        
        # ===== 任务追踪：开始 =====
        tracker = self._task_tracker
        tracker.start_task(scratch_session_id, query)
        
        # ===== 会话记忆：加载历史 =====
        chat_history = self._history.load_chat_history(session_id, orch_log)
        if chat_history:
            context.chat_history = chat_history
            context.session_id = session_id
        
        try:
            # ============================================================
            # 0. [前置] 轻量分类：快速识别非数据分析问题
            # 在所有重操作（虚拟字段检索、语义解析）之前，用极短的 LLM 调用判断问题类型
            # ============================================================
            query_type = await self._quick_classify(query, chat_history)
            if query_type == "chat":
                orch_log.info(f"前置分类: chat，走快速响应路径")
                state = await self._handle_chat_query(state, context, orch_log)
                result = self._build_result(state, query, start_time)
                self._history.save_to_history(session_id, query, state.summary or "", state)
                self._result_cache.put(session_id, query, result)
                task_log.done(state.summary or "")
                tracker.end_task(summary=state.summary or "")
                if self._event_sink is not None:
                    await self._event_sink.put(None)
                    self._event_sink = None
                return result
            
            # ============================================================
            # 0.5 [前置] 虚拟字段检索（在语义解析之前）
            # 将用户查询与 YML 虚拟字段做文本相似度匹配，召回结果注入后续各阶段 prompt
            # ============================================================
            self._retrieve_virtual_fields(state, query, orch_log)
            
            # ============================================================
            # 1. [前置 workflow] 语义解析 + 查询改写（合并为一次 LLM 调用）
            # SemanticParser 同时完成：指代消解、问题补全、意图提取
            # ============================================================
            orch_log.info("1. [前置] 语义解析...")
            await self._emit("step_start", {"name": "semantic_parse"})
            sp_step = tracker.start_step("semantic_parse", {"query": query})
            _sp_call_id = uuid.uuid4().hex
            await self._emit("tool_start", {"call_id": _sp_call_id, "tool_name": "semantic_parse"})
            await self._emit("tool_args", {"call_id": _sp_call_id, "args": {"query": query}})
            await self._semantic_parse_tool(state, context)
            
            # 取出改写后的查询，替换后续流程中的 user_query
            if state.intent and state.intent.rewritten_query and state.intent.rewritten_query != query:
                rewritten = state.intent.rewritten_query
                orch_log.info(f"查询改写: {query} → {rewritten}")
                state.user_query = rewritten
                context.user_query = rewritten
                state.rewritten_query = rewritten  # type: ignore[attr-defined]
                tracker.set_rewritten_query(rewritten)
                
                # ===== 用改写后的 query 重新检索 =====
                if self._context_retriever.has_text_index or self._context_retriever.has_example_store:
                    retrieval = self._context_retriever.retrieve_all(
                        rewritten, table_name=state.table_name,
                    )
                    if not retrieval.is_empty():
                        await self._inject_column_stats(retrieval, state.table_name)
                        state.retrieval_context = retrieval
                        orch_log.debug(f"改写后重新检索: {len(retrieval.value_matches)} 值匹配, "
                                       f"{len(retrieval.few_shot_examples)} 示例")
                
                # ===== 用改写后的 query 重新检索虚拟字段 =====
                self._retrieve_virtual_fields(state, rewritten, orch_log)
            
            sp_output: dict[str, Any] = {}
            if state.intent:
                # ★ 读取 LLM 判断的 research_mode 写入 state
                state.research_mode = getattr(state.intent, "research_mode", False)
                if state.research_mode:
                    orch_log.info("LLM 判断: research_mode=True（深度分析模式）")
                
                task_log.intent(
                    intent_type=state.intent.intent_type,
                    metrics=state.intent.metrics or [],
                    dimensions=state.intent.dimensions or [],
                    filters=state.intent.filter_refs or [],
                )
                sp_output = {
                    "intent_type": state.intent.intent_type,
                    "metrics": state.intent.metrics or [],
                    "dimensions": state.intent.dimensions or [],
                    "table": state.table_name,
                }
            else:
                raise ValueError("语义解析未返回 Intent，无法继续执行分析流程")
            tracker.end_step(sp_step, sp_output)
            await self._emit("tool_result", {
                "call_id": _sp_call_id, "message_id": uuid.uuid4().hex,
                "result": sp_output,
            })
            await self._emit("tool_end", {"call_id": _sp_call_id})
            await self._emit("step_end", {"name": "semantic_parse"})
            
            # ============================================================
            # 1.5 检查是否为非数据分析请求
            # ============================================================
            if state.intent and state.intent.is_other_query():
                orch_log.info("检测到非数据分析请求（mode=other），直接生成响应...")
                state = await self._handle_other_query(state, context, orch_log)
                if state.summary:
                    task_log.done(state.summary)
                result = self._build_result(state, query, start_time)
                self._history.save_to_history(session_id, query, state.summary or "", state)
                self._result_cache.put(session_id, query, result)
                tracker.end_task(summary=state.summary or "")
                return result
            
            # ============================================================
            # 2. Planner 生成分析计划（或恢复持久化计划）
            # ============================================================
            await self._emit("step_start", {"name": "planner"})
            planner_step = tracker.start_step("planner", {"query": state.user_query})
            _plan_call_id = uuid.uuid4().hex
            await self._emit("tool_start", {"call_id": _plan_call_id, "tool_name": "planner"})
            await self._emit("tool_args", {"call_id": _plan_call_id, "args": {"query": state.user_query}})
            plan = await self._get_or_create_plan(
                state, context, scratch_session_id, query, orch_log,
            )
            orch_log.info(f"分析计划:\n{plan.to_display()}")
            tracker.set_plan(plan.to_display())
            tracker.end_step(planner_step, {
                "task_count": len(plan.tasks),
                "plan": plan.to_display(),
            })
            await self._emit("tool_result", {
                "call_id": _plan_call_id, "message_id": uuid.uuid4().hex,
                "result": {"task_count": len(plan.tasks), "plan": plan.to_display()},
            })
            await self._emit("tool_end", {"call_id": _plan_call_id})
            await self._emit("step_end", {"name": "planner"})
            
            # ============================================================
            # 3. 按计划执行 SQLAgent
            # ============================================================
            orch_log.info(f"3. SQL 分析流程... (Agent: {self._sql_agent.display_name})")
            await self._emit("step_start", {"name": "sql_execute"})
            exec_step = tracker.start_step("sql_execute", {
                "plan_task_count": len(plan.tasks),
                "plan_tasks": [t.id for t in plan.tasks],
            })
            await self._execute_plan(state, context, orch_log, scratch_session_id)
            exec_output: dict[str, Any] = {}
            if state.executed_sqls:
                exec_output["sql_count"] = len(state.executed_sqls)
                exec_output["task_ids"] = list(state.executed_sqls.keys())
            elif state.final_sql or state.current_sql:
                exec_output["sql_count"] = 1
                exec_output["sql_preview"] = (state.final_sql or state.current_sql)[:200]
            if state.execute_result:
                exec_output["row_count"] = state.execute_result.get("row_count", 0)
            if state.error:
                exec_output["error"] = str(state.error)[:200]
            tracker.end_step(exec_step, exec_output or None)
            await self._emit("step_end", {"name": "sql_execute"})
            
            # Task View: SQL
            if state.current_sql:
                task_log.sql(state.current_sql, 1)
            
            # Task View: 执行
            if state.execute_result:
                row_count = state.execute_result.get("row_count", 0)
                task_log.execute(row_count, success=state.error is None or state.error_type == ErrorType.NO_DATA)
            
            # ============================================================
            # 4. 生成总结
            # ============================================================
            # ★ 多阶段分析：无条件走 summary（即使 Planner 提前 D 结束）
            # ★ 单阶段分析：仅在没有 summary 时生成
            need_summary = (
                state.temp_results  # 多阶段：始终需要 summary
                or (not state.summary and (state.has_result or state.error_type == ErrorType.NO_DATA))
            )
            if need_summary:
                await self._emit("step_start", {"name": "summarize"})
                sum_input: dict[str, Any] = {
                    "has_result": state.has_result,
                    "row_count": state.execute_result.get("row_count", 0) if state.execute_result else 0,
                }
                if state.executed_sqls:
                    sum_input["sql_count"] = len(state.executed_sqls)
                if state.temp_results:
                    sum_input["multi_stage"] = True
                    sum_input["task_count"] = len(state.temp_results)
                sum_step = tracker.start_step("summarize", sum_input)
                state = await self._generate_summary(state, query)
                tracker.end_step(sum_step, {"summary": (state.summary or "")[:200]})
                await self._emit("step_end", {"name": "summarize"})
            
            # 标记完成
            if state.has_result or state.summary:
                state.phase = ReActPhase.DONE
            
            # Task View: 完成
            if state.summary:
                task_log.done(state.summary)
            
            result = self._build_result(state, query, start_time)
            self._history.save_to_history(session_id, query, state.summary or "", state)
            self._result_cache.put(session_id, query, result)
            
            # 合并所有执行过的 SQL（多步骤时记录完整）
            if state.executed_sqls:
                all_sql_parts = [
                    f"-- [{tid}]\n{sql}" for tid, sql in state.executed_sqls.items()
                ]
                combined_sql = "\n\n".join(all_sql_parts)
            else:
                combined_sql = state.final_sql or state.current_sql
            
            tracker.end_task(
                final_sql=combined_sql,
                summary=state.summary or "",
            )
            
            # ★ 清理临时表（普通表，必须主动清理）
            await self._scratch_pad.cleanup_temp_tables(scratch_session_id)
            
            # AG-UI: 流程结束信号
            if self._event_sink is not None:
                await self._event_sink.put(None)
                self._event_sink = None
            
            return result
            
        except Exception as e:
            logger.error(f"[Orchestrator] 处理失败: {e}")
            tracker.end_task(error=str(e))
            # AG-UI: 错误 + 结束信号
            await self._emit("error", {"error": str(e)})
            if self._event_sink is not None:
                await self._event_sink.put(None)
                self._event_sink = None
            return {
                "success": False, "query": query, "sql": state.current_sql,
                "result": [], "error": str(e),
                "debug": state.get_debug_info() if self.debug else None,
            }

    async def _execute_plan(
        self,
        state: ReActState,
        context: AgentContext,
        orch_log,
        scratch_session_id: str = "",
    ) -> None:
        """
        执行 Planner 生成的分析计划
        
        通信模式（Orchestrator 模式 + Scratch Pad）：
        1. Orchestrator 构建 TaskRequest 消息 → 发给 SQLAgent
        2. SQLAgent 返回 TaskResponse 消息 → Orchestrator 写入文件
        3. collected_results 只保存摘要 + 文件引用（不保存全量数据）
        4. Orchestrator 将 collected_results 传给 Planner → Planner 基于摘要决策
        
        并行执行：
        - 同一层的 ready 任务（依赖均已满足）使用 asyncio.gather 并行执行
        - 每批任务完成后统一做一次 Planner 决策
        
        状态管理：
        - collected_results: Orchestrator 本地变量（精简版：摘要 + 文件引用）
        - state.temp_results: 同步更新（兼容旧接口如 _build_result / _generate_summary）
        """
        # Orchestrator 拥有的局部状态（精简版，含文件引用）
        collected_results: dict[str, list[dict[str, Any]]] = {}
        
        # 恢复已完成任务的结果（Plan Persistence 场景）
        if state.plan_resumed and self.planner.analysis_plan:
            collected_results = self._restore_completed_results(
                scratch_session_id, self.planner.analysis_plan,
            )
            if collected_results:
                state.temp_results = collected_results
        
        while state.plan_step < state.max_plan_steps:
            # 1. 获取所有 ready 任务
            ready_tasks = self.planner.get_ready_tasks(collected_results)
            if not ready_tasks:
                orch_log.info("没有更多任务")
                break
            
            # 2. 死循环检测 + validation 限制
            filtered_tasks = []
            for t in ready_tasks:
                if state.mark_task_repeat(t.id):
                    orch_log.warn(f"任务 {t.id} 重复执行，强制标记失败")
                    if self.planner.analysis_plan:
                        self.planner.analysis_plan.mark_failed(t.id, "重复执行超限")
                    continue
                if t.type.value == "validation":
                    if not state.bump_validation():
                        orch_log.warn("validation 任务达到上限，跳过更多诊断")
                        if self.planner.analysis_plan:
                            self.planner.analysis_plan.mark_skipped(t.id, "validation上限")
                        continue
                filtered_tasks.append(t)
            
            if not filtered_tasks:
                continue
            
            # 4. 并行或串行执行
            if len(filtered_tasks) == 1:
                # 单任务：串行执行（保持原有逻辑）
                task = filtered_tasks[0]
                current_step = state.inc_plan_step()
                task_dict = task.to_dict()
                task_id = task_dict.get("id", "")
                task_type = task_dict.get("type", "")
                orch_log.info(f"执行任务 {current_step}: [{task_type}] {task_dict['description'][:80]}...")
                
                await self._execute_single_task(
                    state, context, task, collected_results,
                    orch_log, scratch_session_id,
                )
            else:
                # 多任务：并行执行（每个任务使用独立的 state 深拷贝避免冲突）
                task_ids = [t.id for t in filtered_tasks]
                orch_log.info(f"并行执行 {len(filtered_tasks)} 个任务: {task_ids}")
                
                # 先批量分配 step 编号（在 await 前完成，避免并发修改 exec_meta）
                task_steps: dict[str, int] = {}
                for t in filtered_tasks:
                    task_steps[t.id] = state.inc_plan_step()
                
                async def _run_one(t: Any) -> None:
                    step = task_steps[t.id]
                    td = t.to_dict()
                    orch_log.info(f"  [并行] 任务 {step}: [{td.get('type', '')}] {td['description'][:80]}...")
                    # 使用 deepcopy 确保嵌套可变对象（exec_meta, thoughts, error_context 等）完全隔离
                    task_state = copy.deepcopy(state)
                    task_state.execute_result = None
                    task_state.current_sql = ""
                    task_state.final_sql = ""
                    task_state.execution_error = None
                    task_state.error = None
                    task_state.error_type = ErrorType.NONE
                    task_state.error_context = {}
                    await self._execute_single_task(
                        task_state, context, t, collected_results,
                        orch_log, scratch_session_id,
                        is_parallel=True,
                    )
                    # 回写最后执行的 SQL 到主 state（用于日志/调试）
                    if task_state.current_sql:
                        state.current_sql = task_state.current_sql
                    if task_state.final_sql:
                        state.final_sql = task_state.final_sql
                    if task_state.execute_result:
                        state.execute_result = task_state.execute_result
                    # 合并多步 SQL 记录
                    state.executed_sqls.update(task_state.executed_sqls)
                
                await asyncio.gather(*[_run_one(t) for t in filtered_tasks])
            
            # 5. 同步 temp_results
            state.temp_results = collected_results
            
            # 6. Planner 决策（一批任务完成后统一决策一次）
            decision = await self.planner.decide_next_action(state, context, collected_results)
            await self._emit("custom", {
                "name": "planner_decision",
                "value": {"action": decision.get("action", ""), "reason": decision.get("reason", "")},
            })
            if await self._handle_planner_decision(state, context, decision, orch_log):
                break
        
        # 最终同步
        state.temp_results = collected_results
        
        if collected_results:
            orch_log.info(f"完成 {len(collected_results)} 个任务的数据收集")

    async def _execute_single_task(
        self,
        state: ReActState,
        context: AgentContext,
        task: Any,
        collected_results: dict[str, list[dict[str, Any]]],
        orch_log,
        scratch_session_id: str,
        is_parallel: bool = False,
    ) -> None:
        """执行单个任务（从原 _execute_plan 循环体提取）"""
        task_dict = task.to_dict()
        task_id = task_dict.get("id", "")
        task_type = task_dict.get("type", "query")
        task_desc = task_dict.get("description", "")
        depends_on = task_dict.get("depends_on", [])
        
        # AG-UI: 任务开始
        _sql_call_id = uuid.uuid4().hex
        await self._emit("tool_start", {
            "call_id": _sql_call_id, "tool_name": f"sql_task_{task_id}",
        })
        await self._emit("tool_args", {
            "call_id": _sql_call_id,
            "args": {"task_id": task_id, "type": task_type, "description": task_desc},
        })
        
        # ===== 追踪：开始 plan node =====
        tracker = self._task_tracker
        node_id = tracker.start_node(
            step_id=tracker.active_step_id or "",
            plan_task_id=task_id,
            task_type=task_type,
            description=task_desc,
            depends_on=depends_on,
            is_parallel=is_parallel,
        )
        
        request = self._build_task_request(task_dict, collected_results, state)
        
        try:
            response = await self._sql_agent.run_task(state, context, request)
            
            await self._store_response_to_scratch(
                collected_results, response,
                scratch_session_id,
            )
            
            task_has_error = self._check_task_has_error(response, collected_results, task_id)
            
            temp_summary = await self.planner.inspect_temp_results(state, collected_results)
            if temp_summary:
                orch_log.debug(f"temp_results 摘要:\n{temp_summary[:200]}...")
            
            # 收集 node 结果信息
            node_sql = state.current_sql or ""
            node_row_count = 0
            node_sample: list[dict[str, Any]] = []
            for r in collected_results.get(task_id, []):
                node_sql = node_sql or r.get("sql", "")
                node_row_count += r.get("row_count", 0)
                node_sample.extend(r.get("examples", [])[:5])
            
            if task_has_error:
                orch_log.warn(f"任务 {task_id} 存在 SQL 执行错误，交由 Planner 决策")
                if self.planner.analysis_plan:
                    self.planner.analysis_plan.mark_failed(task_id, task_has_error)
                self._persist_task_completion(
                    scratch_session_id, task_id, "failed",
                )
                tracker.end_node(node_id, sql=node_sql, row_count=node_row_count,
                                 result_sample=node_sample[:5], error=task_has_error)
                await self._emit("tool_result", {
                    "call_id": _sql_call_id, "message_id": uuid.uuid4().hex,
                    "result": {"task_id": task_id, "error": task_has_error},
                })
            else:
                if node_sql:
                    state.executed_sqls[task_id] = node_sql
                
                if self.planner.analysis_plan:
                    self.planner.analysis_plan.mark_completed(task_id)
                
                result_file = ""
                for r in collected_results.get(task_id, []):
                    ref = r.get("_file_ref", {})
                    if ref.get("path"):
                        result_file = ref["path"]
                        break
                self._persist_task_completion(
                    scratch_session_id, task_id, "completed", result_file,
                )
                tracker.end_node(node_id, sql=node_sql, row_count=node_row_count,
                                 result_sample=node_sample[:5])
                await self._emit("tool_result", {
                    "call_id": _sql_call_id, "message_id": uuid.uuid4().hex,
                    "result": {
                        "task_id": task_id, "sql": node_sql,
                        "row_count": node_row_count,
                    },
                })
            
            await self._emit("tool_end", {"call_id": _sql_call_id})
            
        except Exception as e:
            orch_log.warn(f"任务 {task_id} 执行失败: {e}")
            if self.planner.analysis_plan:
                self.planner.analysis_plan.mark_failed(task_id, str(e))
            self._persist_task_completion(
                scratch_session_id, task_id, "failed",
            )
            tracker.end_node(node_id, error=str(e))
            await self._emit("tool_result", {
                "call_id": _sql_call_id, "message_id": uuid.uuid4().hex,
                "result": {"task_id": task_id, "error": str(e)},
            })
            await self._emit("tool_end", {"call_id": _sql_call_id})

    @staticmethod
    def _check_task_has_error(
        response: Any,
        collected_results: dict[str, list[dict[str, Any]]],
        task_id: str,
    ) -> str:
        """检查任务执行结果是否包含 SQL 错误。
        
        Returns:
            错误描述字符串（空字符串表示无错误）
        """
        # 1. 检查 TaskResponse 级别的错误
        if hasattr(response, 'error') and response.error:
            return response.error
        if hasattr(response, 'success') and response.success is False:
            return response.error or "任务执行失败"
        
        # 2. 检查 collected_results 中该任务的 issues
        for r in collected_results.get(task_id, []):
            issues = r.get("issues", [])
            for issue in issues:
                if isinstance(issue, str) and (
                    issue.startswith("sql_error:") or
                    issue.startswith("error:") or
                    "no_execute_result" in issue
                ):
                    return issue
        
        return ""

    def _build_task_request(
        self,
        task_dict: dict[str, Any],
        collected_results: dict[str, list[dict[str, Any]]],
        state: "ReActState | None" = None,
    ) -> TaskRequest:
        """
        构建 TaskRequest 消息（包含上游结果摘要 + 文件引用）
        
        Scratch Pad 模式下：
        - parent_results_summary 包含摘要文本 + 文件路径
        - previous_results 传精简版（含 _file_ref），SQLAgent 按需读取
        """
        depends_on = task_dict.get("depends_on", [])
        
        parent_summary_parts: list[str] = []
        all_previous: list[dict[str, Any]] = []
        
        # ★ 注入 Planner 的承上启下分析（如果有）
        if state and state.transition_context:
            parent_summary_parts.append(
                f"[Planner 分析结论] {state.transition_context}"
            )
            # 用完后清空，避免影响更后面的任务
            state.transition_context = ""
        
        # 标注依赖任务状态，让下游 Agent 感知
        if self.planner.analysis_plan:
            for dep_id in depends_on:
                dep_task = self.planner.analysis_plan.get_task(dep_id)
                if dep_task and dep_task.status in ("failed", "skipped"):
                    parent_summary_parts.append(
                        f"⚠ [{dep_id}] 状态={dep_task.status}"
                        + (f"({dep_task.skip_reason})" if dep_task.skip_reason else "")
                    )
        
        INLINE_ROW_LIMIT = 30
        
        # ★ 收集上游临时表信息（含列名，方便下游 SQL 引用）
        upstream_temp_tables: dict[str, dict[str, Any]] = {}  # {task_id: {name: str, columns: list}}
        
        for dep_id in depends_on:
            for r in collected_results.get(dep_id, []):
                stats = r.get("stats", {})
                file_ref = r.get("_file_ref")
                temp_table = r.get("_temp_table")
                row_count = r.get("row_count", 0)
                examples = r.get("examples", [])
                
                # 记录临时表（含列名）
                if temp_table:
                    cols = list(examples[0].keys()) if examples else []
                    upstream_temp_tables[dep_id] = {
                        "name": temp_table,
                        "columns": cols,
                        "row_count": row_count,
                    }
                
                # ★ 回退：examples 为空但文件存在时，从文件读取
                if not examples and file_ref:
                    examples = self._read_examples_from_file(file_ref)
                
                if stats.get("available_years"):
                    parent_summary_parts.append(f"可用年份: {stats['available_years']}")
                if stats.get("top_contributor"):
                    parent_summary_parts.append(f"top贡献: {stats['top_contributor']}")
                
                # 内联上游数据
                if examples:
                    display_rows = examples if row_count <= INLINE_ROW_LIMIT else examples[:INLINE_ROW_LIMIT]
                    data_lines = [f"[{dep_id}] 上游返回 {row_count} 行，数据如下:"]
                    for ex in display_rows:
                        row_str = ", ".join(f"{k}={v}" for k, v in ex.items())
                        data_lines.append(f"  - {row_str}")
                    if row_count > INLINE_ROW_LIMIT:
                        data_lines.append(f"  ... 共 {row_count} 行，已展示前 {INLINE_ROW_LIMIT} 行")
                    parent_summary_parts.append("\n".join(data_lines))
                elif row_count:
                    parent_summary_parts.append(f"上游返回 {row_count} 行")
                    if file_ref:
                        summary = file_ref.get("summary", "")
                        if summary:
                            parent_summary_parts.append(f"[{dep_id}] {summary}")
                
                all_previous.append(r)
        
        if not depends_on:
            task_id = task_dict.get("id", "")
            for tid, results in collected_results.items():
                if tid != task_id:
                    all_previous.extend(results)
        
        # ★ 注入临时表信息到 TaskRequest（通过构造参数传递，frozen dataclass 不可后赋值）
        request = TaskRequest.from_planner_task(
            task_dict,
            parent_results_summary="\n".join(parent_summary_parts),
            previous_results=all_previous,
            upstream_temp_tables=upstream_temp_tables,
        )
        
        return request
    
    def _read_examples_from_file(self, file_ref: dict[str, Any]) -> list[dict[str, Any]]:
        """从 scratch pad 文件回退读取 examples"""
        file_path = file_ref.get("path", "")
        if not file_path:
            return []
        full_data = self._scratch_pad.read_task_result(file_path)
        if not full_data:
            return []
        return full_data.get("examples", [])
    
    @staticmethod
    def _store_response(
        collected_results: dict[str, list[dict[str, Any]]],
        response: TaskResponse,
    ) -> None:
        """将 TaskResponse 存入 collected_results（旧接口，全量内存）"""
        task_id = response.task_id
        if task_id not in collected_results:
            collected_results[task_id] = []
        collected_results[task_id].extend(response.to_results_dicts())

    async def _store_response_to_scratch(
        self,
        collected_results: dict[str, list[dict[str, Any]]],
        response: TaskResponse,
        scratch_session_id: str,
    ) -> None:
        """
        将 TaskResponse 写入 Scratch Pad 文件和临时表，collected_results 只存精简版
        
        精简版包含：摘要 + 文件路径 + 临时表名 + 统计 + 样例（前30行）
        完整数据通过文件引用或临时表按需读取
        """
        task_id = response.task_id
        if task_id not in collected_results:
            collected_results[task_id] = []
        
        for result_obj in response.results:
            result_entry = result_obj.to_dict()
            # 用完整行数据替换截断后的 examples（供文件保存和临时表使用）
            all_rows = result_obj.all_rows
            if all_rows:
                result_entry["examples"] = all_rows
            # 通过 ScratchPadManager 写入文件并获取精简版
            slim_result = self._scratch_pad.save_task_result(
                session_id=scratch_session_id,
                task_id=task_id,
                result_entry=result_entry,
            )
            
            # ★ 创建临时表（使用完整行数据）
            rows = all_rows or result_entry.get("examples", [])
            if rows and self._scratch_pad.db_connector:
                try:
                    table_name = await self._scratch_pad.save_result_to_temp_table(
                        scratch_session_id, task_id, rows
                    )
                    if table_name:
                        slim_result["_temp_table"] = table_name
                except Exception as e:
                    logger.warn(f"创建临时表失败 (task={task_id}): {e}")
            
            collected_results[task_id].append(slim_result)

    async def _handle_planner_decision(
        self,
        state: ReActState,
        context: AgentContext,
        decision: dict[str, Any],
        orch_log,
    ) -> bool:
        """
        处理 Planner 的决策
        
        返回 True 表示结束整个计划执行；False 表示继续后续任务
        """
        action = decision.get("action", "done")
        
        if action == "done":
            reason = decision.get("reason", "")
            conclusion = decision.get("conclusion", "")
            orch_log.info(f"Planner 决定结束: {reason}")
            if conclusion:
                orch_log.info(f"Planner 结论（供 summary 参考）: {conclusion[:100]}...")
                # ★ 不写入 state.summary，留给后续 summary 流程统一生成
                state.planner_conclusion = conclusion
            return True
        
        if action == "intervention":
            intervention = decision.get("intervention", {})
            orch_log.info(f"Planner 请求人类介入: {intervention.get('question', '')[:80]}")
            state.intervention = intervention
            state.intervention_step_id = state.plan_step
            await self._emit("custom", {
                "name": "human_intervention",
                "value": intervention,
            })
            return True  # 中断执行循环
        
        if action == "adjust":
            if state.bump_adjust():
                reason = decision.get("reason", "")
                adjustment = decision.get("adjustment", {})
                max_adjust = state.exec_meta["max_adjust"]
                adjust_count = state.exec_meta["adjust_count"]
                orch_log.info(f"Planner 决定调整 ({adjust_count}/{max_adjust}): {reason}")
                self._apply_adjustment(state, context, adjustment, orch_log)
            else:
                orch_log.warn("达到最大调整次数，继续下一任务")
                state.reset_adjust()
            return False
        
        if action == "retry":
            retry_hint = decision.get("retry_hint", "")
            orch_log.info(f"重试任务，提示: {retry_hint}")
            state.dec_plan_step()  # 不计入步数
            return False
        
        # action == "continue" 或未知值：默认继续下一任务
        state.reset_adjust()
        # ★ 将 Planner 的承上启下分析存入 state，供下一步 SQL Agent 参考
        transition_context = decision.get("transition_context", "")
        if transition_context:
            state.transition_context = transition_context
            orch_log.info(f"Planner 承上启下: {transition_context[:120]}...")
        # ★ 传递 Planner 选择的表列表到 state，下游 SQL 生成据此裁剪上下文
        selected_tables = decision.get("selected_tables")
        if selected_tables and isinstance(selected_tables, list):
            state.selected_tables = selected_tables
        else:
            state.selected_tables = None  # 未指定时默认注入源表

        # ★ 强制注入：当下一个任务有 depends_on 时，将上游临时表合并到 selected_tables
        next_task = decision.get("task")
        if next_task is not None:
            # ★ 传递 Planner decide 指定的虚拟字段到下一个任务的 meta
            decide_vfs = decision.get("virtual_fields")
            if decide_vfs is not None and isinstance(decide_vfs, list):
                if hasattr(next_task, "meta") and isinstance(next_task.meta, dict):
                    next_task.meta["virtual_fields"] = decide_vfs
                    orch_log.info(f"Planner decide 指定虚拟字段: {decide_vfs}")

            depends_on = (
                next_task.depends_on
                if hasattr(next_task, "depends_on")
                else next_task.get("depends_on", []) if isinstance(next_task, dict) else []
            )
            if depends_on and state.temp_results:
                upstream_temp_names: list[str] = []
                for dep_id in depends_on:
                    for r in state.temp_results.get(dep_id, []):
                        tt = r.get("_temp_table")
                        if tt:
                            upstream_temp_names.append(tt)
                if upstream_temp_names:
                    current = list(state.selected_tables) if state.selected_tables else []
                    merged = current + [t for t in upstream_temp_names if t not in current]
                    state.selected_tables = merged
                    orch_log.info(
                        f"强制注入上游临时表到 selected_tables: "
                        f"{upstream_temp_names} → 最终 {state.selected_tables}"
                    )

        if state.selected_tables:
            orch_log.info(f"Planner 选定表: {state.selected_tables}")
        return False

    # ============================================================
    # Plan Persistence（方案二）
    # ============================================================

    async def _get_or_create_plan(
        self,
        state: ReActState,
        context: AgentContext,
        scratch_session_id: str,
        query: str,
        orch_log,
    ):
        """
        获取或创建分析计划（支持跨轮次复用）

        流程：
        1. 检查 scratch/{session_id}/plan.json 是否存在且有未完成任务
        2. 如果有 → 恢复计划，跳过 LLM 重新规划
        3. 如果没有 → 正常生成新计划并持久化
        """
        from chatdb.agents.planner import AnalysisPlan

        plan_path = str(self._scratch_pad.base_path / scratch_session_id / "plan.json")

        # 尝试加载持久化计划
        existing_plan_data = self._scratch_pad.load_plan(scratch_session_id)
        if existing_plan_data and existing_plan_data.get("status") != "completed":
            existing_plan = AnalysisPlan.from_dict(existing_plan_data)
            if existing_plan.has_pending_tasks():
                orch_log.info(
                    f"2. 恢复持久化计划 ({existing_plan.progress_summary().splitlines()[0]})"
                )
                # 注入到 Planner，让后续流程正常使用
                self.planner._analysis_plan = existing_plan
                orch_log.info(f"恢复的计划:\n{existing_plan.to_display()}")
                # 标记 state
                state.plan_resumed = True
                state.persistent_plan_path = plan_path
                return existing_plan

        # 正常生成新计划
        orch_log.info("2. Planner 生成分析计划...")
        plan = await self.planner.generate_analysis_plan(state, context)
        orch_log.info(f"分析计划:\n{plan.to_display()}")

        # 持久化新计划
        rewritten = getattr(state, "rewritten_query", query)
        plan.original_query = query
        plan.rewritten_query = rewritten if isinstance(rewritten, str) else query
        from datetime import datetime as _dt
        plan.created_at = _dt.now().isoformat()
        self._scratch_pad.save_plan(scratch_session_id, plan.to_dict())
        state.persistent_plan_path = plan_path

        return plan

    def _persist_task_completion(
        self,
        scratch_session_id: str,
        task_id: str,
        status: str = "completed",
        result_file: str = "",
    ) -> None:
        """将任务完成状态同步到持久化的 plan.json"""
        if scratch_session_id:
            self._scratch_pad.update_plan_task_status(
                scratch_session_id, task_id, status, result_file,
            )

    def _restore_completed_results(
        self,
        scratch_session_id: str,
        plan: AnalysisPlan,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        从磁盘恢复已完成任务的精简结果（Plan Persistence 恢复场景）

        遍历计划中已完成的任务，从 manifest / 结果文件重建 collected_results，
        使后续任务的 depends_on 能正确引用上游结果。
        """
        collected: dict[str, list[dict[str, Any]]] = {}
        
        for task in plan.tasks:
            if task.status != "completed":
                continue
            result_file = task.meta.get("result_file", "")
            if not result_file:
                continue
            full_data = self._scratch_pad.read_task_result(result_file)
            if not full_data:
                continue
            # 构建精简版结果（与 _store_response_to_scratch 格式一致）
            summary = self._scratch_pad._generate_summary(full_data)
            row_count = full_data.get("row_count", 0)
            all_examples = full_data.get("examples", [])
            # ★ 少量数据保留全部行，大量数据保留前 30 行
            inline_threshold = 30
            examples_slim = all_examples if row_count <= inline_threshold else all_examples[:inline_threshold]
            slim = {
                "subtask": full_data.get("subtask", ""),
                "sql": full_data.get("sql", ""),
                "row_count": row_count,
                "examples": examples_slim,
                "stats": full_data.get("stats", {}),
                "issues": full_data.get("issues", []),
                "_file_ref": {
                    "path": result_file,
                    "full_row_count": row_count,
                    "summary": summary,
                },
            }
            collected.setdefault(task.id, []).append(slim)
        
        return collected

    def _apply_adjustment(
        self,
        state: ReActState,
        context: AgentContext,
        adjustment: dict,
        orch_log,
    ) -> None:
        """
        应用 Planner 的调整策略
        
        adjustment 格式（由 LLM 动态生成）:
        - strategy: 调整策略描述
        - change: 具体改动描述
        """
        if not adjustment:
            return
        
        strategy = adjustment.get("strategy", "")
        change = adjustment.get("change", "")
        
        orch_log.info(f"调整策略: {strategy}")
        if change:
            orch_log.info(f"具体改动: {change}")
        
        # 将调整信息存入 state，让 SQLAgent 在下次执行时参考
        state.extra_context = f"[调整策略] {strategy}\n[具体改动] {change}"
    
    async def _init_state(self, query: str) -> ReActState:
        """初始化 State"""
        state = ReActState(user_query=query, max_steps=self.max_steps)
        return state
    
    async def _init_context(self, query: str, state: ReActState) -> AgentContext:
        """初始化 Context（精简版：数据写入 state，context 仅保留 API 兼容字段）"""
        if self.tables_meta:
            # ★ 即使外部传入 tables_meta，也要过滤掉临时表（防止上一轮遗留的 temp_ 表污染）
            tables_info = [t for t in self.tables_meta if not t.get("table_name", "").startswith("temp_")]
            for table in tables_info:
                columns = table.get("columns_info") or table.get("columns", [])
                state.available_columns.extend(columns)
        else:
            # ★ 清理历史遗留的临时表（避免干扰语义解析）
            await self._cleanup_legacy_temp_tables()
            
            # 每次查询实时从数据库反射 schema，确保与最新数据一致
            schema_info = await self.schema_inspector.get_schema_info()
            tables_info = get_tables_info(schema_info)
            
            # ★ 过滤掉临时表，只保留源表
            tables_info = [t for t in tables_info if not t.get("table_name", "").startswith("temp_")]
        
        # ★ 使用 ContextRetriever 召回最相关的表（而不是传递所有表给语义解析器）
        relevant_tables_info = await self._retrieve_relevant_tables(query, tables_info)
        
        # 构建精简的 schema_text（只包含召回的表）
        schema_text = self._build_schema_text_for_tables(relevant_tables_info or tables_info)
        
        state.schema_text = schema_text
        state.table_name = select_best_table(query, relevant_tables_info or tables_info)
        state.available_tables = tables_info  # 保留完整列表供后续使用

        # ===== 表理解文本（从 meta_data.db 缓存加载，或动态生成）=====
        state.table_understanding = await self._load_table_understanding(
            state.table_name, tables_info
        )

        # ===== 检索增强（Context Engineering）=====
        # 初次检索（原始 query），辅助 SemanticParser 识别意图
        # 语义解析后会用改写 query 重新检索，更新 Planner / SQLTool 的上下文
        if self._context_retriever.has_text_index or self._context_retriever.has_example_store:
            retrieval = self._context_retriever.retrieve_all(
                query, table_name=state.table_name,
            )
            if not retrieval.is_empty():
                # ★ 注入实时列描述统计，丰富 schema hint 输出
                await self._inject_column_stats(retrieval, state.table_name)
                state.retrieval_context = retrieval
        
        return AgentContext(
            user_query=query,
            schema_text=schema_text,
        )

    def _retrieve_virtual_fields(
        self, state: ReActState, query: str, orch_log,
    ) -> None:
        """
        在语义解析之前，将用户查询与 YML 虚拟字段做文本相似度检索。
        
        检索文本由 description + synonyms + expr 组成。
        召回的虚拟字段按得分降序排列，存入 state.retrieved_virtual_fields。
        """
        yml_config_dict = self._load_yml_config_dict()
        if not yml_config_dict or not yml_config_dict.get("virtual_fields"):
            return
        
        from chatdb.config.virtual_field import VirtualFieldRetriever
        retriever = VirtualFieldRetriever(yml_config_dict)
        result = retriever.retrieve(query)
        
        if not result.is_empty():
            state.retrieved_virtual_fields = result
            field_ids = [f.id for f in result.fields]
            orch_log.info(
                f"0.5 虚拟字段检索: {len(result.fields)} 个召回 — "
                f"{', '.join(field_ids[:8])}{'...' if len(field_ids) > 8 else ''}"
            )

    async def _cleanup_legacy_temp_tables(self) -> None:
        """清理历史遗留的临时表（避免干扰语义解析）"""
        try:
            # 查询所有临时表
            result = await self.db_connector.execute_query(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'temp_%'"
            )
            temp_tables = [row.get("table_name") for row in result if row.get("table_name")]
            
            if temp_tables:
                self._log.info(f"清理 {len(temp_tables)} 个历史临时表: {', '.join(temp_tables[:5])}{'...' if len(temp_tables) > 5 else ''}")
                for table_name in temp_tables:
                    try:
                        await self.db_connector.execute_query(f"DROP TABLE IF EXISTS {table_name}")
                    except Exception as e:
                        self._log.warn(f"删除临时表 {table_name} 失败: {e}")
        except Exception as e:
            self._log.warn(f"清理临时表失败: {e}")

    async def _retrieve_relevant_tables(
        self, query: str, tables_info: list[dict[str, Any]], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        使用 ContextRetriever 召回最相关的表
        
        如果没有索引或召回失败，返回所有表（兜底）
        """
        if not self._context_retriever.has_text_index:
            return tables_info
        
        try:
            # 使用 ContextRetriever 召回相关表
            retrieval = self._context_retriever.retrieve_schema_hints(
                query, table_name=None, top_k_tables=top_k, top_k_columns=0
            )
            
            if retrieval.relevant_tables:
                recalled_names = {t["table_name"] for t in retrieval.relevant_tables}
                relevant = [t for t in tables_info if t.get("table_name") in recalled_names]
                
                self._log.debug(
                    f"Schema 召回: {len(relevant)} 表 (from {len(tables_info)} total) - "
                    f"{', '.join(recalled_names)}"
                )
                return relevant
        except Exception as e:
            self._log.warn(f"表召回失败，使用全部表: {e}")
        
        return tables_info

    def _build_schema_text_for_tables(self, tables_info: list[dict[str, Any]]) -> str:
        """为指定的表列表构建 schema_text"""
        from chatdb.utils.common import build_schema_text
        return build_schema_text(tables_info)

    async def _inject_column_stats(
        self, retrieval: "RetrievalResult", table_name: str | None,
    ) -> None:
        """为 RetrievalResult 注入实时列描述统计，丰富 schema hint 输出。"""
        if not table_name:
            return
        try:
            from chatdb.database.duckdb.duckdb import DuckDBConnector
            if isinstance(self.db_connector, DuckDBConnector):
                stats = await self.db_connector.get_column_stats_async(table_name)
                if stats:
                    retrieval.column_stats_map[table_name] = stats
        except Exception:
            pass  # 获取失败不影响检索流程

    async def _load_table_understanding(
        self,
        table_name: str | None,
        tables_info: list[dict[str, Any]],
    ) -> str:
        """
        加载表理解文本。

        优先从 tables_meta 中取（如果调用方已传入）；
        否则从 meta_data.db 读取缓存；
        如果都没有，尝试用 LLM 动态生成并缓存。
        """
        if not table_name:
            return ""

        # 1. 尝试从 tables_info 中取（调用方传入的 tables_meta 可能已含此字段）
        target = next((t for t in tables_info if t.get("table_name") == table_name), None)
        if target and target.get("table_understanding"):
            return target["table_understanding"]

        # 2. 从 meta_data.db 缓存读取
        try:
            from chatdb.storage import MetaDataStore
            meta_store = MetaDataStore()
            meta = meta_store.get_by_table_name(table_name)
            if meta and meta.get("table_understanding"):
                return meta["table_understanding"]

            # 3. 缓存未命中 → 用 LLM 动态生成
            column_profiles = (
                (meta.get("column_profiles") if meta else None)
                or (target.get("column_profiles") if target else None)
                or []
            )
            table_description = (
                (meta.get("table_description") if meta else None)
                or (target.get("table_description") if target else None)
                or ""
            )
            row_count = (
                (meta.get("row_count") if meta else 0)
                or (target.get("row_count") if target else 0)
                or 0
            )
            column_count = (
                (meta.get("column_count") if meta else 0)
                or (target.get("column_count") if target else 0)
                or 0
            )

            if column_profiles:
                from chatdb.preprocessing.table_understanding import generate_table_understanding
                orch_log = get_component_logger("orchestrator")
                orch_log.info(f"表 '{table_name}' 无理解文本缓存，正在用 LLM 生成...")
                # 加载 yml_config 用于提供额外业务上下文
                yml_config_dict = self._load_yml_config_dict()
                understanding = await generate_table_understanding(
                    llm=self.llm,
                    table_name=table_name,
                    table_description=table_description,
                    row_count=row_count,
                    column_count=column_count,
                    column_profiles=column_profiles,
                    yml_config=yml_config_dict,
                )
                # 写回缓存
                if meta:
                    h = meta.get("content_hash") or meta.get("table_hash") or meta.get("file_hash")
                    if h:
                        source_type = meta.get("source_type", "excel")
                        meta_store.update_table_understanding(h, understanding, source_type)
                orch_log.info(f"表 '{table_name}' 理解文本已生成并缓存 ({len(understanding)} 字)")
                return understanding

        except Exception as e:
            orch_log = get_component_logger("orchestrator")
            orch_log.warning(f"加载表理解文本失败: {e}")

        return ""

    def _load_yml_config_dict(self) -> dict | None:
        """将 self.yml_config（可能是路径或 dict）加载为 dict，加载失败返回 None"""
        if not self.yml_config:
            return None
        if isinstance(self.yml_config, dict):
            return self.yml_config
        from pathlib import Path
        yml_path = Path(self.yml_config)
        if yml_path.exists():
            try:
                import yaml
                from chatdb.config.metrics_loader import preprocess_yaml_config
                with open(yml_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                return preprocess_yaml_config(raw) if raw else None
            except Exception:
                return None
        return None
    
    def invalidate_result_cache(self, session_id: str | None = None) -> int:
        """手动清除结果缓存"""
        return self._result_cache.invalidate(session_id)
    
    async def _quick_classify(self, query: str, chat_history: list | None) -> str:
        """
        前置轻量分类：判断问题是否和数据分析有关。
        
        在所有重操作（虚拟字段检索、语义解析）之前执行，
        用极短的 LLM 调用快速分流。
        
        Returns:
            "analysis" | "chat" | "ambiguous"
        """
        history_hint = ""
        if chat_history:
            last = chat_history[-1] if chat_history else None
            if last:
                q = (last.get("query") or "")[:60]
                a = (last.get("answer") or "")[:60]
                history_hint = f"\n上一轮对话: Q: {q} A: {a}"
        
        prompt = f"""判断用户问题的类型，只输出一个单词：
- analysis: 需要查询数据库、做数据分析（如流水、收入、趋势、排名、对比等）
- chat: 闲聊、常识问答、与数据分析无关（如天气、地理、打招呼等）
- ambiguous: 不确定，可能和数据有关也可能无关

用户问题: {query}{history_hint}

类型:"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="你是问题分类器。只输出 analysis/chat/ambiguous 其中一个单词，不要输出其他内容。",
                caller_name="quick_classify",
            )
            result = response.strip().lower().split()[0] if response else "analysis"
            return result if result in ("analysis", "chat", "ambiguous") else "analysis"
        except Exception:
            return "analysis"  # 分类失败时走正常分析流程
    
    async def _handle_chat_query(
        self, state: ReActState, context: AgentContext, orch_log,
    ) -> ReActState:
        """
        处理纯闲聊/常识问题，极简路径。
        
        跳过虚拟字段检索、语义解析、Planner 等全部重流程，
        直接用 LLM 回复。
        """
        orch_log.info(f"闲聊快速响应: {state.user_query[:50]}...")
        
        history_context = HistoryHelper.format_history_for_prompt(context.chat_history)
        prompt = f"""用户问题: {state.user_query}
{history_context}
请回答用户的问题。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt=(
                    "你是 ChatDB 数据分析助手。友好地回答用户问题。"
                    "如果问题和数据分析无关，正常回答即可，"
                    "可以自然地提及你的数据分析能力，但不要强行引导。"
                ),
                caller_name="chat_response",
            )
            state.summary = response
            state.phase = ReActPhase.DONE
        except Exception as e:
            orch_log.error(f"闲聊响应失败: {e}")
            state.summary = f"抱歉，处理时出错了: {e}"
        
        return state

    async def _handle_other_query(self, state: ReActState, context: AgentContext, orch_log) -> ReActState:
        """
        处理非数据分析请求（mode=other）
        
        SemanticParser 已判断这不是数据分析请求，直接让 LLM 响应。
        other_request 字段包含了 SemanticParser 对用户意图的理解。
        """
        other_request = state.intent.other_request if state.intent else None
        orch_log.info(f"非数据分析请求: {other_request or state.user_query[:50]}...")
        
        # 构建上下文：始终提供数据能力说明，让 LLM 知道自己能做什么
        data_context = self._build_data_context(context, state)
        history_context = HistoryHelper.format_history_for_prompt(context.chat_history)
        
        prompt = f"""用户问题: {state.user_query}
{f"用户意图: {other_request}" if other_request else ""}

{history_context}{data_context}

请回应用户。"""

        try:
            response = await self.llm.chat(
                prompt=prompt,
                system_prompt="""你是 ChatDB 数据分析助手。
你的核心能力是：将自然语言问题转换为 SQL 查询，分析数据并给出洞察。
对于非数据分析问题，自然友好地回应，并在合适时引导用户使用你的数据分析能力。""",
                caller_name="meta_query",
            )
            state.summary = response
            state.phase = ReActPhase.DONE
        except Exception as e:
            orch_log.error(f"处理失败: {e}")
            state.summary = f"抱歉，处理时出错了: {e}"
        
        return state
    
    def _build_data_context(self, context: AgentContext, state: ReActState) -> str:
        """构建数据上下文，让 LLM 了解可用的数据能力"""
        import yaml
        
        lines = ["## 当前数据环境"]
        
        # 表信息
        if state.table_name:
            lines.append(f"- 当前数据表: {state.table_name}")
        
        # Schema 摘要
        if state.schema_text:
            schema_brief = state.schema_text[:500]
            lines.append(f"- 表结构摘要:\n{schema_brief}")
        
        # YAML 业务配置
        if self.yml_config:
            yml_config_dict = self._load_yml_config_dict()
            
            if yml_config_dict:
                metrics = list(yml_config_dict.get("metrics", {}).keys())[:8]
                dims = list(yml_config_dict.get("dimensions", {}).keys())[:8]
                if metrics:
                    lines.append(f"- 可分析指标: {', '.join(metrics)}")
                if dims:
                    lines.append(f"- 可用维度: {', '.join(dims)}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    async def _generate_summary(self, state: ReActState, query: str) -> ReActState:
        """生成总结（统一走多阶段汇总路径）"""
        
        # 如果有 temp_results，使用多阶段汇总
        if state.temp_results:
            summary_context = self._build_summary_context(state)
            state.extra_context = summary_context
            await self._summarize_tool(state, AgentContext(user_query=query))
            return state
        
        # 空结果处理
        rows = state.execute_result.get("rows", []) if state.execute_result else []
        if not rows or len(rows) == 0:
            diagnosis = state.error_context.get("no_data_diagnosis", {})
            guidance = state.error_context.get("user_guidance", "")
            conclusion = diagnosis.get("conclusion", "") if diagnosis else ""
            if conclusion or guidance:
                state.summary = "查询未返回结果。"
                if conclusion:
                    state.summary += f"\n原因：{conclusion}"
                if guidance:
                    state.summary += f"\n\n下一步建议：\n{guidance}"
            else:
                state.summary = f"在表『{state.table_name}』中，按当前筛选条件未查询到数据。"
            return state
        
        # 单结果场景：构建简易 summary context，统一走 summarize 路径
        from chatdb.utils.common import format_rows
        lines = ["## 多阶段分析结果汇总\n"]
        lines.append(f"用户问题: {state.user_query}\n")
        
        # ★ 注入指标单位信息
        unit_section = self._build_metric_unit_section(state)
        if unit_section:
            lines.append(unit_section)
            lines.append("")
        
        lines.append("### 阶段: query")
        if state.final_sql or state.current_sql:
            sql = state.final_sql or state.current_sql
            lines.append(f"执行SQL:\n```sql\n{sql}\n```")
        lines.append(f"返回行数: {len(rows)}")
        lines.append("查询结果:")
        for ex in rows[:30]:
            ex_str = ", ".join(f"{k}={v}" for k, v in list(ex.items()))
            lines.append(f"  - {ex_str}")
        if len(rows) > 30:
            lines.append(f"  ... 共 {len(rows)} 行，已展示前 30 行")
        
        state.extra_context = "\n".join(lines)
        await self._summarize_tool(state, AgentContext(user_query=query))
        return state
    
    def _build_summary_context(self, state: ReActState) -> str:
        """
        从 temp_results + 分析计划 构建多阶段汇总上下文
        
        包含每个任务的：描述、SQL、查询结果，让 LLM 能智能综合所有分支结果。
        """
        # 展开文件引用为完整数据
        expanded = self._scratch_pad.expand_results(state.temp_results)
        
        # 获取计划中的任务描述信息
        task_descriptions: dict[str, str] = {}
        if self.planner.analysis_plan:
            for t in self.planner.analysis_plan.tasks:
                task_descriptions[t.id] = t.description
        
        lines = ["## 多阶段分析结果汇总\n"]
        lines.append(f"用户问题: {state.user_query}\n")
        
        # 改写后的问题（如有）
        rewritten = getattr(state, "rewritten_query", "")
        if rewritten and rewritten != state.user_query:
            lines.append(f"改写后的问题: {rewritten}\n")
        
        # ★ 注入指标单位信息，帮助 LLM 正确理解原始数据的量纲
        unit_section = self._build_metric_unit_section(state)
        if unit_section:
            lines.append(unit_section)
            lines.append("")
        
        for task_id, results in expanded.items():
            desc = task_descriptions.get(task_id, "")
            lines.append(f"### 阶段: {task_id}")
            if desc:
                lines.append(f"任务描述: {desc}")
            
            for i, r in enumerate(results):
                subtask = r.get("subtask", "")
                row_count = r.get("row_count", 0)
                sql = r.get("sql", "")
                examples = r.get("examples", [])
                stats = r.get("stats", {})
                issues = r.get("issues", [])
                
                if subtask:
                    lines.append(f"子任务: {subtask}")
                
                if sql:
                    lines.append(f"执行SQL:\n```sql\n{sql}\n```")
                
                lines.append(f"返回行数: {row_count}")
                
                if examples:
                    lines.append("查询结果:")
                    for ex in examples[:30]:
                        ex_str = ", ".join(f"{k}={v}" for k, v in list(ex.items()))
                        lines.append(f"  - {ex_str}")
                    if len(examples) > 30:
                        lines.append(f"  ... 共 {len(examples)} 行，已展示前 30 行")
                
                if stats:
                    stats_str = ", ".join(f"{k}={v}" for k, v in stats.items())
                    lines.append(f"统计: {stats_str}")
                
                if issues:
                    lines.append(f"备注: {', '.join(issues)}")
            
            lines.append("")
        
        # ★ 如果 Planner 有结论，作为参考信息注入
        if state.planner_conclusion:
            lines.append("### Planner 分析结论（供参考）")
            lines.append(state.planner_conclusion)
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_metric_unit_section(self, state: ReActState) -> str:
        """构建指标单位说明段，注入到 summary context 中。
        
        从 yml_config 的 virtual_fields 中提取 metric 类型字段的 unit 和 display_divisor，
        告诉 summarize LLM：SQL 查询返回的原始数值是什么单位，展示时应该如何换算。
        """
        yml_config = state.yml_config
        if not yml_config:
            return ""
        
        virtual_fields = yml_config.get("virtual_fields", {})
        if not virtual_fields:
            return ""
        
        # 从 virtual_fields 提取 metric 类型
        unit_lines = []
        for fid, fdef in virtual_fields.items():
            if not isinstance(fdef, dict) or fdef.get("field_type") != "metric":
                continue
            unit = fdef.get("unit", "")
            display_divisor = fdef.get("display_divisor")
            if not unit and not display_divisor:
                continue
            desc = fdef.get("description", fid)
            parts = [f"- {desc}（{fid}）: "]
            if display_divisor:
                parts.append(f"SQL原始值单位=元, 展示单位={unit}, 换算除数={display_divisor}")
            elif unit:
                parts.append(f"展示单位={unit}")
            unit_lines.append("".join(parts))
        
        if not unit_lines:
            return ""
        
        return "### ★ 指标单位说明\n" + (
            "以下指标在业务配置中定义了展示单位和换算规则。\n"
            "**SQL 返回的原始数值是未经换算的原始值**，回答时必须按 display_divisor 换算后再展示：\n"
        ) + "\n".join(unit_lines)
    
    def _build_result(self, state: ReActState, query: str, start_time: float) -> dict[str, Any]:
        """构建返回结果（统一响应结构）"""
        duration_ms = (time.time() - start_time) * 1000
        success = state.phase == ReActPhase.DONE and (
            state.has_result or bool(state.analysis_slices) or bool(state.temp_results)
        )
        
        rewritten = getattr(state, "rewritten_query", None)
        result: dict[str, Any] = {
            "success": success,
            "query": query,
            "rewritten_query": rewritten if rewritten and rewritten != query else None,
            "sql": state.final_sql or state.current_sql,
            "result": state.execute_result.get("rows", []) if state.execute_result else [],
            "row_count": state.execute_result.get("row_count", 0) if state.execute_result else 0,
            "summary": state.summary,
            "table_name": state.table_name,
            "intent": state.intent.to_dict() if state.intent and hasattr(state.intent, 'to_dict') else None,
        }
        
        # ★ 统一 status 字段：completed / need_clarification
        if state.intervention:
            intervention = state.intervention
            result["status"] = "need_clarification"
            result["clarification_request"] = {
                "reason": intervention.get("reason", ""),
                "question": intervention.get("question", ""),
                "options": intervention.get("options", []),
                "free_input_allowed": intervention.get("free_input_allowed", True),
            }
            result["success"] = False
            result["run_context"] = {
                "step_id": state.intervention_step_id,
                "plan_state": "paused",
            }
        else:
            result["status"] = "completed"
        
        # ★ research_mode 下解析结构化研究结论
        if state.research_mode and state.summary:
            parsed = self._try_parse_research_json(state.summary)
            if parsed:
                result["confidence"] = parsed.get("confidence")
                result["key_findings"] = parsed.get("key_findings", [])
                result["limitations"] = parsed.get("limitations", [])
                result["suggested_follow_ups"] = parsed.get("suggested_follow_ups", [])
        
        # 添加 temp_results（展开文件引用为完整数据）
        if state.temp_results:
            result["temp_results"] = self._scratch_pad.expand_results(state.temp_results)
        
        if state.analysis_slices:
            result["analysis_slices"] = [s.to_dict() for s in state.analysis_slices]
            result["analysis_results"] = [s.to_dict() for s in state.analysis_slices]
            result["analysis_summary"] = state.get_analysis_summary()
        
        if not result["success"] and not state.intervention:
            result["error"] = state.error
        
        if self.debug:
            debug_info = state.get_debug_info()
            debug_info["duration_ms"] = duration_ms
            debug_info["steps"] = state.step
            debug_info["explored_dimensions"] = list(state.explored_dimensions)
            debug_info["temp_results_keys"] = list(state.temp_results.keys()) if state.temp_results else []
            if debug_info.get("reasoning_trace"):
                logger.debug("[AgentOrchestrator] %s", debug_info["reasoning_trace"])
            result["debug"] = debug_info

        return result
    
    @staticmethod
    def _try_parse_research_json(summary: str) -> dict[str, Any] | None:
        """尝试从 summary 末尾提取 research 结构化 JSON"""
        import re as _re
        match = _re.search(r'```json\s*(\{.*?\})\s*```', summary, _re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        return None


# 便捷函数
async def run_query(
    query: str,
    llm: BaseLLM,
    db_connector: BaseDatabaseConnector,
    yml_config: Optional[Union[str, Path]] = None,
    tables_meta: Optional[list[dict[str, Any]]] = None,
    debug: bool = False,
    max_steps: int = 10,
    session_id: Optional[str] = None,
    history_db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """
    运行查询
    
    Args:
        query: 用户查询
        llm: LLM 实例
        db_connector: 数据库连接器
        yml_config: YAML 配置文件路径（提供后自动启用领域分析能力）
        tables_meta: 表元数据
        debug: 是否启用调试模式
        max_steps: 最大循环步数
        session_id: 会话 ID（启用多轮对话记忆）
        history_db_path: 历史数据库路径（启用持久化记忆）
    
    Returns:
        查询结果
    
    Example:
        # 带领域配置（推荐）
        result = await run_query(
            "王者荣耀流水增长来自哪里",
            llm, db,
            yml_config="data/yml/ieg.yml",
        )
        
        # 多轮对话模式
        result = await run_query(
            "再查和平精英对比",
            llm, db,
            yml_config="data/yml/ieg.yml",
            session_id="my-session-1",
            history_db_path="data/pilot/history.db",
        )
    """
    orchestrator = AgentOrchestrator(
        llm=llm,
        db_connector=db_connector,
        yml_config=yml_config,
        tables_meta=tables_meta,
        debug=debug,
        max_steps=max_steps,
        history_db_path=history_db_path,
    )
    return await orchestrator.process_query(query, session_id=session_id)
