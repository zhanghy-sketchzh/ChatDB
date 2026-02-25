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
import hashlib
import time
import uuid

from chatdb.agents.base import AgentContext, AgentStatus
from chatdb.core.messages import TaskRequest, TaskResponse
from chatdb.core.react_state import ReActState, ReActPhase, ErrorType
from chatdb.core.scratch_pad import ScratchPadManager
from chatdb.agents.planner import AnalysisPlan, PlannerAgent
from chatdb.agents.sql_agent import SQLAgent
from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.schema import SchemaInspector
from chatdb.llm.base import BaseLLM
from chatdb.storage.chat_history import ChatHistoryDB, ChatHistoryManager, HistoryConfig
from chatdb.tools import ToolRegistry, UnixTool
from chatdb.core.semantic_parse import SemanticParseTool
from chatdb.core.summarize import SummarizeAnswerTool
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
    ):
        self.llm = llm
        self.db_connector = db_connector
        self.schema_inspector = SchemaInspector(db_connector)
        self.yml_config = yml_config
        self.tables_meta = tables_meta
        self.debug = debug
        self.max_steps = max_steps
        
        if debug:
            set_log_level_to_debug()
            enable_llm_debug(True, show_input=True)  # debug 显示完整输入输出
        
        # 初始化 ToolRegistry
        self.registry = ToolRegistry()
        
        # SQLAgent（核心 Agent，接受 yml_config）
        self._sql_agent = SQLAgent(self.llm, self.db_connector, yml_config)
        
        # 语义解析 Tool（前置 workflow）
        self._semantic_parse_tool = SemanticParseTool(llm, yml_config)
        
        # 总结 Tool
        self._summarize_tool = SummarizeAnswerTool(llm)
        
        # 文件系统工具（供 Planner 观察 scratch 数据）
        self._unix_tool = UnixTool(workspace=".", readonly=True)

        # Planner（生成分析型 ToDo，持有 UnixTool 观察结果）
        self.planner = PlannerAgent(llm, unix_tool=self._unix_tool)
        
        # ===== 会话记忆 =====
        self._history_manager: ChatHistoryManager | None = None
        if history_db_path:
            db = ChatHistoryDB(history_db_path)
            self._history_manager = ChatHistoryManager(
                db, history_config or HistoryConfig()
            )
            self._history_manager.set_agent("orchestrator")
        
        # ===== Scratch Pad（文件暂存）=====
        self._scratch_pad = ScratchPadManager(base_path="data/scratch")
        
        # ===== Schema 缓存 =====
        self._schema_cache: dict[str, Any] | None = None
        
        # ===== 查询结果缓存 =====
        # key: (session_id, query_hash) → value: {"result": dict, "timestamp": float}
        self._result_cache: dict[str, dict[str, Any]] = {}
        self._cache_max_size: int = 50  # 最大缓存条目
        self._cache_ttl: float = 300.0  # 缓存过期时间（秒）
    
    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        use_cache: bool = True,
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
        
        流程：
        1. [前置] 检查结果缓存
        2. [前置] 加载会话历史
        3. [前置] 语义解析（不属于 Planner 调度）
        4. Planner 生成分析计划
        5. SQLAgent 按计划执行
        6. 生成总结
        7. [后置] 保存本轮结果到历史 + 写入缓存
        """
        start_time = time.time()
        
        # Task View: 开始任务
        task_log.start(query)
        orch_log = get_component_logger("Orchestrator")
        
        # ===== 结果缓存：命中检查 =====
        if use_cache:
            cached = self._get_cached_result(session_id, query)
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
        
        # ===== 会话记忆：加载历史 =====
        chat_history = self._load_chat_history(session_id, orch_log)
        if chat_history:
            context.chat_history = chat_history
            context.session_id = session_id
        
        try:
            # ============================================================
            # 1. [前置 workflow] 语义解析 + 查询改写（合并为一次 LLM 调用）
            # SemanticParser 同时完成：指代消解、问题补全、意图提取
            # ============================================================
            orch_log.info("1. [前置] 语义解析...")
            await self._semantic_parse_tool(state, context)
            
            # 取出改写后的查询，替换后续流程中的 user_query
            if state.intent and state.intent.rewritten_query and state.intent.rewritten_query != query:
                rewritten = state.intent.rewritten_query
                orch_log.info(f"查询改写: {query} → {rewritten}")
                state.user_query = rewritten
                context.user_query = rewritten
                state.rewritten_query = rewritten  # type: ignore[attr-defined]
            
            if state.intent:
                # 简化：使用 intent_type 属性（由 Planner 设置分析模式后才有意义）
                task_log.intent(
                    intent_type=state.intent.intent_type,
                    metrics=state.intent.metrics or [],
                    dimensions=state.intent.dimensions or [],
                    filters=state.intent.filter_refs or [],
                )
            else:
                orch_log.warn("语义解析未返回 Intent，继续执行")
            
            # ============================================================
            # 1.5 检查是否为非数据分析请求
            # 使用 is_other_query() 方法判断
            # ============================================================
            if state.intent and state.intent.is_other_query():
                orch_log.info("检测到非数据分析请求（mode=other），直接生成响应...")
                state = await self._handle_other_query(state, context, orch_log)
                if state.summary:
                    task_log.done(state.summary)
                result = self._build_result(state, query, start_time)
                self._save_to_history(session_id, query, state.summary or "", state)
                self._cache_result(session_id, query, result)
                return result
            
            # ============================================================
            # 2. Planner 生成分析计划（或恢复持久化计划）
            # ============================================================
            plan = await self._get_or_create_plan(
                state, context, scratch_session_id, query, orch_log,
            )
            orch_log.info(f"分析计划:\n{plan.to_display()}")
            
            # ============================================================
            # 3. 按计划执行 SQLAgent
            # ============================================================
            orch_log.info(f"3. SQL 分析流程... (Agent: {self._sql_agent.display_name})")
            await self._execute_plan(state, context, orch_log, scratch_session_id)
            
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
            if not state.summary and (state.has_result or state.error_type == ErrorType.NO_DATA):
                state = await self._generate_summary(state, query)
            
            # 标记完成
            if state.has_result or state.summary:
                state.phase = ReActPhase.DONE
            
            # Task View: 完成
            if state.summary:
                task_log.done(state.summary)
            
            result = self._build_result(state, query, start_time)
            self._save_to_history(session_id, query, state.summary or "", state)
            self._cache_result(session_id, query, result)
            return result
            
        except Exception as e:
            logger.error(f"[Orchestrator] 处理失败: {e}")
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
        
        Agent 之间不直接共享可变状态，由 Orchestrator 路由消息。
        大数据通过文件传递，prompt 只传摘要和文件路径。
        
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
            current_step = state.inc_plan_step()
            
            # 1. 获取当前任务
            current_task = self.planner.get_current_task(collected_results)
            if not current_task:
                orch_log.info("没有更多任务")
                break
            
            task_dict = current_task.to_dict()
            task_id = task_dict.get("id", "")
            task_type = task_dict.get("type", "")
            
            # 2. 死循环检测
            if state.mark_task_repeat(task_id):
                orch_log.warn(f"任务 {task_id} 重复执行，强制跳过")
                self.planner.advance_plan(collected_results)
                continue
            
            # 3. validation 任务数量限制
            if task_type == "validation":
                if not state.bump_validation():
                    orch_log.warn("validation 任务达到上限，跳过更多诊断")
                    self.planner.advance_plan(collected_results)
                    continue
            
            orch_log.info(f"执行任务 {current_step}: [{task_type}] {task_dict['description'][:50]}...")
            
            # 4. summary 任务由 Orchestrator 处理
            if task_type == "summary":
                self.planner.advance_plan(collected_results)
                continue
            
            # 5. 构建 TaskRequest 消息（Orchestrator → SQLAgent）
            request = self._build_task_request(task_dict, collected_results)
            
            try:
                # 6. SQLAgent 执行任务，返回 TaskResponse 消息
                response = await self._sql_agent.run_task(state, context, request)
                
                # 7. Orchestrator 写入 Scratch Pad 文件 + 存储精简结果
                self._store_response_to_scratch(
                    collected_results, response,
                    scratch_session_id,
                )
                
                # 同步到 state.temp_results（兼容旧接口）
                state.temp_results = collected_results
                
                # 8. 将结果传给 Planner 查看（基于摘要，不传全量数据）
                temp_summary = await self.planner.inspect_temp_results(state, collected_results)
                if temp_summary:
                    orch_log.debug(f"temp_results 摘要:\n{temp_summary[:200]}...")
                
                # 标记当前任务完成
                self.planner.advance_plan(collected_results)
                
                # 持久化任务完成状态到 plan.json
                result_file = ""
                for r in collected_results.get(task_id, []):
                    ref = r.get("_file_ref", {})
                    if ref.get("path"):
                        result_file = ref["path"]
                        break
                self._persist_task_completion(
                    scratch_session_id, task_id, "completed", result_file,
                )
                
                # 9. Planner 决策（接收精简版 collected_results）
                decision = await self.planner.decide_next_action(state, context, collected_results)
                if await self._handle_planner_decision(state, context, decision, orch_log):
                    break
                
            except Exception as e:
                orch_log.warn(f"任务执行失败: {e}")
                self.planner.mark_task_failed(str(e), collected_results)
                self.planner.advance_plan(collected_results)
                self._persist_task_completion(
                    scratch_session_id, task_id, "failed",
                )
        
        # 最终同步
        state.temp_results = collected_results
        
        if collected_results:
            orch_log.info(f"完成 {len(collected_results)} 个任务的数据收集")

    def _build_task_request(
        self,
        task_dict: dict[str, Any],
        collected_results: dict[str, list[dict[str, Any]]],
    ) -> TaskRequest:
        """
        构建 TaskRequest 消息（包含上游结果摘要 + 文件引用）
        
        Scratch Pad 模式下：
        - parent_results_summary 包含摘要文本 + 文件路径
        - previous_results 传精简版（含 _file_ref），SQLAgent 按需读取
        """
        depends_on = task_dict.get("depends_on", [])
        
        # 为依赖任务构建结果摘要
        parent_summary_parts = []
        all_previous = []
        for dep_id in depends_on:
            for r in collected_results.get(dep_id, []):
                stats = r.get("stats", {})
                file_ref = r.get("_file_ref")
                
                if stats.get("available_years"):
                    parent_summary_parts.append(f"可用年份: {stats['available_years']}")
                if stats.get("top_contributor"):
                    parent_summary_parts.append(f"top贡献: {stats['top_contributor']}")
                if r.get("row_count"):
                    parent_summary_parts.append(f"上游返回 {r['row_count']} 行")
                
                # 添加文件引用信息到摘要
                if file_ref:
                    summary = file_ref.get("summary", "")
                    file_path = file_ref.get("path", "")
                    if summary:
                        parent_summary_parts.append(f"[{dep_id}] {summary}")
                    if file_path:
                        parent_summary_parts.append(f"完整数据: {file_path}")
                
                all_previous.append(r)
        
        # 如果没有显式依赖，收集所有之前的结果（用于 drilldown 等）
        if not depends_on:
            task_id = task_dict.get("id", "")
            for tid, results in collected_results.items():
                if tid != task_id:
                    all_previous.extend(results)
        
        return TaskRequest.from_planner_task(
            task_dict,
            parent_results_summary="; ".join(parent_summary_parts),
            previous_results=all_previous,
        )

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

    def _store_response_to_scratch(
        self,
        collected_results: dict[str, list[dict[str, Any]]],
        response: TaskResponse,
        scratch_session_id: str,
    ) -> None:
        """
        将 TaskResponse 写入 Scratch Pad 文件，collected_results 只存精简版
        
        精简版包含：摘要 + 文件路径 + 统计 + 样例（前5行）
        完整数据通过文件引用按需读取
        """
        task_id = response.task_id
        if task_id not in collected_results:
            collected_results[task_id] = []
        
        for result_entry in response.to_results_dicts():
            # 通过 ScratchPadManager 写入文件并获取精简版
            slim_result = self._scratch_pad.save_task_result(
                session_id=scratch_session_id,
                task_id=task_id,
                result_entry=result_entry,
            )
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
                orch_log.info(f"Planner 结论: {conclusion[:100]}...")
                state.summary = conclusion
            return True
        
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
            slim = {
                "subtask": full_data.get("subtask", ""),
                "sql": full_data.get("sql", ""),
                "row_count": full_data.get("row_count", 0),
                "examples": full_data.get("examples", [])[:5],
                "stats": full_data.get("stats", {}),
                "issues": full_data.get("issues", []),
                "_file_ref": {
                    "path": result_file,
                    "full_row_count": full_data.get("row_count", 0),
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
            tables_info = self.tables_meta
            schema_text = build_schema_text(self.tables_meta)
            for table in self.tables_meta:
                columns = table.get("columns_info") or table.get("columns", [])
                state.available_columns.extend(columns)
        elif self._schema_cache:
            # 命中缓存，避免重复反射
            tables_info = self._schema_cache["tables_info"]
            schema_text = self._schema_cache["schema_text"]
        else:
            schema_info = await self.schema_inspector.get_schema_info()
            tables_info = get_tables_info(schema_info)
            schema_text = schema_info.to_prompt_text()
            # 缓存 schema（同一 Orchestrator 实例内复用）
            self._schema_cache = {
                "tables_info": tables_info,
                "schema_text": schema_text,
            }
        
        state.schema_text = schema_text
        state.table_name = select_best_table(query, tables_info)
        state.available_tables = tables_info
        
        return AgentContext(
            user_query=query,
            schema_text=schema_text,
        )
    
    # ===== 会话记忆 =====
    
    def _load_chat_history(
        self,
        session_id: str | None,
        orch_log,
    ) -> list[dict[str, str]]:
        """加载会话历史，返回 chat 格式的历史列表"""
        if not self._history_manager or not session_id:
            return []
        
        self._history_manager.start_session(session_id)
        history = self._history_manager.get_history_as_chat_format()
        if history:
            orch_log.info(f"已加载 {len(history) // 2} 轮历史对话 (session={session_id[:8]}...)")
        return history
    
    def _save_to_history(
        self,
        session_id: str | None,
        query: str,
        summary: str,
        state: ReActState,
    ) -> None:
        """将本轮结果保存到会话历史"""
        if not self._history_manager or not session_id:
            return
        
        # 构建 metadata：保存关键中间结果供后续轮次参考
        metadata: dict[str, Any] = {}
        if state.intent and hasattr(state.intent, "to_dict"):
            metadata["intent"] = state.intent.to_dict()
        if state.table_name:
            metadata["table_name"] = state.table_name
        if state.final_sql or state.current_sql:
            metadata["sql"] = state.final_sql or state.current_sql
        if state.temp_results:
            metadata["task_results_summary"] = {
                tid: len(results) for tid, results in state.temp_results.items()
            }
        
        # 构建 assistant_output：summary + 关键结果数据 + SQL
        # 关键：将查询结果中的关键值写入 output，供后续轮次指代消解使用
        output_parts = []
        if summary:
            output_parts.append(summary)
        
        # 附加结构化结果摘要：让后续轮次能精确引用本轮的查询结果值
        result_data_brief = self._extract_result_data_brief(state)
        if result_data_brief:
            output_parts.append(result_data_brief)
        
        sql = state.final_sql or state.current_sql
        if sql:
            output_parts.append(f"[SQL] {sql}")
        
        self._history_manager.add_interaction(
            user_input=query,
            assistant_output="\n".join(output_parts) if output_parts else "(无结果)",
            metadata=metadata,
        )
    
    @staticmethod
    def _extract_result_data_brief(state: ReActState) -> str:
        """
        从查询结果中提取关键数据摘要，用于写入历史记录。
        
        目的：后续轮次通过历史上下文能精确知道上轮查询返回了哪些具体值，
        从而正确理解"这几个产品"、"上面那个"等指代。
        """
        rows = state.execute_result.get("rows", []) if state.execute_result else []
        if not rows:
            return ""
        
        # 最多展示前 10 行的关键字段值
        brief_rows = rows[:10]
        parts = ["[查询结果数据]"]
        for i, row in enumerate(brief_rows, 1):
            # 每行最多展示前 5 个字段
            items = list(row.items())[:5]
            row_str = ", ".join(f"{k}={v}" for k, v in items)
            parts.append(f"  {i}. {row_str}")
        
        if len(rows) > 10:
            parts.append(f"  ...共 {len(rows)} 行")
        
        return "\n".join(parts)
    
    def invalidate_schema_cache(self) -> None:
        """手动使 Schema 缓存失效（当数据库结构变更时调用）"""
        self._schema_cache = None
    
    # ===== 查询结果缓存 =====
    
    @staticmethod
    def _query_cache_key(session_id: str | None, query: str) -> str:
        """生成查询缓存 key: session_id + query 的哈希"""
        raw = f"{session_id or ''}:{query.strip().lower()}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def _get_cached_result(self, session_id: str | None, query: str) -> dict[str, Any] | None:
        """获取缓存的查询结果，返回 None 表示缓存未命中"""
        key = self._query_cache_key(session_id, query)
        entry = self._result_cache.get(key)
        if not entry:
            return None
        # 检查 TTL
        if time.time() - entry["timestamp"] > self._cache_ttl:
            del self._result_cache[key]
            return None
        return entry["result"]
    
    def _cache_result(self, session_id: str | None, query: str, result: dict[str, Any]) -> None:
        """缓存查询结果"""
        # 仅缓存成功的结果
        if not result.get("success"):
            return
        key = self._query_cache_key(session_id, query)
        # 淘汰最早条目（超出最大缓存）
        if len(self._result_cache) >= self._cache_max_size:
            oldest_key = min(self._result_cache, key=lambda k: self._result_cache[k]["timestamp"])
            del self._result_cache[oldest_key]
        self._result_cache[key] = {
            "result": result,
            "timestamp": time.time(),
        }
    
    def invalidate_result_cache(self, session_id: str | None = None) -> int:
        """
        手动清除结果缓存
        
        Args:
            session_id: 指定时仅清除该会话的缓存，不指定则清除全部
        
        Returns:
            被清除的缓存条目数
        """
        if session_id is None:
            count = len(self._result_cache)
            self._result_cache.clear()
            return count
        # 按 session_id 前缀匹配清除（key 是 hash 无法精确匹配，直接全清）
        count = len(self._result_cache)
        self._result_cache.clear()
        return count
    
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
        history_context = self._format_history_for_prompt(context.chat_history)
        
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
            yml_config_dict = None
            if isinstance(self.yml_config, (str, Path)):
                try:
                    yml_path = Path(self.yml_config)
                    if yml_path.exists():
                        with open(yml_path, "r", encoding="utf-8") as f:
                            yml_config_dict = yaml.safe_load(f)
                except Exception:
                    pass
            elif isinstance(self.yml_config, dict):
                yml_config_dict = self.yml_config
            
            if yml_config_dict:
                metrics = list(yml_config_dict.get("metrics", {}).keys())[:8]
                dims = list(yml_config_dict.get("dimensions", {}).keys())[:8]
                if metrics:
                    lines.append(f"- 可分析指标: {', '.join(metrics)}")
                if dims:
                    lines.append(f"- 可用维度: {', '.join(dims)}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    @staticmethod
    def _format_history_for_prompt(chat_history: list[dict[str, str]]) -> str:
        """将 chat_history 格式化为 prompt 注入文本"""
        if not chat_history:
            return ""
        
        lines = ["## 历史对话\n"]
        for msg in chat_history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
            lines.append("")
        
        return "\n".join(lines) + "\n"
    
    async def _generate_summary(self, state: ReActState, query: str) -> ReActState:
        """生成总结（利用 temp_results）"""
        rows = state.execute_result.get("rows", []) if state.execute_result else []
        
        # 如果有 temp_results，优先使用其进行总结
        if state.temp_results:
            summary_context = self._build_summary_context(state)
            state.extra_context = summary_context
            await self._summarize_tool(state, AgentContext(user_query=query))
            return state
        
        # 空结果处理
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
        
        # 使用 summarize Tool
        await self._summarize_tool(state, AgentContext(user_query=query))
        return state
    
    def _build_summary_context(self, state: ReActState) -> str:
        """
        从 temp_results 构建总结上下文
        
        Scratch Pad 模式下：从文件读取完整数据用于生成总结
        （总结需要完整数据以确保准确性）
        """
        # 展开文件引用为完整数据
        expanded = self._scratch_pad.expand_results(state.temp_results)
        
        lines = ["## 分析结果汇总\n"]
        
        for task_id, results in expanded.items():
            lines.append(f"### {task_id}")
            for r in results:
                subtask = r.get("subtask", "")
                row_count = r.get("row_count", 0)
                examples = r.get("examples", [])
                stats = r.get("stats", {})
                issues = r.get("issues", [])
                
                lines.append(f"- 子任务: {subtask}, 行数: {row_count}")
                
                if examples:
                    lines.append("  示例数据:")
                    for ex in examples[:5]:
                        ex_str = ", ".join(f"{k}={v}" for k, v in list(ex.items())[:4])
                        lines.append(f"    {ex_str}")
                
                if stats:
                    stats_str = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:5])
                    lines.append(f"  统计: {stats_str}")
                
                if issues:
                    lines.append(f"  注意: {', '.join(issues)}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_result(self, state: ReActState, query: str, start_time: float) -> dict[str, Any]:
        """构建返回结果"""
        duration_ms = (time.time() - start_time) * 1000
        success = state.phase == ReActPhase.DONE and (
            state.has_result or bool(state.analysis_slices) or bool(state.temp_results)
        )
        
        rewritten = getattr(state, "rewritten_query", None)
        result = {
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
        
        # 添加 temp_results（展开文件引用为完整数据）
        if state.temp_results:
            result["temp_results"] = self._scratch_pad.expand_results(state.temp_results)
        
        if state.analysis_slices:
            result["analysis_slices"] = [s.to_dict() for s in state.analysis_slices]
            result["analysis_results"] = state.analysis_results
            result["analysis_summary"] = state.get_analysis_summary()
        
        if not success:
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


# 兼容旧名称
run_react_query = run_query
run_domain_query = run_query  # 不再区分，统一入口
ReActOrchestrator = AgentOrchestrator
