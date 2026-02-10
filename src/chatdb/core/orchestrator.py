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
import time

from chatdb.agents.base import AgentContext, AgentStatus
from chatdb.core.react_state import ReActState, ReActPhase, ErrorType
from chatdb.agents.planner import PlannerAgent
from chatdb.agents.sql_agent import SQLAgent
from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.schema import SchemaInspector
from chatdb.llm.base import BaseLLM
from chatdb.tools import (
    ToolRegistry,
    GetSchemaTool,
    SemanticParseTool,
    SummarizeAnswerTool,
)
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
            enable_llm_debug(True, show_input=False)  # debug 默认只显示输出
        
        # 初始化 ToolRegistry
        self.registry = ToolRegistry()
        self.registry.register(GetSchemaTool(self.db_connector))
        
        # SQLAgent（核心 Agent，接受 yml_config）
        self._sql_agent = SQLAgent(self.llm, self.db_connector, yml_config)
        
        # 语义解析 Tool（前置 workflow）
        self._semantic_parse_tool = SemanticParseTool(llm, yml_config)
        
        # 总结 Tool
        self._summarize_tool = SummarizeAnswerTool(llm)
        
        # Planner（生成分析型 ToDo）
        self.planner = PlannerAgent(llm)
    
    async def process_query(self, query: str) -> dict[str, Any]:
        """
        处理用户查询
        
        流程：
        1. [前置] 语义解析（不属于 Planner 调度）
        2. Planner 生成分析计划
        3. SQLAgent 按计划执行
        4. 生成总结
        """
        start_time = time.time()
        
        # Task View: 开始任务
        task_log.start(query)
        orch_log = get_component_logger("Orchestrator")
        
        # 初始化
        self.planner.clear_history()
        state = await self._init_state(query)
        context = await self._init_context(query, state)
        
        try:
            # ============================================================
            # 1. [前置 workflow] 语义解析
            # 不属于 Planner 调度，是必要的前置步骤
            # ============================================================
            orch_log.info("1. [前置] 语义解析...")
            await self._semantic_parse_tool(state, context)
            
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
            # 1.5 检查是否为元查询（如模拟问题、解释结构等）
            # 使用 is_meta_query() 方法判断
            # ============================================================
            if state.intent and state.intent.is_meta_query():
                orch_log.info("检测到元查询（mode=meta），直接生成响应...")
                state = await self._handle_meta_query(state, context, orch_log)
                if state.summary:
                    task_log.done(state.summary)
                return self._build_result(state, query, start_time)
            
            # ============================================================
            # 2. Planner 生成分析计划
            # ============================================================
            orch_log.info("2. Planner 生成分析计划...")
            plan = await self.planner.generate_analysis_plan(state, context)
            orch_log.info(f"分析计划:\n{plan.to_display()}")
            
            # ============================================================
            # 3. 按计划执行 SQLAgent
            # ============================================================
            orch_log.info(f"3. SQL 分析流程... (Agent: {self._sql_agent.display_name})")
            await self._execute_plan(state, context, orch_log)
            
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
            
            return self._build_result(state, query, start_time)
            
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
    ) -> None:
        """
        执行 Planner 生成的分析计划
        
        流程：
        1. 获取当前任务
        2. SQLAgent.run_task() 执行任务，结果写入 temp_results
        3. Planner.inspect_temp_results() 查看结果
        4. Planner.decide_next_action() 决定下一步:
           - continue: 继续下一任务
           - adjust: 调整当前任务（重新执行）
           - done: 结束，可能带有 conclusion
        5. 循环直到计划完成或达到最大步数
        """
        step_count = 0
        max_plan_steps = 8  # 最多执行 8 个任务（包括动态插入的）
        adjust_count = 0    # 调整次数限制
        max_adjust = 2      # 同一任务最多调整 2 次
        validation_count = 0  # validation 任务计数
        max_validation = 2    # 最多执行 2 个 validation 任务
        last_task_id = None   # 上一个任务 ID，用于检测死循环
        same_task_count = 0   # 同一任务重复执行次数
        
        while step_count < max_plan_steps:
            step_count += 1
            
            # 获取当前任务
            current_task = self.planner.get_current_task(state.temp_results)
            if not current_task:
                orch_log.info("没有更多任务")
                break
            
            task_dict = current_task.to_dict()
            task_id = task_dict.get("id", "")
            task_type = task_dict.get("type", "")
            
            # 死循环检测：同一任务重复执行
            if task_id == last_task_id:
                same_task_count += 1
                if same_task_count >= 2:
                    orch_log.warn(f"任务 {task_id} 重复执行 {same_task_count} 次，强制跳过")
                    self.planner.advance_plan(state.temp_results)
                    same_task_count = 0
                    continue
            else:
                last_task_id = task_id
                same_task_count = 0
            
            # validation 任务数量限制
            if task_type == "validation":
                validation_count += 1
                if validation_count > max_validation:
                    orch_log.warn(f"已执行 {validation_count-1} 个 validation 任务，跳过更多诊断")
                    self.planner.advance_plan(state.temp_results)
                    continue
            
            orch_log.info(f"执行任务 {step_count}: [{task_type}] {task_dict['description'][:50]}...")
            
            # summary 任务由 Orchestrator 处理
            if task_type == "summary":
                self.planner.advance_plan(state.temp_results)
                continue
            
            # SQLAgent 执行任务（结果写入 temp_results）
            try:
                await self._sql_agent.run_task(state, context, task_dict)
                
                # 查看执行结果
                temp_summary = self.planner.inspect_temp_results(state)
                if temp_summary:
                    orch_log.debug(f"temp_results 摘要:\n{temp_summary[:200]}...")
                
                # 标记当前任务完成
                self.planner.advance_plan(state.temp_results)
                
                # 让 Planner 决定下一步
                decision = await self.planner.decide_next_action(state, context)
                action = decision.get("action", "done")
                
                if action == "done":
                    reason = decision.get("reason", "")
                    conclusion = decision.get("conclusion", "")
                    orch_log.info(f"Planner 决定结束: {reason}")
                    
                    # 如果 Planner 给出了结论，直接使用
                    if conclusion:
                        orch_log.info(f"Planner 结论: {conclusion[:100]}...")
                        state.summary = conclusion
                    break
                    
                elif action == "adjust":
                    adjust_count += 1
                    reason = decision.get("reason", "")
                    adjustment = decision.get("adjustment", {})
                    
                    orch_log.info(f"Planner 决定调整 ({adjust_count}/{max_adjust}): {reason}")
                    
                    if adjust_count >= max_adjust:
                        orch_log.warn("达到最大调整次数，继续下一任务")
                        adjust_count = 0
                    else:
                        # 应用调整策略
                        self._apply_adjustment(state, context, adjustment, orch_log)
                        
                elif action == "retry":
                    # 重试任务（不增加 step_count）
                    retry_hint = decision.get("retry_hint", "")
                    orch_log.info(f"重试任务，提示: {retry_hint}")
                    step_count -= 1  # 不计入步数
                    
                else:
                    # continue - 继续下一任务
                    adjust_count = 0  # 重置调整计数
                
            except Exception as e:
                orch_log.warn(f"任务执行失败: {e}")
                self.planner.mark_task_failed(str(e), state.temp_results)
                self.planner.advance_plan(state.temp_results)
        
        # 日志：最终 temp_results 状态
        if state.temp_results:
            orch_log.info(f"完成 {len(state.temp_results)} 个任务的数据收集")

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
        
        # 将调整信息存入 context，让 SQLAgent 在下次执行时参考
        if hasattr(context, 'extra_context'):
            context.extra_context = f"[调整策略] {strategy}\n[具体改动] {change}"
    
    async def _init_state(self, query: str) -> ReActState:
        """初始化 State"""
        state = ReActState(user_query=query, max_steps=self.max_steps)
        state.think("开始处理查询")
        return state
    
    async def _init_context(self, query: str, state: ReActState) -> AgentContext:
        """初始化 Context"""
        if self.tables_meta:
            tables_info = self.tables_meta
            schema_text = build_schema_text(self.tables_meta)
            for table in self.tables_meta:
                columns = table.get("columns_info") or table.get("columns", [])
                state.available_columns.extend(columns)
        else:
            schema_info = await self.schema_inspector.get_schema_info()
            tables_info = get_tables_info(schema_info)
            schema_text = schema_info.to_prompt_text()
        
        state.schema_text = schema_text
        state.table_name = select_best_table(query, tables_info)
        state.think(f"选择表: {state.table_name}")
        
        return AgentContext(
            user_query=query,
            schema_text=schema_text,
            available_tables=tables_info,
            selected_tables=[state.table_name] if state.table_name else [],
        )
    
    async def _handle_meta_query(self, state: ReActState, context: AgentContext, orch_log) -> ReActState:
        """
        处理非数据分析请求（mode=meta）
        
        SemanticParser 已判断这不是数据分析请求，直接让 LLM 响应。
        meta_request 字段包含了 SemanticParser 对用户意图的理解。
        """
        meta_request = state.intent.meta_request if state.intent else None
        orch_log.info(f"非数据分析请求: {meta_request or context.user_query[:50]}...")
        
        # 构建上下文：始终提供数据能力说明，让 LLM 知道自己能做什么
        data_context = self._build_data_context(context, state)
        
        prompt = f"""用户问题: {context.user_query}
{f"用户意图: {meta_request}" if meta_request else ""}

{data_context}

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
        if context.schema_text:
            schema_brief = context.schema_text[:500]
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
    
    async def _generate_summary(self, state: ReActState, query: str) -> ReActState:
        """生成总结（利用 temp_results）"""
        rows = state.execute_result.get("rows", []) if state.execute_result else []
        
        # 如果有 temp_results，优先使用其进行总结
        if state.temp_results:
            summary_context = self._build_summary_context(state)
            # 将 temp_results 摘要加入 context
            context = AgentContext(
                user_query=query,
                extra_context=summary_context,
            )
            await self._summarize_tool(state, context)
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
        """从 temp_results 构建总结上下文"""
        lines = ["## 分析结果汇总\n"]
        
        for task_id, results in state.temp_results.items():
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
                    for ex in examples[:3]:
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
        
        result = {
            "success": success,
            "query": query,
            "sql": state.final_sql or state.current_sql,
            "result": state.execute_result.get("rows", []) if state.execute_result else [],
            "row_count": state.execute_result.get("row_count", 0) if state.execute_result else 0,
            "summary": state.summary,
            "table_name": state.table_name,
            "intent": state.intent.to_dict() if state.intent and hasattr(state.intent, 'to_dict') else None,
        }
        
        # 添加 temp_results（新的分析结果存储）
        if state.temp_results:
            result["temp_results"] = state.temp_results
        
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
    
    Returns:
        查询结果
    
    Example:
        # 带领域配置（推荐）
        result = await run_query(
            "王者荣耀流水增长来自哪里",
            llm, db,
            yml_config="data/yml/ieg.yml",
        )
        
        # 基础模式
        result = await run_query("IEG本部流水是多少", llm, db)
    """
    orchestrator = AgentOrchestrator(
        llm=llm,
        db_connector=db_connector,
        yml_config=yml_config,
        tables_meta=tables_meta,
        debug=debug,
        max_steps=max_steps,
    )
    return await orchestrator.process_query(query)


# 兼容旧名称
run_react_query = run_query
run_domain_query = run_query  # 不再区分，统一入口
ReActOrchestrator = AgentOrchestrator
