"""
ReportOrchestrator — 报告生成主编排器

对标 chatdb/core/orchestrator.py 的 AgentOrchestrator：
- chatdb Orchestrator 编排 SemanticParse → Planner → SQLAgent → Summarize
- ReportOrchestrator 编排 Outline → SectionDAG → (SubQuestions → DataQuery → Write) → Assemble

核心流程（todo.md 第十章）：
```
ReportOrchestrator.generate_report()
  │
  ├─ [Stage 1] ReportPlanner.generate_outline()
  │     → 写入 scratch/{session}/report_outline.json
  │     → event_sink("outline_end", {outline})
  │
  ├─ SectionDAG(sections)  — 拓扑排序，并行/串行编排
  │
  ├─ 逐批执行章节:
  │     ├─ event_sink("section_start", ...)
  │     ├─ [Stage 2] ReportPlanner.plan_sub_questions(section)
  │     ├─ [Stage 3] DataService.query(sub_question)  × N
  │     │     └─ EvidenceCollector.record(...)
  │     ├─ [Stage 4a] ReportWriter.write_section(section)
  │     ├─ ReportWriter.verify(section, mode="section")
  │     ├─ TransitionPlanner.plan_transition(section)
  │     └─ event_sink("section_end", ...)
  │
  ├─ ReportWriter.verify(mode="global")
  │
  └─ ReportWriter.assemble_report()
       → event_sink("report_chunk", ...)
       → event_sink(None)  — 结束信号
       → 返回 ReportResult
```

使用方式：
```python
orch = ReportOrchestrator(
    llm=llm,
    data_service=data_service,
    config=ReportConfig(),
)
result = await orch.generate_report("帮我生成一份2024年流水分析报告")
```
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from lib.core.base_orchestrator import BaseOrchestrator
from lib.llm import BaseLLM

from chatreport.agents.planner import ReportPlanner
from chatreport.agents.writer import ReportWriter
from chatreport.config.report_config import ReportConfig
from chatreport.core.evidence import EvidenceCollector
from chatreport.core.react_state import (
    DataResult,
    ReportPhase,
    ReportState,
    SectionResult,
    SectionSpec,
    SectionStatus,
)
from chatreport.core.section_dag import SectionDAG
from chatreport.core.summarize import TransitionPlanner
from chatreport.tools.data_service import DataService


class ReportOrchestrator(BaseOrchestrator):
    """
    报告生成主编排器 — 继承 BaseOrchestrator 复用 _emit / save_scratch / event_sink 管理

    关键设计：
    1. event_sink 机制继承自 BaseOrchestrator
    2. SectionDAG 控制章节间并行/串行
    3. 章节内子问题串行（共享 session 上下文）
    4. Scratch Pad 持久化中间结果
    """

    def __init__(
        self,
        llm: BaseLLM,
        data_service: DataService,
        config: ReportConfig | None = None,
        scratch_base: str = "data/scratch",
    ):
        super().__init__(
            llm=llm,
            component_name="ReportOrchestrator",
            scratch_base=scratch_base,
        )
        self.data_service = data_service
        self.config = config or ReportConfig()

        # 子 Agent
        self._planner = ReportPlanner(llm, self.config)
        self._writer = ReportWriter(llm, self.config)
        self._transition = TransitionPlanner(llm, self.config)

        # 证据链收集器（每次报告生成重置）
        self._evidence: EvidenceCollector | None = None

    # ================================================================
    # Template Method 实现（仅 init_state / build_result 真正需要）
    # generate_report 才是真正入口，覆写 run 的流程
    # ================================================================

    async def init_state(self, query: str, session_id: str | None, **kwargs: Any) -> ReportState:
        return ReportState(
            user_query=query,
            session_id=session_id or uuid.uuid4().hex,
            max_sections=self.config.max_sections,
            max_sub_questions_per_section=self.config.max_sub_questions_per_section,
            verify_enabled=self.config.verify_enabled,
        )

    async def pre_process(self, state: Any, **kwargs: Any) -> Any:
        return state

    async def plan(self, state: Any, **kwargs: Any) -> Any:
        return state

    async def execute(self, state: Any, **kwargs: Any) -> Any:
        return state

    async def post_process(self, state: Any, **kwargs: Any) -> Any:
        return state

    def build_result(self, state: Any, query: str, start_time: float) -> dict[str, Any]:
        return {}

    # ================================================================
    # 主入口
    # ================================================================

    async def generate_report(
        self,
        query: str,
        session_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
        event_sink: asyncio.Queue | None = None,
    ) -> dict[str, Any]:
        """
        生成完整报告 — 对标 chatdb Orchestrator.process_query()

        复用 BaseOrchestrator 的 event_sink 管理和 _emit 机制。
        """
        self._event_sink = event_sink
        start_time = time.time()

        # 初始化状态
        state = await self.init_state(query, session_id)
        self._evidence = EvidenceCollector()

        self._log.info(f"开始生成报告: {query[:80]}... session={state.session_id[:12]}")

        try:
            # ---- Stage 1: 大纲生成 ----
            await self._stage_outline(state, chat_history)

            # ---- Stage 2-4: 按 DAG 执行章节 ----
            await self._stage_sections(state)

            # ---- Stage 5: 全局审计 ----
            await self._stage_global_verify(state)

            # ---- Stage 6: 全局拼装 ----
            await self._stage_assembly(state)

            state.phase = ReportPhase.DONE
            elapsed = time.time() - start_time

            self._log.info(
                f"报告生成完成: {state.title}, "
                f"{state.section_count} 章节, "
                f"{self._evidence.total_count} 条证据, "
                f"耗时 {elapsed:.1f}s"
            )

            result = {
                "success": True,
                "title": state.title,
                "markdown": state.final_markdown,
                "outline": state.outline,
                "section_count": state.section_count,
                "evidence_count": self._evidence.total_count,
                "global_verification": state.global_verification,
                "elapsed_seconds": round(elapsed, 2),
                "session_id": state.session_id,
            }

            # 持久化最终报告
            self.save_scratch(state.session_id, "final_report.md", state.final_markdown)

            return result

        except Exception as e:
            state.phase = ReportPhase.ERROR
            state.error = str(e)
            elapsed = time.time() - start_time
            self._log.error(f"报告生成失败: {e}")

            await self._emit("error", {"error": str(e)})

            return {
                "success": False,
                "error": str(e),
                "title": state.title,
                "markdown": "",
                "outline": state.outline,
                "elapsed_seconds": round(elapsed, 2),
                "session_id": state.session_id,
            }

        finally:
            # 对齐 BaseOrchestrator 的 finally 清理
            if self._event_sink is not None:
                await self._event_sink.put(None)
                self._event_sink = None

    # ================================================================
    # Stage 1: 大纲生成
    # ================================================================

    async def _stage_outline(
        self,
        state: ReportState,
        chat_history: list[dict[str, str]] | None = None,
    ) -> None:
        """Stage 1: 生成报告大纲"""
        state.phase = ReportPhase.OUTLINE
        await self._emit("step_start", {"name": "outline_generation"})

        # 获取数据上下文
        data_context = self.data_service.get_data_context()

        outline = await self._planner.generate_outline(
            query=state.user_query,
            data_context=data_context,
            chat_history=chat_history,
            max_sections=state.max_sections,
        )

        state.set_outline(outline)

        # 持久化大纲
        self.save_scratch(
            state.session_id, "report_outline.json",
            json.dumps(outline, ensure_ascii=False, indent=2),
        )

        await self._emit("step_end", {"name": "outline_generation"})
        await self._emit("custom", {
            "name": "outline_end",
            "value": {
                "title": state.title,
                "sections": [s.to_dict() for s in state.section_specs],
            },
        })

        self._log.info(f"大纲完成: {state.title}, {state.section_count} 个章节")

    # ================================================================
    # Stage 2-4: 按 DAG 执行章节
    # ================================================================

    async def _stage_sections(self, state: ReportState) -> None:
        """按 SectionDAG 拓扑顺序执行所有章节"""
        dag = SectionDAG(state.section_specs)

        self._log.info(f"章节 DAG 构建完成:\n{dag}")

        for batch in dag.iter_batches():
            batch_specs = [
                spec for spec in state.section_specs if spec.id in batch
            ]

            # 并行执行当前批次（受 max_parallel_sections 限制）
            max_parallel = self.config.max_parallel_sections
            if len(batch_specs) <= max_parallel:
                tasks = [
                    self._execute_section(state, spec, dag)
                    for spec in batch_specs
                ]
                await asyncio.gather(*tasks)
            else:
                # 超过并行限制，分批
                for i in range(0, len(batch_specs), max_parallel):
                    sub_batch = batch_specs[i:i + max_parallel]
                    tasks = [
                        self._execute_section(state, spec, dag)
                        for spec in sub_batch
                    ]
                    await asyncio.gather(*tasks)

    async def _execute_section(
        self,
        state: ReportState,
        spec: SectionSpec,
        dag: SectionDAG,
    ) -> None:
        """执行单个章节的完整流程：规划 → 查数 → 写作 → 审计 → 承上启下"""
        section_id = spec.id
        sr = state.get_section(section_id)
        if not sr:
            return

        section_start = time.time()
        dag.mark_running(section_id)

        await self._emit("custom", {
            "name": "section_start",
            "value": {"section_id": section_id, "title": spec.title},
        })

        try:
            # ---- 2. 规划子问题 ----
            state.update_section_status(section_id, SectionStatus.PLANNING)
            previous_transitions = state.get_previous_transitions(section_id)

            sub_questions = await self._planner.plan_sub_questions(
                section=spec,
                previous_transitions=previous_transitions,
                outline=state.outline,
                data_context=self.data_service.get_data_context(),
            )
            sr.sub_questions = sub_questions

            # 持久化子问题
            self.save_scratch(
                state.session_id, f"{section_id}/sub_questions.json",
                json.dumps([q.to_dict() for q in sub_questions], ensure_ascii=False, indent=2),
            )

            await self._emit("custom", {
                "name": "sub_question_plan",
                "value": {
                    "section_id": section_id,
                    "questions": [q.to_dict() for q in sub_questions],
                },
            })

            # ---- 3. 数据查询 ----
            state.update_section_status(section_id, SectionStatus.QUERYING)
            data_results = await self._query_sub_questions(
                state, section_id, sub_questions,
            )
            sr.data_results = data_results

            # ---- 4a. 撰写章节 ----
            state.update_section_status(section_id, SectionStatus.WRITING)
            evidence_summary = self._evidence.section_summary(section_id)

            content = await self._writer.write_section(
                section=spec,
                data_results=data_results,
                previous_transitions=previous_transitions,
                evidence_summary=evidence_summary,
            )
            sr.content = content

            # 持久化章节内容
            self.save_scratch(state.session_id, f"{section_id}/content.md", content)

            await self._emit("custom", {
                "name": "section_write_end",
                "value": {
                    "section_id": section_id,
                    "content_length": len(content),
                },
            })

            # ---- 4b. 章节级审计 ----
            if self.config.verify_enabled and self.config.verify_section_level:
                state.update_section_status(section_id, SectionStatus.VERIFYING)
                verification = await self._writer.verify(
                    content=content,
                    data_results=data_results,
                    mode="section",
                    section_spec=spec,
                )
                sr.verification = verification

                self.save_scratch(
                    state.session_id, f"{section_id}/verification.json",
                    json.dumps(verification, ensure_ascii=False, indent=2),
                )

                await self._emit("custom", {
                    "name": "verification_result",
                    "value": {"section_id": section_id, **verification},
                })

                # ---- 4b+. 补充查询 + 内容修正（若审计未通过） ----
                if not verification.get("passed", True):
                    content = await self._handle_verification_remediation(
                        state, spec, sr, content, data_results, verification,
                    )
                    sr.content = content
                    self.save_scratch(
                        state.session_id, f"{section_id}/content.md", content
                    )

            # ---- 4c. 承上启下 ----
            transition = await self._transition.plan_transition(
                section=spec,
                section_content=content,
                data_results=data_results,
                outline=state.outline,
            )
            sr.transition = transition

            self.save_scratch(
                state.session_id, f"{section_id}/transition.json",
                json.dumps(transition, ensure_ascii=False, indent=2),
            )

            await self._emit("custom", {
                "name": "transition_generated",
                "value": {"section_id": section_id, **transition},
            })

            # ---- 标记完成 ----
            sr.status = SectionStatus.COMPLETED
            sr.elapsed_ms = int((time.time() - section_start) * 1000)
            dag.mark_done(section_id)

            await self._emit("custom", {
                "name": "section_end",
                "value": {
                    "section_id": section_id,
                    "status": "completed",
                    "elapsed_ms": sr.elapsed_ms,
                },
            })

            self._log.info(
                f"章节完成: [{section_id}] {spec.title}, "
                f"{len(data_results)} 查询, {len(content)} 字符, "
                f"耗时 {sr.elapsed_ms}ms"
            )

        except Exception as e:
            sr.status = SectionStatus.FAILED
            sr.error = str(e)
            sr.elapsed_ms = int((time.time() - section_start) * 1000)
            dag.mark_failed(section_id)

            self._log.error(f"章节失败: [{section_id}] {spec.title}: {e}")

            await self._emit("custom", {
                "name": "section_end",
                "value": {
                    "section_id": section_id,
                    "status": "failed",
                    "error": str(e),
                },
            })

    # ================================================================
    # Stage 4b+: 审计修正（补充查询 + 内容修正）
    # ================================================================

    async def _handle_verification_remediation(
        self,
        state: ReportState,
        spec: SectionSpec,
        sr: SectionResult,
        content: str,
        data_results: list[DataResult],
        verification: dict[str, Any],
    ) -> str:
        """
        审计未通过时的修正流程：
        1. 提取补充查询建议 → 执行补充查询
        2. 调用 writer.revise_section() 修正内容
        3. 限制最多一轮修正（避免无限循环）
        """
        section_id = spec.id

        self._log.info(
            f"章节审计未通过 [{section_id}], "
            f"{len(verification.get('issues', []))} 个问题, "
            f"开始修正..."
        )

        await self._emit("custom", {
            "name": "section_remediation_start",
            "value": {"section_id": section_id},
        })

        # 1. 执行补充查询
        supp_queries = verification.get("supplementary_queries", [])
        supplementary_data: list[DataResult] = []

        if supp_queries:
            self._log.info(
                f"  [{section_id}] 执行 {len(supp_queries)} 条补充查询"
            )

            for i, query_text in enumerate(supp_queries[:self.config.max_query_retries]):
                await self._emit("custom", {
                    "name": "supplementary_query_start",
                    "value": {
                        "section_id": section_id,
                        "query_index": i,
                        "question": query_text,
                    },
                })

                result = await self.data_service.query(
                    question=query_text,
                    session_id=state.session_id,
                )

                supplementary_data.append(result)

                # 记录证据
                if result.success and self._evidence:
                    eid = self._evidence.record(result, section_id)
                    sr.evidence_ids.append(eid)

                await self._emit("custom", {
                    "name": "supplementary_query_end",
                    "value": {
                        "section_id": section_id,
                        "query_index": i,
                        "success": result.success,
                        "row_count": result.row_count,
                    },
                })

            # 持久化补充查询结果
            if supplementary_data:
                self.save_scratch(
                    state.session_id,
                    f"{section_id}/supplementary_results.json",
                    json.dumps(
                        [r.to_dict() for r in supplementary_data],
                        ensure_ascii=False, indent=2,
                    ),
                )

        # 2. 内容修正
        revised_content = await self._writer.revise_section(
            original_content=content,
            verification=verification,
            supplementary_data=supplementary_data or None,
            section_spec=spec,
        )

        self.save_scratch(
            state.session_id, f"{section_id}/content_revised.md", revised_content
        )

        # 更新审计结果
        sr.verification["remediation_applied"] = True
        sr.verification["supplementary_query_count"] = len(supplementary_data)

        await self._emit("custom", {
            "name": "section_remediation_end",
            "value": {
                "section_id": section_id,
                "supplementary_queries": len(supplementary_data),
                "original_length": len(content),
                "revised_length": len(revised_content),
            },
        })

        self._log.info(
            f"  [{section_id}] 修正完成: "
            f"{len(content)} → {len(revised_content)} 字符, "
            f"{len(supplementary_data)} 条补充查询"
        )

        return revised_content

    # ================================================================
    # Stage 3 细化: 子问题查询
    # ================================================================

    async def _query_sub_questions(
        self,
        state: ReportState,
        section_id: str,
        sub_questions: list,
    ) -> list[DataResult]:
        """
        串行执行子问题查询（共享 session 上下文）

        串行原因：后续问题可能引用前序结果，共享 session 确保上下文连贯。
        """
        data_results: list[DataResult] = []

        for i, sq in enumerate(sub_questions):
            self._log.info(
                f"  [{section_id}] 查询 {i + 1}/{len(sub_questions)}: {sq.question[:60]}..."
            )

            await self._emit("custom", {
                "name": "data_query_start",
                "value": {
                    "section_id": section_id,
                    "question_index": i,
                    "question": sq.question,
                },
            })

            # 调用 DataService（ChatDB）
            result = await self.data_service.query(
                question=sq.question,
                session_id=state.session_id,
            )

            # 记录证据
            if result.success and self._evidence:
                eid = self._evidence.record(result, section_id)
                sr = state.get_section(section_id)
                if sr:
                    sr.evidence_ids.append(eid)

                await self._emit("custom", {
                    "name": "evidence_recorded",
                    "value": {
                        "section_id": section_id,
                        "evidence_id": eid,
                        "question": sq.question,
                    },
                })

            # 回填结果到子问题
            sq.result = result
            data_results.append(result)

            # 持久化查询结果
            self.save_scratch(
                state.session_id,
                f"{section_id}/query_results/q{i + 1}_result.json",
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            )

            await self._emit("custom", {
                "name": "data_query_end",
                "value": {
                    "section_id": section_id,
                    "question_index": i,
                    "success": result.success,
                    "row_count": result.row_count,
                    "elapsed_ms": result.elapsed_ms,
                },
            })

            if not result.success:
                self._log.warn(
                    f"  [{section_id}] 查询 {i + 1} 失败: {result.error}"
                )

        # 持久化证据链
        if self._evidence:
            section_evidence = self._evidence.get_section_evidence(section_id)
            if section_evidence:
                self.save_scratch(
                    state.session_id, f"{section_id}/evidence.json",
                    json.dumps(
                        [e.to_dict() for e in section_evidence],
                        ensure_ascii=False, indent=2,
                    ),
                )

        return data_results

    # ================================================================
    # Stage 5: 全局审计
    # ================================================================

    async def _stage_global_verify(self, state: ReportState) -> None:
        """全局质量审计"""
        if not self.config.verify_enabled or not self.config.verify_global_level:
            return

        state.phase = ReportPhase.VERIFICATION
        await self._emit("step_start", {"name": "global_verification"})

        # 收集所有数据结果
        all_data_results: list[DataResult] = []
        all_section_results: list[SectionResult] = []
        for spec in state.section_specs:
            sr = state.get_section(spec.id)
            if sr:
                all_data_results.extend(sr.data_results)
                all_section_results.append(sr)

        # 拼装所有章节内容作为审计输入
        all_content = "\n\n".join(
            f"## {sr.title}\n{sr.content}"
            for sr in all_section_results
            if sr.content
        )

        verification = await self._writer.verify(
            content=all_content,
            data_results=all_data_results,
            mode="global",
            all_sections=all_section_results,
        )
        state.global_verification = verification

        self.save_scratch(
            state.session_id, "global_verification.json",
            json.dumps(verification, ensure_ascii=False, indent=2),
        )

        await self._emit("step_end", {"name": "global_verification"})
        await self._emit("custom", {
            "name": "global_verification",
            "value": verification,
        })

        if verification.get("issues"):
            self._log.warn(
                f"全局审计发现 {len(verification['issues'])} 个问题"
            )

    # ================================================================
    # Stage 6: 全局拼装
    # ================================================================

    async def _stage_assembly(self, state: ReportState) -> None:
        """全局拼装最终报告"""
        state.phase = ReportPhase.ASSEMBLY
        await self._emit("step_start", {"name": "report_assembly"})

        final_markdown = await self._writer.assemble_report(
            state=state,
            evidence_collector=self._evidence,
        )
        state.final_markdown = final_markdown

        await self._emit("step_end", {"name": "report_assembly"})

        # 流式推送报告内容（分块）
        chunk_size = 500
        for i in range(0, len(final_markdown), chunk_size):
            chunk = final_markdown[i:i + chunk_size]
            await self._emit("text_chunk", {
                "message_id": f"report_{state.session_id[:8]}",
                "delta": chunk,
                "start": i == 0,
                "end": i + chunk_size >= len(final_markdown),
            })

        self._log.info(f"报告拼装完成: {len(final_markdown)} 字符")

    # ================================================================
    # 便捷方法
    # ================================================================

    @property
    def evidence_collector(self) -> EvidenceCollector | None:
        return self._evidence
