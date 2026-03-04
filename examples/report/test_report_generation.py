#!/usr/bin/env python3
"""
报告生成功能测试（ReportOrchestrator + ReportAGUIAdapter）

测试目标：
  1. ReportPlanner 正确生成报告大纲（Outline）
  2. SectionDAG 按拓扑序编排章节执行
  3. DataService 调用 ChatDB 查询数据
  4. ReportWriter 撰写/审计/拼装最终报告
  5. ReportAGUIAdapter AG-UI 事件流完整性

用法：
    python examples/report/test_report_generation.py [-v|-vv] [--query N] [--all]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="报告生成功能测试（ReportOrchestrator）")
    p.add_argument("-v", "--verbose", action="store_true", help="显示详细步骤 + AG-UI 摘要")
    p.add_argument("-vv", dest="verbose_llm", action="store_true", help="显示完整 LLM I/O")
    p.add_argument("--query", type=int, default=None, help="只运行第 N 个问题（1-based）")
    p.add_argument("--all", action="store_true", help="运行全部问题（默认前 2 个）")
    args = p.parse_args()
    if args.verbose_llm:
        args.verbose = True
    return args

ARGS = parse_args()

# ─── 测试用例 ─────────────────────────────────────────────────────────────────

QUERIES = [
    # (id, query, expected_keywords, min_sections, description)
    (1,
     "帮我写一份2025年IEG流水走势及变化原因分析报告",
     ["趋势", "变化", "原因"], 3,
     "趋势+归因的经典报告，需要先看趋势再拆解原因"),
    (2,
     "生成一份国内外产品流水对比分析报告，重点分析差异原因",
     ["国内", "海外", "对比", "差异"], 3,
     "按地区维度对比，需要分拆国内/海外数据再做差异分析"),
    (3,
     "写一份2024年和2025年各产品大类表现对比分析报告",
     ["产品", "对比", "同比"], 3,
     "跨年份+产品大类维度的综合分析报告"),
    (4,
     "请分析IEG的成本结构并生成详细报告，包含各项成本占比和变化趋势",
     ["成本", "占比", "趋势"], 3,
     "成本结构拆解+占比+趋势的多维度报告"),
    (5,
     "帮我写一份2025年Top5产品深度分析报告，包含流水、用户、利润等多维度分析",
     ["Top5", "流水", "用户"], 4,
     "排名+多维度下钻的综合深度报告"),
]


def _check(b: bool) -> str:
    return "✓" if b else "✗"


# ─── SSE 解析工具 ─────────────────────────────────────────────────────────────

def parse_sse_payloads(raw_lines: list[str]) -> list[dict]:
    """从 SSE 原始行中提取所有 JSON payload。"""
    payloads: list[dict] = []
    for raw in raw_lines:
        for line in raw.strip().split("\n"):
            if not line.startswith("data:"):
                continue
            try:
                payloads.append(json.loads(line[5:].strip()))
            except (json.JSONDecodeError, IndexError):
                pass
    return payloads


# ─── AG-UI 事件统计 ───────────────────────────────────────────────────────────

_BOOL_EVENT_MAP = {
    "RUN_STARTED": "has_run_started",
    "RUN_FINISHED": "has_run_finished",
    "RUN_ERROR": "has_run_error",
    "TEXT_MESSAGE_START": "has_text_message",
    "STATE_SNAPSHOT": "has_state_snapshot",
}

_LIST_EVENT_MAP = {
    "STEP_STARTED": ("steps", "stepName"),
}


@dataclass
class AGUIEventStats:
    """AG-UI 事件流统计收集器"""
    event_counts: dict[str, int] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    custom_events: list[str] = field(default_factory=list)
    has_run_started: bool = False
    has_run_finished: bool = False
    has_run_error: bool = False
    has_text_message: bool = False
    has_state_snapshot: bool = False
    raw_events: list[str] = field(default_factory=list)

    # 报告特有事件追踪
    has_outline: bool = False
    section_starts: list[str] = field(default_factory=list)
    section_ends: list[str] = field(default_factory=list)
    sub_question_plans: int = 0
    data_queries: int = 0
    verification_results: int = 0

    def record(self, sse_line: str):
        self.raw_events.append(sse_line)
        for payload in parse_sse_payloads([sse_line]):
            evt = payload.get("type", "")
            self.event_counts[evt] = self.event_counts.get(evt, 0) + 1

            if evt in _BOOL_EVENT_MAP:
                setattr(self, _BOOL_EVENT_MAP[evt], True)
            elif evt in _LIST_EVENT_MAP:
                attr, key = _LIST_EVENT_MAP[evt]
                getattr(self, attr).append(payload.get(key, "unknown"))

            # 报告特有的 CUSTOM 事件
            if evt == "CUSTOM":
                custom_name = payload.get("name", "")
                self.custom_events.append(custom_name)
                if custom_name == "outline_end":
                    self.has_outline = True
                elif custom_name == "section_start":
                    sec_id = payload.get("value", {}).get("section_id", "?")
                    self.section_starts.append(sec_id)
                elif custom_name == "section_end":
                    sec_id = payload.get("value", {}).get("section_id", "?")
                    self.section_ends.append(sec_id)
                elif custom_name == "sub_question_plan":
                    self.sub_question_plans += 1
                elif custom_name in ("data_query_start", "data_query_end"):
                    self.data_queries += 1
                elif custom_name == "verification_result":
                    self.verification_results += 1

    @property
    def is_complete(self) -> bool:
        return self.has_run_started and (self.has_run_finished or self.has_run_error)

    def summary_line(self) -> str:
        total = sum(self.event_counts.values())
        return (
            f"事件 {total} 条 │ 步骤: {self.steps} │ "
            f"大纲: {_check(self.has_outline)} │ "
            f"章节: {len(self.section_ends)}/{len(self.section_starts)} │ "
            f"查询: {self.data_queries} │ "
            f"审计: {self.verification_results} │ "
            f"文本: {_check(self.has_text_message)} │ "
            f"快照: {_check(self.has_state_snapshot)} │ "
            f"完整: {_check(self.is_complete)}"
        )


# ─── 测试结果 ─────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """单条测试执行结果"""
    qid: int
    query: str
    expected_keywords: list[str]
    min_sections: int
    report_title: str = ""
    report_markdown: str = ""
    section_count: int = 0
    evidence_count: int = 0
    outline: list[dict] = field(default_factory=list)
    success: bool = False
    error: str = ""
    elapsed: float = 0.0
    session_id: str = ""
    agui_stats: AGUIEventStats = field(default_factory=AGUIEventStats)

    @property
    def has_report_format(self) -> bool:
        return any(k in self.report_markdown for k in ["# ", "## ", "摘要", "结论", "数据口径"])

    @property
    def keyword_hits(self) -> int:
        return sum(1 for k in self.expected_keywords if k in self.report_markdown)

    @property
    def sections_sufficient(self) -> bool:
        return self.section_count >= self.min_sections


# ─── 核心执行 ─────────────────────────────────────────────────────────────────

async def run_single_query(report_orch_factory, q: tuple) -> TestResult:
    """通过 ReportAGUIAdapter 运行单个测试问题。"""
    from chatreport.core.agui_adapter import ReportAGUIAdapter

    qid, query, expected_kw, min_sections, _ = q
    tr = TestResult(qid, query, expected_kw, min_sections)
    tr.session_id = f"rpt-{uuid.uuid4().hex[:8]}"

    report_orch = report_orch_factory(tr.session_id)
    adapter = ReportAGUIAdapter()

    t0 = time.time()
    try:
        async for sse_chunk in adapter.stream(
            orchestrator=report_orch,
            query=query,
            thread_id=uuid.uuid4().hex,
            run_id=uuid.uuid4().hex,
            session_id=tr.session_id,
        ):
            tr.agui_stats.record(sse_chunk)
            if ARGS.verbose_llm:
                print(f"    [AG-UI] {sse_chunk.strip()[:120]}")

        tr.elapsed = time.time() - t0
        _extract_result_from_events(tr)

    except Exception as e:
        tr.elapsed = time.time() - t0
        tr.error = str(e)
        traceback.print_exc()

    return tr


def _extract_result_from_events(tr: TestResult):
    """从事件流提取结果信息。"""
    text_chunks: list[str] = []
    for payload in parse_sse_payloads(tr.agui_stats.raw_events):
        evt = payload.get("type")

        if evt == "STATE_SNAPSHOT":
            snapshot = payload.get("snapshot", {})
            tr.success = snapshot.get("success", False)
            tr.report_title = snapshot.get("title", "")
            tr.section_count = snapshot.get("section_count", 0)
            tr.evidence_count = snapshot.get("evidence_count", 0)

        elif evt == "TEXT_MESSAGE_CONTENT":
            delta = payload.get("delta", "")
            if delta:
                text_chunks.append(delta)

        elif evt == "CUSTOM":
            name = payload.get("name", "")
            if name == "outline_end":
                value = payload.get("value", {})
                tr.report_title = value.get("title", "")
                tr.outline = value.get("sections", [])

    tr.report_markdown = "".join(text_chunks)

    # 从 scratch 目录补充（如果事件流中未包含完整报告）
    if not tr.report_markdown:
        report_path = Path(f"data/scratch/{tr.session_id}/final_report.md")
        if report_path.exists():
            tr.report_markdown = report_path.read_text(encoding="utf-8")

    # 从 outline 文件补充
    if not tr.outline:
        outline_path = Path(f"data/scratch/{tr.session_id}/report_outline.json")
        if outline_path.exists():
            try:
                outline_data = json.loads(outline_path.read_text(encoding="utf-8"))
                tr.outline = outline_data.get("sections", [])
                tr.report_title = outline_data.get("title", tr.report_title)
            except (json.JSONDecodeError, KeyError):
                pass


# ─── 输出格式 ─────────────────────────────────────────────────────────────────

_W = 70

def print_test_result(tr: TestResult, q: tuple):
    """格式化打印单个测试结果。"""
    _, _, _, _, desc = q
    print(f"\n  ┌─ 问题 #{tr.qid} {'─' * 50}")
    print(f"  │ {tr.query}")
    print(f"  │ 说明: {desc}")

    print(f"  ├─ 大纲生成 {'─' * 46}")
    print(f"  │ 标题: {tr.report_title or '(无)'}")
    print(f"  │ 大纲章节: {len(tr.outline)} (期望≥{tr.min_sections})")
    for i, sec in enumerate(tr.outline):
        title = sec.get("title", sec.get("id", "?"))
        deps = sec.get("dependencies", [])
        dep_str = f" ← [{','.join(deps)}]" if deps else ""
        print(f"  │   [{i+1}] {title}{dep_str}")

    print(f"  ├─ 执行结果 {'─' * 46}")
    print(f"  │ 成功: {_check(tr.success)} │ "
          f"耗时: {tr.elapsed:.1f}s │ "
          f"章节: {tr.section_count} │ "
          f"证据: {tr.evidence_count}")

    # 关键词命中
    hit = tr.keyword_hits
    total = len(tr.expected_keywords)
    print(f"  ├─ 内容检查 {'─' * 46}")
    print(f"  │ 关键词命中: {hit}/{total} │ "
          f"报告格式: {_check(tr.has_report_format)} │ "
          f"报告长度: {len(tr.report_markdown)} 字符")

    # AG-UI 事件
    stats = tr.agui_stats
    print(f"  ├─ AG-UI 事件流 {'─' * 42}")
    print(f"  │ {stats.summary_line()}")

    if tr.error:
        print(f"  │ ⚠ 错误: {tr.error[:80]}")
    elif tr.report_markdown:
        preview = tr.report_markdown.replace("\n", " ")[:150]
        print(f"  │ 摘要: {preview}...")

    print(f"  └{'─' * (_W - 5)}")


def print_summary_report(results: list[TestResult]):
    """打印汇总统计表格。"""
    total = len(results)
    if total == 0:
        return

    print(f"\n\n{'═' * _W}")
    print("  报告生成测试 · 汇总（ReportOrchestrator）")
    print(f"{'═' * _W}")
    print(f"{'#':>3} {'成功':>4} {'大纲':>4} {'章节':>4} {'证据':>4} {'格式':>4} {'AG-UI':>6} {'耗时':>7}")
    print("─" * _W)

    for tr in results:
        err = " ERR" if tr.error else ""
        print(f" {tr.qid:>2}     {_check(tr.success)}   "
              f"{len(tr.outline):>2}     {tr.section_count:>2}     "
              f"{tr.evidence_count:>2}     {_check(tr.has_report_format)}      "
              f"{_check(tr.agui_stats.is_complete)}  {tr.elapsed:>5.1f}s{err}")

    ok = sum(1 for r in results if r.success)
    fmt = sum(1 for r in results if r.has_report_format)
    agui_ok = sum(1 for r in results if r.agui_stats.is_complete)
    outline_ok = sum(1 for r in results if r.agui_stats.has_outline)
    errs = sum(1 for r in results if r.error)
    pct = lambda n: f"{n}/{total} ({n / total * 100:.0f}%)"

    print(f"\n  总计 {total} 个问题")
    print(f"  成功生成:       {pct(ok)}")
    print(f"  大纲生成:       {pct(outline_ok)}")
    print(f"  报告格式输出:   {pct(fmt)}")
    print(f"  AG-UI 事件完整: {pct(agui_ok)}")
    print(f"  平均章节数:     {sum(r.section_count for r in results) / total:.1f}")
    print(f"  平均证据数:     {sum(r.evidence_count for r in results) / total:.1f}")
    print(f"  平均耗时:       {sum(r.elapsed for r in results) / total:.1f}s")
    if errs:
        print(f"  执行失败:       {pct(errs)}")


# ─── 日志配置 ─────────────────────────────────────────────────────────────────

def setup_logging():
    from lib.utils.logger import enable_llm_debug, set_log_level_to_debug, set_log_level_to_info

    if ARGS.verbose_llm:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=True)
    elif ARGS.verbose:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=False)
    else:
        set_log_level_to_info()
        enable_llm_debug(enable=False)


# ─── 入口 ─────────────────────────────────────────────────────────────────────

async def main():
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.database.duckdb import DuckDBConnector
    from chatreport.config.report_config import ReportConfig
    from chatreport.core.orchestrator import ReportOrchestrator
    from chatreport.tools.data_service import DataService
    from lib.llm import LLMFactory
    from lib.storage.chat_history import HistoryConfig

    setup_logging()

    db_path = "data/duckdb/csv_7dbb24bf.duckdb"
    yml_config = "data/yml/metrics_config.yml"
    history_db_path = "data/pilot/test_history.db"

    if not Path(db_path).exists():
        print(f"数据库不存在: {db_path}")
        return

    # Banner
    mode = "LLM详细" if ARGS.verbose_llm else ("详细" if ARGS.verbose else "精简")
    print(f"{'=' * _W}\n  报告生成功能测试（ReportOrchestrator）\n{'=' * _W}")
    print(f"  数据库: {db_path}  |  YAML: {yml_config}  |  日志: {mode}\n")

    # 连接数据库
    db = DuckDBConnector(database=db_path)
    await db.connect()

    try:
        tables_meta = db.get_tables_meta()
        for t in tables_meta:
            print(f"  表: {t['table_name']} ({t['row_count']} 行)")
        print()

        llm = LLMFactory.create(provider="hunyuan", temperature=1.0, top_p=0.95)
        report_config = ReportConfig(
            verify_enabled=True,
            transition_enabled=True,
            max_parallel_sections=2,
        )

        def make_report_orchestrator(session_id: str) -> ReportOrchestrator:
            # ChatDB Orchestrator 作为 DataService 的后端
            chatdb_orch = AgentOrchestrator(
                llm, db,
                yml_config=yml_config,
                tables_meta=tables_meta,
                debug=ARGS.verbose,
                history_db_path=history_db_path,
                history_config=HistoryConfig(num_history_runs=3),
            )
            data_service = DataService(chatdb_orch)
            return ReportOrchestrator(
                llm=llm,
                data_service=data_service,
                config=report_config,
            )

        # 选择问题
        if ARGS.query is not None:
            queries = [q for q in QUERIES if q[0] == ARGS.query]
            if not queries:
                print(f"未找到问题 #{ARGS.query}")
                return
        elif ARGS.all:
            queries = QUERIES
        else:
            queries = QUERIES[:2]
            print(f"  默认运行前 2 个问题（加 --all 运行全部 {len(QUERIES)} 个）\n")

        # 逐个运行
        results: list[TestResult] = []
        for q in queries:
            print(f"{'━' * _W}\n  运行问题 #{q[0]}/{len(QUERIES)}\n{'─' * _W}")
            tr = await run_single_query(make_report_orchestrator, q)
            print_test_result(tr, q)
            results.append(tr)

        print_summary_report(results)

    finally:
        await db.disconnect()

    print(f"\n{'═' * _W}\n  测试完成\n{'═' * _W}")


if __name__ == "__main__":
    asyncio.run(main())
