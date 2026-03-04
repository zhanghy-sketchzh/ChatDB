#!/usr/bin/env python3
"""
多步骤规划与推理能力测试（AG-UI 协议版）

测试目标：
  1. Planner 为复杂问题生成多任务 DAG（带 depends_on）
  2. Orchestrator 按拓扑序执行、传递上游结果
  3. decide_next_action 动态调整计划
  4. AG-UI 事件流完整性（RUN_STARTED → STEP/TOOL → TEXT → RUN_FINISHED）

用法：
    python examples/test_multi_step_plan.py [-v|-vv] [--query N] [--all]
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多步骤规划测试（AG-UI）")
    p.add_argument("-v", "--verbose", action="store_true", help="显示 ReAct 步骤 + AG-UI 摘要")
    p.add_argument("-vv", dest="verbose_llm", action="store_true", help="显示完整 LLM I/O")
    p.add_argument("--query", type=int, default=None, help="只运行第 N 个问题（1-based）")
    p.add_argument("--all", action="store_true", help="运行全部问题（默认前 3 个）")
    args = p.parse_args()
    if args.verbose_llm:
        args.verbose = True
    return args

ARGS = parse_args()

# ─── 测试用例 ─────────────────────────────────────────────────────────────────

QUERIES = [
    # (id, query, expected_pattern, min_tasks, why_multi_step)
    (1,
     "2025年流水最高的3个产品是哪些？然后告诉我这3个产品各自在国内和海外的流水分别是多少",
     "ranking → drilldown/source", 2,
     "下钻任务需要先知道Top3是谁，才能展开"),
    (2,
     "分别统计2024年和2025年IEG本部的总流水，然后计算同比增长率，并按产品大类拆解增长来源",
     "comparison → source", 2,
     "先要算出总量同比，再按产品大类拆解增量贡献"),
    (3,
     "先给我看2022到2025年IEG每年的总流水趋势，然后找出流水下降最多的年份，并分析该年份哪些产品的流水跌幅最大",
     "trend → comparison → source/ranking", 2,
     "需要先看趋势找到下降年份，再针对该年份做产品级分析"),
    (4,
     "2024年各产品大类在总流水中的占比是多少？2025年的占比又是多少？哪些品类占比在上升？",
     "ratio(2024) + ratio(2025) → comparison", 2,
     "需要分别算两年占比，再对比变化方向"),
    (5,
     "2025年IEG的递延后利润总额是多少？利润最高的Top5产品分别是什么？这5个产品的利润率（利润/流水）各是多少？",
     "basic → ranking → ratio(drilldown)", 3,
     "三个独立子问题，后续问题依赖前面的结果"),
    (6,
     "对比一下2025年实际数据和预测数据的总流水差异，然后找出差异最大的5个产品",
     "comparison(实际vs预测) → ranking(差异Top5)", 2,
     "先算总体差异，再对差异做产品级排名"),
    (7,
     "2025年IEG的各项成本（按大盘报表项）分别是多少？运营成本占总流水的比例是多少？跟2024年比变化如何？",
     "source(成本拆解) → ratio(成本率) → comparison(同比)", 2,
     "先拆解成本结构，再算成本率，再做同比"),
    (8,
     "2023年、2024年、2025年，国内流水和海外流水分别是多少？国内和海外的年增速分别是多少？哪边增速更快？",
     "source(国内/海外×年) → comparison(增速) → summary", 2,
     "先拆出国内/海外三年数据，再计算各自增速并对比"),
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

# 事件类型 → 需要设置的布尔属性名
_BOOL_EVENT_MAP = {
    "RUN_STARTED": "has_run_started",
    "RUN_FINISHED": "has_run_finished",
    "RUN_ERROR": "has_run_error",
    "TEXT_MESSAGE_START": "has_text_message",
    "STATE_SNAPSHOT": "has_state_snapshot",
}

# 事件类型 → (追加到的列表属性, payload 中取的 key)
_LIST_EVENT_MAP = {
    "STEP_STARTED": ("steps", "stepName"),
    "TOOL_CALL_START": ("tool_calls", "toolCallName"),
}


@dataclass
class AGUIEventStats:
    """AG-UI 事件流统计收集器"""

    event_counts: dict[str, int] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    has_run_started: bool = False
    has_run_finished: bool = False
    has_run_error: bool = False
    has_text_message: bool = False
    has_state_snapshot: bool = False
    raw_events: list[str] = field(default_factory=list)

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

    @property
    def is_complete(self) -> bool:
        return self.has_run_started and (self.has_run_finished or self.has_run_error)

    def summary_line(self) -> str:
        total = sum(self.event_counts.values())
        return (
            f"事件 {total} 条 │ 步骤: {self.steps} │ "
            f"工具: {len(self.tool_calls)} │ "
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
    expected: str
    min_tasks: int
    actual_tasks: list[dict] = field(default_factory=list)
    has_deps: bool = False
    summary: str = ""
    error: str = ""
    elapsed: float = 0.0
    session_id: str = ""
    had_adjustment: bool = False
    agui_stats: AGUIEventStats = field(default_factory=AGUIEventStats)

    @property
    def n_tasks(self) -> int:
        return len(self.actual_tasks)

    @property
    def is_multi(self) -> bool:
        return self.n_tasks >= 2


# ─── 核心执行 ─────────────────────────────────────────────────────────────────

async def run_single_query(orchestrator_factory, q: tuple) -> TestResult:
    """通过 AG-UI 适配器运行单个测试问题。"""
    from chatdb.core.agui_adapter import AGUIAdapter

    qid, query, expected, min_tasks, _ = q
    tr = TestResult(qid, query, expected, min_tasks)
    tr.session_id = f"msp-{uuid.uuid4().hex[:8]}"

    orchestrator = orchestrator_factory(tr.session_id)
    adapter = AGUIAdapter()

    t0 = time.time()
    try:
        async for sse_chunk in adapter.stream(
            orchestrator=orchestrator,
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
    """从事件流和 plan 文件中提取 summary / tasks / deps 等信息。"""
    # summary: 优先取 STATE_SNAPSHOT，其次取首条 TEXT_MESSAGE_CONTENT
    for payload in parse_sse_payloads(tr.agui_stats.raw_events):
        evt = payload.get("type")
        if evt == "STATE_SNAPSHOT":
            tr.summary = payload.get("snapshot", {}).get("query", "")
        elif evt == "TEXT_MESSAGE_CONTENT" and not tr.summary:
            delta = payload.get("delta", "")
            if delta:
                tr.summary = delta

    # plan.json
    plan_path = Path(f"data/scratch/{tr.session_id}/plan.json")
    if plan_path.exists():
        tasks = json.loads(plan_path.read_text(encoding="utf-8")).get("tasks", [])
        tr.actual_tasks = tasks
        tr.has_deps = any(bool(t.get("depends_on")) for t in tasks)

    tr.had_adjustment = any(
        t.get("meta", {}).get("retry_hint") or t.get("meta", {}).get("adjusted")
        for t in tr.actual_tasks
    )


# ─── 输出格式 ─────────────────────────────────────────────────────────────────

_W = 70  # 输出宽度

_STATUS_ICONS = {"completed": "✓", "failed": "✗", "skipped": "○"}


def print_test_result(tr: TestResult, q: tuple):
    """格式化打印单个测试结果。"""
    _, _, _, _, why = q
    print(f"\n  ┌─ 问题 #{tr.qid} {'─' * 50}")
    print(f"  │ {tr.query[:80]}{'...' if len(tr.query) > 80 else ''}")
    print(f"  │ 预期: {tr.expected}")
    print(f"  │ 原因: {why}")
    print(f"  ├─ 执行结果 {'─' * 46}")
    print(f"  │ 耗时: {tr.elapsed:.1f}s │ 任务数: {tr.n_tasks} (期望≥{tr.min_tasks}) │ "
          f"多步: {_check(tr.is_multi)} │ 依赖: {_check(tr.has_deps)}")

    for t in tr.actual_tasks:
        deps = t.get("depends_on", [])
        dep_str = f" ← [{','.join(deps)}]" if deps else ""
        icon = _STATUS_ICONS.get(t.get("status", ""), "?")
        print(f"  │   {icon} {t.get('id', '?'):15s} "
              f"[{t.get('type', '?'):12s}] "
              f"{t.get('description', '')[:55]}{dep_str}")

    # AG-UI 事件
    stats = tr.agui_stats
    print(f"  ├─ AG-UI 事件流 {'─' * 42}")
    print(f"  │ {stats.summary_line()}")
    if stats.event_counts:
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(stats.event_counts.items()))
        print(f"  │ 事件明细: {detail}")

    if tr.error:
        print(f"  │ ⚠ 错误: {tr.error[:80]}")
    elif tr.summary:
        print(f"  │ 总结: {tr.summary.replace(chr(10), ' ')[:150]}...")

    print(f"  └{'─' * (_W - 5)}")


def print_summary_report(results: list[TestResult]):
    """打印汇总统计表格 + 诊断建议。"""
    total = len(results)
    if total == 0:
        return

    print(f"\n\n{'═' * _W}")
    print("  汇总报告")
    print(f"{'═' * _W}")
    print(f"{'#':>3} {'任务':>4} {'依赖':>4} {'多步':>4} {'AG-UI':>6} {'耗时':>7}  预期模式")
    print("─" * _W)

    for tr in results:
        err = " ERR" if tr.error else ""
        print(f" {tr.qid:>2}   {tr.n_tasks:>2}     {_check(tr.has_deps)}     "
              f"{_check(tr.is_multi)}      {_check(tr.agui_stats.is_complete)}  "
              f"{tr.elapsed:>5.1f}s  {tr.expected}{err}")

    multi = sum(1 for r in results if r.is_multi)
    dep = sum(1 for r in results if r.has_deps)
    agui_ok = sum(1 for r in results if r.agui_stats.is_complete)
    errs = sum(1 for r in results if r.error)
    pct = lambda n: f"{n}/{total} ({n / total * 100:.0f}%)"

    print(f"\n  总计 {total} 个问题")
    print(f"  多步骤 (≥2 任务): {pct(multi)}")
    print(f"  有依赖关系:       {pct(dep)}")
    print(f"  AG-UI 事件完整:   {pct(agui_ok)}")
    print(f"  平均任务数:       {sum(r.n_tasks for r in results) / total:.1f}")
    print(f"  平均耗时:         {sum(r.elapsed for r in results) / total:.1f}s")
    if errs:
        print(f"  执行失败:         {pct(errs)}")

    if multi < total * 0.5:
        print(f"\n  💡 诊断: 多步骤率较低 ({multi / total * 100:.0f}%)。")
        print("     可能原因:")
        print("     1. Planner prompt 中 '简单问题一个任务解决' 规则过于强势")
        print("     2. SemanticParser 只输出单一 task_type，限制了多类型拆解")
        print("     3. 各 task_type prompt 强调 '一步到位原则'")
        print("     建议: 在 Planner prompt 中增加复合问题多步拆解的引导")


def print_tracker_report(history_db_path: str, session_ids: list[str]):
    """从 TaskHistoryDB 读取并打印执行记录。"""
    from lib.storage.task_history import TaskHistoryDB

    db = TaskHistoryDB(history_db_path)

    print(f"\n\n{'═' * _W}")
    print("  TaskTracker 执行记录")
    print(f"{'═' * _W}")

    total_calls, total_tokens = 0, 0
    for sid in session_ids:
        for task in db.get_tasks_by_session(sid):
            print(task.display())
            total_calls += len(task.llm_calls)
            total_tokens += sum(c.input_tokens + c.output_tokens for c in task.llm_calls)

    if total_calls:
        print(f"\n  LLM 总计: {total_calls} 次调用, {total_tokens:,} tokens")


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
    from lib.llm import LLMFactory
    from chatdb.preprocessing.text_index import TextIndex
    from chatdb.preprocessing.vector_store import ExampleVectorStore
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
    print(f"{'=' * _W}\n  多步骤规划与推理能力测试（AG-UI 协议）\n{'=' * _W}")
    print(f"  数据库: {db_path}  |  YAML: {yml_config}  |  日志: {mode}\n")

    # 检索增强
    text_index_path = Path("data/pilot/text_index.db")
    text_index = TextIndex(db_path=str(text_index_path)) if text_index_path.exists() else None
    example_store = ExampleVectorStore()

    # 连接数据库
    db = DuckDBConnector(database=db_path)
    await db.connect()

    session_ids: list[str] = []
    try:
        tables_meta = db.get_tables_meta()
        for t in tables_meta:
            print(f"  表: {t['table_name']} ({t['row_count']} 行)")
        print()

        # llm = LLMFactory.create(provider="venus", model="glm-5", top_p=0.95)
        llm=LLMFactory.create(provider="hunyuan", temperature=1.0, top_p=0.95)
        def make_orchestrator(sid: str) -> AgentOrchestrator:
            return AgentOrchestrator(
                llm, db,
                yml_config=yml_config,
                tables_meta=tables_meta,
                debug=ARGS.verbose,
                history_db_path=history_db_path,
                history_config=HistoryConfig(num_history_runs=3),
                text_index=text_index,
                example_store=example_store,
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
            queries = QUERIES[:3]
            print(f"  默认运行前 3 个问题（加 --all 运行全部 {len(QUERIES)} 个）\n")

        # 逐个运行
        results: list[TestResult] = []
        for q in queries:
            print(f"{'━' * _W}\n  运行问题 #{q[0]}/{len(QUERIES)}\n{'─' * _W}")
            tr = await run_single_query(make_orchestrator, q)
            print_test_result(tr, q)
            results.append(tr)
            session_ids.append(tr.session_id)

        print_summary_report(results)
        print_tracker_report(history_db_path, session_ids)

    finally:
        await db.disconnect()

    print(f"\n{'═' * _W}\n  测试完成\n{'═' * _W}")


if __name__ == "__main__":
    asyncio.run(main())
