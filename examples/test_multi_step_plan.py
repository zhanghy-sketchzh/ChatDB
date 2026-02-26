#!/usr/bin/env python3
"""
多步骤规划与推理能力测试

测试目标：
  1. 验证 Planner 能否为复杂问题生成多任务 DAG（带 depends_on）
  2. 验证 Orchestrator 按拓扑序执行、传递上游结果的能力
  3. 验证 decide_next_action 动态调整计划的能力

设计思路：
  当前系统中 SemanticParser 只输出单一 task_type，Planner prompt 也倾向于
  "一个任务解决"。因此测试问题从两个角度设计：

  A. 同类型但必须多步（一个 SQL 解不了）：
     - 先查Top产品 → 再对Top结果下钻
     - 先查两个不同口径的数据 → 再对比

  B. 复合问题（包含不同分析需求的子问题）：
     - "流水是多少？主要来自哪里？趋势如何？"
     - 这些问题即使 Planner 只生成1个任务，也可以观察 decide_next_action
       是否会在执行后追加任务

用法：
    python examples/test_multi_step_plan.py [-v|-vv] [--query N] [--all]

    -v      显示 ReAct 步骤
    -vv     显示完整 LLM I/O
    --query N   只运行第 N 个问题（1-based）
    --all       运行所有问题（默认只运行前3个作为快速检查）
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── CLI args ────────────────────────────────────────────────────────────────
_flags = [a for a in sys.argv[1:] if a.startswith("-")]
VERBOSE = "-v" in _flags or "--verbose" in _flags or "-vv" in _flags
VERBOSE_LLM = "-vv" in _flags
RUN_ALL = "--all" in _flags

ONLY_QUERY = None
for i, a in enumerate(sys.argv[1:], 1):
    if a == "--query" and i < len(sys.argv) - 1:
        ONLY_QUERY = int(sys.argv[i + 1])
        break

# ─── 测试用例 ──────────────────────────────────────────────────────────────────

MULTI_STEP_QUERIES = [
    # ━━ 1. 排名 + 下钻（必须两步：先找Top，再对Top结果细查）━━
    {
        "id": 1,
        "query": "2025年流水最高的3个产品是哪些？然后告诉我这3个产品各自在国内和海外的流水分别是多少",
        "expected_pattern": "ranking → drilldown/source",
        "expected_min_tasks": 2,
        "why_multi_step": "下钻任务需要先知道Top3是谁，才能展开",
    },
    # ━━ 2. 两个指标对比（不同口径，一个SQL难以同时搞定）━━
    {
        "id": 2,
        "query": "分别统计2024年和2025年IEG本部的总流水，然后计算同比增长率，并按产品大类拆解增长来源",
        "expected_pattern": "comparison → source",
        "expected_min_tasks": 2,
        "why_multi_step": "先要算出总量同比，再按产品大类拆解增量贡献",
    },
    # ━━ 3. 趋势 + 异常定位 ━━
    {
        "id": 3,
        "query": "先给我看2022到2025年IEG每年的总流水趋势，然后找出流水下降最多的年份，并分析该年份哪些产品的流水跌幅最大",
        "expected_pattern": "trend → comparison → source/ranking",
        "expected_min_tasks": 2,
        "why_multi_step": "需要先看趋势找到下降年份，再针对该年份做产品级分析",
    },
    # ━━ 4. 占比 + 跨年对比 ━━
    {
        "id": 4,
        "query": "2024年各产品大类在总流水中的占比是多少？2025年的占比又是多少？哪些品类占比在上升？",
        "expected_pattern": "ratio(2024) + ratio(2025) → comparison",
        "expected_min_tasks": 2,
        "why_multi_step": "需要分别算两年占比，再对比变化方向",
    },
    # ━━ 5. 基础 + 排名 + 利润率（三步递进）━━
    {
        "id": 5,
        "query": "2025年IEG的递延后利润总额是多少？利润最高的Top5产品分别是什么？这5个产品的利润率（利润/流水）各是多少？",
        "expected_pattern": "basic → ranking → ratio(drilldown)",
        "expected_min_tasks": 3,
        "why_multi_step": "三个独立子问题，后续问题依赖前面的结果",
    },
    # ━━ 6. 实际 vs 预测数据对比 ━━
    {
        "id": 6,
        "query": "对比一下2025年实际数据和预测数据的总流水差异，然后找出差异最大的5个产品",
        "expected_pattern": "comparison(实际vs预测) → ranking(差异Top5)",
        "expected_min_tasks": 2,
        "why_multi_step": "先算总体差异，再对差异做产品级排名",
    },
    # ━━ 7. 成本结构拆解 + 成本率对比 ━━
    {
        "id": 7,
        "query": "2025年IEG的各项成本（按大盘报表项）分别是多少？运营成本占总流水的比例是多少？跟2024年比变化如何？",
        "expected_pattern": "source(成本拆解) → ratio(成本率) → comparison(同比)",
        "expected_min_tasks": 2,
        "why_multi_step": "先拆解成本结构，再算成本率，再做同比",
    },
    # ━━ 8. 国内/海外 × 年份的矩阵分析 ━━
    {
        "id": 8,
        "query": "2023年、2024年、2025年，国内流水和海外流水分别是多少？国内和海外的年增速分别是多少？哪边增速更快？",
        "expected_pattern": "source(国内/海外×年) → comparison(增速) → summary",
        "expected_min_tasks": 2,
        "why_multi_step": "先拆出国内/海外三年数据，再计算各自增速并对比",
    },
]


class TestResult:
    """单个测试用例的执行结果"""

    def __init__(self, qid: int, query: str, expected: str, min_tasks: int):
        self.qid = qid
        self.query = query
        self.expected = expected
        self.min_tasks = min_tasks
        self.actual_tasks: list[dict] = []
        self.has_deps = False
        self.summary = ""
        self.error = ""
        self.elapsed = 0.0
        self.session_id = ""
        # decide_next_action 是否追加过任务
        self.had_adjustment = False

    @property
    def n_tasks(self) -> int:
        return len(self.actual_tasks)

    @property
    def is_multi(self) -> bool:
        return self.n_tasks >= 2


def print_divider(char="━", width=70):
    print(char * width)


async def run_single_query(
    orchestrator_factory, q: dict, db
) -> TestResult:
    """运行单个测试问题"""
    tr = TestResult(q["id"], q["query"], q["expected_pattern"], q["expected_min_tasks"])
    session_id = f"msp-{uuid.uuid4().hex[:8]}"
    tr.session_id = session_id

    orchestrator = orchestrator_factory(session_id)

    t0 = time.time()
    try:
        result = await orchestrator.process_query(query=q["query"], session_id=session_id)
        tr.elapsed = time.time() - t0
        tr.summary = result.get("summary", "")

        # 提取 Plan 信息
        plan_path = Path(f"data/scratch/{session_id}/plan.json")
        if plan_path.exists():
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
            tasks = plan_data.get("tasks", [])
            tr.actual_tasks = tasks
            tr.has_deps = any(bool(t.get("depends_on")) for t in tasks)

        # 检查 plan 中是否有 adjust 过的任务（meta 中有 retry_hint 等）
        for t in tr.actual_tasks:
            meta = t.get("meta", {})
            if meta.get("retry_hint") or meta.get("adjusted"):
                tr.had_adjustment = True
                break

    except Exception as e:
        tr.elapsed = time.time() - t0
        tr.error = str(e)

    return tr


def print_test_result(tr: TestResult, q: dict):
    """格式化打印单个测试结果"""
    print(f"\n  ┌─ 问题 #{tr.qid} {'─' * 50}")
    print(f"  │ {tr.query[:80]}{'...' if len(tr.query) > 80 else ''}")
    print(f"  │ 预期: {tr.expected}")
    print(f"  │ 原因: {q['why_multi_step']}")
    print(f"  ├─ 执行结果 {'─' * 46}")
    print(f"  │ 耗时: {tr.elapsed:.1f}s │ 任务数: {tr.n_tasks} (期望≥{tr.min_tasks}) │ "
          f"多步: {'✓' if tr.is_multi else '✗'} │ 依赖: {'✓' if tr.has_deps else '✗'}")

    if tr.actual_tasks:
        for t in tr.actual_tasks:
            deps = t.get("depends_on", [])
            dep_str = f" ← [{','.join(deps)}]" if deps else ""
            status_icon = {"completed": "✓", "failed": "✗", "skipped": "○"}.get(
                t.get("status", ""), "?")
            print(f"  │   {status_icon} {t.get('id', '?'):15s} "
                  f"[{t.get('type', '?'):12s}] "
                  f"{t.get('description', '')[:55]}{dep_str}")

    if tr.error:
        print(f"  │ ⚠ 错误: {tr.error[:80]}")
    elif tr.summary:
        preview = tr.summary.replace("\n", " ")[:150]
        print(f"  │ 总结: {preview}...")

    print(f"  └{'─' * 65}")


def print_tracker_report(history_db_path: str, session_ids: list[str]):
    """从 TaskHistoryDB 读取并打印所有测试的执行记录"""
    from chatdb.storage.task_history import TaskHistoryDB

    db = TaskHistoryDB(history_db_path)

    print(f"\n\n{'═' * 70}")
    print("  TaskTracker 执行记录")
    print(f"{'═' * 70}")

    total_llm_calls = 0
    total_tokens = 0

    for sid in session_ids:
        tasks = db.get_tasks_by_session(sid)
        for task in tasks:
            print(task.display())

            # 累计 LLM 统计
            total_llm_calls += len(task.llm_calls)
            for c in task.llm_calls:
                total_tokens += c.input_tokens + c.output_tokens

    if total_llm_calls > 0:
        print(f"\n  LLM 总计: {total_llm_calls} 次调用, {total_tokens:,} tokens")


async def main():
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.llm.factory import LLMFactory
    from chatdb.preprocessing.text_index import TextIndex
    from chatdb.preprocessing.vector_store import ExampleVectorStore
    from chatdb.storage.chat_history import HistoryConfig
    from chatdb.utils.logger import (
        enable_llm_debug,
        set_log_level_to_debug,
        set_log_level_to_info,
    )

    if VERBOSE_LLM:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=True)
    elif VERBOSE:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=False)
    else:
        set_log_level_to_info()
        enable_llm_debug(enable=False)

    # ── 配置 ──
    db_path = "data/duckdb/csv_7dbb24bf.duckdb"
    yml_config = "data/yml/metrics_config.yml"
    history_db_path = "data/pilot/test_history.db"

    if not Path(db_path).exists():
        print(f"数据库不存在: {db_path}")
        return

    print("=" * 70)
    print("  多步骤规划与推理能力测试")
    print("=" * 70)
    print(f"  数据库:   {db_path}")
    print(f"  YAML:     {yml_config}")
    mode_label = "LLM详细" if VERBOSE_LLM else ("详细" if VERBOSE else "精简")
    print(f"  日志模式: {mode_label}  (-v 看步骤, -vv 看LLM)")
    print()

    # ── 检索增强 ──
    text_index_path = Path("data/pilot/text_index.db")
    text_index = TextIndex(db_path=str(text_index_path)) if text_index_path.exists() else None
    example_store = ExampleVectorStore()

    # ── 连接 ──
    db = DuckDBConnector(database=db_path)
    await db.connect()

    session_ids: list[str] = []

    try:
        tables_meta = db.get_tables_meta()
        for t in tables_meta:
            print(f"  表: {t['table_name']} ({t['row_count']} 行)")
        print()

        llm = LLMFactory.create(provider="hunyuan")

        # Orchestrator 工厂（每个问题创建新实例，避免状态污染）
        def make_orchestrator(_sid: str) -> AgentOrchestrator:
            return AgentOrchestrator(
                llm, db,
                yml_config=yml_config,
                tables_meta=tables_meta,
                debug=VERBOSE,
                history_db_path=history_db_path,
                history_config=HistoryConfig(num_history_runs=3),
                text_index=text_index,
                example_store=example_store,
            )

        # ── 选择问题 ──
        if ONLY_QUERY is not None:
            queries = [q for q in MULTI_STEP_QUERIES if q["id"] == ONLY_QUERY]
            if not queries:
                print(f"未找到问题 #{ONLY_QUERY}")
                return
        elif RUN_ALL:
            queries = MULTI_STEP_QUERIES
        else:
            queries = MULTI_STEP_QUERIES[:3]
            print(f"  默认运行前 3 个问题（加 --all 运行全部 {len(MULTI_STEP_QUERIES)} 个）")
            print()

        results: list[TestResult] = []
        for q in queries:
            print_divider()
            print(f"  运行问题 #{q['id']}/{len(MULTI_STEP_QUERIES)}")
            print_divider("─")

            tr = await run_single_query(make_orchestrator, q, db)
            print_test_result(tr, q)
            results.append(tr)
            session_ids.append(tr.session_id)

        # ═══ 汇总报告 ═══
        print(f"\n\n{'═' * 70}")
        print("  汇总报告")
        print(f"{'═' * 70}")

        header = f"{'#':>3} {'任务':>4} {'依赖':>4} {'多步':>4} {'耗时':>7}  预期模式"
        print(header)
        print("─" * 70)

        for tr in results:
            m = "✓" if tr.is_multi else "✗"
            d = "✓" if tr.has_deps else "✗"
            e = " ERR" if tr.error else ""
            print(f" {tr.qid:>2}   {tr.n_tasks:>2}     {d}     {m}   "
                  f"{tr.elapsed:>5.1f}s  {tr.expected}{e}")

        total = len(results)
        multi = sum(1 for r in results if r.is_multi)
        dep = sum(1 for r in results if r.has_deps)
        avg_t = sum(r.n_tasks for r in results) / max(1, total)
        avg_s = sum(r.elapsed for r in results) / max(1, total)
        errs = sum(1 for r in results if r.error)

        print(f"\n  总计 {total} 个问题")
        print(f"  多步骤 (≥2 任务): {multi}/{total} ({multi/total*100:.0f}%)")
        print(f"  有依赖关系:       {dep}/{total} ({dep/total*100:.0f}%)")
        print(f"  平均任务数:       {avg_t:.1f}")
        print(f"  平均耗时:         {avg_s:.1f}s")
        if errs:
            print(f"  执行失败:         {errs}/{total}")

        # 诊断建议
        if multi < total * 0.5:
            print(f"\n  💡 诊断: 多步骤率较低 ({multi/total*100:.0f}%)。")
            print("     可能原因:")
            print("     1. Planner prompt 中 '简单问题一个任务解决' 规则过于强势")
            print("     2. SemanticParser 只输出单一 task_type，限制了多类型拆解")
            print("     3. 各 task_type prompt 强调 '一步到位原则'")
            print("     建议: 在 Planner prompt 中增加复合问题多步拆解的引导")

        # ═══ TaskTracker 执行记录 ═══
        print_tracker_report(history_db_path, session_ids)

    finally:
        await db.disconnect()

    print(f"\n{'═' * 70}")
    print("  测试完成")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
