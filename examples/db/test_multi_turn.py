#!/usr/bin/env python3
"""
多轮对话测试示例

演示功能：
  1. 同一 session_id 下的多轮对话记忆
  2. 跨轮次上下文感知（指代消解、省略补全）
  3. 查询结果缓存命中
  4. 会话历史持久化（SQLite）

用法：
    python examples/test_multi_turn.py [-v|-vv] [--all]
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
    p = argparse.ArgumentParser(description="多轮对话测试")
    p.add_argument("-v", "--verbose", action="store_true", help="显示 ReAct 步骤 + AG-UI 摘要")
    p.add_argument("-vv", dest="verbose_llm", action="store_true", help="显示完整 LLM I/O")
    p.add_argument("--all", action="store_true", help="运行全部对话轮次（默认前 2 个）")
    args = p.parse_args()
    if args.verbose_llm:
        args.verbose = True
    return args


ARGS = parse_args()

# ─── 测试用例 ─────────────────────────────────────────────────────────────────

CONVERSATION_SETS = [
    # (id, turns: [(query, expected_pattern, why)], description)
    (1,
     [
         ("流水最高的5个产品是哪些？", "ranking", "基础排名查询"),
         ("这几个产品的流水分别占总流水的多少？", "ratio(drilldown)", "指代消解: 'this几个产品'指前一轮的Top5"),
     ],
     "多轮指代消解与上下文感知"),
]


# ─── 对话轮次结果 ─────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """单轮对话的执行结果"""

    turn_id: int
    query: str
    expected: str
    elapsed: float = 0.0
    summary: str = ""
    error: str = ""
    task_info: dict = field(default_factory=dict)  # 任务统计信息
    cache_hit: bool = False


@dataclass
class ConversationResult:
    """多轮对话的汇总结果"""

    conv_id: int
    description: str
    turns: list[TurnResult] = field(default_factory=list)
    session_id: str = ""
    total_elapsed: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return all(not t.error for t in self.turns)

    @property
    def cache_hits(self) -> int:
        return sum(1 for t in self.turns if t.cache_hit)


# ─── 核心执行 ─────────────────────────────────────────────────────────────────

async def run_single_turn(orchestrator, query: str, session_id: str) -> TurnResult:
    """执行单个对话轮次。"""
    tr = TurnResult(turn_id=0, query=query, expected="")
    t0 = time.time()

    try:
        result = await orchestrator.process_query(query, session_id=session_id)
        tr.elapsed = time.time() - t0

        # 提取结果摘要
        if isinstance(result, dict):
            tr.summary = result.get("summary", "")
            task_info = result.get("task_info", {})
            tr.task_info = task_info
            tr.cache_hit = task_info.get("cache_hit", False)
        else:
            tr.summary = str(result)[:200]

    except Exception as e:
        tr.elapsed = time.time() - t0
        tr.error = str(e)
        traceback.print_exc()

    return tr


async def run_conversation(orchestrator_factory, conv_spec: tuple) -> ConversationResult:
    """执行一个完整的多轮对话。"""
    conv_id, turns_spec, description = conv_spec
    cr = ConversationResult(conv_id=conv_id, description=description)
    cr.session_id = f"mturn-{uuid.uuid4().hex[:8]}"

    # 对每个数据库连接创建一次 Orchestrator（保证会话状态一致）
    orchestrator = orchestrator_factory(cr.session_id)

    t0 = time.time()
    try:
        for turn_idx, (query, expected, why) in enumerate(turns_spec, start=1):
            if ARGS.verbose:
                print(f"    轮次 {turn_idx}: {query[:60]}...")

            tr = await run_single_turn(orchestrator, query, cr.session_id)
            tr.turn_id = turn_idx
            tr.expected = expected
            cr.turns.append(tr)

            if ARGS.verbose_llm:
                print(f"      摘要: {tr.summary[:100]}")
                print(f"      耗时: {tr.elapsed:.2f}s, 缓存命中: {tr.cache_hit}")

        cr.total_elapsed = time.time() - t0

    except Exception as e:
        cr.total_elapsed = time.time() - t0
        cr.error = str(e)

    return cr


# ─── 输出格式 ─────────────────────────────────────────────────────────────────

_W = 80  # 输出宽度

def _check(b: bool) -> str:
    return "✓" if b else "✗"


def print_conversation_result(cr: ConversationResult):
    """格式化打印单个多轮对话结果。"""
    print(f"\n  ┌─ 对话 #{cr.conv_id}: {cr.description[:50]} {'─' * 20}")
    print(f"  │ Session: {cr.session_id}")
    print(f"  │ 轮数: {len(cr.turns)} │ 成功: {_check(cr.success)} │ 缓存命中: {cr.cache_hits}/{len(cr.turns)} │ "
          f"总耗时: {cr.total_elapsed:.1f}s")

    for tr in cr.turns:
        status = "✓" if not tr.error else "✗"
        cache_tag = " [缓存]" if tr.cache_hit else ""
        print(f"  │ ")
        print(f"  │   {status} 轮次 {tr.turn_id}: {tr.query[:60]}{'...' if len(tr.query) > 60 else ''}")
        print(f"  │     预期: {tr.expected} │ 耗时: {tr.elapsed:.2f}s{cache_tag}")
        if tr.summary:
            summary_line = tr.summary.replace("\n", " ")[:120]
            print(f"  │     摘要: {summary_line}...")
        if tr.error:
            print(f"  │     ⚠ 错误: {tr.error[:100]}")

    print(f"  └{'─' * (_W - 5)}")


def print_summary_report(results: list[ConversationResult]):
    """打印汇总统计表格。"""
    total = len(results)
    if total == 0:
        return

    print(f"\n\n{'═' * _W}")
    print("  汇总报告")
    print(f"{'═' * _W}")
    print(f"{'#':>3} {'轮数':>4} {'缓存':>6} {'成功':>4} {'总耗时':>8}  对话描述")
    print("─" * _W)

    for cr in results:
        cache_pct = f"{cr.cache_hits}/{len(cr.turns)}" if cr.turns else "0/0"
        err = " ERR" if cr.error else ""
        print(f" {cr.conv_id:>2}    {len(cr.turns):>2}   {cache_pct:>6}   {_check(cr.success):>3}  "
              f"{cr.total_elapsed:>6.1f}s  {cr.description}{err}")

    success = sum(1 for r in results if r.success)
    total_turns = sum(len(r.turns) for r in results)
    total_cache = sum(r.cache_hits for r in results)
    total_elapsed = sum(r.total_elapsed for r in results)

    print(f"\n  总计 {total} 个对话，{total_turns} 轮")
    print(f"  成功率:        {success}/{total} ({success / total * 100:.0f}%)")
    print(f"  缓存命中率:    {total_cache}/{total_turns} ({total_cache / total_turns * 100:.0f}%)" if total_turns > 0 else "  缓存命中率:    0/0")
    print(f"  平均对话耗时:  {total_elapsed / total:.1f}s")


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
    from chatdb.database.csv import CSVConnector
    from lib.llm import LLMFactory
    from chatdb.preprocessing.text_index import TextIndex
    from chatdb.preprocessing.vector_store import ExampleVectorStore
    from lib.storage.chat_history import HistoryConfig

    setup_logging()

    csv_path = "data/excel/脚本测试数据.csv"
    yml_config = "data/yml/metrics_config.yml"
    history_db_path = "data/pilot/test_history.db"

    if not Path(csv_path).exists():
        print(f"CSV 文件不存在: {csv_path}")
        return

    # Banner
    mode = "LLM详细" if ARGS.verbose_llm else ("详细" if ARGS.verbose else "精简")
    print(f"{'=' * _W}\n  多轮对话测试\n{'=' * _W}")
    print(f"  CSV: {csv_path}")
    print(f"  YAML: {yml_config}")
    print(f"  历史: {history_db_path}")
    print(f"  日志: {mode}\n")

    # 检索增强
    text_index_path = Path("data/pilot/text_index.db")
    text_index = TextIndex(db_path=str(text_index_path)) if text_index_path.exists() else None
    example_store = ExampleVectorStore()

    # 连接数据库
    async with CSVConnector(csv_path) as db:
        print(f"  表: {db.table_name} ({db.import_info['row_count']} 行)\n")

        llm = LLMFactory.create(provider="hunyuan", model="hunyuan-2.0-thinking-20251109", temprature=1.0, top_p=0.95)
        # llm = LLMFactory.create(provider="venus", model="glm-5", temprature=1.0,top_p=0.95)
        def make_orchestrator(session_id: str) -> AgentOrchestrator:
            return AgentOrchestrator(
                llm, db,
                yml_config=yml_config,
                tables_meta=db.get_tables_meta(),
                debug=ARGS.verbose,
                history_db_path=history_db_path,
                history_config=HistoryConfig(num_history_runs=5),
                text_index=text_index,
                example_store=example_store,
            )

        # 选择对话集
        if ARGS.all:
            conversations = CONVERSATION_SETS
        else:
            conversations = CONVERSATION_SETS[:1]
            print(f"  默认运行第 1 个对话集（加 --all 运行全部 {len(CONVERSATION_SETS)} 个）\n")

        # 逐个运行
        results: list[ConversationResult] = []
        for conv_spec in conversations:
            print(f"{'━' * _W}\n  运行对话 #{conv_spec[0]}/{len(CONVERSATION_SETS)}\n{'─' * _W}")
            cr = await run_conversation(make_orchestrator, conv_spec)
            print_conversation_result(cr)
            results.append(cr)

        print_summary_report(results)

    print(f"\n{'═' * _W}\n  测试完成\n{'═' * _W}")


if __name__ == "__main__":
    asyncio.run(main())
