#!/usr/bin/env python3
"""
无 YML 配置的简单查询测试（AG-UI 协议）

验证不带 yml_config 时，整体流程（语义解析 → Planner → SQL → 总结）是否正常。

用法：
    python examples/test_no_yml.py [--query "自定义问题"] [--db path/to/file.duckdb] [-v] [-vv]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="无 YML 简单查询测试")
    p.add_argument("--query", type=str, default=None, help="自定义查询问题")
    p.add_argument("--db", type=str, default="data/duckdb/csv_7dbb24bf.duckdb", help="DuckDB 路径")
    p.add_argument("-v", "--verbose", action="store_true", help="显示详细日志（仅模型输出）")
    p.add_argument("-vv", dest="verbose_llm", action="store_true", help="显示完整 LLM 输入输出")
    args = p.parse_args()
    if args.verbose_llm:
        args.verbose = True
    return args


QUERIES = [
    "这张表一共有多少行数据？",
    "按年份统计总流水",
    "流水最高的5个产品是哪些？",
]


def parse_sse_events(raw_lines: list[str]) -> list[dict]:
    """从 SSE 原始行提取 JSON payload。"""
    events = []
    for raw in raw_lines:
        for line in raw.strip().split("\n"):
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except (json.JSONDecodeError, IndexError):
                    pass
    return events


async def run_query(orchestrator, query: str, verbose: bool) -> dict:
    """通过 AG-UI 适配器执行查询，返回统计信息。"""
    from chatdb.core.agui_adapter import AGUIAdapter

    adapter = AGUIAdapter()
    raw_events: list[str] = []
    t0 = time.time()

    async for chunk in adapter.stream(
        orchestrator=orchestrator,
        query=query,
        thread_id=uuid.uuid4().hex,
        run_id=uuid.uuid4().hex,
    ):
        raw_events.append(chunk)
        if verbose:
            print(f"  [SSE] {chunk.strip()[:120]}")

    elapsed = time.time() - t0
    events = parse_sse_events(raw_events)

    # 提取关键信息
    summary = ""
    snapshot = {}
    event_types = {}
    steps = []
    for e in events:
        t = e.get("type", "")
        event_types[t] = event_types.get(t, 0) + 1
        if t == "TEXT_MESSAGE_CONTENT":
            summary += e.get("delta", "")
        elif t == "STATE_SNAPSHOT":
            snapshot = e.get("snapshot", {})
        elif t == "STEP_STARTED":
            steps.append(e.get("stepName", "?"))

    has_error = any(e.get("type") == "RUN_ERROR" for e in events)
    is_complete = event_types.get("RUN_STARTED", 0) > 0 and (
        event_types.get("RUN_FINISHED", 0) > 0 or has_error
    )

    return {
        "elapsed": elapsed,
        "event_count": len(events),
        "event_types": event_types,
        "steps": steps,
        "summary": summary,
        "snapshot": snapshot,
        "is_complete": is_complete,
        "has_error": has_error,
    }


async def main():
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.llm.factory import LLMFactory
    from chatdb.utils.logger import set_log_level_to_debug, set_log_level_to_info, enable_llm_debug

    args = parse_args()

    if args.verbose_llm:
        set_log_level_to_debug()
        enable_llm_debug(True, show_input=True)
    elif args.verbose:
        set_log_level_to_debug()
        enable_llm_debug(True, show_input=False)
    else:
        set_log_level_to_info()
        enable_llm_debug(False)

    db_path = args.db
    if not Path(db_path).exists():
        print(f"数据库不存在: {db_path}")
        return

    queries = [args.query] if args.query else QUERIES

    # 连接数据库
    db = DuckDBConnector(database=db_path)
    await db.connect()

    try:
        tables_meta = db.get_tables_meta()
        llm = LLMFactory.create(provider="hunyuan", temperature=1.0, top_p=0.95)

        print(f"{'=' * 60}")
        print(f"  无 YML 配置查询测试")
        print(f"{'=' * 60}")
        print(f"  数据库: {db_path}")
        for t in tables_meta:
            print(f"  表: {t['table_name']} ({t['row_count']} 行)")
        print()

        for i, q in enumerate(queries, 1):
            print(f"{'─' * 60}")
            print(f"  [{i}/{len(queries)}] {q}")
            print(f"{'─' * 60}")

            orchestrator = AgentOrchestrator(
                llm, db,
                yml_config=None,  # 不传 YML
                tables_meta=tables_meta,
            )

            result = await run_query(orchestrator, q, args.verbose)

            # 输出结果
            ok = "✓" if result["is_complete"] and not result["has_error"] else "✗"
            print(f"  {ok} 耗时: {result['elapsed']:.1f}s | "
                  f"事件: {result['event_count']} | "
                  f"步骤: {result['steps']}")

            if result["snapshot"]:
                s = result["snapshot"]
                print(f"  数据: {s.get('row_count', 0)} 行 | SQL: {s.get('sql', '')[:80]}")

            if result["summary"]:
                print(f"  回答: {result['summary'][:150]}...")

            if result["has_error"]:
                print(f"  ⚠ 执行出错")
            print()

        # 汇总
        print(f"{'=' * 60}")
        print(f"  测试完成，共 {len(queries)} 个问题")
        print(f"{'=' * 60}")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
