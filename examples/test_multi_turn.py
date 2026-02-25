#!/usr/bin/env python3
"""
多轮对话测试示例

演示功能：
1. 同一 session_id 下的多轮对话记忆
2. 跨轮次上下文感知（指代消解、省略补全）
3. 查询结果缓存命中
4. 会话历史持久化（SQLite）

用法：
    python examples/test_multi_turn.py [csv_path] [yml_config] [-v|--verbose]

日志模式：
    默认    只显示关键结果（Query / SQL / 行数 / 总结）
    -v      额外显示 ReAct 步骤 trace（DEBUG 级别）
    -vv     显示完整 LLM 输入输出（最详细）
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── 解析 verbose 标志（在导入 chatdb 之前，避免日志初始化顺序问题）─────────────
_args = [a for a in sys.argv[1:] if a.startswith("-")]
VERBOSE = "-v" in _args or "--verbose" in _args or "-vv" in _args
VERBOSE_LLM = "-vv" in _args
# 从 sys.argv 中去除 flag，剩余部分作为位置参数
_pos_args = [a for a in sys.argv[1:] if not a.startswith("-")]


async def main():
    from chatdb.database.csv import CSVConnector
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.llm.factory import LLMFactory
    from chatdb.storage.chat_history import HistoryConfig
    from chatdb.utils.logger import (
        enable_llm_debug,
        set_log_level_to_debug,
        set_log_level_to_info,
    )

    # ─── 按 verbose 级别配置日志 ──────────────────────────────────────────────
    if VERBOSE_LLM:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=True)   # 显示完整 prompt + response
    elif VERBOSE:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=False)  # 只显示 response
    else:
        set_log_level_to_info()
        enable_llm_debug(enable=False)                   # 静默 LLM 细节

    # ===== 配置 =====
    csv_path = _pos_args[0] if len(_pos_args) > 0 else "data/excel/脚本测试数据.csv"
    yml_config = _pos_args[1] if len(_pos_args) > 1 else "data/yml/metrics_config.yml"
    history_db_path = "data/pilot/test_history.db"
    session_id = f"test-{uuid.uuid4().hex[:8]}"

    mode_label = "详细(LLM)" if VERBOSE_LLM else ("详细" if VERBOSE else "精简")
    print(f"CSV:     {csv_path}")
    print(f"YAML:    {yml_config}")
    print(f"History: {history_db_path}")
    print(f"Session: {session_id}")
    print(f"日志模式: {mode_label}  (加 -v 看步骤, 加 -vv 看完整 LLM 输入输出)")
    print("=" * 60)

    queries = [
        "流水最高的5个产品是哪些？",
        "这几个产品的流水分别占总流水的多少？",
    ]

    async with CSVConnector(csv_path) as db:
        print(f"表名: {db.table_name}, 行数: {db.import_info['row_count']}\n")
        llm = LLMFactory.create(provider="hunyuan")
        orchestrator = AgentOrchestrator(
            llm, db,
            yml_config=yml_config,
            tables_meta=db.get_tables_meta(),
            debug=VERBOSE,
            history_db_path=history_db_path,
            history_config=HistoryConfig(num_history_runs=5),
        )

        for i, query in enumerate(queries, 1):
            print(f"\n{'═' * 60}")
            print(f"  轮次 {i} / {len(queries)}   {query}")
            print(f"{'═' * 60}")
            result = await orchestrator.process_query(query, session_id=session_id)


    print(f"\n{'═' * 60}")
    print("  测试完成")


if __name__ == "__main__":
    asyncio.run(main())
