#!/usr/bin/env python3
"""
检索增强端到端测试

验证完整链路：TextIndex 检索 → ContextRetriever → prompt 注入 → LLM 生成 SQL → 执行
重点观察：检索召回了什么、最终 SQL 是什么、查询结果是否正确

用法：
    python examples/test_retrieval_e2e.py [-v|-vv]

    -v    显示 ReAct 步骤
    -vv   显示完整 LLM 输入输出（可看到检索内容被注入 prompt 的位置）
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 解析 verbose 标志
_args = [a for a in sys.argv[1:] if a.startswith("-")]
VERBOSE = "-v" in _args or "--verbose" in _args or "-vv" in _args
VERBOSE_LLM = "-vv" in _args


async def main():
    from chatdb.database.csv import CSVConnector
    from chatdb.core.orchestrator import AgentOrchestrator
    from lib.llm import LLMFactory
    from chatdb.preprocessing.text_index import TextIndex
    from chatdb.preprocessing.vector_store import ExampleVectorStore
    from chatdb.core.context_retriever import ContextRetriever
    from lib.utils.logger import (
        enable_llm_debug,
        set_log_level_to_debug,
        set_log_level_to_info,
    )

    # ─── 日志配置 ─────────────────────────────────────────────
    if VERBOSE_LLM:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=True)
    elif VERBOSE:
        set_log_level_to_debug()
        enable_llm_debug(enable=True, show_input=False)
    else:
        set_log_level_to_info()
        enable_llm_debug(enable=False)

    # ─── 配置 ─────────────────────────────────────────────────
    csv_path = "data/excel/脚本测试数据.csv"
    yml_config = "data/yml/metrics_config.yml"
    session_id = f"retrieval-test-{uuid.uuid4().hex[:8]}"
    query = "王者荣耀的流水是多少"

    print("=" * 70)
    print("  ChatDB 检索增强端到端测试")
    print("=" * 70)
    print(f"  CSV:       {csv_path}")
    print(f"  YAML:      {yml_config}")
    print(f"  Session:   {session_id}")
    print(f"  Query:     {query}")
    mode = "完整LLM" if VERBOSE_LLM else ("详细" if VERBOSE else "精简")
    print(f"  日志模式:   {mode}  (加 -v 看步骤, -vv 看完整 LLM prompt)")
    print("=" * 70)

    # ─── Step 1: 加载 TextIndex ───────────────────────────────
    print("\n[Step 1] 加载 TextIndex (BM25 索引)")
    text_index_path = Path("data/pilot/text_index.db")
    if not text_index_path.exists():
        print(f"  [ERROR] text_index.db 不存在: {text_index_path}")
        print("  请先运行 DataPreprocessor 构建索引")
        return

    text_index = TextIndex(db_path=str(text_index_path))
    import sqlite3
    with sqlite3.connect(str(text_index_path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"  [OK] 文档总数: {total}")

    # ─── Step 2: 预览检索结果 ─────────────────────────────────
    print(f"\n[Step 2] 预览检索召回（query = \"{query}\"）")
    retriever = ContextRetriever(text_index=text_index)
    preview = retriever.retrieve_all(query)

    print(f"\n  --- Schema 召回 ---")
    if preview.relevant_tables:
        for t in preview.relevant_tables:
            print(f"    表: {t['table_name']} (score={t['score']:.3f}) {t.get('description', '')}")
    else:
        print(f"    (无)")

    if preview.relevant_columns:
        for tbl, cols in preview.relevant_columns.items():
            print(f"    列: {tbl} → {', '.join(cols)}")
    else:
        print(f"    列: (无)")

    print(f"\n  --- 值匹配召回 ---")
    if preview.value_matches:
        # 按 table.column 分组显示
        grouped = {}
        for m in preview.value_matches:
            key = f'{m["table_name"]}."{m["column_name"]}"'
            grouped.setdefault(key, []).append(m["matched_value"])
        for col_key, values in grouped.items():
            unique = list(dict.fromkeys(values))[:5]
            print(f"    {col_key}: {', '.join(repr(v) for v in unique)}")
    else:
        print(f"    (无)")

    print(f"\n  --- Few-shot 召回 ---")
    if preview.few_shot_examples:
        for ex in preview.few_shot_examples:
            print(f"    Q: {ex['question']}")
            print(f"    SQL: {ex['sql'][:80]}...")
    else:
        print(f"    (无 — 未配置 ExampleVectorStore)")

    # 打印格式化后的 prompt 段落
    print(f"\n  --- 注入 prompt 的文本 ---")
    for label, text in [
        ("Schema Hint (→ SemanticParser/Planner)", preview.format_schema_hint()),
        ("Value Hint (→ SQLTool)", preview.format_value_hint()),
        ("Few-shot (→ Planner)", preview.format_few_shot()),
    ]:
        if text:
            print(f"\n  [{label}]:")
            for line in text.splitlines():
                print(f"    {line}")
        else:
            print(f"\n  [{label}]: (空)")

    # ─── Step 3: 完整查询（带检索增强）──────────────────────────
    print(f"\n\n{'='*70}")
    print(f"[Step 3] 完整查询 — 启用检索增强")
    print(f"{'='*70}")

    async with CSVConnector(csv_path) as db:
        print(f"  表名: {db.table_name}, 行数: {db.import_info['row_count']}")

        llm = LLMFactory.create(provider="hunyuan")
        orchestrator = AgentOrchestrator(
            llm, db,
            yml_config=yml_config,
            tables_meta=db.get_tables_meta(),
            debug=VERBOSE,
            text_index=text_index,        # ← 启用检索增强
        )

        print(f"\n  查询: {query}")
        print(f"  {'─'*50}")
        result = await orchestrator.process_query(query, session_id=session_id)

        # ─── 输出结果 ─────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  查询结果")
        print(f"{'='*70}")
        print(f"  SQL:  {result.get('sql', 'N/A')}")

        data = result.get("result", [])
        if isinstance(data, list):
            print(f"  行数: {len(data)}")
            for row in data[:10]:
                print(f"    {row}")
            if len(data) > 10:
                print(f"    ... 共 {len(data)} 行")
        else:
            print(f"  结果: {data}")

        print(f"\n  总结: {result.get('summary', 'N/A')}")

    # ─── Step 4: 对比（不启用检索增强）──────────────────────────
    print(f"\n\n{'='*70}")
    print(f"[Step 4] 对比查询 — 不启用检索增强")
    print(f"{'='*70}")

    async with CSVConnector(csv_path) as db:
        llm2 = LLMFactory.create(provider="hunyuan")
        orchestrator_no_retrieval = AgentOrchestrator(
            llm2, db,
            yml_config=yml_config,
            tables_meta=db.get_tables_meta(),
            debug=VERBOSE,
            # 不传 text_index → 无检索增强
        )

        print(f"  查询: {query}")
        print(f"  {'─'*50}")
        session_id2 = f"no-retrieval-{uuid.uuid4().hex[:8]}"
        result2 = await orchestrator_no_retrieval.process_query(query, session_id=session_id2)

        print(f"\n{'='*70}")
        print(f"  查询结果（无检索增强）")
        print(f"{'='*70}")
        print(f"  SQL:  {result2.get('sql', 'N/A')}")

        data2 = result2.get("result", [])
        if isinstance(data2, list):
            print(f"  行数: {len(data2)}")
            for row in data2[:10]:
                print(f"    {row}")
        else:
            print(f"  结果: {data2}")

        print(f"\n  总结: {result2.get('summary', 'N/A')}")

    # ─── 对比总结 ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  对比总结")
    print(f"{'='*70}")
    print(f"  启用检索增强:")
    print(f"    SQL: {result.get('sql', 'N/A')}")
    print(f"  未启用检索增强:")
    print(f"    SQL: {result2.get('sql', 'N/A')}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
