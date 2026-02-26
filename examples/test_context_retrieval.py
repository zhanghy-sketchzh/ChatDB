"""
检索增强（Context Engineering）功能测试

测试三级检索管道：
1. TextIndex (BM25) — Schema/列/值 召回
2. ExampleVectorStore — Few-shot 示例召回
3. ContextRetriever — 统一检索门面 + RetrievalResult 格式化
"""

import sys
import os

# 确保项目 src 在 path 上
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_text_index():
    """测试 1: TextIndex BM25 检索（使用已有的 text_index.db）"""
    separator("测试 1: TextIndex BM25 检索")
    
    from chatdb.preprocessing.text_index import TextIndex

    # 加载已有的索引库
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "pilot", "text_index.db")
    if not os.path.exists(db_path):
        print(f"[SKIP] text_index.db 不存在: {db_path}")
        print("  请先运行 DataPreprocessor 构建索引")
        return None

    text_index = TextIndex(db_path=db_path)
    print(f"[OK] TextIndex 加载成功，数据库: {db_path}")

    # 统计文档数
    import sqlite3
    with sqlite3.connect(str(db_path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        types = conn.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type").fetchall()
    print(f"  索引文档总数: {total}")
    for doc_type, count in types:
        print(f"  - {doc_type}: {count}")

    # 测试几个典型查询
    test_queries = [
        "各地区销售额",
        "排名前十",
        "华东大区",
        "占比",
    ]

    for q in test_queries:
        print(f"\n--- 查询: '{q}' ---")

        # 表召回
        tables = text_index.search_tables(q, top_k=3)
        if tables:
            print(f"  表召回 ({len(tables)}):")
            for r in tables:
                print(f"    - {r.table_name} (score={r.score:.3f})")

        # 列召回
        cols = text_index.search_columns(q, top_k=5)
        if cols:
            print(f"  列召回 ({len(cols)}):")
            for r in cols[:5]:
                print(f"    - {r.table_name}.{r.column_name} (score={r.score:.3f})")

        # 关键词提取
        keywords = text_index.extract_keywords(q, top_k=5)
        print(f"  关键词: {keywords}")

        # 值匹配
        value_matches = text_index.match_keywords_to_values(q, top_k_keywords=3, top_k_values=5)
        if value_matches:
            print(f"  值匹配 ({len(value_matches)}):")
            for m in value_matches[:5]:
                print(f"    - '{m.keyword}' → {m.table_name}.{m.column_name} = '{m.matched_value}' (score={m.score:.3f})")

    print("\n[OK] TextIndex 测试通过 ✓")
    return text_index


def test_example_vector_store():
    """测试 2: ExampleVectorStore BM25 检索"""
    separator("测试 2: ExampleVectorStore Few-shot 检索")
    
    from chatdb.preprocessing.vector_store import ExampleVectorStore

    # 使用临时内存模式
    store = ExampleVectorStore()
    print(f"[OK] ExampleVectorStore 创建成功 (内存模式)")
    assert store.count == 0, "初始应为空"

    # 添加示例
    examples = [
        {
            "question": "各地区销售额是多少",
            "sql": 'SELECT "大区", SUM("销售额") AS total_sales FROM sales GROUP BY "大区" ORDER BY total_sales DESC',
            "explanation": "按大区分组汇总销售额",
        },
        {
            "question": "排名前十的销售人员",
            "sql": 'SELECT "销售人员", SUM("销售额") AS total FROM sales GROUP BY "销售人员" ORDER BY total DESC LIMIT 10',
            "explanation": "按销售额降序取前10",
        },
        {
            "question": "各月份销售趋势",
            "sql": "SELECT EXTRACT(MONTH FROM \"日期\") AS month, SUM(\"销售额\") FROM sales GROUP BY month ORDER BY month",
            "explanation": "按月份聚合分析趋势",
        },
        {
            "question": "华东大区的产品占比",
            "sql": 'SELECT "产品类型", ROUND(SUM("销售额")*100.0 / (SELECT SUM("销售额") FROM sales WHERE "大区"=\'华东\'), 2) AS pct FROM sales WHERE "大区"=\'华东\' GROUP BY "产品类型"',
            "explanation": "筛选华东大区，计算各产品占比",
        },
        {
            "question": "同比增长率",
            "sql": 'SELECT year, SUM("销售额") AS total, LAG(SUM("销售额")) OVER (ORDER BY year) AS prev, ROUND((SUM("销售额") - LAG(SUM("销售额")) OVER (ORDER BY year)) * 100.0 / LAG(SUM("销售额")) OVER (ORDER BY year), 2) AS yoy_growth FROM sales GROUP BY year',
            "explanation": "使用 LAG 窗口函数计算同比增长",
        },
    ]

    for ex in examples:
        doc_id = store.add_example(**ex)
        print(f"  添加: '{ex['question'][:20]}...' → doc_id={doc_id}")

    print(f"  示例总数: {store.count}")
    assert store.count == len(examples), f"期望 {len(examples)}，实际 {store.count}"

    # 测试 few-shot 检索
    test_queries = [
        "各大区的销售额排名",
        "华东占比多少",
        "增长趋势",
    ]

    for q in test_queries:
        print(f"\n--- Few-shot 查询: '{q}' ---")
        results = store.get_few_shot_examples(q, top_k=2)
        for r in results:
            print(f"  命中: '{r['question']}'")
            print(f"    SQL: {r['sql'][:80]}...")

    # 测试 SQLite 持久化
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name

    try:
        store_persist = ExampleVectorStore(db_path=tmp_path)
        for ex in examples[:2]:
            store_persist.add_example(**ex)
        print(f"\n  持久化写入: {store_persist.count} 条 → {tmp_path}")

        # 重新加载
        store_reload = ExampleVectorStore(db_path=tmp_path)
        print(f"  重新加载: {store_reload.count} 条")
        assert store_reload.count == 2, f"期望 2，实际 {store_reload.count}"

        # 检索
        results = store_reload.get_few_shot_examples("销售额", top_k=2)
        assert len(results) > 0, "重新加载后应能检索到结果"
        print(f"  持久化后检索: {len(results)} 结果 ✓")
    finally:
        os.unlink(tmp_path)

    print("\n[OK] ExampleVectorStore 测试通过 ✓")
    return store


def test_context_retriever(text_index=None, example_store=None):
    """测试 3: ContextRetriever 统一检索 + RetrievalResult 格式化"""
    separator("测试 3: ContextRetriever 统一检索门面")
    
    from chatdb.core.context_retriever import ContextRetriever, RetrievalResult

    # --- 3a: 空检索器（无 index / store） ---
    print("--- 3a: 空检索器（两个依赖均为 None）---")
    empty_retriever = ContextRetriever()
    assert not empty_retriever.has_text_index
    assert not empty_retriever.has_example_store
    result = empty_retriever.retrieve_all("任意查询")
    assert result.is_empty(), "空检索器应返回空结果"
    assert result.format_schema_hint() == ""
    assert result.format_value_hint() == ""
    assert result.format_few_shot() == ""
    print("  [OK] 空检索器优雅降级 ✓")

    # --- 3b: 仅 TextIndex ---
    if text_index is not None:
        print("\n--- 3b: TextIndex 检索 ---")
        retriever = ContextRetriever(text_index=text_index)
        assert retriever.has_text_index
        assert not retriever.has_example_store

        result = retriever.retrieve_schema_hints("销售额排名", top_k_tables=3, top_k_columns=5)
        print(f"  Schema 召回: {len(result.relevant_tables)} 表, {sum(len(v) for v in result.relevant_columns.values())} 列")
        if result.has_schema_hints():
            hint = result.format_schema_hint()
            print(f"  格式化输出:\n{_indent(hint, 4)}")

        result_v = retriever.retrieve_value_hints("华东")
        print(f"\n  值召回: {len(result_v.value_matches)} 匹配")
        if result_v.has_value_hints():
            hint = result_v.format_value_hint()
            print(f"  格式化输出:\n{_indent(hint, 4)}")
    else:
        print("\n--- 3b: [SKIP] TextIndex 不可用 ---")

    # --- 3c: 仅 ExampleVectorStore ---
    if example_store is not None:
        print("\n--- 3c: ExampleVectorStore 检索 ---")
        retriever = ContextRetriever(example_store=example_store)
        assert retriever.has_example_store

        result = retriever.retrieve_few_shot("各地区的销售额", top_k=2)
        print(f"  Few-shot 召回: {len(result.few_shot_examples)} 示例")
        if result.has_few_shot():
            hint = result.format_few_shot()
            print(f"  格式化输出:\n{_indent(hint, 4)}")
    else:
        print("\n--- 3c: [SKIP] ExampleVectorStore 不可用 ---")

    # --- 3d: 完整三级检索 ---
    if text_index is not None or example_store is not None:
        print("\n--- 3d: retrieve_all 全量检索 ---")
        full_retriever = ContextRetriever(
            text_index=text_index,
            example_store=example_store,
        )

        query = "华东大区销售额排名前十"
        merged = full_retriever.retrieve_all(query)
        print(f"  查询: '{query}'")
        print(f"  结果:")
        print(f"    - 表: {len(merged.relevant_tables)}")
        print(f"    - 列: {sum(len(v) for v in merged.relevant_columns.values())}")
        print(f"    - 值匹配: {len(merged.value_matches)}")
        print(f"    - Few-shot: {len(merged.few_shot_examples)}")
        print(f"    - is_empty: {merged.is_empty()}")

        # 打印完整格式化输出
        for label, fmt in [
            ("Schema Hint", merged.format_schema_hint),
            ("Value Hint", merged.format_value_hint),
            ("Few-shot", merged.format_few_shot),
        ]:
            text = fmt()
            if text:
                print(f"\n  [{label}]:")
                print(_indent(text, 4))
    else:
        print("\n--- 3d: [SKIP] 无可用检索组件 ---")

    # --- 3e: RetrievalResult 数据类验证 ---
    print("\n--- 3e: RetrievalResult 数据类 ---")
    r = RetrievalResult(
        relevant_tables=[{"table_name": "test_table", "score": 1.0, "description": "测试表"}],
        relevant_columns={"test_table": ["col_a", "col_b"]},
        value_matches=[{"keyword": "华东", "matched_value": "华东", "table_name": "t", "column_name": "c", "score": 0.9}],
        few_shot_examples=[{"question": "测试问题", "sql": "SELECT 1", "explanation": "测试"}],
    )
    assert r.has_schema_hints()
    assert r.has_value_hints()
    assert r.has_few_shot()
    assert not r.is_empty()
    assert "test_table" in r.format_schema_hint()
    assert "华东" in r.format_value_hint()
    assert "测试问题" in r.format_few_shot()
    print("  [OK] RetrievalResult 数据类验证通过 ✓")

    print("\n[OK] ContextRetriever 测试通过 ✓")


def test_react_state_integration():
    """测试 4: ReActState 集成 — retrieval_context 字段"""
    separator("测试 4: ReActState.retrieval_context 字段")
    
    from chatdb.core.react_state import ReActState
    from chatdb.core.context_retriever import RetrievalResult

    state = ReActState(user_query="测试查询")
    assert state.retrieval_context is None, "默认应为 None"
    print("  [OK] 默认值为 None ✓")

    # 赋值
    rc = RetrievalResult(
        relevant_tables=[{"table_name": "sales", "score": 0.8, "description": ""}],
    )
    state.retrieval_context = rc
    assert state.retrieval_context is rc
    assert state.retrieval_context.has_schema_hints()
    print("  [OK] 可正常赋值和读取 RetrievalResult ✓")

    # 通过 getattr 安全访问（模拟各 Agent 的使用方式）
    rc_safe = getattr(state, "retrieval_context", None)
    assert rc_safe is not None
    assert hasattr(rc_safe, "format_schema_hint")
    hint = rc_safe.format_schema_hint()
    assert "sales" in hint
    print(f"  [OK] getattr 安全访问模式验证通过 ✓")
    print(f"  格式化: {hint[:50]}...")

    print("\n[OK] ReActState 集成测试通过 ✓")


def _indent(text: str, spaces: int) -> str:
    """辅助：缩进文本"""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def main():
    print("=" * 60)
    print("  ChatDB 检索增强（Context Engineering）功能测试")
    print("=" * 60)

    # 1. TextIndex
    text_index = test_text_index()

    # 2. ExampleVectorStore
    example_store = test_example_vector_store()

    # 3. ContextRetriever（核心集成测试）
    test_context_retriever(text_index=text_index, example_store=example_store)

    # 4. ReActState 集成
    test_react_state_integration()

    # 总结
    separator("测试总结")
    print("  所有测试通过 ✓")
    print()
    print("  组件状态:")
    print(f"    TextIndex:          {'可用' if text_index else '跳过 (无 text_index.db)'}")
    print(f"    ExampleVectorStore: 可用")
    print(f"    ContextRetriever:   可用")
    print(f"    ReActState 集成:     可用")
    print()


if __name__ == "__main__":
    main()
