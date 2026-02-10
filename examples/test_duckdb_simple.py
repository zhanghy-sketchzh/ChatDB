#!/usr/bin/env python3
"""直接查询 DuckDB 数据库（无需 Excel 导入）"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    from chatdb.utils.logger import setup_logging, enable_llm_debug
    setup_logging()
    
    if "--debug" in sys.argv or "-d" in sys.argv:
        enable_llm_debug(True)
    
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.llm.factory import LLMFactory
    
    # 查找数据库文件
    db_files = list(Path("data/duckdb").glob("*.duckdb"))
    if not db_files:
        print("未找到 DuckDB 数据库，请先运行 test_excel_simple.py")
        return
    
    db_path = str(db_files[0])
    print(f"数据库: {db_path}")
    
    # 连接数据库
    db = DuckDBConnector(database=db_path)
    await db.connect()
    
    try:
        tables_meta = db.get_tables_meta()
        for t in tables_meta:
            print(f"  - {t['table_name']}: {t['row_count']}行")
        
        # 创建 LLM 和协调器
        llm = LLMFactory.create(provider="hunyuan")
        orchestrator = AgentOrchestrator(llm, db, tables_meta=tables_meta)
        
        # 执行查询
        query = "今年奖金最高的人是谁"
        print(f"\n查询: {query}")
        
        res = await orchestrator.process_query(query)
        print(f"\nSQL: {res['sql']}")
        print(f"结果: {res['result']}")
        print(f"总结: {res['summary']}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
