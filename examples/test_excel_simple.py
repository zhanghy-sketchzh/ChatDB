#!/usr/bin/env python3
"""精简测试：今年奖金最高的人是谁"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    from chatdb.utils.logger import setup_logging, enable_llm_debug
    setup_logging()
    
    # 启用 LLM Debug 模式，查看完整的模型输入输出
    # 通过命令行参数控制: python test_excel_simple.py --debug 或 -d
    if "--debug" in sys.argv or "-d" in sys.argv:
        enable_llm_debug(True)
    
    from chatdb.database.excel.register import ExcelConnector
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.llm.factory import LLMFactory
    
    excel_file = "data/excel/奖金数据模版5.xlsx"
    
    # 创建 LLM
    llm = LLMFactory.create(provider="hunyuan")
    
    # 导入 Excel 数据到 DuckDB
    excel_service = ExcelConnector(llm_client=llm)
    result = await excel_service.process_excel_multi_tables(
        excel_file_path=excel_file,
        original_filename=Path(excel_file).name,
        force_reimport=False,
    )
    db_path = result['db_path']
    tables_meta = result.get('tables', [])  # 获取表元数据
    print(f"数据库: {db_path}")
    
    # 使用 DuckDB 连接器查询
    db = DuckDBConnector(database=db_path)
    async with db:
        # 传入表元数据，包含 row_count、table_description 等
        orchestrator = AgentOrchestrator(llm, db, tables_meta=tables_meta)
        query = "今年奖金最高的人是谁"
        print(f"\n查询: {query}")
        
        res = await orchestrator.process_query(query)
        
        print(f"\nSQL: {res['sql']}")
        print(f"结果: {res['result']}")
        print(f"总结: {res['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
