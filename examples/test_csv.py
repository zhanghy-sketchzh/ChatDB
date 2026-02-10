#!/usr/bin/env python3
"""CSV 文件查询示例"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    from chatdb.database.csv import CSVConnector
    from chatdb.core.orchestrator import AgentOrchestrator
    from chatdb.llm.factory import LLMFactory
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/excel/脚本测试数据.csv"
    query = sys.argv[2] if len(sys.argv) > 2 else "手游在 IEG 总流水中的占比是多少"
    yml_config = sys.argv[3] if len(sys.argv) > 3 else "data/yml/metrics_config.yml"
    
    async with CSVConnector(csv_path) as db:
        print(f"CSV: {csv_path}")
        print(f"表名: {db.table_name}, 行数: {db.import_info['row_count']}")
        print(f"YAML: {yml_config}")
        
        llm = LLMFactory.create(provider="hunyuan")
        orchestrator = AgentOrchestrator(
            llm, 
            db, 
            yml_config=yml_config,
            tables_meta=db.get_tables_meta(),
            debug=True,  # debug 模式默认不显示 LLM 输入
        )
        
        print(f"\n查询: {query}")
        result = await orchestrator.process_query(query)
        
        print(f"SQL: {result['sql']}")
        print(f"结果: {result['result'][:5]}..." if len(result['result']) > 5 else f"结果: {result['result']}")
        print(f"总结: {result['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
