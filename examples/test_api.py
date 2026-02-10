#!/usr/bin/env python3
"""API 测试脚本"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_chat_api():
    """测试 Chat API（直接调用，无需启动服务器）"""
    from chatdb.api.routes.chat import chat_query, get_tables, ChatRequest
    from chatdb.utils.logger import setup_logging
    
    setup_logging()
    
    # 查找数据库
    db_files = list(Path("data/duckdb").glob("*.duckdb"))
    if not db_files:
        print("未找到 DuckDB 数据库")
        return
    
    db_path = str(db_files[0])
    print(f"数据库: {db_path}\n")
    
    # 1. 测试获取表信息
    print("=== 获取表信息 ===")
    tables_result = await get_tables(db_path)
    for t in tables_result["tables"]:
        print(f"  {t['table_name']}: {t['row_count']}行")
    
    # 2. 测试查询
    print("\n=== 执行查询 ===")
    request = ChatRequest(query="今年奖金最高的人是谁", db_path=db_path)
    response = await chat_query(request)
    
    print(f"SQL: {response.sql}")
    print(f"结果: {response.result}")
    print(f"总结: {response.summary}")


async def test_http_api():
    """测试 HTTP API（需要先启动服务器: uvicorn chatdb.api.app:app）"""
    import httpx
    
    base_url = "http://127.0.0.1:8000"
    
    # 查找数据库
    db_files = list(Path("data/duckdb").glob("*.duckdb"))
    db_path = str(db_files[0]) if db_files else ""
    
    async with httpx.AsyncClient() as client:
        # 健康检查
        r = await client.get(f"{base_url}/health")
        print(f"健康检查: {r.json()}")
        
        # 获取表信息
        r = await client.get(f"{base_url}/chat/tables", params={"db_path": db_path})
        print(f"表信息: {r.json()}")
        
        # 执行查询
        r = await client.post(f"{base_url}/chat/", json={
            "query": "今年奖金最高的人是谁",
            "db_path": db_path
        })
        print(f"查询结果: {r.json()}")


if __name__ == "__main__":
    if "--http" in sys.argv:
        asyncio.run(test_http_api())
    else:
        asyncio.run(test_chat_api())
