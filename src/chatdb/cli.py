"""
命令行入口

提供 ChatDB 的命令行接口。
"""

import argparse
import asyncio
import sys


def run_server(host: str, port: int, reload: bool = False) -> None:
    """启动 API 服务器"""
    import uvicorn

    uvicorn.run(
        "chatdb.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


async def interactive_query(query: str, yml_config: str | None = None) -> None:
    """交互式查询"""
    from chatdb.core import AgentOrchestrator
    from chatdb.utils.logger import setup_logging
    from chatdb.database.base import create_connector
    from chatdb.llm.factory import LLMFactory

    setup_logging()

    connector = create_connector()
    async with connector:
        llm = LLMFactory.create()
        orchestrator = AgentOrchestrator(
            llm, connector, 
            yml_config=yml_config,
        )

        result = await orchestrator.process_query(query)

        print("\n" + "=" * 60)
        print("📝 原始查询:", result["query"])
        print("\n💾 生成的 SQL:")
        print(result["sql"])
        print("\n📊 查询结果:")
        if result["result"]:
            for row in result["result"][:10]:
                print(row)
            if len(result["result"]) > 10:
                print(f"... 共 {result['row_count']} 条记录")
        else:
            print("无数据")
        print("\n📋 结果总结:")
        print(result["summary"])
        print("=" * 60)


def main() -> None:
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="ChatDB - 基于 LLM 多智能体的自然语言数据库查询系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  chatdb serve                    # 启动 API 服务器
  chatdb serve --port 9000        # 指定端口启动
  chatdb query "查询所有用户"      # 交互式查询
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务器")
    serve_parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")
    serve_parser.add_argument("--reload", action="store_true", help="开发模式（热重载）")

    # query 命令
    query_parser = subparsers.add_parser("query", help="交互式查询")
    query_parser.add_argument("text", help="自然语言查询")
    query_parser.add_argument("--yml", type=str, default=None, help="YAML 配置文件路径")

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args.host, args.port, args.reload)
    elif args.command == "query":
        asyncio.run(interactive_query(args.text, yml_config=args.yml))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

