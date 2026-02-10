#!/usr/bin/env python3
"""
Excel 数据处理示例

演示如何使用 ChatDB 处理 Excel 文件并存储到 DuckDB
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 初始化日志系统
from chatdb.utils.logger import setup_logging
setup_logging()

from chatdb.database.excel.register import ExcelConnector
from chatdb.llm.factory import LLMFactory


async def main():
    """主函数"""
    # Excel 文件路径
    excel_file_path = "data/excel/奖金数据模版5.xlsx"
    
    if not Path(excel_file_path).exists():
        print(f"错误: Excel 文件不存在: {excel_file_path}")
        return
    
    # 创建混元 LLM 客户端（自动从 config.toml 或环境变量读取配置）
    llm_client = LLMFactory.create(provider="hunyuan")
    
    print(f"使用混元 LLM: {llm_client.model}")
    
    # 创建 Excel 服务并处理文件
    excel_service = ExcelConnector(llm_client=llm_client)
    result = await excel_service.process_excel_multi_tables(
        excel_file_path=excel_file_path,
        original_filename=Path(excel_file_path).name,
        force_reimport=True,  # 强制重新导入，触发 LLM 调用
    )
    
    # 打印结果摘要
    print(f"\n处理完成: {result['status']}")
    print(f"数据库路径: {result['db_path']}")
    print(f"表数量: {len(result['tables'])}")
    
    for table in result['tables']:
        print(f"  - {table['table_name']} ({table['row_count']}行, {table['column_count']}列)")


if __name__ == "__main__":
    asyncio.run(main())
