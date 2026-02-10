#!/usr/bin/env python3
"""
简单的 DuckDB SQL 执行脚本
直接在代码中修改数据库路径和SQL语句
"""
import duckdb
from pathlib import Path

# ========== 配置区域 - 在这里修改 ==========

# 数据库文件路径
DB_PATH = "/Users/luchun/Desktop/work/ChatDB/data/duckdb/excel_26253606.duckdb"

# 表名
TABLE_NAME = "年终奖分配明细表_国内"

# 要执行的SQL (支持多条,用列表)
SQL_QUERIES = [
    '''SELECT * FROM "年终奖分配明细表_国内" LIMIT 5''',
    '''SELECT DISTINCT "职级标签" FROM "年终奖分配明细表_国内" LIMIT 20''',
]

# ========== 执行区域 - 一般不需要修改 ==========

def main():
    # 检查数据库文件
    if not Path(DB_PATH).exists():
        print(f"❌ 错误: 数据库文件不存在: {DB_PATH}")
        return
    
    print(f"📦 连接数据库: {DB_PATH}")
    print(f"📋 表名: {TABLE_NAME}")
    print("=" * 80)
    
    try:
        # 连接数据库
        conn = duckdb.connect(DB_PATH)
        
        # 执行所有SQL
        for i, sql in enumerate(SQL_QUERIES, 1):
            sql = sql.strip()
            if not sql or sql.startswith('#'):
                continue
                
            print(f"\n【查询 {i}】")
            print("-" * 80)
            print(sql)
            print("-" * 80)
            
            try:
                # 执行SQL
                result = conn.execute(sql)
                df = result.df()
                
                print(f"\n✅ 返回 {len(df)} 行数据:\n")
                print(df.to_string())
                print("\n" + "=" * 80)
                
            except Exception as e:
                print(f"\n❌ SQL执行失败:")
                print(f"错误: {e}")
                print("=" * 80)
        
        # 关闭连接
        conn.close()
        print("\n✅ 完成!")
        
    except Exception as e:
        print(f"\n❌ 连接数据库失败: {e}")


if __name__ == "__main__":
    main()
