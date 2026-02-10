#!/bin/bash
# 清理缓存并运行 Excel 导入示例

echo "🧹 清理 Python 缓存..."
find /Users/luchun/Desktop/work/ChatDB -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /Users/luchun/Desktop/work/ChatDB -type f -name "*.pyc" -delete 2>/dev/null || true

echo "🧹 清理数据缓存..."
cd /Users/luchun/Desktop/work/ChatDB
echo "yes" | python scripts/clear_duckdb_meta.py clear-all

echo ""
echo "🚀 运行 Excel 导入示例..."
python examples/excel_to_duckdb_example.py

echo ""
echo "✅ 完成！"


