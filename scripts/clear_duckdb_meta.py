#!/usr/bin/env python3
"""
清除 DuckDB 和 Meta Data 数据工具
用于清空 DuckDB 数据库文件和元数据数据库
"""
import sqlite3
import sys
from pathlib import Path
from typing import List


def _get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 从脚本位置计算项目根目录
    # scripts/clear_duckdb_meta.py -> 项目根目录
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def _get_duckdb_dir() -> Path:
    """
    获取 DuckDB 数据库目录
    
    Returns:
        DuckDB 数据库目录路径
    """
    project_root = _get_project_root()
    return project_root / "data" / "pilot" / "databases"


def _get_excel_meta_db() -> Path:
    """
    获取 Excel 元数据数据库路径
    
    Returns:
        Excel 元数据数据库路径
    """
    project_root = _get_project_root()
    return project_root / "data" / "pilot" / "excel_meta_data.db"


def _get_data_metadata_db() -> Path:
    """
    获取数据元数据数据库路径
    
    Returns:
        数据元数据数据库路径
    """
    project_root = _get_project_root()
    return project_root / "src" / "chatdb" / "database" / "meta" / "data_metadata.db"


def list_duckdb_files():
    """列出所有 DuckDB 数据库文件"""
    duckdb_dir = _get_duckdb_dir()
    
    if not duckdb_dir.exists():
        print(f"📭 DuckDB 数据库目录不存在: {duckdb_dir}")
        return
    
    duckdb_files = list(duckdb_dir.glob("*.duckdb"))
    
    if not duckdb_files:
        print(f"📭 DuckDB 数据库目录为空")
        return
    
    print(f"\n📊 DuckDB 数据库文件 (共 {len(duckdb_files)} 个):\n")
    for db_file in sorted(duckdb_files):
        size = db_file.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"  - {db_file.name} ({size_mb:.2f} MB)")


def clear_duckdb_files(auto_confirm: bool = False):
    """清除所有 DuckDB 数据库文件
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    duckdb_dir = _get_duckdb_dir()
    
    if not duckdb_dir.exists():
        print(f"📭 DuckDB 数据库目录不存在: {duckdb_dir}")
        return
    
    duckdb_files = list(duckdb_dir.glob("*.duckdb"))
    
    if not duckdb_files:
        print(f"📭 DuckDB 数据库目录为空")
        return
    
    print(f"\n📊 发现 {len(duckdb_files)} 个 DuckDB 数据库文件")
    for db_file in duckdb_files:
        size = db_file.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"  - {db_file.name} ({size_mb:.2f} MB)")
    
    if not auto_confirm:
        choice = input("\n⚠️  确认要清除所有 DuckDB 数据库文件吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    deleted_count = 0
    for db_file in duckdb_files:
        try:
            db_file.unlink()
            deleted_count += 1
            print(f"✅ 已删除: {db_file.name}")
        except Exception as e:
            print(f"❌ 删除文件失败 {db_file.name}: {e}")
    
    print(f"\n✅ 总计清除 {deleted_count}/{len(duckdb_files)} 个 DuckDB 数据库文件")


def list_excel_meta_data():
    """列出 Excel 元数据数据库中的记录"""
    excel_meta_db = _get_excel_meta_db()
    
    if not excel_meta_db.exists():
        print(f"📭 Excel 元数据数据库不存在: {excel_meta_db}")
        return
    
    conn = sqlite3.connect(str(excel_meta_db))
    cursor = conn.cursor()
    
    try:
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='table_meta'")
        if not cursor.fetchone():
            print("📭 表 table_meta 不存在")
            conn.close()
            return
        
        cursor.execute("""
            SELECT 
                id,
                file_hash,
                table_name,
                sheet_name,
                file_name,
                db_name,
                db_path,
                row_count,
                column_count,
                created_at,
                last_accessed,
                access_count
            FROM table_meta
            ORDER BY last_accessed DESC
        """)
        
        records = cursor.fetchall()
        
        if not records:
            print("📭 当前没有元数据记录")
        else:
            print(f"\n📊 Excel 元数据记录 (共 {len(records)} 条):\n")
            for record in records:
                print(f"{'='*80}")
                print(f"ID: {record[0]}")
                print(f"文件哈希: {record[1]}")
                print(f"表名: {record[2]}")
                print(f"Sheet名: {record[3]}")
                print(f"文件名: {record[4]}")
                print(f"数据库名: {record[5]}")
                print(f"数据库路径: {record[6]}")
                print(f"数据规模: {record[7]}行 × {record[8]}列")
                print(f"创建时间: {record[9]}")
                print(f"最后访问: {record[10]}")
                print(f"访问次数: {record[11]}")
    except sqlite3.OperationalError as e:
        print(f"❌ 查询失败: {e}")
    finally:
        conn.close()


def clear_excel_meta_data(auto_confirm: bool = False):
    """清除 Excel 元数据数据库中的所有记录
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    excel_meta_db = _get_excel_meta_db()
    
    if not excel_meta_db.exists():
        print(f"📭 Excel 元数据数据库不存在: {excel_meta_db}")
        return
    
    conn = sqlite3.connect(str(excel_meta_db))
    cursor = conn.cursor()
    
    try:
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='table_meta'")
        if not cursor.fetchone():
            print("📭 表 table_meta 不存在")
            conn.close()
            return
        
        # 统计记录数
        cursor.execute("SELECT COUNT(*) FROM table_meta")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📭 当前没有元数据记录")
            conn.close()
            return
        
        print(f"\n📊 当前有 {count} 条元数据记录")
        
        if not auto_confirm:
            choice = input("\n⚠️  确认要清除所有 Excel 元数据记录吗？(yes/no): ")
            if choice.lower() != 'yes':
                print("❌ 取消操作")
                conn.close()
                return
        
        cursor.execute("DELETE FROM table_meta")
        deleted = cursor.rowcount
        conn.commit()
        
        print(f"✅ 已清除 {deleted} 条 Excel 元数据记录")
    except sqlite3.OperationalError as e:
        print(f"❌ 操作失败: {e}")
    finally:
        conn.close()


def clear_excel_meta_by_id(record_id: int):
    """根据 ID 删除 Excel 元数据记录
    
    Args:
        record_id: 记录 ID
    """
    excel_meta_db = _get_excel_meta_db()
    
    if not excel_meta_db.exists():
        print(f"📭 Excel 元数据数据库不存在: {excel_meta_db}")
        return
    
    conn = sqlite3.connect(str(excel_meta_db))
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM table_meta WHERE id = ?", (record_id,))
        deleted = cursor.rowcount
        conn.commit()
        
        if deleted > 0:
            print(f"✅ 已删除 ID={record_id} 的元数据记录")
        else:
            print(f"⚠️ 未找到 ID={record_id} 的元数据记录")
    except sqlite3.OperationalError as e:
        print(f"❌ 操作失败: {e}")
    finally:
        conn.close()


def list_data_metadata():
    """列出数据元数据数据库中的记录"""
    data_metadata_db = _get_data_metadata_db()
    
    if not data_metadata_db.exists():
        print(f"📭 数据元数据数据库不存在: {data_metadata_db}")
        return
    
    conn = sqlite3.connect(str(data_metadata_db))
    cursor = conn.cursor()
    
    try:
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print("📭 数据库中没有表")
            conn.close()
            return
        
        print(f"\n📊 数据元数据数据库表 (共 {len(tables)} 个):\n")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table}: {count} 条记录")
    except sqlite3.OperationalError as e:
        print(f"❌ 查询失败: {e}")
    finally:
        conn.close()


def clear_data_metadata(auto_confirm: bool = False):
    """清除数据元数据数据库中的所有记录
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    data_metadata_db = _get_data_metadata_db()
    
    if not data_metadata_db.exists():
        print(f"📭 数据元数据数据库不存在: {data_metadata_db}")
        return
    
    conn = sqlite3.connect(str(data_metadata_db))
    cursor = conn.cursor()
    
    try:
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print("📭 数据库中没有表")
            conn.close()
            return
        
        # 统计总记录数
        total_count = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_count += count
        
        if total_count == 0:
            print("📭 当前没有元数据记录")
            conn.close()
            return
        
        print(f"\n📊 当前有 {total_count} 条元数据记录（分布在 {len(tables)} 个表中）")
        
        if not auto_confirm:
            choice = input("\n⚠️  确认要清除所有数据元数据记录吗？(yes/no): ")
            if choice.lower() != 'yes':
                print("❌ 取消操作")
                conn.close()
                return
        
        deleted_total = 0
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            deleted = cursor.rowcount
            deleted_total += deleted
            if deleted > 0:
                print(f"✅ 已清除表 {table}: {deleted} 条记录")
        
        conn.commit()
        print(f"\n✅ 总计清除 {deleted_total} 条数据元数据记录")
    except sqlite3.OperationalError as e:
        print(f"❌ 操作失败: {e}")
    finally:
        conn.close()


def clear_all(auto_confirm: bool = False):
    """清除所有 DuckDB 和 Meta Data 数据
    
    Args:
        auto_confirm: 是否自动确认（用于批量清除）
    """
    print("\n⚠️  警告: 此操作将清除以下所有数据:")
    print("  1. DuckDB 数据库文件（.duckdb）")
    print("  2. Excel 元数据数据库记录（excel_meta_data.db）")
    print("  3. 数据元数据数据库记录（data_metadata.db）")
    
    if not auto_confirm:
        choice = input("\n⚠️  确认要清除所有数据吗？(yes/no): ")
        if choice.lower() != 'yes':
            print("❌ 取消操作")
            return
    
    # 清除 DuckDB 文件
    print("\n1️⃣ 清除 DuckDB 数据库文件...")
    clear_duckdb_files(auto_confirm=True)
    
    # 清除 Excel 元数据
    print("\n2️⃣ 清除 Excel 元数据记录...")
    clear_excel_meta_data(auto_confirm=True)
    
    # 清除数据元数据
    print("\n3️⃣ 清除数据元数据记录...")
    clear_data_metadata(auto_confirm=True)
    
    print("\n✅ 所有数据清除完成！")


if __name__ == "__main__":
    print("🗑️  DuckDB 和 Meta Data 清理工具\n")
    
    if len(sys.argv) == 1:
        # 无参数：显示帮助信息
        print("使用方法:")
        print("\n📊 DuckDB 相关:")
        print("  python scripts/clear_duckdb_meta.py duckdb-list        # 列出所有 DuckDB 数据库文件")
        print("  python scripts/clear_duckdb_meta.py duckdb-clear      # 清除所有 DuckDB 数据库文件")
        
        print("\n📋 Excel 元数据相关:")
        print("  python scripts/clear_duckdb_meta.py excel-meta-list    # 列出 Excel 元数据记录")
        print("  python scripts/clear_duckdb_meta.py excel-meta-clear   # 清除所有 Excel 元数据记录")
        print("  python scripts/clear_duckdb_meta.py excel-meta-clear-id <ID>  # 根据 ID 删除 Excel 元数据记录")
        
        print("\n📋 数据元数据相关:")
        print("  python scripts/clear_duckdb_meta.py data-meta-list     # 列出数据元数据记录")
        print("  python scripts/clear_duckdb_meta.py data-meta-clear    # 清除所有数据元数据记录")
        
        print("\n🗑️  全部清除:")
        print("  python scripts/clear_duckdb_meta.py clear-all          # 清除所有数据")
    
    elif len(sys.argv) >= 2:
        command = sys.argv[1]
        
        # DuckDB 相关命令
        if command == "duckdb-list":
            list_duckdb_files()
        
        elif command == "duckdb-clear":
            clear_duckdb_files()
        
        # Excel 元数据相关命令
        elif command == "excel-meta-list":
            list_excel_meta_data()
        
        elif command == "excel-meta-clear":
            clear_excel_meta_data()
        
        elif command == "excel-meta-clear-id" and len(sys.argv) == 3:
            try:
                record_id = int(sys.argv[2])
                clear_excel_meta_by_id(record_id)
            except ValueError:
                print("❌ 无效的 ID，请输入数字")
        
        # 数据元数据相关命令
        elif command == "data-meta-list":
            list_data_metadata()
        
        elif command == "data-meta-clear":
            clear_data_metadata()
        
        # 全部清除
        elif command == "clear-all":
            clear_all()
        
        else:
            print("❌ 无效的命令")
            print("\n使用方法:")
            print("  python scripts/clear_duckdb_meta.py duckdb-list        # 列出 DuckDB 文件")
            print("  python scripts/clear_duckdb_meta.py excel-meta-list    # 列出 Excel 元数据")
            print("  python scripts/clear_duckdb_meta.py clear-all          # 清除所有数据")
            print("\n使用 'python scripts/clear_duckdb_meta.py' 查看完整帮助")


