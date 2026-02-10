# 数据库连接器使用指南

## 架构设计

数据库连接器采用继承模式设计：

- **BaseDatabaseConnector**: 基类，定义通用接口和共同方法
- **PostgreSQLConnector**: PostgreSQL 数据库连接器
- **MySQLConnector**: MySQL 数据库连接器
- **SQLiteConnector**: SQLite 数据库连接器
- **DuckDBConnector**: DuckDB 数据库连接器
- **ExcelConnector**: Excel 文件连接器（通过 DuckDB 读取）

## 使用方式

### 方式一：使用工厂函数（推荐）

```python
from chatdb.database.factory import create_connector

# 使用默认配置
connector = create_connector()

# 指定数据库类型
connector = create_connector(db_type="mysql")

# 指定连接参数
connector = create_connector(
    db_type="postgresql",
    host="localhost",
    port=5432,
    user="postgres",
    password="password",
    database="mydb"
)

# Excel 文件
connector = create_connector(
    db_type="excel",
    file_path="/path/to/file.xlsx",
    sheet_name="Sheet1"
)

# 使用连接
async with connector:
    result = await connector.execute_safe_query("SELECT * FROM users LIMIT 10")
```

### 方式二：直接使用连接器类

```python
from chatdb.database.connector import (
    PostgreSQLConnector,
    MySQLConnector,
    SQLiteConnector,
    DuckDBConnector,
    ExcelConnector,
)

# PostgreSQL
connector = PostgreSQLConnector(
    host="localhost",
    port=5432,
    user="postgres",
    password="password",
    database="mydb"
)

# MySQL
connector = MySQLConnector(
    host="localhost",
    port=3306,
    user="root",
    password="password",
    database="mydb"
)

# SQLite
connector = SQLiteConnector(database="./data/mydb.db")

# DuckDB
connector = DuckDBConnector(database="./data/mydb.duckdb")
# 或使用内存数据库
connector = DuckDBConnector()

# Excel
connector = ExcelConnector(
    file_path="./data/sales.xlsx",
    sheet_name="Sheet1"
)

# 使用连接
async with connector:
    result = await connector.execute_safe_query("SELECT * FROM users")
```

### 方式三：使用连接 URL

```python
from chatdb.database.factory import create_connector

# 直接使用连接 URL
connector = create_connector(
    connection_url="postgresql+asyncpg://user:password@localhost:5432/mydb"
)

async with connector:
    result = await connector.execute_safe_query("SELECT * FROM users")
```

## 支持的操作

所有连接器都支持以下方法：

- `connect()`: 建立数据库连接
- `disconnect()`: 关闭数据库连接
- `execute_query(sql, params, fetch)`: 执行 SQL 查询
- `execute_safe_query(sql, params, max_rows)`: 安全执行 SQL（仅 SELECT）
- `get_session()`: 获取数据库会话

## Excel 连接器说明

Excel 连接器通过以下方式工作：

1. 使用 `pandas` 读取 Excel 文件
2. 将数据加载到临时 DuckDB 数据库中
3. 通过 SQL 查询数据
4. 连接关闭时自动清理临时文件

**注意**：
- 需要安装 `pandas`、`openpyxl` 和 `duckdb`
- Excel 文件会被加载到内存中的临时数据库
- 支持 `.xlsx` 和 `.xls` 格式

## 扩展新的连接器

如果需要添加新的数据库类型，可以继承 `BaseDatabaseConnector`：

```python
from chatdb.database.connector import BaseDatabaseConnector

class MyDatabaseConnector(BaseDatabaseConnector):
    """自定义数据库连接器"""
    
    def _get_default_connection_url(self) -> str:
        """实现连接 URL 生成"""
        return "mydb://user:pass@host:port/db"
    
    # 可选：重写其他方法以自定义行为
    def _get_pool_size(self) -> int:
        return 10
    
    async def _test_connection(self) -> None:
        # 自定义连接测试逻辑
        pass
```

然后在工厂函数中注册：

```python
from chatdb.database.factory import create_connector
from chatdb.database.connector import BaseDatabaseConnector

# 注册新连接器
BaseDatabaseConnector._providers["mydb"] = MyDatabaseConnector

# 使用
connector = create_connector(db_type="mydb")
```


