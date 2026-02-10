"""
RunSQL 工具 - 简单通用的 SQL 执行工具

设计理念：
- 让 AI 直接传入 SQL 和表名就能执行
- 不做复杂的任务类型分发，只做基础的安全检查
- 返回清晰的执行结果

使用场景：
- AI 需要验证某个 SQL 是否正确
- AI 需要快速查看数据
- AI 需要诊断问题（检查字段值分布等）
"""

from typing import Any, Optional

from chatdb.database.base import BaseDatabaseConnector
from chatdb.tools.base import BaseTool, ToolParameter, ToolResult
from chatdb.utils.logger import get_component_logger


# 危险关键词（禁止写操作）
DANGEROUS_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "CREATE"]


class RunSQLTool(BaseTool):
    """
    简单通用的 SQL 执行工具
    
    功能：
    - 执行只读 SQL 查询
    - 返回结果行数、列名、数据示例
    - 基础安全检查（禁止写操作）
    """
    
    name = "run_sql"
    description = "执行 SQL 查询并返回结果。只支持 SELECT 查询，禁止写操作。"
    category = "database"
    
    parameters = [
        ToolParameter(
            name="sql",
            type="string",
            description="要执行的 SQL 语句（必须是 SELECT 查询）",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="返回结果的最大行数，默认 100",
            required=False,
            default=100,
        ),
    ]
    
    def __init__(self, db_connector: Optional[BaseDatabaseConnector] = None):
        super().__init__()
        self.db_connector = db_connector
        self._log = get_component_logger("RunSQLTool")
    
    def _check_sql_safety(self, sql: str) -> list[str]:
        """安全检查：只允许 SELECT，禁止写操作"""
        errors = []
        sql_upper = sql.upper().strip()
        
        # 必须以 SELECT 开头
        if not sql_upper.startswith("SELECT"):
            errors.append("只支持 SELECT 查询")
        
        # 禁止危险关键词
        for kw in DANGEROUS_KEYWORDS:
            if kw in sql_upper:
                errors.append(f"禁止使用 {kw} 语句")
        
        return errors
    
    async def run(self, **kwargs: Any) -> ToolResult:
        """执行 SQL 查询"""
        sql = kwargs.get("sql", "").strip()
        limit = kwargs.get("limit", 100)
        
        # 参数校验
        if not sql:
            return ToolResult.fail("SQL 不能为空")
        
        if not self.db_connector:
            return ToolResult.fail("未配置数据库连接")
        
        # 安全检查
        errors = self._check_sql_safety(sql)
        if errors:
            return ToolResult.fail(f"安全检查失败: {'; '.join(errors)}")
        
        # 执行 SQL
        self._log.info(f"执行 SQL: {sql[:80]}...")
        
        try:
            rows = await self.db_connector.execute_query(sql)
            
            # 提取列名
            columns = list(rows[0].keys()) if rows else []
            
            # 限制返回行数
            result_rows = rows[:limit] if limit else rows
            
            self._log.info(f"查询成功: 返回 {len(rows)} 行")
            
            return ToolResult.ok(
                data={
                    "rows": result_rows,
                    "row_count": len(rows),
                    "columns": columns,
                    "sql": sql,
                },
                message=f"查询成功，返回 {len(rows)} 行数据",
            )
            
        except Exception as e:
            error_msg = str(e)
            self._log.error(f"SQL 执行失败: {error_msg}")
            
            return ToolResult.fail(
                message=f"SQL 执行失败: {error_msg}",
                data={"sql": sql, "error": error_msg},
            )
    
    def __call__(self, sql: str, limit: int = 100) -> ToolResult:
        """同步调用接口（便于简单场景使用）"""
        import asyncio
        return asyncio.run(self.run(sql=sql, limit=limit))


# 便捷函数
def create_run_sql_tool(db_connector: BaseDatabaseConnector) -> RunSQLTool:
    """创建 RunSQL 工具实例"""
    return RunSQLTool(db_connector)
