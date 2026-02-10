"""
GetSchema Tool - 获取数据库 Schema 信息

获取表结构、列信息等元数据。
"""

from typing import Any

from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.schema import SchemaInspector
from chatdb.tools.base import BaseTool, ToolParameter, ToolResult
from chatdb.utils.logger import logger


class GetSchemaTool(BaseTool):
    """
    Schema 获取工具
    
    功能：
    - 获取数据库所有表信息
    - 获取指定表的列信息
    - 格式化输出供 LLM 使用
    """
    
    def __init__(self, db_connector: BaseDatabaseConnector):
        self.db_connector = db_connector
        self.schema_inspector = SchemaInspector(db_connector)
    
    @property
    def name(self) -> str:
        return "get_schema"
    
    @property
    def description(self) -> str:
        return """获取数据库的 Schema 信息（表结构、列信息）。

使用场景：
- 了解数据库结构
- 获取可用的表和列信息
- 为 SQL 生成提供上下文

输入：
- table_name: 指定表名（可选，不指定则返回所有表）

输出：
- tables: 表信息列表
- schema_text: 格式化的 Schema 文本"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="table_name",
                type="string",
                description="指定表名（可选）",
                required=False,
            ),
        ]
    
    async def execute(
        self,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """获取 Schema 信息"""
        logger.info(f"[get_schema] 获取 Schema: {table_name or '全部'}")
        
        try:
            schema_info = await self.schema_inspector.get_schema_info()
            
            if table_name:
                # 获取指定表
                table = next(
                    (t for t in schema_info.tables if t.name == table_name),
                    None,
                )
                if not table:
                    return ToolResult.fail(f"表不存在: {table_name}")
                
                tables_data = [{
                    "table_name": table.name,
                    "columns": [
                        {
                            "name": col.name,
                            "type": col.data_type,
                            "comment": col.comment,
                        }
                        for col in table.columns
                    ],
                    "comment": table.comment,
                }]
            else:
                # 获取所有表
                tables_data = [
                    {
                        "table_name": table.name,
                        "columns": [
                            {
                                "name": col.name,
                                "type": col.data_type,
                                "comment": col.comment,
                            }
                            for col in table.columns
                        ],
                        "comment": table.comment,
                    }
                    for table in schema_info.tables
                ]
            
            schema_text = schema_info.to_prompt_text()
            
            return ToolResult.ok(
                data={
                    "tables": tables_data,
                    "schema_text": schema_text,
                },
                message=f"获取到 {len(tables_data)} 个表的 Schema",
            )
        except Exception as e:
            logger.error(f"[get_schema] 获取失败: {e}")
            return ToolResult.fail(f"获取 Schema 失败: {e}")
