"""
数据库路由

处理数据库连接和 Schema 相关的 API 请求。
"""

from fastapi import APIRouter, Depends, HTTPException

from chatdb.api.dependencies import get_db_connector, get_schema_inspector
from chatdb.api.schemas import SchemaResponse, TableInfoResponse, ColumnInfoResponse
from chatdb.utils.exceptions import SchemaError
from chatdb.utils.logger import logger
from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.schema import SchemaInspector

router = APIRouter(prefix="/database", tags=["Database"])


@router.get(
    "/schema",
    response_model=SchemaResponse,
    summary="获取数据库 Schema",
    description="获取当前连接数据库的完整表结构信息",
)
async def get_schema(
    inspector: SchemaInspector = Depends(get_schema_inspector),
) -> SchemaResponse:
    """获取数据库 Schema 信息"""
    try:
        schema_info = await inspector.get_schema_info()

        # 转换为响应模型
        tables = []
        for table in schema_info.tables:
            columns = [
                ColumnInfoResponse(
                    name=col.name,
                    type=col.type,
                    nullable=col.nullable,
                    primary_key=col.primary_key,
                    default=col.default,
                    comment=col.comment,
                )
                for col in table.columns
            ]
            tables.append(
                TableInfoResponse(
                    name=table.name,
                    columns=columns,
                    primary_keys=table.primary_keys,
                    foreign_keys=table.foreign_keys,
                    comment=table.comment,
                )
            )

        return SchemaResponse(
            database_name=schema_info.database_name,
            tables=tables,
        )

    except SchemaError as e:
        logger.error(f"获取 Schema 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/tables",
    response_model=list[str],
    summary="获取表列表",
    description="获取数据库中所有表的名称",
)
async def get_tables(
    inspector: SchemaInspector = Depends(get_schema_inspector),
) -> list[str]:
    """获取表列表"""
    try:
        schema_info = await inspector.get_schema_info()
        return [table.name for table in schema_info.tables]
    except SchemaError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/tables/{table_name}/sample",
    summary="获取表示例数据",
    description="获取指定表的示例数据",
)
async def get_table_sample(
    table_name: str,
    limit: int = 5,
    inspector: SchemaInspector = Depends(get_schema_inspector),
):
    """获取表的示例数据"""
    try:
        sample_data = await inspector.get_table_sample(table_name, limit)
        return {
            "table": table_name,
            "sample_data": sample_data,
            "row_count": len(sample_data),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/execute",
    summary="执行原始 SQL",
    description="直接执行 SQL 查询（仅支持 SELECT）",
)
async def execute_sql(
    sql: str,
    connector: BaseDatabaseConnector = Depends(get_db_connector),
):
    """执行原始 SQL 查询"""
    try:
        result = await connector.execute_query(sql)
        return {
            "success": True,
            "result": result,
            "row_count": len(result),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

