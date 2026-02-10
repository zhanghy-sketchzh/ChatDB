"""
Chat 路由 - 支持动态数据库查询

前端可传入数据库路径/CSV文件路径和查询语句，无需预先配置连接。
支持：
- DuckDB 数据库文件 (.duckdb)
- CSV 文件 (.csv) - 自动导入到 DuckDB
"""

import hashlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatdb.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["Chat"])


def _import_csv_to_duckdb(csv_path: str) -> str:
    """将 CSV 文件导入 DuckDB，返回数据库路径"""
    import duckdb
    
    csv_path = Path(csv_path).resolve()
    file_hash = hashlib.md5(str(csv_path).encode()).hexdigest()[:8]
    
    # 数据库存储目录
    db_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "duckdb"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / f"csv_{file_hash}.duckdb")
    
    table_name = csv_path.stem.replace(" ", "_").replace("-", "_")
    
    conn = duckdb.connect(db_path)
    try:
        # 检查表是否已存在
        existing = conn.execute("SHOW TABLES").fetchall()
        if (table_name,) not in existing:
            logger.info(f"正在导入 CSV: {csv_path}")
            conn.execute(f'''
                CREATE TABLE "{table_name}" AS 
                SELECT * FROM read_csv_auto('{csv_path}', header=true, sample_size=-1)
            ''')
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            logger.info(f"CSV 导入完成: {row_count} 行")
    finally:
        conn.close()
    
    return db_path


class ChatRequest(BaseModel):
    """Chat 请求"""
    query: str = Field(..., description="自然语言查询", min_length=1)
    db_path: str = Field(..., description="数据库或 CSV 文件路径")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "今年奖金最高的人是谁",
                    "db_path": "data/duckdb/excel_26253606.duckdb"
                },
                {
                    "query": "有哪些产品大类",
                    "db_path": "data/excel/脚本测试数据.csv"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Chat 响应"""
    success: bool
    query: str
    sql: str = ""
    result: list[dict[str, Any]] = []
    row_count: int = 0
    summary: str = ""
    error: str | None = None


@router.post("/", response_model=ChatResponse, summary="执行自然语言查询")
async def chat_query(request: ChatRequest) -> ChatResponse:
    """
    对指定数据库/CSV文件执行自然语言查询
    
    - **query**: 自然语言查询语句
    - **db_path**: DuckDB 数据库文件 (.duckdb) 或 CSV 文件 (.csv)
    """
    from chatdb.database.duckdb import DuckDBConnector
    from chatdb.core import AgentOrchestrator
    from chatdb.llm.factory import LLMFactory
    
    file_path = Path(request.db_path)
    
    # 验证文件存在
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件不存在: {request.db_path}")
    
    # 根据文件类型处理
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        db_path = _import_csv_to_duckdb(request.db_path)
    elif suffix == ".duckdb":
        db_path = request.db_path
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}，支持 .duckdb 和 .csv")
    
    db = DuckDBConnector(database=db_path)
    
    try:
        await db.connect()
        tables_meta = db.get_tables_meta()
        
        llm = LLMFactory.create(provider="hunyuan")
        orchestrator = AgentOrchestrator(llm, db, tables_meta=tables_meta)
        
        result = await orchestrator.process_query(request.query)
        
        return ChatResponse(
            success=result.get("success", True),
            query=request.query,
            sql=result.get("sql", ""),
            result=result.get("result", []),
            row_count=result.get("row_count", 0),
            summary=result.get("summary", ""),
        )
    except Exception as e:
        logger.exception(f"查询错误: {e}")
        return ChatResponse(
            success=False,
            query=request.query,
            error=str(e),
        )
    finally:
        await db.disconnect()


@router.get("/tables", summary="获取数据库表信息")
async def get_tables(db_path: str) -> dict[str, Any]:
    """获取指定数据库/CSV文件的表信息"""
    from chatdb.database.duckdb import DuckDBConnector
    
    file_path = Path(db_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件不存在: {db_path}")
    
    # 根据文件类型处理
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        actual_db_path = _import_csv_to_duckdb(db_path)
    elif suffix == ".duckdb":
        actual_db_path = db_path
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")
    
    db = DuckDBConnector(database=actual_db_path)
    
    try:
        await db.connect()
        tables_meta = db.get_tables_meta()
        return {"tables": tables_meta, "db_path": actual_db_path}
    finally:
        await db.disconnect()
