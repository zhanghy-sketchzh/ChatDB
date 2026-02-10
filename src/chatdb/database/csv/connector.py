"""
CSV 数据库连接器

基于 DuckDB 实现高性能 CSV 文件查询：
- 自动将 CSV 导入 DuckDB（利用原生 read_csv_auto）
- 支持缓存，避免重复导入
- 继承 DuckDB 的全部查询能力
"""

import hashlib
from pathlib import Path
from typing import Any

import duckdb

from chatdb.database.duckdb import DuckDBConnector
from chatdb.utils.logger import logger


class CSVConnector(DuckDBConnector):
    """
    CSV 文件连接器
    
    用法：
        async with CSVConnector("data.csv") as db:
            result = await db.execute_query("SELECT * FROM data LIMIT 10")
    """
    
    def __init__(
        self,
        csv_path: str | Path,
        table_name: str | None = None,
        db_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        """
        初始化 CSV 连接器
        
        Args:
            csv_path: CSV 文件路径
            table_name: 表名（默认使用文件名）
            db_path: DuckDB 文件路径（默认自动生成）
            cache_dir: 缓存目录（默认 data/duckdb）
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")
        
        # 表名
        self.table_name = table_name or self.csv_path.stem.replace(" ", "_").replace("-", "_")
        
        # 数据库路径
        if db_path is None:
            file_hash = hashlib.md5(str(self.csv_path).encode()).hexdigest()[:8]
            cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parents[4] / "data" / "duckdb"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / f"csv_{file_hash}.duckdb"
        
        self._db_path = Path(db_path)
        self._import_info: dict[str, Any] | None = None
        
        super().__init__(database=str(self._db_path))
    
    async def connect(self) -> None:
        """连接并导入 CSV（先导入再连接，避免连接冲突）"""
        self._import_info = self._import_csv()
        await super().connect()
    
    def _import_csv(self) -> dict[str, Any]:
        """将 CSV 导入 DuckDB"""
        conn = duckdb.connect(str(self._db_path))
        
        try:
            # 检查表是否已存在
            existing = conn.execute("SHOW TABLES").fetchall()
            if (self.table_name,) in existing:
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{self.table_name}"').fetchone()[0]
                columns = conn.execute(f'DESCRIBE "{self.table_name}"').fetchall()
                logger.info(f"[CSVConnector] 使用缓存: {self.table_name} ({row_count} 行)")
                return {
                    "status": "cached",
                    "table_name": self.table_name,
                    "row_count": row_count,
                    "columns": [{"name": c[0], "type": c[1]} for c in columns],
                }
            
            # 导入 CSV
            logger.info(f"[CSVConnector] 导入 CSV: {self.csv_path.name}")
            conn.execute(f'''
                CREATE TABLE "{self.table_name}" AS 
                SELECT * FROM read_csv_auto('{self.csv_path}', header=true, sample_size=-1)
            ''')
            
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{self.table_name}"').fetchone()[0]
            columns = conn.execute(f'DESCRIBE "{self.table_name}"').fetchall()
            
            logger.info(f"[CSVConnector] 导入完成: {row_count} 行, {len(columns)} 列")
            return {
                "status": "imported",
                "table_name": self.table_name,
                "row_count": row_count,
                "columns": [{"name": c[0], "type": c[1]} for c in columns],
            }
        finally:
            conn.close()
    
    @property
    def import_info(self) -> dict[str, Any]:
        """获取导入信息"""
        return self._import_info or {}
    
    @property
    def db_path(self) -> Path:
        """获取 DuckDB 文件路径"""
        return self._db_path
