"""DuckDB 数据库连接器模块"""

from chatdb.database.duckdb.duckdb import DuckDBConnector
from chatdb.database.duckdb.syntax_rules import (
    get_duckdb_syntax_rules,
    get_analysis_constraints,
    DUCKDB_MULTI_TABLE_STRATEGIES,
)

__all__ = [
    "DuckDBConnector",
    "get_duckdb_syntax_rules",
    "get_analysis_constraints",
    "DUCKDB_MULTI_TABLE_STRATEGIES",
]
