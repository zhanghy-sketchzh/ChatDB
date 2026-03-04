"""
chatdb.core.scratch_pad — ScratchPadManager

继承 lib.core.scratch_pad.BaseScratchPadManager，
扩展 DB 临时表管理功能（DuckDB 场景特有）。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from lib.core.scratch_pad import BaseScratchPadManager
from pathlib import Path

if TYPE_CHECKING:
    from chatdb.database.base import BaseDatabaseConnector


class ScratchPadManager(BaseScratchPadManager):
    """
    SQL 结果文件暂存管理器

    在 BaseScratchPadManager 基础上扩展：
    - 临时表管理（用于任务间数据传递，DuckDB 普通表）
    """

    def __init__(
        self,
        base_path: str | Path = "data/scratch",
        max_age_hours: float = 24.0,
        db_connector: "BaseDatabaseConnector | None" = None,
    ):
        super().__init__(base_path=base_path, max_age_hours=max_age_hours)
        self.db_connector = db_connector
        # 记录 session 创建的临时表
        self._temp_tables: dict[str, list[str]] = {}  # {session_id: [table_name, ...]}

    # ============================================================
    # 临时表管理（用于任务间数据传递）
    # ============================================================

    def get_temp_table_name(self, session_id: str, task_id: str) -> str:
        """生成临时表名：temp_{session_short}_{task_safe}"""
        session_short = session_id[:8] if len(session_id) > 8 else session_id
        task_safe = self._safe_filename(task_id)
        return f"temp_{session_short}_{task_safe}"

    async def save_result_to_temp_table(
        self,
        session_id: str,
        task_id: str,
        rows: list[dict[str, Any]],
    ) -> str | None:
        """
        将任务结果保存为临时表

        Args:
            session_id: 会话 ID
            task_id: 任务 ID
            rows: 查询结果行

        Returns:
            临时表名（成功）或 None（失败）
        """
        if not self.db_connector or not rows:
            return None

        table_name = self.get_temp_table_name(session_id, task_id)

        try:
            # 先删除同名表（如果存在）
            drop_sql = f'DROP TABLE IF EXISTS "{table_name}"'
            await self.db_connector.execute_query(drop_sql)

            # ★ 用 CREATE TABLE AS SELECT ... FROM VALUES 一次性完成建表+插入
            # DuckDB SQLAlchemy 每次 connect() 是独立连接，TEMPORARY TABLE 跨连接不可见
            # 因此使用普通表，清理时主动 DROP
            def _format_value(v: Any) -> str:
                if v is None:
                    return "NULL"
                elif isinstance(v, bool):
                    return "TRUE" if v else "FALSE"
                elif isinstance(v, (int, float)):
                    return str(v)
                elif isinstance(v, str):
                    return f"'{v.replace(chr(39), chr(39)+chr(39))}'"
                else:
                    return f"'{str(v)}'"

            # ★ 统一所有行的列名集合
            col_names_ordered: list[str] = []
            col_names_seen: set[str] = set()
            for row in rows:
                for key in row.keys():
                    if key not in col_names_seen:
                        col_names_ordered.append(key)
                        col_names_seen.add(key)

            col_aliases = ", ".join(
                f'col{i} AS "{name}"' for i, name in enumerate(col_names_ordered)
            )
            
            value_rows = []
            for row in rows:
                vals = ", ".join(
                    _format_value(row.get(col)) for col in col_names_ordered
                )
                value_rows.append(f"({vals})")
            
            values_str = ", ".join(value_rows)
            create_as_sql = (
                f'CREATE TABLE "{table_name}" AS '
                f'SELECT {col_aliases} FROM (VALUES {values_str})'
            )
            await self.db_connector.execute_query(create_as_sql)

            # 记录临时表
            if session_id not in self._temp_tables:
                self._temp_tables[session_id] = []
            if table_name not in self._temp_tables[session_id]:
                self._temp_tables[session_id].append(table_name)

            self._log.info(f"临时表已创建: {table_name} ({len(rows)} 行)")
            return table_name

        except Exception as e:
            self._log.error(f"创建临时表失败 ({table_name}): {e}")
            return None

    def get_temp_tables(self, session_id: str) -> list[str]:
        """获取某会话的所有临时表"""
        return self._temp_tables.get(session_id, [])

    async def cleanup_temp_tables(self, session_id: str) -> int:
        """清理某会话的所有临时表"""
        if not self.db_connector:
            return 0

        tables = self._temp_tables.get(session_id, [])
        cleaned = 0

        for table_name in tables:
            try:
                drop_sql = f'DROP TABLE IF EXISTS "{table_name}"'
                await self.db_connector.execute_query(drop_sql)
                cleaned += 1
                self._log.debug(f"已删除临时表: {table_name}")
            except Exception as e:
                self._log.warn(f"删除临时表失败 ({table_name}): {e}")

        if session_id in self._temp_tables:
            del self._temp_tables[session_id]

        return cleaned

    async def cleanup_all_temp_tables(self) -> int:
        """清理所有临时表"""
        total_cleaned = 0
        session_ids = list(self._temp_tables.keys())

        for session_id in session_ids:
            cleaned = await self.cleanup_temp_tables(session_id)
            total_cleaned += cleaned

        return total_cleaned
