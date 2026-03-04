"""
DataService — ChatDB 适配层

对标 chatdb/tools/sql.py 的角色：
- chatdb/tools/sql.py 直接执行 SQL
- chatreport/tools/data_service.py 调用 ChatDB 的 process_query 问数

这是 chatreport 与 chatdb 之间的**唯一跨包依赖点**。
ChatReport 的其他模块不直接 import chatdb 的任何内容。

使用方式：
    from chatdb.core.orchestrator import AgentOrchestrator
    data_service = DataService(orchestrator)
    result = await data_service.query("2024年各月流水总额")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from chatreport.core.react_state import DataResult

if TYPE_CHECKING:
    from chatdb.core.orchestrator import AgentOrchestrator


class DataService:
    """
    ChatDB 薄适配层 — 将自然语言问题转发给 ChatDB，返回结构化结果。

    设计原则：
    1. ChatReport 只通过 DataService 与 ChatDB 交互
    2. DataService 屏蔽 ChatDB 内部细节（SQL Agent、Planner 等）
    3. 返回统一的 DataResult，供 Writer / Evidence 使用
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orch = orchestrator

    async def query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> DataResult:
        """
        向 ChatDB 提问一个自然语言问题，获取数据结果。

        Args:
            question: 自然语言问题（如 "2024年各月流水总额是多少"）
            session_id: 共享 session，使后续问题可引用前序结果

        Returns:
            DataResult 封装的查询结果
        """
        t0 = time.time()
        try:
            result = await self._orch.process_query(
                query=question,
                session_id=session_id,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            return DataResult(
                success=result.get("success", False),
                summary=result.get("summary", ""),
                data=result.get("result", []),
                sql=result.get("sql", ""),
                row_count=result.get("row_count", 0),
                error=result.get("error"),
                question=question,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            return DataResult(
                success=False,
                error=str(e),
                question=question,
                elapsed_ms=elapsed_ms,
            )

    async def batch_query(
        self,
        questions: list[str],
        session_id: str | None = None,
    ) -> list[DataResult]:
        """
        批量查询（串行，共享 session 上下文）。

        串行原因：后续问题可能引用前序结果（如 "上面那个表按月展开"），
        共享 session 确保 ChatDB 的历史上下文连贯。

        Args:
            questions: 自然语言问题列表
            session_id: 共享会话 ID

        Returns:
            按顺序对应的 DataResult 列表
        """
        results: list[DataResult] = []
        for q in questions:
            r = await self.query(q, session_id=session_id)
            results.append(r)
        return results

    def get_data_context(self) -> dict[str, Any]:
        """
        获取数据上下文信息（表描述、可用指标/维度），
        供 ReportPlanner 在生成大纲时了解数据能力边界。

        Returns:
            包含 tables_meta / yml_config 等信息的字典
        """
        ctx: dict[str, Any] = {}

        # 表元数据
        if hasattr(self._orch, "tables_meta") and self._orch.tables_meta:
            ctx["tables_meta"] = self._orch.tables_meta

        # YML 配置（虚拟字段 / 指标维度）
        if hasattr(self._orch, "yml_config") and self._orch.yml_config:
            ctx["yml_config"] = str(self._orch.yml_config)

        return ctx
