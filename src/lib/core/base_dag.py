"""
lib/core/base_dag.py — DAG 拓扑调度框架

从 chatdb 的 AnalysisPlan 和 chatreport 的 SectionDAG 提取的公共 DAG 管理能力。

提供：
- NodeStatus: 节点状态枚举
- DAGNode: DAG 节点基类
- BaseDAG: DAG 拓扑排序 + 批次执行框架

特性：
- add_node() / get_node() — 节点管理
- get_ready_nodes() — 获取依赖已满足的节点
- mark_done() / mark_failed() / mark_skipped() — 状态更新
- iter_batches() — 逐批返回可并行节点
- has_cycle() — Kahn 算法环路检测
- progress_summary() — 进度摘要
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator


class NodeStatus(str, Enum):
    """DAG 节点状态"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """DAG 节点基类"""
    id: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "meta": self.meta,
        }


class BaseDAG:
    """
    DAG 拓扑排序 + 批次执行框架

    chatdb 的 AnalysisPlan 和 chatreport 的 SectionDAG 都可以继承此类，
    获得统一的依赖管理和批次调度能力。
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    # ================================================================
    # 节点管理
    # ================================================================

    def add_node(self, node: DAGNode) -> None:
        """添加节点"""
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> DAGNode | None:
        """获取节点"""
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> list[DAGNode]:
        """所有节点列表"""
        return list(self._nodes.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ================================================================
    # 状态管理
    # ================================================================

    def get_ready_nodes(self) -> list[DAGNode]:
        """获取所有依赖已满足的待执行节点"""
        ready = []
        completed_ids = {
            nid for nid, n in self._nodes.items()
            if n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
        }
        for node in self._nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in node.dependencies):
                ready.append(node)
        return ready

    def mark_running(self, node_id: str) -> None:
        """标记节点为运行中"""
        if node_id in self._nodes:
            self._nodes[node_id].status = NodeStatus.RUNNING

    def mark_done(self, node_id: str) -> None:
        """标记节点为完成"""
        if node_id in self._nodes:
            self._nodes[node_id].status = NodeStatus.COMPLETED

    def mark_failed(self, node_id: str) -> None:
        """标记节点为失败"""
        if node_id in self._nodes:
            self._nodes[node_id].status = NodeStatus.FAILED

    def mark_skipped(self, node_id: str) -> None:
        """标记节点为跳过"""
        if node_id in self._nodes:
            self._nodes[node_id].status = NodeStatus.SKIPPED

    # ================================================================
    # 批次调度
    # ================================================================

    def iter_batches(self) -> Generator[list[DAGNode], None, None]:
        """
        逐批返回可并行执行的节点。

        每次 yield 一批依赖已满足的节点，调用方执行完毕后
        需调用 mark_done/mark_failed 更新状态，然后继续迭代。
        """
        while True:
            ready = self.get_ready_nodes()
            if not ready:
                break
            yield ready

    @property
    def all_done(self) -> bool:
        """所有节点是否已完成（含失败/跳过）"""
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)
            for n in self._nodes.values()
        )

    # ================================================================
    # 环路检测
    # ================================================================

    def has_cycle(self) -> bool:
        """Kahn 算法检测环路"""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.id] = in_degree.get(node.id, 0) + 1

        # 注意：这里需要从 0 入度节点开始
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            nid = queue.pop(0)
            visited += 1
            # 找到所有以 nid 为依赖的节点，减少其入度
            for node in self._nodes.values():
                if nid in node.dependencies:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        queue.append(node.id)

        return visited != len(self._nodes)

    # ================================================================
    # 进度报告
    # ================================================================

    def progress_summary(self) -> str:
        """进度摘要"""
        status_counts: dict[str, int] = {}
        for node in self._nodes.values():
            status_counts[node.status.value] = status_counts.get(node.status.value, 0) + 1

        parts = [f"{k}: {v}" for k, v in sorted(status_counts.items())]
        return f"DAG({self.node_count} nodes): {', '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "node_count": self.node_count,
            "all_done": self.all_done,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
        }
