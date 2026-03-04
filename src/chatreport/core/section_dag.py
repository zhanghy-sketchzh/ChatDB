"""
SectionDAG — 章节级 DAG 拓扑管理

对标 chatdb/agents/planner.py 的 AnalysisPlan，
但管理的是章节间依赖关系而非 SQL 任务间依赖。

核心能力：
1. 根据章节 dependencies 构建 DAG
2. 返回可并行执行的章节批次（拓扑排序）
3. 检测循环依赖
4. 管理章节执行状态

示例：
    s1(无依赖) ──┐
    s2(无依赖) ──┤─→ s3(依赖s1,s2) ─→ s4(依赖s3)
    (可并行)       (串行等待)

    dag = SectionDAG(section_specs)
    for batch in dag.iter_batches():
        await asyncio.gather(*[run_section(s) for s in batch])
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from chatreport.core.react_state import SectionSpec, SectionStatus


class DAGCycleError(Exception):
    """DAG 存在循环依赖"""
    pass


@dataclass
class SectionNode:
    """DAG 节点 — 封装章节及其依赖关系"""
    spec: SectionSpec
    status: SectionStatus = SectionStatus.PENDING
    # 自动由 DAG 构建填充
    in_degree: int = 0
    dependents: list[str] = field(default_factory=list)  # 依赖本节点的下游 section_id

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def is_ready(self) -> bool:
        """无前置依赖 or 所有前置已完成 → 可执行"""
        return self.status == SectionStatus.PENDING and self.in_degree == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "in_degree": self.in_degree,
            "dependents": self.dependents,
            "dependencies": self.spec.dependencies,
        }


class SectionDAG:
    """
    章节级 DAG — 管理章节间并行/串行执行顺序

    使用 Kahn 算法做拓扑排序，支持：
    - iter_batches(): 逐批返回可并行的章节
    - mark_done(section_id): 标记完成并释放下游
    - mark_failed(section_id): 标记失败
    """

    def __init__(self, section_specs: list[SectionSpec]):
        self._nodes: dict[str, SectionNode] = {}
        self._build(section_specs)

    def _build(self, specs: list[SectionSpec]) -> None:
        """构建 DAG"""
        # 1. 创建所有节点
        for spec in specs:
            self._nodes[spec.id] = SectionNode(spec=spec)

        # 2. 建立依赖边
        for spec in specs:
            node = self._nodes[spec.id]
            for dep_id in spec.dependencies:
                if dep_id not in self._nodes:
                    continue  # 忽略不存在的依赖（容错）
                node.in_degree += 1
                self._nodes[dep_id].dependents.append(spec.id)

        # 3. 检测循环
        self._check_cycle()

    def _check_cycle(self) -> None:
        """Kahn 算法检测循环依赖"""
        in_deg = {nid: n.in_degree for nid, n in self._nodes.items()}
        queue = deque(nid for nid, d in in_deg.items() if d == 0)
        visited = 0

        while queue:
            nid = queue.popleft()
            visited += 1
            for dep_id in self._nodes[nid].dependents:
                in_deg[dep_id] -= 1
                if in_deg[dep_id] == 0:
                    queue.append(dep_id)

        if visited != len(self._nodes):
            cycle_nodes = [nid for nid, d in in_deg.items() if d > 0]
            raise DAGCycleError(
                f"章节依赖存在循环: {cycle_nodes}"
            )

    # ---- 执行控制 ----

    def get_ready_sections(self) -> list[SectionNode]:
        """获取当前可执行的章节（入度为 0 且状态为 PENDING）"""
        return [n for n in self._nodes.values() if n.is_ready]

    def mark_done(self, section_id: str) -> list[str]:
        """
        标记章节完成，释放下游依赖

        Returns:
            新变为可执行的下游 section_id 列表
        """
        node = self._nodes.get(section_id)
        if not node:
            return []

        node.status = SectionStatus.COMPLETED
        newly_ready = []
        for dep_id in node.dependents:
            dep_node = self._nodes[dep_id]
            dep_node.in_degree -= 1
            if dep_node.is_ready:
                newly_ready.append(dep_id)
        return newly_ready

    def mark_failed(self, section_id: str) -> None:
        """标记章节失败"""
        node = self._nodes.get(section_id)
        if node:
            node.status = SectionStatus.FAILED

    def mark_running(self, section_id: str) -> None:
        """标记章节正在执行"""
        node = self._nodes.get(section_id)
        if node:
            node.status = SectionStatus.PLANNING

    def iter_batches(self):
        """
        生成器：逐批返回可并行执行的章节 ID 列表

        每 batch 内的章节互相无依赖，可 asyncio.gather 并行。
        调用方需在每批执行完后 mark_done 所有成功的章节。

        Yields:
            list[str]: 一批可并行的 section_id
        """
        # 复制 in_degree 做模拟（不影响实际状态）
        in_deg = {nid: n.in_degree for nid, n in self._nodes.items()}
        queue = deque(nid for nid, d in in_deg.items() if d == 0)

        while queue:
            batch = list(queue)
            queue.clear()
            yield batch
            for nid in batch:
                for dep_id in self._nodes[nid].dependents:
                    in_deg[dep_id] -= 1
                    if in_deg[dep_id] == 0:
                        queue.append(dep_id)

    # ---- 查询 ----

    def get_node(self, section_id: str) -> SectionNode | None:
        return self._nodes.get(section_id)

    @property
    def all_done(self) -> bool:
        return all(
            n.status in (SectionStatus.COMPLETED, SectionStatus.FAILED)
            for n in self._nodes.values()
        )

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "total": self.node_count,
            "all_done": self.all_done,
        }

    def __repr__(self) -> str:
        parts = []
        for nid, node in self._nodes.items():
            deps = node.spec.dependencies
            dep_str = f" (depends: {deps})" if deps else ""
            parts.append(f"  {nid}: {node.title}{dep_str} [{node.status.value}]")
        return f"SectionDAG(\n" + "\n".join(parts) + "\n)"
