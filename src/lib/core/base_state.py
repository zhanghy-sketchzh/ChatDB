"""
lib/core/base_state.py — 通用状态机基类

从 chatdb.core.react_state.ReActState 和 chatreport.core.react_state.ReportState
中提取的公共部分。

提供：
- BasePhase: 基础阶段枚举
- BaseState: 通用状态机基类（user_query, phase, step, summary, error, ReAct日志, exec_meta）

继承关系：
- chatdb.core.react_state.ReActState(BaseState) — 增加 SQL/intent/schema/分析切片
- chatreport.core.react_state.ReportState(BaseState) — 增加 outline/sections/evidence
- 未来 chathtml/chatpython 等场景各自扩展
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lib.utils.logger import get_component_logger


class BasePhase(str, Enum):
    """基础阶段枚举 — 子类可扩展"""
    INIT = "init"
    DONE = "done"
    ERROR = "error"


@dataclass
class BaseState:
    """
    通用状态机基类

    提供场景无关的核心字段和方法：
    - user_query / session_id: 输入标识
    - phase / step / max_steps: 阶段和步数控制
    - summary / error: 输出
    - thoughts / actions / observations / reflections: ReAct 日志
    - exec_meta: 执行元信息（plan_step, max_plan_steps 等）
    - temp_results: 中间结果存储
    """

    # ===== 输入 =====
    user_query: str = ""
    session_id: str = ""

    # ===== 阶段控制 =====
    phase: Any = BasePhase.INIT  # 子类可用自己的 Phase 枚举
    step: int = 0
    max_steps: int = 10

    # ===== 输出 =====
    summary: str = ""
    error: str | None = None

    # ===== ReAct 日志 =====
    thoughts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)

    # ===== 执行元信息 =====
    exec_meta: dict[str, Any] = field(default_factory=lambda: {
        "plan_step": 0,
        "max_plan_steps": 8,
    })

    # ===== 中间结果 =====
    temp_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # ===== 日志组件名（子类可覆写）=====
    _component_name: str = field(default="State", init=False, repr=False)

    # ===== 方法 =====

    def _get_logger(self):
        """获取组件日志器（延迟加载避免循环导入）"""
        return get_component_logger(self._component_name)

    def think(self, thought: str) -> None:
        """记录思考（推理/分析/决策）"""
        self.thoughts.append(f"[Step {self.step}] THINK: {thought}")
        self._get_logger().think(f"(Step {self.step}) {thought}")

    def act(self, action: str, tool: str = "", params: dict[str, Any] | None = None) -> None:
        """记录行动"""
        self.actions.append(f"[Step {self.step}] ACT: {action}")
        self._get_logger().act(action)

    def observe(self, observation: str) -> None:
        """记录观察（工具返回结果/执行结果）"""
        self.observations.append(f"[Step {self.step}] OBSERVE: {observation}")
        self._get_logger().observe(f"(Step {self.step}) {observation}")

    def reflect(self, reflection: str) -> None:
        """记录反思（错误分析/策略调整）"""
        self.reflections.append(f"[Step {self.step}] REFLECT: {reflection}")
        self._get_logger().reflect(f"(Step {self.step}) {reflection}")

    def next_step(self) -> None:
        """进入下一个循环步骤"""
        self.step += 1

    def set_error(self, error: str, **kwargs: Any) -> None:
        """
        设置错误。

        子类可扩展此方法添加 error_type、error_context 等。
        基类仅设置 error 字符串。
        """
        self.error = error
        self.reflect(f"错误: {error}")

    def clear_error(self) -> None:
        """清除错误"""
        self.error = None

    @property
    def is_done(self) -> bool:
        """是否完成"""
        phase_val = self.phase.value if isinstance(self.phase, Enum) else self.phase
        return phase_val in ("done", "error", "give_up")

    @property
    def can_continue(self) -> bool:
        """是否可以继续"""
        return not self.is_done and self.step < self.max_steps

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（子类应覆写扩展）"""
        phase_val = self.phase.value if isinstance(self.phase, Enum) else self.phase
        return {
            "user_query": self.user_query,
            "session_id": self.session_id,
            "phase": phase_val,
            "step": self.step,
            "summary": self.summary,
            "error": self.error,
        }

    def get_debug_info(self) -> dict[str, Any]:
        """获取调试信息（子类应覆写扩展）"""
        return {
            "reasoning_trace": self.get_reasoning_trace(),
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
            "reflections": self.reflections,
            "exec_meta": self.exec_meta,
        }

    def get_reasoning_trace(self) -> str:
        """将 THINK/ACT/OBSERVE/REFLECT 按步数合并为 ReAct 过程回放"""
        entries: list[tuple[int, str, str]] = []
        kind_order = {"THINK": 0, "ACT": 1, "OBSERVE": 2, "REFLECT": 3}

        for s in self.thoughts:
            m = re.match(r"\[Step (\d+)\] THINK: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "THINK", m.group(2).strip()))

        for s in self.actions:
            m = re.match(r"\[Step (\d+)\] ACT: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "ACT", m.group(2).strip()))

        for s in self.observations:
            m = re.match(r"\[Step (\d+)\] OBSERVE: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "OBSERVE", m.group(2).strip()))

        for s in self.reflections:
            m = re.match(r"\[Step (\d+)\] REFLECT: (.+)", s, re.DOTALL)
            if m:
                entries.append((int(m.group(1)), "REFLECT", m.group(2).strip()))

        entries.sort(key=lambda x: (x[0], kind_order.get(x[1], 9)))

        if not entries:
            return ""

        lines = ["ReAct 过程回放："]
        current_step = -1

        for step_num, kind, msg in entries:
            if step_num != current_step:
                current_step = step_num
                lines.append(f"\n[Step {step_num}]")
            prefix = {"THINK": "💭", "ACT": "⚡", "OBSERVE": "👁️", "REFLECT": "🔄"}.get(kind, "•")
            lines.append(f"  {prefix} {kind}: {msg}")

        return "\n".join(lines)

    # ===== exec_meta helpers =====

    def inc_plan_step(self) -> int:
        """增加计划步数，返回当前步数"""
        self.exec_meta["plan_step"] = self.exec_meta.get("plan_step", 0) + 1
        return self.exec_meta["plan_step"]

    @property
    def plan_step(self) -> int:
        """当前计划步数"""
        return self.exec_meta.get("plan_step", 0)

    @property
    def max_plan_steps(self) -> int:
        """最大计划步数"""
        return self.exec_meta.get("max_plan_steps", 8)
