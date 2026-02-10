"""
任务追踪器

记录任务执行过程中的所有步骤和结果，最终保存为结构化日志文件。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TaskStep:
    """任务步骤记录"""

    step_name: str
    step_type: str  # "llm_call", "agent_call", "tool_call", "db_query", etc.
    timestamp: str
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskTracer:
    """任务追踪器 - 记录任务执行全流程"""

    task_id: str
    task_name: str
    task_description: str
    task_file_name: str | None = None
    ground_truth: str | None = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None
    status: str = "running"  # running, completed, interrupted, failed
    final_answer: str | None = None
    steps: list[TaskStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        step_name: str,
        step_type: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加任务步骤"""
        step = TaskStep(
            step_name=step_name,
            step_type=step_type,
            timestamp=datetime.now().isoformat(),
            input_data=input_data or {},
            output_data=output_data or {},
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )
        self.steps.append(step)

    def set_status(self, status: str) -> None:
        """设置任务状态"""
        self.status = status
        if status in ["completed", "interrupted", "failed"]:
            self.end_time = datetime.now().isoformat()

    def set_final_answer(self, answer: str) -> None:
        """设置最终答案"""
        self.final_answer = answer

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def save(self, file_path: str | Path) -> None:
        """保存日志到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        from chatdb.utils.logger import logger

        logger.info(f"任务日志已保存: {file_path}")

