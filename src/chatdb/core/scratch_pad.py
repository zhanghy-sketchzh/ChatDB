"""
ScratchPadManager - SQL 结果文件暂存

设计理念（Filesystem Context Engineering）：
- SQLAgent 执行完后，完整结果行写入文件，prompt 只传摘要和文件路径
- 写一次，按需读：Planner/Summary 各自按需读取，不通过消息链传递大数据
- 文件即接口：Agent 之间不直接传递大数据，通过文件路径引用

文件结构：
    data/scratch/{session_id}/
        task_{task_id}_result.json   # SQLAgent 写入的完整结果
    data/scratch/{session_id}/manifest.json  # 结果索引清单

collected_results 格式变化：
    原始格式（内存全量）:
        {"task_id": [{"subtask", "sql", "row_count", "examples", "stats", "issues"}, ...]}

    新格式（文件引用 + 摘要）:
        {"task_id": [{
            "subtask", "sql", "row_count",
            "examples": [...前5行...],         # 保留少量样例
            "stats": {...},                     # 保留统计信息
            "issues": [...],                    # 保留问题诊断
            "_file_ref": {                      # 新增：文件引用
                "path": "data/scratch/.../task_xxx_result.json",
                "full_row_count": 200,
                "summary": "共 200 行，Top3: ..."
            }
        }, ...]}
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from chatdb.utils.logger import get_component_logger


class ScratchPadManager:
    """
    SQL 结果文件暂存管理器

    核心职责：
    1. 将 SQLAgent 的完整执行结果写入 JSON 文件
    2. 生成紧凑的摘要（供 Planner prompt 使用）
    3. 提供按需读取接口（供 Summary/深查使用）
    4. 管理临时文件的生命周期（清理）
    """

    def __init__(
        self,
        base_path: str | Path = "data/scratch",
        max_age_hours: float = 24.0,
    ):
        self.base_path = Path(base_path)
        self.max_age_hours = max_age_hours
        self._log = get_component_logger("ScratchPad")

    # ============================================================
    # 写入
    # ============================================================

    def save_task_result(
        self,
        session_id: str,
        task_id: str,
        result_entry: dict[str, Any],
    ) -> dict[str, Any]:
        """
        将单个任务结果写入文件，返回文件引用。

        Args:
            session_id: 会话 ID
            task_id: 任务 ID
            result_entry: 原始的 TaskResultEntry.to_dict() 结果

        Returns:
            带 _file_ref 的精简结果（用于 collected_results）
        """
        session_dir = self._ensure_session_dir(session_id)

        # 构建完整结果（包含全部行数据）
        full_result = {
            "task_id": task_id,
            "subtask": result_entry.get("subtask", ""),
            "sql": result_entry.get("sql", ""),
            "row_count": result_entry.get("row_count", 0),
            "examples": result_entry.get("examples", []),
            "stats": result_entry.get("stats", {}),
            "issues": result_entry.get("issues", []),
            "saved_at": datetime.now().isoformat(),
        }

        # 写入文件
        subtask = result_entry.get("subtask", "result")
        safe_subtask = self._safe_filename(subtask)
        filename = f"task_{self._safe_filename(task_id)}_{safe_subtask}.json"
        file_path = session_dir / filename
        file_path.write_text(json.dumps(full_result, ensure_ascii=False, indent=2))

        # 更新 manifest
        self._update_manifest(session_dir, task_id, subtask, str(file_path), full_result)

        self._log.debug(f"结果已写入: {file_path} ({full_result['row_count']} 行)")

        # 生成摘要
        summary = self._generate_summary(full_result)

        # 返回精简版结果（替代原来的全量数据）
        row_count = result_entry.get("row_count", 0)
        all_examples = result_entry.get("examples", [])
        # ★ 少量数据保留全部行，大量数据保留前 30 行（供 Planner 直接内联到上下文）
        inline_threshold = 30
        examples_slim = all_examples if row_count <= inline_threshold else all_examples[:inline_threshold]
        
        slim_result = {
            "subtask": result_entry.get("subtask", ""),
            "sql": result_entry.get("sql", ""),
            "row_count": row_count,
            "examples": examples_slim,
            "stats": result_entry.get("stats", {}),
            "issues": result_entry.get("issues", []),
            "_file_ref": {
                "path": str(file_path),
                "full_row_count": row_count,
                "summary": summary,
            },
        }
        return slim_result

    # ============================================================
    # 读取
    # ============================================================

    def read_task_result(self, file_path: str | Path) -> dict[str, Any] | None:
        """
        按需读取完整任务结果

        Args:
            file_path: 文件路径（来自 _file_ref.path）

        Returns:
            完整结果 dict，或 None（文件不存在）
        """
        path = Path(file_path)
        if not path.exists():
            self._log.warn(f"结果文件不存在: {file_path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self._log.warn(f"读取结果文件失败: {e}")
            return None

    def read_all_results(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        """
        读取某会话的全部完整结果

        Returns:
            {task_id: [full_result_dict, ...]}
        """
        session_dir = self.base_path / session_id
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            return {}

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        all_results: dict[str, list[dict[str, Any]]] = {}
        for entry in manifest.get("entries", []):
            task_id = entry["task_id"]
            file_path = entry["file_path"]
            result = self.read_task_result(file_path)
            if result:
                all_results.setdefault(task_id, []).append(result)

        return all_results

    def get_file_ref(
        self,
        collected_results: dict[str, list[dict[str, Any]]],
        task_id: str,
    ) -> list[dict[str, Any]]:
        """从 collected_results 中获取某任务的文件引用列表"""
        refs = []
        for r in collected_results.get(task_id, []):
            file_ref = r.get("_file_ref")
            if file_ref:
                refs.append(file_ref)
        return refs

    # ============================================================
    # 摘要生成
    # ============================================================

    def _generate_summary(self, full_result: dict[str, Any]) -> str:
        """
        从完整结果生成紧凑摘要（用于 Planner prompt）

        摘要策略：
        - 行数 + Top N（维度=指标） + 变化趋势洞察
        - 控制在 400 字符以内
        """
        parts: list[str] = []
        row_count = full_result.get("row_count", 0)
        parts.append(f"共 {row_count} 行")

        examples = full_result.get("examples", [])
        if examples:
            top_items = self._extract_top_items(examples, n=3)
            if top_items:
                parts.append(f"Top: {top_items}")
            
            # 时序变化洞察（多行数值数据时自动计算）
            trend_insight = self._extract_trend_insight(examples)
            if trend_insight:
                parts.append(trend_insight)

        # 从 stats 提取关键统计（排除易误导的聚合字段）
        stats = full_result.get("stats", {})
        skip_keys = {"row_count", "available_years", "year_count"}
        stat_parts = [
            f"{k}={v}" for k, v in list(stats.items())[:3]
            if k not in skip_keys
        ]
        if stat_parts:
            parts.append(", ".join(stat_parts))

        summary = "; ".join(parts)
        if len(summary) > 400:
            summary = summary[:397] + "..."
        return summary

    @staticmethod
    def _extract_top_items(examples: list[dict[str, Any]], n: int = 3) -> str:
        """从样例行中提取 Top N 条目的简要描述"""
        if not examples:
            return ""

        dim_col, val_col = None, None
        for key, val in examples[0].items():
            if dim_col is None and isinstance(val, str):
                dim_col = key
            if val_col is None and isinstance(val, float):
                val_col = key
            if dim_col and val_col:
                break

        # 整数维度列（如年份）不应被当作指标
        if not dim_col:
            for key, val in examples[0].items():
                if isinstance(val, int) and not isinstance(val, bool):
                    dim_col = key
                    break

        if not val_col:
            items = [str(list(ex.values())[0]) for ex in examples[:n] if ex]
            return ", ".join(items)

        items: list[str] = []
        for ex in examples[:n]:
            dim_val = ex.get(dim_col, "?") if dim_col else "?"
            num_val = ex.get(val_col, 0)
            if isinstance(num_val, (int, float)) and abs(num_val) >= 1e8:
                items.append(f"{dim_val}={num_val/1e8:.2f}亿")
            elif isinstance(num_val, (int, float)) and abs(num_val) >= 1e4:
                items.append(f"{dim_val}={num_val/1e4:.2f}万")
            else:
                items.append(f"{dim_val}={num_val}")
        return ", ".join(items)

    @staticmethod
    def _extract_trend_insight(examples: list[dict[str, Any]]) -> str:
        """从时序数据中提取变化趋势洞察（最大涨幅/跌幅）"""
        if len(examples) < 2:
            return ""

        dim_col, val_col = None, None
        for key, val in examples[0].items():
            if val_col is None and isinstance(val, float):
                val_col = key
            elif dim_col is None:
                dim_col = key

        if not dim_col or not val_col:
            return ""

        max_drop_pct, max_drop_label = 0.0, ""
        for i in range(1, len(examples)):
            prev = examples[i - 1].get(val_col, 0)
            curr = examples[i].get(val_col, 0)
            if not isinstance(prev, (int, float)) or not isinstance(curr, (int, float)) or prev == 0:
                continue
            pct = (curr - prev) / prev
            if pct < max_drop_pct:
                max_drop_pct = pct
                prev_dim = examples[i - 1].get(dim_col, "?")
                curr_dim = examples[i].get(dim_col, "?")
                max_drop_label = f"{prev_dim}→{curr_dim}"

        if max_drop_pct < -0.01:
            return f"最大降幅: {max_drop_label}（{abs(max_drop_pct):.1%}）"
        return ""

    def generate_planner_summary(
        self,
        collected_results: dict[str, list[dict[str, Any]]],
    ) -> str:
        """
        为 Planner 决策生成结果摘要

        格式：
            ### 任务: base_query
              [trend_analysis] 返回 200 行
              摘要: 共 200 行; Top: 王者荣耀=2.3亿, 和平精英=1.8亿
              完整数据: data/scratch/.../task_base_result.json
        """
        if not collected_results:
            return "（尚无）"

        lines = []
        for task_id, task_results in collected_results.items():
            lines.append(f"### 任务: {task_id}")
            for i, r in enumerate(task_results):
                subtask = r.get("subtask", f"步骤{i+1}")
                row_count = r.get("row_count", 0)
                stats = r.get("stats", {})
                issues = r.get("issues", [])
                file_ref = r.get("_file_ref")

                lines.append(f"  [{subtask}] 返回 {row_count} 行")

                # SQL 错误时显示失败的 SQL
                has_sql_error = any("error:" in issue for issue in issues)
                sql = r.get("sql", "")
                if has_sql_error and sql:
                    sql_preview = sql[:300] + "..." if len(sql) > 300 else sql
                    lines.append(f"  **失败的SQL**: `{sql_preview}`")

                # 摘要（来自文件引用）
                if file_ref:
                    summary = file_ref.get("summary", "")
                    file_path = file_ref.get("path", "")
                    if summary:
                        lines.append(f"  摘要: {summary}")
                    if file_path:
                        lines.append(f"  完整数据: {file_path}")
                else:
                    # 兼容：没有文件引用时，显示 examples
                    examples = r.get("examples", [])
                    if examples:
                        lines.append("  示例:")
                        for ex in examples[:3]:
                            items = list(ex.items())[:4]
                            lines.append(f"    - {', '.join(f'{k}={v}' for k, v in items)}")

                if stats:
                    stat_items = list(stats.items())
                    lines.append(f"  统计: {', '.join(f'{k}={v}' for k, v in stat_items)}")
                if issues:
                    lines.append(f"  备注: {', '.join(issues)}")
            lines.append("")
        return "\n".join(lines).strip()

    # ============================================================
    # 文件管理
    # ============================================================

    def _ensure_session_dir(self, session_id: str) -> Path:
        """确保会话目录存在"""
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _update_manifest(
        self,
        session_dir: Path,
        task_id: str,
        subtask: str,
        file_path: str,
        result: dict[str, Any],
    ) -> None:
        """更新会话的结果索引清单"""
        manifest_path = session_dir / "manifest.json"

        manifest: dict[str, Any]
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {"entries": []}
        else:
            manifest = {"entries": [], "created_at": datetime.now().isoformat()}

        entries: list[dict[str, Any]] = manifest.get("entries", [])
        entries.append({
            "task_id": task_id,
            "subtask": subtask,
            "file_path": file_path,
            "row_count": result.get("row_count", 0),
            "saved_at": result.get("saved_at", ""),
        })
        manifest["entries"] = entries
        manifest["updated_at"] = datetime.now().isoformat()

        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    @staticmethod
    def _safe_filename(name: str) -> str:
        """将字符串转为安全的文件名"""
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        return safe[:50]  # 限制长度

    def cleanup_session(self, session_id: str) -> None:
        """清理指定会话的 scratch 文件"""
        session_dir = self.base_path / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            self._log.debug(f"已清理会话 scratch: {session_id}")

    def cleanup_expired(self) -> int:
        """清理过期的 scratch 文件"""
        if not self.base_path.exists():
            return 0

        cutoff = time.time() - (self.max_age_hours * 3600)
        cleaned = 0

        for session_dir in self.base_path.iterdir():
            if not session_dir.is_dir():
                continue
            # 检查目录修改时间
            if session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir)
                cleaned += 1
                self._log.debug(f"清理过期 scratch: {session_dir.name}")

        return cleaned

    # ============================================================
    # 从 collected_results 恢复完整数据（供 Summary 使用）
    # ============================================================

    def expand_results(
        self,
        collected_results: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        将 collected_results 中的文件引用展开为完整数据

        用于 _generate_summary 等需要完整数据的场景。
        对于没有 _file_ref 的结果，原样返回。
        """
        expanded: dict[str, list[dict[str, Any]]] = {}
        for task_id, results in collected_results.items():
            expanded[task_id] = []
            for r in results:
                file_ref = r.get("_file_ref")
                if file_ref:
                    full = self.read_task_result(file_ref["path"])
                    if full:
                        expanded[task_id].append(full)
                    else:
                        # 读取失败，回退到精简版
                        expanded[task_id].append(r)
                else:
                    expanded[task_id].append(r)
        return expanded

    # ============================================================
    # Plan 持久化（方案二：Plan Persistence）
    # ============================================================

    def save_plan(
        self,
        session_id: str,
        plan_dict: dict[str, Any],
    ) -> Path:
        """
        将分析计划持久化到文件

        Args:
            session_id: 会话 ID
            plan_dict: AnalysisPlan.to_dict() 的结果

        Returns:
            plan 文件路径
        """
        session_dir = self._ensure_session_dir(session_id)
        plan_path = session_dir / "plan.json"
        plan_dict["updated_at"] = datetime.now().isoformat()
        plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2))
        self._log.debug(f"计划已保存: {plan_path}")
        return plan_path

    def load_plan(self, session_id: str) -> dict[str, Any] | None:
        """
        加载会话的持久化计划

        Returns:
            plan dict，或 None（不存在/解析失败）
        """
        plan_path = self.base_path / session_id / "plan.json"
        if not plan_path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
            return data
        except Exception as e:
            self._log.warn(f"加载计划失败: {e}")
            return None

    def update_plan_task_status(
        self,
        session_id: str,
        task_id: str,
        status: str,
        result_file: str = "",
    ) -> None:
        """
        更新持久化计划中单个任务的状态

        Args:
            session_id: 会话 ID
            task_id: 任务 ID
            status: 新状态（completed/failed/skipped）
            result_file: 结果文件路径（可选，任务完成时关联）
        """
        plan_data = self.load_plan(session_id)
        if not plan_data:
            return

        for task in plan_data.get("tasks", []):
            if task.get("id") == task_id:
                task["status"] = status
                if result_file:
                    task.setdefault("meta", {})["result_file"] = result_file
                break

        # 检查整体完成状态
        all_done = all(
            t.get("status") in ("completed", "failed", "skipped")
            for t in plan_data.get("tasks", [])
        )
        if all_done:
            plan_data["status"] = "completed"

        self.save_plan(session_id, plan_data)
