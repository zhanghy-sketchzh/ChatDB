"""
UnixTool - 文件系统操作工具

提供文件读写、目录遍历、内容搜索等能力，
配合 ScratchPadManager 实现 Filesystem Context Engineering。

子工具：
- unix.read_file:  读取文件（支持行范围）
- unix.write_file: 写入文件
- unix.list_dir:   列出目录结构
- unix.search:     搜索文件内容（grep）
- unix.file_stat:  获取文件元信息
"""

from __future__ import annotations

import fnmatch
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from chatdb.tools.base import BaseTool, SubToolDef, ToolMetadata, ToolParameter, ToolResult
from lib.utils.logger import get_component_logger

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext

# 安全沙箱：只允许操作的目录白名单前缀
_DEFAULT_ALLOWED_ROOTS = ("data/", "output/", "scratch/", "tmp/")
# 禁止写入的文件后缀
_DENY_WRITE_EXTS = {".py", ".sh", ".env", ".yaml", ".yml", ".toml", ".cfg"}

MAX_READ_LINES = 2000
MAX_SEARCH_MATCHES = 200
MAX_LIST_ENTRIES = 500


class UnixTool(BaseTool):
    """
    文件系统操作工具

    核心能力：
    - 读取/写入文件（支持行范围切片）
    - 列出目录树
    - 正则搜索文件内容
    - 获取文件元信息
    - 与 ScratchPad 联动，按需加载大结果
    """

    def __init__(
        self,
        workspace: str | Path = ".",
        allowed_roots: tuple[str, ...] = _DEFAULT_ALLOWED_ROOTS,
        readonly: bool = False,
    ):
        metadata = ToolMetadata(
            name="unix",
            description="文件系统操作：读取、写入、目录遍历、内容搜索、元信息查询",
            category="filesystem",
            inputs={
                "action": {
                    "type": "str",
                    "description": "操作类型: read_file|write_file|list_dir|search|file_stat",
                },
                "path": {"type": "str", "description": "目标路径（相对于 workspace）"},
            },
            outputs={
                "content": {"type": "str", "description": "文件内容或搜索结果"},
                "entries": {"type": "list", "description": "目录条目列表"},
            },
            subtools=[
                "unix.read_file",
                "unix.write_file",
                "unix.list_dir",
                "unix.search",
                "unix.file_stat",
            ],
            is_core=False,
        )
        super().__init__(metadata)

        self.workspace = Path(workspace).resolve()
        self.allowed_roots = allowed_roots
        self.readonly = readonly
        self._log = get_component_logger("UnixTool")

    # ------------------------------------------------------------------
    # BaseTool 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "unix"

    @property
    def description(self) -> str:
        return (
            "文件系统操作工具：读取文件、写入文件、列目录、搜索内容、获取元信息。\n"
            "适用于 ScratchPad 文件的按需读取和结果持久化。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("action", "string", "操作类型", enum=[
                "read_file", "write_file", "list_dir", "search", "file_stat",
            ]),
            ToolParameter("path", "string", "目标路径（相对于 workspace）"),
            ToolParameter("content", "string", "写入内容（write_file 时必填）", required=False),
            ToolParameter("start_line", "number", "起始行号（1-based，read_file）", required=False),
            ToolParameter("end_line", "number", "结束行号（含，read_file）", required=False),
            ToolParameter("pattern", "string", "正则表达式（search）", required=False),
            ToolParameter("glob", "string", "文件名 glob 过滤（search/list_dir）", required=False),
            ToolParameter("recursive", "boolean", "是否递归（list_dir/search）", required=False),
        ]

    @property
    def subtool_defs(self) -> list[SubToolDef]:
        """LLM 可选的原子指令集"""
        return [
            SubToolDef(
                name="read_file",
                description="读取文件内容。支持行范围切片，适合按需查看 scratch 结果文件",
                parameters=[
                    ToolParameter("path", "string", "文件路径（相对于 workspace）"),
                    ToolParameter("start_line", "number", "起始行号（1-based）", required=False),
                    ToolParameter("end_line", "number", "结束行号（含）", required=False),
                ],
            ),
            SubToolDef(
                name="write_file",
                description="将内容写入文件。仅限 data/output/scratch/tmp 目录",
                parameters=[
                    ToolParameter("path", "string", "文件路径（相对于 workspace）"),
                    ToolParameter("content", "string", "写入内容"),
                ],
            ),
            SubToolDef(
                name="list_dir",
                description="列出目录内容，查看 scratch 会话目录下有哪些结果文件",
                parameters=[
                    ToolParameter("path", "string", "目录路径（相对于 workspace）"),
                    ToolParameter("glob", "string", "文件名过滤模式，如 *.json", required=False),
                    ToolParameter("recursive", "boolean", "是否递归子目录", required=False),
                ],
            ),
            SubToolDef(
                name="search",
                description="在文件中搜索内容（正则匹配），适合在结果文件中查找特定数据",
                parameters=[
                    ToolParameter("path", "string", "搜索起始路径"),
                    ToolParameter("pattern", "string", "正则表达式"),
                    ToolParameter("glob", "string", "文件名过滤，默认 *.json", required=False),
                    ToolParameter("recursive", "boolean", "是否递归搜索", required=False),
                ],
            ),
            SubToolDef(
                name="file_stat",
                description="获取文件/目录元信息（大小、修改时间、子项数量）",
                parameters=[
                    ToolParameter("path", "string", "目标路径"),
                ],
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        dispatch = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "search": self._search,
            "file_stat": self._file_stat,
        }
        handler = dispatch.get(action)
        if not handler:
            return ToolResult.fail(f"未知操作: {action}，支持: {list(dispatch)}")
        return await handler(**kwargs)

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    async def _read_file(self, **kw: Any) -> ToolResult:
        """读取文件，支持行范围切片"""
        path = self._resolve(kw.get("path", ""))
        if not path:
            return ToolResult.fail("路径不能为空")
        if not path.is_file():
            return ToolResult.fail(f"文件不存在: {path}")

        start = kw.get("start_line")
        end = kw.get("end_line")

        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            return ToolResult.fail("非文本文件，无法读取")

        total = len(lines)

        if start or end:
            s = max((start or 1) - 1, 0)
            e = min(end or total, total)
            selected = lines[s:e]
        else:
            selected = lines[:MAX_READ_LINES]

        truncated = len(selected) < total and not (start or end)
        content = "".join(selected)

        return ToolResult.ok(
            data={
                "content": content,
                "total_lines": total,
                "returned_lines": len(selected),
                "truncated": truncated,
                "path": str(path.relative_to(self.workspace)),
            },
            message=f"读取 {len(selected)}/{total} 行",
        )

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    async def _write_file(self, **kw: Any) -> ToolResult:
        """写入文件（仅限允许目录，禁止写代码文件）"""
        if self.readonly:
            return ToolResult.fail("当前为只读模式")

        rel_path = kw.get("path", "")
        content = kw.get("content", "")
        if not rel_path:
            return ToolResult.fail("路径不能为空")

        path = self._resolve(rel_path)
        if not path:
            return ToolResult.fail("路径不能为空")

        if not self._is_allowed_write(path):
            return ToolResult.fail(
                f"安全限制：不允许写入 {rel_path}，"
                f"允许的目录前缀: {self.allowed_roots}"
            )

        if path.suffix in _DENY_WRITE_EXTS:
            return ToolResult.fail(f"安全限制：禁止写入 {path.suffix} 文件")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        self._log.debug(f"已写入: {path} ({len(content)} 字符)")
        return ToolResult.ok(
            data={"path": str(path.relative_to(self.workspace)), "bytes": len(content.encode("utf-8"))},
            message=f"写入成功: {rel_path}",
        )

    # ------------------------------------------------------------------
    # list_dir
    # ------------------------------------------------------------------

    async def _list_dir(self, **kw: Any) -> ToolResult:
        """列出目录内容"""
        path = self._resolve(kw.get("path", "."))
        if not path or not path.is_dir():
            return ToolResult.fail(f"目录不存在: {kw.get('path', '.')}")

        recursive = kw.get("recursive", False)
        glob_pat = kw.get("glob", "*")

        entries: list[dict[str, Any]] = []
        iterator = path.rglob(glob_pat) if recursive else path.glob(glob_pat)

        for item in sorted(iterator):
            if len(entries) >= MAX_LIST_ENTRIES:
                break
            rel = item.relative_to(self.workspace)
            entry: dict[str, Any] = {
                "name": item.name,
                "path": str(rel),
                "type": "dir" if item.is_dir() else "file",
            }
            if item.is_file():
                entry["size"] = item.stat().st_size
            entries.append(entry)

        return ToolResult.ok(
            data={"entries": entries, "count": len(entries), "truncated": len(entries) >= MAX_LIST_ENTRIES},
            message=f"共 {len(entries)} 个条目",
        )

    # ------------------------------------------------------------------
    # search (grep)
    # ------------------------------------------------------------------

    async def _search(self, **kw: Any) -> ToolResult:
        """在文件中搜索内容（正则）"""
        base = self._resolve(kw.get("path", "."))
        pattern = kw.get("pattern", "")
        glob_pat = kw.get("glob", "*.json")
        recursive = kw.get("recursive", True)

        if not pattern:
            return ToolResult.fail("搜索需要 pattern 参数")
        if not base:
            return ToolResult.fail("路径不能为空")

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult.fail(f"正则表达式错误: {e}")

        matches: list[dict[str, Any]] = []
        files_to_scan: list[Path] = []

        if base.is_file():
            files_to_scan = [base]
        elif base.is_dir():
            iterator = base.rglob(glob_pat) if recursive else base.glob(glob_pat)
            files_to_scan = [f for f in iterator if f.is_file()]
        else:
            return ToolResult.fail(f"路径不存在: {base}")

        for fp in sorted(files_to_scan):
            if len(matches) >= MAX_SEARCH_MATCHES:
                break
            try:
                lines = fp.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append({
                        "file": str(fp.relative_to(self.workspace)),
                        "line": i,
                        "text": line.strip()[:300],
                    })
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        break

        return ToolResult.ok(
            data={
                "matches": matches,
                "match_count": len(matches),
                "truncated": len(matches) >= MAX_SEARCH_MATCHES,
            },
            message=f"找到 {len(matches)} 处匹配",
        )

    # ------------------------------------------------------------------
    # file_stat
    # ------------------------------------------------------------------

    async def _file_stat(self, **kw: Any) -> ToolResult:
        """获取文件/目录元信息"""
        path = self._resolve(kw.get("path", ""))
        if not path or not path.exists():
            return ToolResult.fail(f"路径不存在: {kw.get('path', '')}")

        stat = path.stat()
        info: dict[str, Any] = {
            "path": str(path.relative_to(self.workspace)),
            "type": "dir" if path.is_dir() else "file",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }

        if path.is_dir():
            children = list(path.iterdir())
            info["child_count"] = len(children)
            info["child_files"] = sum(1 for c in children if c.is_file())
            info["child_dirs"] = sum(1 for c in children if c.is_dir())

        return ToolResult.ok(data=info, message=f"{'目录' if path.is_dir() else '文件'}: {info['path']}")

    # ------------------------------------------------------------------
    # ReAct 模式
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        **kwargs: Any,
    ) -> None:
        result = await self.execute(**kwargs)
        if not result.success:
            state.set_error(result.error or "文件操作失败")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path | None:
        """将相对路径解析为绝对路径，并校验不逃逸出 workspace"""
        if not rel_path:
            return None
        resolved = (self.workspace / rel_path).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            self._log.warning(f"路径逃逸: {rel_path}")
            return None
        return resolved

    def _is_allowed_write(self, path: Path) -> bool:
        """检查写入路径是否在允许的目录下"""
        rel = str(path.relative_to(self.workspace))
        return any(rel.startswith(root) for root in self.allowed_roots)
