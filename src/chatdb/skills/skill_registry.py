"""
SkillRegistry - 分析技能注册中心

设计理念：
1. 渐进披露（Progressive Disclosure）：不同消费者只获取需要的信息层
   - SemanticParser: 识别层（keywords + when_to_use）
   - Planner:        规划层（planner_prompt + examples）
   - SQLAgent/Tool:  执行层（sql_instruction + sql_meta）
2. 文件驱动：每个 Skill 是一个目录（SKILL.md + assets/），新增技能无需改代码
3. 统一接口：注册后系统各组件通过 SkillRegistry 按需获取

目录约定：
  skills/
    trend/
      SKILL.md          # 技能定义（YAML front-matter + Markdown 分节）
      assets/
        examples.json   # 任务示例（可选）
    comparison/
      SKILL.md
      assets/
        examples.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatdb.utils.logger import logger


# =============================================================================
# Skill 数据模型
# =============================================================================

@dataclass
class SkillMeta:
    """
    Skill 执行元信息（原 SQLTaskMeta 的职责）
    
    控制 SQLAgent 的前置校验和参数补全逻辑。
    """
    requires_time: bool = False
    requires_dimension: bool = False
    requires_previous_results: bool = False
    default_time_granularity: str = ""
    intent_hint_template: str = ""
    stats_fields: list[str] = field(default_factory=list)
    issue_fields: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """
    一个完整的分析技能定义
    
    各字段按消费者分层（渐进披露）：
    ┌──────────────────┬───────────────────────────────────────┐
    │  消费者           │  需要的字段                           │
    ├──────────────────┼───────────────────────────────────────┤
    │  SemanticParser  │  id, label, description, keywords,    │
    │                  │  when_to_use, not_for, disambiguation │
    ├──────────────────┼───────────────────────────────────────┤
    │  Planner         │  planner_prompt, examples, sql_hints  │
    ├──────────────────┼───────────────────────────────────────┤
    │  SQLTool         │  sql_instruction, meta                │
    ├──────────────────┼───────────────────────────────────────┤
    │  SQLAgent        │  meta (requires_*, stats/issue)       │
    └──────────────────┴───────────────────────────────────────┘
    """
    # === 基础标识 ===
    id: str                          # 技能 ID (如 "trend", "comparison")
    label: str                       # 中文名 (如 "趋势分析")
    description: str                 # 一句话描述

    # === 识别层 (SemanticParser) ===
    keywords: list[str] = field(default_factory=list)       # 触发关键词
    when_to_use: str = ""            # 适用场景
    not_for: str = ""                # 不适用场景
    disambiguation: list[str] = field(default_factory=list) # 区分易混淆类型的规则

    # === 规划层 (Planner) ===
    planner_prompt: str = ""         # Planner LLM 的完整 Prompt 模板
    examples: list[dict] = field(default_factory=list)  # 任务示例 JSON
    sql_hints: dict[str, Any] = field(default_factory=dict)  # SQL 提示

    # === 执行层 (SQLTool + SQLAgent) ===
    sql_instruction: str = ""        # SQL 生成时的详细指导
    meta: SkillMeta = field(default_factory=SkillMeta)

    # === 原始章节（按需读取任意 Markdown 章节）===
    _sections: dict[str, str] = field(default_factory=dict, repr=False)

    # === 内部 ===
    priority_boost: int = 0          # 优先级加成（特殊技能如 anomaly）
    source_path: str = ""            # SKILL.md 来源路径

    # ----- 渐进披露接口 -----

    def get_semantic_prompt(self) -> str:
        """生成 SemanticParser 用的识别描述（表格行，用于 task_type 判断表）"""
        desc = self.description or self.label
        kw = ", ".join(self.keywords[:6]) if self.keywords else ""
        return f"| {self.id} | {self.label} | {desc} | {kw} |"

    def get_semantic_detail(self) -> str:
        """生成 SemanticParser 用的详细识别信息（含适用场景、排除场景、消歧规则）
        
        从 SKILL.md 的 When to Activate / Not For / Disambiguation 章节提取，
        供模型更准确地判断 task_type。
        """
        parts: list[str] = [f"### {self.id} — {self.label}"]
        
        if self.when_to_use:
            parts.append(f"**适用场景**：\n{self.when_to_use}")
        elif self.description:
            parts.append(f"**适用场景**：{self.description}")
        
        if self.not_for:
            parts.append(f"**不适用**：\n{self.not_for}")
        
        if self.disambiguation:
            parts.append("**消歧规则**：\n" + "\n".join(f"- {r}" for r in self.disambiguation))
        
        return "\n\n".join(parts)

    def get_disambiguation_rules(self) -> str:
        """生成消歧规则文本"""
        if not self.disambiguation:
            return ""
        return "\n".join(f"- {r}" for r in self.disambiguation)

    def get_planner_prompt(self) -> str:
        """生成 Planner 用的完整 Prompt（第二层：详细）"""
        if self.planner_prompt:
            return self.planner_prompt
        # 自动从字段拼装 fallback
        lines = [f"## {self.label} ({self.id})\n"]
        if self.when_to_use:
            lines.append(f"**适用场景**：{self.when_to_use}")
        if self.not_for:
            lines.append(f"**不适用**：{self.not_for}")
        if self.examples:
            lines.append(f"\n**任务示例**：\n```json\n{json.dumps(self.examples[0], ensure_ascii=False, indent=2)}\n```")
        return "\n".join(lines)

    def get_sql_instruction(self) -> str:
        """生成 SQLTool 用的 SQL 生成指导（第三层：执行细节）"""
        return self.sql_instruction

    def get_section(self, section_name: str) -> str:
        """
        按章节名获取 SKILL.md 中的任意内容。
        
        section_name 不区分大小写，例如：
          - "sql instruction"
          - "planner prompt"
          - "when to activate"
          - "disambiguation"
        """
        return self._sections.get(section_name.lower(), "")


# =============================================================================
# SKILL.md 解析器
# =============================================================================

class SkillParser:
    """
    解析 SKILL.md 文件为 Skill 对象
    
    SKILL.md 格式约定：
    ```
    ---
    id: trend
    label: 趋势分析
    description: 按时间聚合，分析变化趋势
    keywords: [趋势, 走势, 逐月, 逐年, 变化]
    priority_boost: 0
    meta:
      requires_time: true
      default_time_granularity: year
      intent_hint_template: "按{granularity}聚合{metric}，观察时间变化趋势"
      stats_fields: [available_years, year_count, growth_rate]
      issue_fields: [only_single_year, missing_time_range]
    ---
    
    # When to Activate
    ...
    
    # Planner Prompt
    ...
    
    # SQL Instruction
    ...
    
    # Examples
    ...
    ```
    """

    # front-matter 正则
    _FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    # section heading 正则 (# Section Name)
    _SECTION_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

    @classmethod
    def parse_file(cls, path: Path) -> Skill:
        """解析单个 SKILL.md 文件"""
        text = path.read_text(encoding="utf-8")

        # 1. 解析 YAML front-matter
        fm_data = cls._parse_front_matter(text)

        # 2. 解析 Markdown 分节
        sections = cls._parse_sections(text)

        # 3. 加载配套 examples.json
        examples = cls._load_examples(path.parent / "assets" / "examples.json")

        # 4. 构建 SkillMeta
        meta_raw = fm_data.pop("meta", {}) or {}
        meta = SkillMeta(
            requires_time=meta_raw.get("requires_time", False),
            requires_dimension=meta_raw.get("requires_dimension", False),
            requires_previous_results=meta_raw.get("requires_previous_results", False),
            default_time_granularity=meta_raw.get("default_time_granularity", ""),
            intent_hint_template=meta_raw.get("intent_hint_template", ""),
            stats_fields=meta_raw.get("stats_fields", []),
            issue_fields=meta_raw.get("issue_fields", []),
        )

        # 5. 构建 Skill
        skill = Skill(
            id=fm_data.get("id", path.parent.name),
            label=fm_data.get("label", ""),
            description=fm_data.get("description", ""),
            keywords=fm_data.get("keywords", []),
            priority_boost=fm_data.get("priority_boost", 0),
            when_to_use=sections.get("when to activate", ""),
            not_for=sections.get("not for", ""),
            disambiguation=cls._parse_list(sections.get("disambiguation", "")),
            planner_prompt=sections.get("planner prompt", ""),
            sql_instruction=sections.get("sql instruction", ""),
            sql_hints=fm_data.get("sql_hints", {}),
            examples=examples,
            meta=meta,
            _sections=sections,
            source_path=str(path),
        )
        return skill

    @classmethod
    def _parse_front_matter(cls, text: str) -> dict[str, Any]:
        """简单的 YAML front-matter 解析（不依赖 pyyaml）"""
        m = cls._FM_RE.match(text)
        if not m:
            return {}

        result: dict[str, Any] = {}
        current_key = ""
        current_indent = 0
        nested: dict[str, Any] = {}

        for line in m.group(1).split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())

            # 检查是否是嵌套结构的子项
            if indent > 0 and current_key and indent > current_indent:
                k, v = cls._parse_kv(stripped)
                if k:
                    nested[k] = cls._parse_value(v)
                continue
            elif current_key and nested:
                result[current_key] = nested
                nested = {}

            k, v = cls._parse_kv(stripped)
            if not k:
                continue

            current_indent = indent
            if v == "":
                # 下一行可能是嵌套 dict
                current_key = k
                nested = {}
            else:
                current_key = ""
                result[k] = cls._parse_value(v)

        # 收尾
        if current_key and nested:
            result[current_key] = nested

        return result

    @staticmethod
    def _parse_kv(line: str) -> tuple[str, str]:
        """解析 key: value"""
        if ":" not in line:
            return ("", "")
        idx = line.index(":")
        key = line[:idx].strip()
        val = line[idx + 1:].strip()
        return (key, val)

    @staticmethod
    def _parse_value(val: str) -> Any:
        """解析值类型"""
        if not val:
            return ""
        # bool
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        # int
        try:
            return int(val)
        except ValueError:
            pass
        # list: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
        # 去除引号的字符串
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            return val[1:-1]
        return val

    @classmethod
    def _parse_sections(cls, text: str) -> dict[str, str]:
        """解析 Markdown sections（去掉 front-matter 后）"""
        # 去掉 front-matter
        body = cls._FM_RE.sub("", text, count=1)

        sections: dict[str, str] = {}
        positions = list(cls._SECTION_RE.finditer(body))

        for i, match in enumerate(positions):
            name = match.group(1).strip().lower()
            start = match.end()
            end = positions[i + 1].start() if i + 1 < len(positions) else len(body)
            content = body[start:end].strip()
            sections[name] = content

        return sections

    @staticmethod
    def _parse_list(text: str) -> list[str]:
        """解析 Markdown 列表"""
        if not text:
            return []
        return [
            line.lstrip("- ").strip()
            for line in text.split("\n")
            if line.strip().startswith("-")
        ]

    @staticmethod
    def _load_examples(path: Path) -> list[dict]:
        """加载 examples.json"""
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else [data]
            except Exception:
                pass
        return []


# =============================================================================
# SkillRegistry - 核心注册中心
# =============================================================================

class SkillRegistry:
    """
    分析技能注册中心
    
    职责：
    1. 从 skills/ 目录自动发现和加载 Skill
    2. 支持运行时动态注册/注销
    3. 按消费者（SemanticParser / Planner / SQLTool）提供渐进披露接口
    4. 为 Planner 的 type 选择和 SQL 引擎行为提供统一知识源
    
    使用方式：
        registry = SkillRegistry()
        registry.load_from_directory("skills/")
        
        # SemanticParser 获取识别表
        table = registry.get_semantic_table()
        
        # Planner 获取某个类型的 Prompt
        prompt = registry.get_planner_prompt("comparison")
        
        # SQLTool 获取 SQL 指导
        instruction = registry.get_sql_instruction("trend")
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._log = logger

    # ── 加载 ──

    def load_from_directory(self, skills_dir: str | Path) -> int:
        """
        从目录递归加载所有 Skill
        
        目录结构：
          skills_dir/
            trend/SKILL.md
            comparison/SKILL.md
            source/SKILL.md
        
        Returns:
            成功加载的 Skill 数量
        """
        skills_path = Path(skills_dir)
        if not skills_path.is_dir():
            self._log.warn(f"Skill 目录不存在: {skills_path}")
            return 0

        count = 0
        for skill_md in sorted(skills_path.rglob("SKILL.md")):
            try:
                skill = SkillParser.parse_file(skill_md)
                self.register(skill)
                count += 1
            except Exception as e:
                self._log.warn(f"加载 Skill 失败 ({skill_md}): {e}")

        self._log.info(f"已加载 {count} 个 Skill: [{', '.join(self._skills.keys())}]")
        return count

    def register(self, skill: Skill) -> None:
        """注册单个 Skill"""
        if skill.id in self._skills:
            self._log.debug(f"覆盖 Skill: {skill.id}")
        self._skills[skill.id] = skill

    def unregister(self, skill_id: str) -> bool:
        """注销 Skill"""
        if skill_id in self._skills:
            del self._skills[skill_id]
            return True
        return False

    def get(self, skill_id: str) -> Skill | None:
        """获取单个 Skill"""
        return self._skills.get(skill_id)

    def list_ids(self) -> list[str]:
        """列出所有已注册的 Skill ID"""
        return list(self._skills.keys())

    def list_skills(self) -> list[Skill]:
        """列出所有已注册的 Skill"""
        return list(self._skills.values())

    # ── 渐进披露接口 ──

    def get_semantic_table(self, skill_ids: list[str] | None = None) -> str:
        """
        生成 SemanticParser 用的 task_type 判断信息
        
        包含两部分：
        1. 速查表（description + keywords）用于快速定位候选类型
        2. 详细识别规则（When to Activate / Not For / Disambiguation）用于精确判断
        """
        skills = self._filter(skill_ids)
        if not skills:
            return ""

        # 第一部分：速查表
        lines = [
            "### task_type 速查表",
            "",
            "| task_type | 含义 | 判断标准 | 关键词 |",
            "|-----------|------|----------|--------|",
        ]
        for s in skills:
            lines.append(s.get_semantic_prompt())
        
        # 第二部分：详细识别规则
        lines.append("")
        lines.append("### 各 task_type 详细判断规则")
        lines.append("")
        for s in skills:
            detail = s.get_semantic_detail()
            if detail:
                lines.append(detail)
                lines.append("")

        return "\n".join(lines)

    def get_planner_prompt(self, skill_id: str) -> str:
        """获取单个 Skill 的 Planner Prompt"""
        skill = self._skills.get(skill_id)
        if skill:
            return skill.get_planner_prompt()
        return ""

    def get_planner_prompts(self, skill_ids: list[str] | None = None) -> str:
        """获取多个 Skill 的 Planner Prompt（用 --- 分隔）"""
        skills = self._filter(skill_ids)
        return "\n\n---\n\n".join(s.get_planner_prompt() for s in skills)

    def get_planner_type_rules(self, skill_ids: list[str] | None = None) -> str:
        """
        生成 Planner 的 type→SQL引擎行为 映射表
        
        用于 Planner Prompt 的"规则第5条"。
        """
        skills = self._filter(skill_ids)
        if not skills:
            return ""

        lines = []
        for s in skills:
            hint = s.sql_hints.get("summary", s.description) if s.sql_hints else s.description
            lines.append(f"   - `{s.id}`: {hint}")
        return "\n".join(lines)

    def get_sql_instruction(self, skill_id: str) -> str:
        """获取单个 Skill 的 SQL 生成指导"""
        skill = self._skills.get(skill_id)
        if skill:
            return skill.get_sql_instruction()
        return ""

    def get_section(self, skill_id: str, section_name: str) -> str:
        """
        按章节名获取某个 Skill 的 SKILL.md 中的任意内容。
        
        Args:
            skill_id: 技能 ID（如 "ratio", "comparison"）
            section_name: 章节名（不区分大小写，如 "sql instruction", "planner prompt"）
        """
        skill = self._skills.get(skill_id)
        if skill:
            return skill.get_section(section_name)
        return ""

    def get_sql_examples(self, skill_id: str, max_examples: int = 3) -> list[dict]:
        """获取 Skill 的 SQL 示例（用于 SQL 生成 prompt）"""
        skill = self._skills.get(skill_id)
        if skill and skill.examples:
            return skill.examples[:max_examples]
        return []

    def get_skill_meta(self, skill_id: str) -> SkillMeta | None:
        """获取 Skill 的执行元信息"""
        skill = self._skills.get(skill_id)
        if skill:
            return skill.meta
        return None

    def get_skill_label(self, skill_id: str) -> str:
        """获取 Skill 中文名"""
        skill = self._skills.get(skill_id)
        return skill.label if skill else skill_id

    def get_skill_description(self, skill_id: str) -> str:
        """获取 Skill 描述"""
        skill = self._skills.get(skill_id)
        return skill.description if skill else ""

    # ── 批量查询 ──

    def get_all_labels(self) -> dict[str, str]:
        """返回 {skill_id: label} 映射"""
        return {s.id: s.label for s in self._skills.values()}

    def get_all_descriptions(self) -> dict[str, str]:
        """返回 {skill_id: description} 映射"""
        return {s.id: s.description for s in self._skills.values()}

    # ── 内部 ──

    def _filter(self, skill_ids: list[str] | None = None) -> list[Skill]:
        """按 ID 列表过滤（自动去重，保序），None 表示全部"""
        if skill_ids is None:
            return list(self._skills.values())
        seen: set[str] = set()
        result: list[Skill] = []
        for sid in skill_ids:
            if sid not in seen and sid in self._skills:
                seen.add(sid)
                result.append(self._skills[sid])
        return result

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def __repr__(self) -> str:
        return f"<SkillRegistry: {list(self._skills.keys())}>"


# =============================================================================
# 全局实例
# =============================================================================

_default_skill_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """获取全局 SkillRegistry 实例"""
    global _default_skill_registry
    if _default_skill_registry is None:
        _default_skill_registry = SkillRegistry()
    return _default_skill_registry


def init_skill_registry(skills_dir: str | Path) -> SkillRegistry:
    """初始化并加载 Skill 目录"""
    registry = get_skill_registry()
    registry.load_from_directory(skills_dir)
    return registry
