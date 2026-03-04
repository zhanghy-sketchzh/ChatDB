"""
技能系统模块（Skill Registry）

提供分析技能的注册、发现和渐进披露接口。
"""

from lib.skills.skill_registry import (
    Skill,
    SkillMeta,
    SkillParser,
    SkillRegistry,
    get_skill_registry,
    init_skill_registry,
)

__all__ = [
    "Skill",
    "SkillMeta",
    "SkillParser",
    "SkillRegistry",
    "get_skill_registry",
    "init_skill_registry",
]
