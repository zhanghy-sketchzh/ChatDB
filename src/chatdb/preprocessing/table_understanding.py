"""
表理解文本生成器

通过 LLM 根据 Schema 信息（列名、类型、统计摘要、唯一值等）生成一段
详细的表内容理解文本，帮助下游 Agent（SemanticParser/Planner/SQLTool）
准确理解该表的业务含义、数据范围和使用注意事项。

生成的文本缓存在 meta_data.db 的 table_understanding 字段中，避免重复调用 LLM。
"""

import logging
from typing import Any

from chatdb.llm.base import BaseLLM, Message

logger = logging.getLogger(__name__)


def _build_schema_summary(
    table_name: str,
    table_description: str,
    row_count: int,
    column_count: int,
    column_profiles: list[dict[str, Any]],
) -> str:
    """将列元信息组装为 LLM prompt 中的 Schema 描述段"""
    lines = [
        f"表名: {table_name}",
        f"简要描述: {table_description or '（无）'}",
        f"数据量: {row_count:,} 行 × {column_count} 列",
        "",
        "列信息:",
    ]
    for p in column_profiles:
        name = p.get("name", "")
        dtype = p.get("dtype", "")
        summary = p.get("summary", "")
        unique_count = p.get("unique_count", 0)
        null_pct = p.get("null_pct", 0)

        parts = [f"  - {name} ({dtype})"]
        if summary:
            parts.append(f"    摘要: {summary}")
        if unique_count:
            parts.append(f"    唯一值数: {unique_count}")
        if null_pct and null_pct > 0:
            parts.append(f"    缺失率: {null_pct:.1f}%")

        # 枚举值（如果有）
        unique_values = p.get("unique_values", [])
        if unique_values:
            vals = [str(v[0]) if isinstance(v, (list, tuple)) else str(v) for v in unique_values[:30]]
            parts.append(f"    枚举值: {', '.join(vals)}")

        lines.append("\n".join(parts))
    return "\n".join(lines)


SYSTEM_PROMPT = """你是一位资深数据分析师，擅长从数据库 Schema 信息中理解表的业务含义。"""

USER_PROMPT_TEMPLATE = """请根据以下数据表的 Schema 信息，生成一段**详细的表理解文本**。

## 要求
1. **表的整体定位**：这张表记录的是什么业务数据？属于哪个业务领域？
2. **核心维度与指标**：哪些列是维度（分类/分组字段），哪些是指标（数值/度量字段）？
3. **数据粒度**：每一行代表什么？是按什么粒度组织的（如按月、按产品、按部门等）？
4. **关键枚举值含义**：重要分类字段的枚举值分别代表什么含义？
5. **数据范围与边界**：时间范围、数据覆盖的业务范围等
6. **使用注意事项**：
   - 表名本身是否具有误导性（如表名宽泛但数据实际只覆盖某个子领域）
   - 哪些列可能容易被误解
   - 查询时需要注意的筛选条件（如有效数据标记列、口径筛选列等）

## Schema 信息
{schema_summary}
{yml_meta_section}
## 输出格式
请直接输出纯文本的分析结果（不需要 JSON 或 Markdown 标记），用清晰的自然语言段落组织。
控制在 300~500 字以内，重点突出对查询有指导意义的信息。"""


async def generate_table_understanding(
    llm: BaseLLM,
    table_name: str,
    table_description: str = "",
    row_count: int = 0,
    column_count: int = 0,
    column_profiles: list[dict[str, Any]] | None = None,
    yml_config: dict[str, Any] | None = None,
) -> str:
    """
    调用 LLM 生成表理解文本。

    Args:
        llm: LLM 实例
        table_name: 表名
        table_description: 表的简要描述
        row_count: 行数
        column_count: 列数
        column_profiles: 列元信息列表（来自 ColumnProfiler）
        yml_config: YML 业务配置（如有），用于提供报表层级、组织范围等结构化信息

    Returns:
        生成的表理解文本
    """
    schema_summary = _build_schema_summary(
        table_name, table_description, row_count, column_count,
        column_profiles or [],
    )

    # 如果有 YML 配置，提取 meta 信息作为额外上下文
    yml_meta_section = ""
    if yml_config:
        from chatdb.config.table_config import format_yml_meta_for_prompt
        meta_text = format_yml_meta_for_prompt(yml_config, level="full")
        if meta_text:
            yml_meta_section = f"\n## 业务配置信息（来自 YML 元数据）\n{meta_text}\n"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema_summary=schema_summary,
        yml_meta_section=yml_meta_section,
    )

    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    logger.info(f"正在为表 '{table_name}' 生成理解文本...")
    response = await llm.generate(messages, temperature=0.1, max_tokens=2000)

    understanding = response.content.strip()
    logger.info(f"表 '{table_name}' 理解文本生成完成 ({len(understanding)} 字)")
    return understanding
