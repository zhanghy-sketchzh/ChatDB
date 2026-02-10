"""
输出格式化器

将 LLM 生成的 SQL、查询结果、数据摘要等整理成最终展示格式。
"""

from typing import Any


class OutputFormatter:
    """输出格式化器"""

    def format_query_result(
        self,
        query: str,
        sql: str,
        result: list[dict[str, Any]],
        summary: str = "",
        row_count: int = 0,
    ) -> str:
        """
        格式化查询结果为最终展示文本

        Args:
            query: 用户原始查询
            sql: 生成的 SQL
            result: 查询结果
            summary: 结果总结
            row_count: 结果行数

        Returns:
            格式化后的文本
        """
        lines = []

        # 用户查询
        lines.append("=" * 60)
        lines.append("📝 用户查询")
        lines.append("=" * 60)
        lines.append(query)
        lines.append("")

        # 生成的 SQL
        lines.append("=" * 60)
        lines.append("💾 生成的 SQL")
        lines.append("=" * 60)
        lines.append(sql)
        lines.append("")

        # 查询结果
        lines.append("=" * 60)
        lines.append(f"📊 查询结果（共 {row_count} 条记录）")
        lines.append("=" * 60)

        if result:
            # 显示表头
            if result:
                headers = list(result[0].keys())
                lines.append(" | ".join(headers))
                lines.append("-" * 60)

                # 显示数据（最多显示前20行）
                for i, row in enumerate(result[:20], 1):
                    values = [str(row.get(h, "")) for h in headers]
                    lines.append(" | ".join(values))

                if len(result) > 20:
                    lines.append(f"... 还有 {len(result) - 20} 条记录未显示")
        else:
            lines.append("无数据")

        lines.append("")

        # 结果总结
        if summary:
            lines.append("=" * 60)
            lines.append("📋 结果总结")
            lines.append("=" * 60)
            lines.append(summary)
            lines.append("")

        return "\n".join(lines)

    def format_agent_pipeline_result(
        self,
        result: dict[str, Any],
        include_details: bool = True,
    ) -> str:
        """
        格式化智能体管道执行结果为最终展示文本

        Args:
            result: 管道执行结果字典
            include_details: 是否包含详细信息

        Returns:
            格式化后的文本
        """
        lines = []

        # 基本信息
        lines.append("=" * 60)
        lines.append("🎯 任务执行结果")
        lines.append("=" * 60)
        lines.append(f"状态: {'✅ 成功' if result.get('success') else '❌ 失败'}")
        lines.append(f"查询: {result.get('query', '')}")
        lines.append("")

        # 表选择信息
        if result.get("table_selection"):
            ts = result["table_selection"]
            lines.append("📋 表选择:")
            lines.append(f"  选中的表: {', '.join(ts.get('selected_tables', []))}")
            lines.append(f"  选择理由: {ts.get('selection_reason', '')}")
            lines.append("")

        # 问题改写信息
        if result.get("query_rewrite"):
            qr = result["query_rewrite"]
            lines.append("✏️ 问题改写:")
            lines.append(f"  改写后: {qr.get('rewritten_query', '')}")
            if qr.get("relevant_columns"):
                lines.append("  相关列:")
                for col in qr["relevant_columns"]:
                    lines.append(f"    - {col.get('column_name')}: {col.get('usage')}")
            if qr.get("analysis_suggestions"):
                lines.append("  分析建议:")
                for suggestion in qr["analysis_suggestions"]:
                    lines.append(f"    • {suggestion}")
            lines.append("")

        # SQL 信息
        if result.get("sql"):
            lines.append("💾 生成的 SQL:")
            lines.append(result["sql"])
            lines.append("")

        # 验证信息
        if result.get("validation"):
            val = result["validation"]
            status = "✅ 通过" if val.get("is_valid") else "❌ 未通过"
            lines.append(f"🔒 SQL 验证: {status}")
            if val.get("message"):
                lines.append(f"  消息: {val['message']}")
            lines.append("")

        # 查询结果
        if result.get("result"):
            lines.append(f"📊 查询结果（共 {result.get('row_count', 0)} 条记录）:")
            if include_details and result["result"]:
                # 显示前5行
                for i, row in enumerate(result["result"][:5], 1):
                    lines.append(f"  行{i}: {row}")
                if len(result["result"]) > 5:
                    lines.append(f"  ... 还有 {len(result['result']) - 5} 条记录")
            lines.append("")

        # 结果总结
        if result.get("summary"):
            lines.append("📋 结果总结:")
            lines.append(result["summary"])
            lines.append("")

        # 错误信息
        if result.get("error"):
            lines.append("❌ 错误信息:")
            lines.append(result["error"])
            lines.append("")

        # 智能体执行结果
        if include_details and result.get("agent_results"):
            lines.append("🤖 智能体执行详情:")
            for agent_name, agent_result in result["agent_results"].items():
                status = agent_result.get("status", "unknown")
                message = agent_result.get("message", "")
                lines.append(f"  {agent_name}: {status} - {message}")
            lines.append("")

        return "\n".join(lines)

    def format_error_result(
        self,
        error: str,
        query: str = "",
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        格式化错误结果为展示文本

        Args:
            error: 错误信息
            query: 用户查询
            context: 上下文信息

        Returns:
            格式化后的错误文本
        """
        lines = []

        lines.append("=" * 60)
        lines.append("❌ 任务执行失败")
        lines.append("=" * 60)

        if query:
            lines.append(f"用户查询: {query}")
            lines.append("")

        lines.append(f"错误信息: {error}")
        lines.append("")

        if context:
            lines.append("执行上下文:")
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"  {key}: {value}")
                elif isinstance(value, list) and len(value) < 10:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        return "\n".join(lines)


