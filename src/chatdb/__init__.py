"""
ChatDB - 基于LLM多智能体的自然语言数据库查询系统

支持通过自然语言与数据库交互，生成SQL查询并返回结果和总结。
"""

__version__ = "0.1.0"
__author__ = "ChatDB Team"

from chatdb.utils.config import settings

__all__ = ["settings", "__version__"]

