"""
日志配置

参考 agno 的设计，使用 Rich 库进行富文本输出，支持多模块日志和文件日志。
日志文件按日期命名，格式：chatdb_YYYY-MM-DD.log
"""

import logging
import sys
from datetime import datetime
from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Any, Literal, Optional

from rich.logging import RichHandler
from rich.text import Text
from rich.console import Console

from chatdb.utils.config import settings

LOGGER_NAME = "chatdb"
EXCEL_LOGGER_NAME = f"{LOGGER_NAME}.excel"
LLM_LOGGER_NAME = f"{LOGGER_NAME}.llm"
DATABASE_LOGGER_NAME = f"{LOGGER_NAME}.database"
API_LOGGER_NAME = f"{LOGGER_NAME}.api"

# 定义不同模块的日志颜色样式
LOG_STYLES = {
    "excel": {
        "debug": "green",
        "info": "blue",
        "warning": "yellow",
        "error": "red",
    },
    "llm": {
        "debug": "magenta",
        "info": "steel_blue1",
        "warning": "orange3",
        "error": "red",
    },
    "database": {
        "debug": "cyan",
        "info": "bright_blue",
        "warning": "yellow",
        "error": "red",
    },
    "api": {
        "debug": "sandy_brown",
        "info": "orange3",
        "warning": "yellow",
        "error": "red",
    },
    "default": {
        "debug": "green",
        "info": "blue",
        "warning": "yellow",
        "error": "red",
    },
}


class ColoredRichHandler(RichHandler):
    """带颜色样式的 Rich Handler"""

    def __init__(self, *args, source_type: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_type = source_type

    def get_level_text(self, record: logging.LogRecord) -> Text:
        """根据 source_type 返回带颜色的日志级别文本"""
        if not record.msg:
            return Text("")

        level_name = record.levelname.lower()
        style_map = LOG_STYLES.get(self.source_type or "default", LOG_STYLES["default"])
        
        if level_name in style_map:
            color = style_map[level_name]
            return Text(record.levelname, style=color)
        
        return super().get_level_text(record)


class ChatDBLogger(logging.Logger):
    """ChatDB 自定义 Logger，支持居中标题"""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)

    def debug(self, msg: object, *args, center: bool = False, symbol: str = "*", **kwargs):  # type: ignore
        """记录 DEBUG 级别日志"""
        if center:
            msg = center_header(str(msg), symbol)
        # 移除自定义参数，避免传递给父类
        kwargs.pop("center", None)
        kwargs.pop("symbol", None)
        super().debug(msg, *args, **kwargs)

    def info(self, msg: object, *args, center: bool = False, symbol: str = "*", **kwargs):  # type: ignore
        """记录 INFO 级别日志"""
        if center:
            msg = center_header(str(msg), symbol)
        # 移除自定义参数，避免传递给父类
        kwargs.pop("center", None)
        kwargs.pop("symbol", None)
        super().info(msg, *args, **kwargs)

    def warning(self, msg: object, *args, center: bool = False, symbol: str = "*", **kwargs):  # type: ignore
        """记录 WARNING 级别日志"""
        if center:
            msg = center_header(str(msg), symbol)
        # 移除自定义参数，避免传递给父类
        kwargs.pop("center", None)
        kwargs.pop("symbol", None)
        super().warning(msg, *args, **kwargs)

    def error(self, msg: object, *args, center: bool = False, symbol: str = "*", **kwargs):  # type: ignore
        """记录 ERROR 级别日志"""
        if center:
            msg = center_header(str(msg), symbol)
        # 移除自定义参数，避免传递给父类
        kwargs.pop("center", None)
        kwargs.pop("symbol", None)
        super().error(msg, *args, **kwargs)


def build_logger(
    logger_name: str,
    source_type: Optional[str] = None,
    enable_file_logging: bool = True,
) -> Any:
    """
    构建日志器
    
    Args:
        logger_name: 日志器名称
        source_type: 源类型（用于颜色样式）
        enable_file_logging: 是否启用文件日志
    
    Returns:
        配置好的日志器
    """
    # 检查日志器是否已经存在
    _logger = logging.getLogger(logger_name)
    
    # 如果日志器已经是 ChatDBLogger 类型且已配置，直接返回
    if isinstance(_logger, ChatDBLogger) and (_logger.handlers or _logger.level != logging.NOTSET):
        return _logger
    
    # 如果日志器存在但不是 ChatDBLogger，需要重新创建
    if _logger.handlers:
        # 移除现有处理器
        for handler in _logger.handlers[:]:
            _logger.removeHandler(handler)
    
    # 设置自定义日志器类
    logging.setLoggerClass(ChatDBLogger)

    # 创建新的日志器（如果已存在，getLogger 会返回同一个实例，但类型会被更新）
    # 需要删除旧的 logger 并重新创建
    if logger_name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[logger_name]
    
    _logger = logging.getLogger(logger_name)

    # 恢复默认日志器类
    logging.setLoggerClass(logging.Logger)

    # 配置控制台输出（使用 Rich）
    is_dev = getenv("CHATDB_DEV", "false").lower() == "true"
    rich_handler = ColoredRichHandler(
        show_time=False,  # 不显示时间戳
        rich_tracebacks=True,
        show_path=is_dev,
        tracebacks_show_locals=is_dev,
        source_type=source_type,
    )
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
        )
    )

    _logger.addHandler(rich_handler)

    # 配置文件日志
    if enable_file_logging:
        file_handler = _create_file_handler(logger_name)
        if file_handler:
            _logger.addHandler(file_handler)

    # 设置日志级别
    log_level = getattr(logging, settings.log.level.upper(), logging.INFO)
    _logger.setLevel(log_level)
    _logger.propagate = False

    return _logger


def _create_file_handler(logger_name: str) -> Optional[logging.Handler]:
    """创建文件日志处理器
    
    所有 logger 共用同一个日志文件，通过日志格式中的 logger name 区分来源
    """
    try:
        from logging.handlers import RotatingFileHandler

        # 生成带日期的日志文件名（统一使用主日志名）
        log_file = Path(settings.log.file)
        log_dir = log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # 从配置文件中提取基础文件名（如 chatdb.log）
        base_name = log_file.name
        if "." in base_name:
            name_part, ext_part = base_name.rsplit(".", 1)
            # 统一日志文件名：chatdb_2026-01-23.log（不再按模块分）
            dated_filename = f"{name_part}_{datetime.now().strftime('%Y-%m-%d')}.{ext_part}"
        else:
            dated_filename = f"{base_name}_{datetime.now().strftime('%Y-%m-%d')}"

        dated_log_file = log_dir / dated_filename

        # 创建文件处理器（每天轮转）
        file_handler = RotatingFileHandler(
            str(dated_log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=30,  # 保留30个备份文件
            encoding="utf-8",
        )

        # 文件日志格式（包含时间戳、级别、模块、消息）
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        return file_handler
    except Exception:
        # 如果文件日志创建失败，不影响控制台日志
        return None


def center_header(message: str, symbol: str = "*") -> str:
    """生成居中标题"""
    try:
        import shutil

        terminal_width = shutil.get_terminal_size().columns
    except Exception:
        terminal_width = 80  # 默认宽度

    header = f" {message} "
    return f"{header.center(terminal_width - 20, symbol)}"


# 创建全局 Console 实例用于美化输出
_console = Console()


def log_llm_interaction(logger_instance: Any, step_name: str, prompt: str, response: str, max_prompt_chars: int = 300, max_response_chars: int = 500):
    """
    封装 LLM 交互日志输出（输入输出在一起）
    
    Args:
        logger_instance: 日志器实例
        step_name: 步骤名称
        prompt: Prompt 内容
        response: LLM 回复内容
        max_prompt_chars: Prompt 最大显示字符数
        max_response_chars: Response 最大显示字符数
    """
    # 显示步骤名称
    logger_instance.info(f"[{step_name}]")
    
    # 显示 Prompt（截断）
    prompt_preview = prompt[:max_prompt_chars] + "..." if len(prompt) > max_prompt_chars else prompt
    logger_instance.info(f"  Prompt: {prompt_preview}")
    
    # 显示 Response（截断）
    response_preview = response[:max_response_chars] + "..." if len(response) > max_response_chars else response
    logger_instance.info(f"  Response: {response_preview}")


def log_step(logger_instance: Any, step_name: str, message: str = ""):
    """
    美化步骤日志输出
    
    Args:
        logger_instance: 日志器实例
        step_name: 步骤名称
        message: 附加消息
    """
    if message:
        logger_instance.info(f"[{step_name}] {message}")
    else:
        logger_instance.info(f"[{step_name}]")


# 创建各个模块的日志器
default_logger: ChatDBLogger = build_logger(LOGGER_NAME, source_type="default")
excel_logger: ChatDBLogger = build_logger(EXCEL_LOGGER_NAME, source_type="excel")
llm_logger: ChatDBLogger = build_logger(LLM_LOGGER_NAME, source_type="llm")
database_logger: ChatDBLogger = build_logger(DATABASE_LOGGER_NAME, source_type="database")
api_logger: ChatDBLogger = build_logger(API_LOGGER_NAME, source_type="api")

# 默认使用 default_logger
logger: ChatDBLogger = default_logger

# Debug 模式控制
debug_on: bool = False
debug_level: Literal[1, 2] = 1

# LLM Debug 模式控制
llm_debug_on: bool = False
llm_debug_show_input: bool = False  # 是否显示 LLM 输入（prompt）

# ============================================================
# 三层日志系统
# ============================================================
# 1. Task View (INFO): 任务级摘要，一次请求 3~5 行
# 2. ReAct View (DEBUG): 步骤级 Trace，带组件名
# 3. LLM View (llm_debug): 完整 prompt/response

class TaskLogger:
    """
    任务级日志器 - 概览级别摘要
    
    一次请求只输出关键节点：
    - 意图解析结果
    - SQL 生成概览
    - 关键步骤序列
    - 最终摘要
    """
    
    def __init__(self):
        self._steps: list[str] = []
        self._query: str = ""
    
    def start(self, query: str) -> None:
        """开始新任务"""
        self._steps = []
        self._query = query
        logger.info(f"📝 Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    
    def intent(self, intent_type: str, metrics: list[str], dimensions: list[str], filters: list[str]) -> None:
        """记录意图解析结果"""
        parts = [f"type={intent_type}"]
        if metrics:
            parts.append(f"metrics={metrics[:3]}")
        if dimensions:
            parts.append(f"dims={dimensions[:3]}")
        if filters:
            parts.append(f"filters={filters[:3]}")
        logger.info(f"🎯 Intent: {', '.join(parts)}")
        self._steps.append("semantic")
    
    def sql(self, sql: str, candidates_count: int = 1) -> None:
        """记录 SQL 生成"""
        sql_preview = sql[:60].replace('\n', ' ')
        suffix = f" (+{candidates_count-1} candidates)" if candidates_count > 1 else ""
        logger.info(f"📊 SQL: {sql_preview}...{suffix}")
        self._steps.append("sql")
    
    def execute(self, row_count: int, success: bool = True) -> None:
        """记录执行结果"""
        if success:
            logger.info(f"✅ Result: {row_count} rows")
        else:
            logger.info(f"❌ Execute failed")
        self._steps.append("exec")
    
    def explore(self, dimension: str, top_value: str, top_count: int, total_categories: int) -> None:
        """记录探索分析"""
        logger.info(f"🔍 Explore [{dimension}]: top={top_value}({top_count}), {total_categories} categories")
        self._steps.append(f"explore:{dimension}")
    
    def error(self, error_type: str, message: str) -> None:
        """记录错误"""
        logger.warning(f"⚠️ Error [{error_type}]: {message[:80]}")
    
    def done(self, summary: str) -> None:
        """任务完成"""
        summary_preview = summary[:100].replace('\n', ' ')
        logger.info(f"📌 Summary: {summary_preview}{'...' if len(summary) > 100 else ''}")
        logger.info(f"🔗 Steps: {' → '.join(self._steps)} → done")


class ComponentLogger:
    """
    组件级日志器 - 带组件名的 ReAct Trace
    
    格式: [Component/Action] message
    """
    
    def __init__(self, component: str):
        self.component = component
    
    def think(self, message: str) -> None:
        """THINK 日志"""
        logger.debug(f"[{self.component}/THINK] {message}")
    
    def observe(self, message: str) -> None:
        """OBSERVE 日志"""
        logger.debug(f"[{self.component}/OBSERVE] {message}")
    
    def reflect(self, message: str) -> None:
        """REFLECT 日志"""
        logger.debug(f"[{self.component}/REFLECT] {message}")
    
    def act(self, action: str, detail: str = "") -> None:
        """ACT 日志"""
        msg = f"[{self.component}/ACT] {action}"
        if detail:
            msg += f": {detail}"
        logger.debug(msg)
    
    def debug(self, message: str) -> None:
        """DEBUG 日志"""
        logger.debug(f"[{self.component}] {message}")
    
    def info(self, message: str) -> None:
        """INFO 日志"""
        logger.info(f"[{self.component}] {message}")
    
    def warn(self, message: str) -> None:
        """WARNING 日志"""
        logger.warning(f"[{self.component}] {message}")
    
    def error(self, message: str) -> None:
        """ERROR 日志"""
        logger.error(f"[{self.component}] {message}")


# 全局任务日志器
task_log = TaskLogger()

# 组件日志器工厂
_component_loggers: dict[str, ComponentLogger] = {}

def get_component_logger(component: str) -> ComponentLogger:
    """获取组件日志器"""
    if component not in _component_loggers:
        _component_loggers[component] = ComponentLogger(component)
    return _component_loggers[component]


def setup_logging() -> None:
    """初始化日志系统（保持向后兼容）"""
    # 日志系统已经在模块加载时自动初始化
    pass


def set_log_level_to_debug(source_type: Optional[str] = None, level: Literal[1, 2] = 1):
    """设置日志级别为 DEBUG"""
    global debug_on, debug_level

    if source_type is None:
        _logger = default_logger
    else:
        _logger = logging.getLogger(f"{LOGGER_NAME}.{source_type}")

    _logger.setLevel(logging.DEBUG)
    debug_on = True
    debug_level = level


def enable_llm_debug(enable: bool = True, show_input: bool = False):
    """
    启用/禁用 LLM Debug 模式
    
    启用后，会输出 LLM 调用的输入输出
    
    Args:
        enable: 是否启用
        show_input: 是否显示输入（prompt），默认 False 只显示输出
    """
    global llm_debug_on, llm_debug_show_input
    llm_debug_on = enable
    llm_debug_show_input = show_input
    if enable:
        if show_input:
            llm_logger.info("🔍 LLM Debug 模式已启用 - 将输出完整的模型输入输出")
        else:
            llm_logger.info("🔍 LLM Debug 模式已启用 - 仅显示模型输出")
    else:
        llm_logger.info("LLM Debug 模式已关闭")


def is_llm_debug_enabled() -> bool:
    """检查 LLM Debug 模式是否启用"""
    return llm_debug_on


def log_llm_debug(
    caller_name: str,
    system_prompt: str | None,
    user_prompt: str,
    response: str,
    model: str = "",
    agent_name: str = "",
):
    """
    输出 LLM 完整的输入输出（仅在 llm_debug_on 时）
    
    Args:
        caller_name: 调用场景名称 (如 extract_intent, generate_sql)
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        response: 模型响应
        model: 模型名称
        agent_name: 智能体名称 (如 SemanticParser, SQLGenerator)
    """
    if not llm_debug_on:
        return
    
    separator = "=" * 80
    sub_separator = "-" * 60
    
    # 构建标题: [Agent/caller_name]
    title = f"[{agent_name}/{caller_name}]" if agent_name else f"[{caller_name}]"
    
    # 使用 llm_logger 输出
    llm_logger.info(f"\n{separator}")
    llm_logger.info(f"🔍 LLM DEBUG - {title}" + (f" (model: {model})" if model else ""))
    llm_logger.info(separator)
    
    # 只有在 llm_debug_show_input 时才显示输入
    if llm_debug_show_input:
        if system_prompt:
            llm_logger.info(f"\n📋 SYSTEM PROMPT:\n{sub_separator}")
            llm_logger.info(system_prompt)
        
        llm_logger.info(f"\n📝 USER PROMPT:\n{sub_separator}")
        llm_logger.info(user_prompt)
    
    llm_logger.info(f"\n🤖 RESPONSE:\n{sub_separator}")
    llm_logger.info(response)
    
    llm_logger.info(f"\n{separator}\n")


def set_log_level_to_info(source_type: Optional[str] = None):
    """设置日志级别为 INFO"""
    global debug_on

    if source_type is None:
        _logger = default_logger
    else:
        _logger = logging.getLogger(f"{LOGGER_NAME}.{source_type}")

    _logger.setLevel(logging.INFO)
    debug_on = False


def set_log_level_to_warning(source_type: Optional[str] = None):
    """设置日志级别为 WARNING"""
    global debug_on

    if source_type is None:
        _logger = default_logger
    else:
        _logger = logging.getLogger(f"{LOGGER_NAME}.{source_type}")

    _logger.setLevel(logging.WARNING)
    debug_on = False


def set_log_level_to_error(source_type: Optional[str] = None):
    """设置日志级别为 ERROR"""
    global debug_on

    if source_type is None:
        _logger = default_logger
    else:
        _logger = logging.getLogger(f"{LOGGER_NAME}.{source_type}")

    _logger.setLevel(logging.ERROR)
    debug_on = False


def use_excel_logger():
    """切换到 Excel 日志器"""
    global logger
    logger = excel_logger


def use_llm_logger():
    """切换到 LLM 日志器"""
    global logger
    logger = llm_logger


def use_database_logger():
    """切换到 Database 日志器"""
    global logger
    logger = database_logger


def use_api_logger():
    """切换到 API 日志器"""
    global logger
    logger = api_logger


def use_default_logger():
    """切换到默认日志器"""
    global logger
    logger = default_logger


@lru_cache(maxsize=128)
def _using_chatdb_logger(logger_instance: Any) -> bool:
    """检查是否使用 ChatDBLogger"""
    return isinstance(logger_instance, ChatDBLogger)


def log_debug(
    msg: str,
    center: bool = False,
    symbol: str = "*",
    log_level: Literal[1, 2] = 1,
    *args,
    **kwargs,
):
    """记录 DEBUG 日志"""
    global logger, debug_on, debug_level

    if debug_on and debug_level >= log_level:
        if _using_chatdb_logger(logger):
            logger.debug(msg, center, symbol, *args, **kwargs)
        else:
            logger.debug(msg, *args, **kwargs)


def log_info(msg: str, center: bool = False, symbol: str = "*", *args, **kwargs):
    """记录 INFO 日志"""
    global logger
    if _using_chatdb_logger(logger):
        logger.info(msg, center, symbol, *args, **kwargs)
    else:
        logger.info(msg, *args, **kwargs)


def log_warning(msg: str, center: bool = False, symbol: str = "*", *args, **kwargs):
    """记录 WARNING 日志"""
    global logger
    if _using_chatdb_logger(logger):
        logger.warning(msg, center, symbol, *args, **kwargs)
    else:
        logger.warning(msg, *args, **kwargs)


def log_error(msg: str, center: bool = False, symbol: str = "*", *args, **kwargs):
    """记录 ERROR 日志"""
    global logger
    if _using_chatdb_logger(logger):
        logger.error(msg, center, symbol, *args, **kwargs)
    else:
        logger.error(msg, *args, **kwargs)


def log_exception(msg: str, *args, **kwargs):
    """记录异常日志"""
    global logger
    logger.exception(msg, *args, **kwargs)


# 导出
__all__ = [
    "logger",
    "default_logger",
    "excel_logger",
    "llm_logger",
    "database_logger",
    "api_logger",
    "setup_logging",
    "set_log_level_to_debug",
    "set_log_level_to_info",
    "set_log_level_to_warning",
    "set_log_level_to_error",
    "use_excel_logger",
    "use_llm_logger",
    "use_database_logger",
    "use_api_logger",
    "use_default_logger",
    "log_debug",
    "log_info",
    "log_warning",
    "log_error",
    "log_exception",
    "log_step",
    "center_header",
    "debug_on",
    "debug_level",
    # LLM Debug
    "llm_debug_on",
    "llm_debug_show_input",
    "enable_llm_debug",
    "is_llm_debug_enabled",
    "log_llm_debug",
    # 三层日志
    "task_log",
    "TaskLogger",
    "ComponentLogger",
    "get_component_logger",
]
