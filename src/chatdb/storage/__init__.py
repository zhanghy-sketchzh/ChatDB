"""
存储模块

提供数据持久化存储功能：
- MetaDataStore: 表元数据存储（data/pilot/meta_data.db）
- ChatHistoryDB: 聊天历史存储（data/pilot/history.db）
- TaskHistoryDB: 任务执行历史存储（data/pilot/history.db）
"""

from chatdb.storage.chat_history import (
    ChatHistoryDB,
    ChatHistoryManager,
    HistoryConfig,
    Message,
    MessageRole,
    RunRecord,
)
from chatdb.storage.meta_data import DataCacheManager, MetaDataStore
from chatdb.storage.task_history import (
    AgentStep,
    TaskHistoryDB,
    TaskRecord,
    TaskStatus,
    TaskTracker,
)

__all__ = [
    # Meta Data
    "MetaDataStore",
    "DataCacheManager",  # 向后兼容
    # Chat History
    "ChatHistoryDB",
    "ChatHistoryManager",
    "HistoryConfig",
    "Message",
    "MessageRole",
    "RunRecord",
    # Task History
    "TaskHistoryDB",
    "TaskTracker",
    "TaskRecord",
    "AgentStep",
    "TaskStatus",
]
