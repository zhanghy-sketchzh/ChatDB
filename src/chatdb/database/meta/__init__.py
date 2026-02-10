"""
数据源元数据缓存管理模块（已迁移）

已迁移到 chatdb.storage.meta_data
此模块保留用于向后兼容
"""

from chatdb.storage.meta_data import DataCacheManager, MetaDataStore

__all__ = ["DataCacheManager", "MetaDataStore"]
