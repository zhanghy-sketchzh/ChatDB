"""
ChatDB 配置模块

提供两种配置加载方式：
1. MetricsConfigLoader - 原有的全量配置加载（一次性加载所有配置）
2. DomainConfigLoader - 渐进式领域配置加载（两阶段加载，节省 Token）
"""

from chatdb.config.metrics_loader import MetricsConfigLoader, load_metrics_config
from chatdb.config.domain_config import (
    DomainConfigLoader,
    TwoStagePromptBuilder,
    load_domain_config,
    create_prompt_builder,
)

__all__ = [
    # 原有接口
    "MetricsConfigLoader", 
    "load_metrics_config",
    # 渐进式加载接口
    "DomainConfigLoader",
    "TwoStagePromptBuilder",
    "load_domain_config",
    "create_prompt_builder",
]
