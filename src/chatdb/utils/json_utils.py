"""
JSON 工具函数

提供 JSON 序列化相关的工具函数。
"""

from datetime import date, datetime
from typing import Any

import pandas as pd


def convert_to_json_serializable(obj: Any) -> Any:
    """
    递归转换对象为可JSON序列化的格式

    Args:
        obj: 要转换的对象

    Returns:
        可JSON序列化的对象
    """
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        if isinstance(obj, pd.Timestamp):
            if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
                return obj.strftime("%Y-%m-%d")
            else:
                return obj.isoformat()
        elif isinstance(obj, datetime):
            if obj.hour == 0 and obj.minute == 0 and obj.second == 0:
                return obj.strftime("%Y-%m-%d")
            else:
                return obj.isoformat()
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        else:
            return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    else:
        return str(obj)


