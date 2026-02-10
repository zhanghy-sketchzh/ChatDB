"""
列摘要生成器

为每个字段生成 LLM 友好的摘要：
- 数值字段：最大值、最小值、中位数、平均值
- 分类字段：所有唯一值 (unique_values)，summary 取 top20
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    """列摘要"""
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    is_id: bool = False
    
    # 数值字段统计
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    median_val: float | None = None
    std_val: float | None = None
    
    # 分类字段：所有唯一值 (value, count) 按频次降序
    unique_values: list[tuple[str, int]] = field(default_factory=list)
    
    def to_text_summary(self, top_k: int = 20) -> str:
        """生成文本摘要（用于 Light Schema）"""
        if self.dtype in ("整数", "小数"):
            if self.min_val is not None:
                return f"范围[{self.min_val:.2f}~{self.max_val:.2f}], 均值{self.mean_val:.2f}, 中位数{self.median_val:.2f}"
        elif self.unique_values:
            # 取 top20 生成摘要文本
            top_strs = [f"'{v}'" for v, _ in self.unique_values[:top_k]]
            suffix = f"...共{self.unique_count}种" if self.unique_count > top_k else ""
            return f"常见值: {', '.join(top_strs)}{suffix}"
        return f"唯一值{self.unique_count}个"
    
    def to_dict(self) -> dict[str, Any]:
        """转为字典（存储到 meta_data.db）"""
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "null_pct": round(self.null_pct, 2),
            "unique_count": self.unique_count,
            "is_id": self.is_id,
            "summary": self.to_text_summary(),
        }
        if self.min_val is not None:
            d["stats"] = {
                "min": self.min_val, "max": self.max_val,
                "mean": self.mean_val, "median": self.median_val,
            }
        if self.unique_values:
            # 存储所有唯一值
            d["unique_values"] = self.unique_values
        return d


class ColumnProfiler:
    """列摘要生成器"""
    
    def __init__(self, summary_top_k: int = 20):
        """
        Args:
            summary_top_k: summary 文本中展示的 top 值数量
        """
        self.summary_top_k = summary_top_k
    
    def profile_column(
        self, 
        series: pd.Series, 
        col_name: str,
        is_id: bool = False
    ) -> ColumnProfile:
        """生成单列摘要"""
        dtype_str = self._get_dtype_str(series.dtype)
        null_count = int(series.isnull().sum())
        null_pct = (null_count / len(series)) * 100 if len(series) > 0 else 0
        unique_count = series.nunique(dropna=True)
        
        profile = ColumnProfile(
            name=col_name,
            dtype=dtype_str,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            is_id=is_id,
        )
        
        # 数值字段统计
        if dtype_str in ("整数", "小数") and not is_id:
            col_data = series.dropna()
            if len(col_data) > 0:
                profile.min_val = float(col_data.min())
                profile.max_val = float(col_data.max())
                profile.mean_val = float(col_data.mean())
                profile.median_val = float(col_data.median())
                profile.std_val = float(col_data.std())
        
        # 分类字段：存储所有唯一值
        elif dtype_str == "文本" and not is_id:
            value_counts = series.value_counts()
            profile.unique_values = [
                (str(v), int(c)) for v, c in value_counts.items()
            ]
        
        return profile
    
    def profile_dataframe(
        self, 
        df: pd.DataFrame, 
        id_columns: list[str] | None = None
    ) -> list[ColumnProfile]:
        """生成 DataFrame 所有列的摘要"""
        id_columns = id_columns or []
        profiles = []
        
        for col in df.columns:
            profile = self.profile_column(
                df[col], col, is_id=(col in id_columns)
            )
            profiles.append(profile)
        
        return profiles
    
    def profiles_to_json(self, profiles: list[ColumnProfile]) -> str:
        """将摘要列表转为 JSON"""
        return json.dumps(
            [p.to_dict() for p in profiles],
            ensure_ascii=False,
            indent=2
        )
    
    @staticmethod
    def _get_dtype_str(dtype) -> str:
        """获取数据类型的中文描述"""
        dtype_str = str(dtype)
        if dtype_str in ("int64", "int32", "Int64"):
            return "整数"
        elif dtype_str in ("float64", "float32"):
            return "小数"
        elif "datetime" in dtype_str:
            return "日期时间"
        else:
            return "文本"
