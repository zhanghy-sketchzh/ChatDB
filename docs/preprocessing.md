# 数据预处理

## 概述

数据预处理是系统运行的前置步骤，将原始数据转换为 LLM 友好的检索资源。

```
原始数据 (CSV/Excel/DB)
        │
        ▼
  DataPreprocessor
        │
        ├─▶ Schema 生成 (DDL + Light)
        ├─▶ 列摘要生成 (ColumnProfile)
        ├─▶ BM25 索引构建
        └─▶ 元数据存储
```

## 使用示例

```python
from chatdb.preprocessing import DataPreprocessor
import pandas as pd

df = pd.read_csv("data/excel/脚本测试数据.csv")

preprocessor = DataPreprocessor()
result = preprocessor.preprocess_dataframe(
    df,
    table_name="脚本测试数据",
    table_description="IEG财务数据表",
)

print(f"行数: {result.row_count}")
print(f"索引文档数: {result.index_doc_count}")
```

## Schema 生成

### DDL Schema

```sql
CREATE TABLE "脚本测试数据" (
    "年" INTEGER,
    "月份" INTEGER,
    "数据集来源" VARCHAR,
    "ieg口径金额-人民币" DOUBLE
);
-- 行数：3340000
```

### Light Schema (Markdown)

```markdown
## 脚本测试数据 (3340000行)

| 字段 | 类型 | 摘要 |
|------|------|------|
| 年 | 整数 | 范围[2024~2025] |
| 数据集来源 | 文本 | 常见值: '一方报表', '投资公司'...共5种 |
```

## 列摘要 (ColumnProfile)

```python
@dataclass
class ColumnProfile:
    name: str
    dtype: str          # 整数/小数/文本/日期
    null_pct: float
    unique_count: int
    
    # 数值字段
    min_val: float | None
    max_val: float | None
    mean_val: float | None
    
    # 文本字段
    unique_values: list[tuple[str, int]]  # (值, 频次)
```

| 字段类型 | 摘要内容 |
|----------|----------|
| 数值 | 范围、均值、中位数 |
| 文本 | Top20 高频值 |
| ID | 唯一值计数 |

## BM25 索引

三级索引文档：

| 类型 | doc_type | 内容 | 用途 |
|------|----------|------|------|
| 表级 | `table` | 表名 + 描述 + 列名 | 表选择 |
| 列级 | `column` | 列名 + 类型 + 摘要 | 字段定位 |
| 值级 | `value` | 分类字段唯一值 | 关键词召回 |

### 检索示例

```python
# 关键词匹配
matches = preprocessor.match_keywords_to_values("王者荣耀流水")
# -> [KeywordMatch(keyword='王者荣耀', column='考核产品', score=0.95)]
```

## 存储位置

```
data/pilot/
├── meta_data.db      # 表元数据
├── text_index.db     # BM25 索引
└── history.db        # 聊天历史
```

## 更新数据

```python
# 重新预处理会自动更新（基于 file_hash）
result = preprocessor.preprocess_dataframe(df, table_name="xxx")

# 清空索引重建
from chatdb.preprocessing import TextIndex
TextIndex().clear()
```
