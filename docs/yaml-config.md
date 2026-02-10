# YAML 配置规则

YAML 配置是业务口径的核心定义，围绕「表」收口。

## 配置结构（6 个核心节点）

| 节点 | 说明 |
|------|------|
| `meta` | 表元信息 |
| `dimensions` | 维度定义（**含业务术语**） |
| `metrics` | 指标定义（**含默认筛选**） |
| `filters` | 基础筛选条件 |
| `examples` | Few-shot 示例 |
| `rules` | 业务规则（可选） |

## 完整示例

```yaml
# data/yml/metrics_config.yml
meta:
  table_name: "脚本测试数据"
  display_name: "IEG财务数据表"
  value_column: "ieg口径金额-人民币"
  grain: "每条记录代表某产品在某部门某月某报表项的金额"

dimensions:
  time:
    label: "时间维度"
    columns: ["年", "月份", "季度"]
    column_map:
      year: "年"
      quarter: "季度"
      month: "月份"
  
  org:
    label: "组织维度"
    columns: ["数据集来源", "考核部门"]
    # 业务术语合并到维度中
    terms:
      ieg_hq:
        term: "IEG本部"
        synonyms: ["本部", "IEG", "总部"]
        filter: '"数据集来源" = ''一方报表'''
      
      invest_company:
        term: "投资公司"
        synonyms: ["投资", "控股公司"]
        filter: '"数据集来源" = ''投资公司-实际'''
  
  product:
    label: "产品维度"
    columns: ["考核产品", "产品大类"]

metrics:
  flow:
    label: "流水"
    expr: 'SUM("ieg口径金额-人民币")'
    # 指标默认筛选
    default_filters:
      - '"大盘报表项" = ''流水'''
      - '"考核口径" = ''流水'''
  
  revenue:
    label: "收入"
    expr: 'SUM("ieg口径金额-人民币")'
    default_filters:
      - '"大盘报表项" = ''收入'''
      - '"考核口径" = ''收入'''
  
  profit_after_deferral:
    label: "递延后利润"
    expr: 'SUM("ieg口径金额-人民币")'
    default_filters:
      - '"大盘报表项" = ''利润'''
      - '"考核口径" = ''利润'''
      - '"特殊口径" = ''递延后'''

filters:
  base_valid_data:
    label: "有效数据筛选"
    expr: |
      ("统一剔除标签" IS NULL OR "统一剔除标签" = '')
    is_default: true

examples:
  - id: ex_001
    query: "IEG本部今年的流水是多少"
    sql: |
      SELECT SUM("ieg口径金额-人民币") / 1e8 AS "流水(亿元)"
      FROM "脚本测试数据"
      WHERE "统一剔除标签" IS NULL
        AND "数据集来源" = '一方报表'
        AND "年" = 2025
        AND "大盘报表项" = '流水'

rules:
  - id: rule_default_profit
    type: "disambiguation"
    match_pattern: "利润"
    default_value: "递延后利润"
    description: "利润默认指递延后利润"
```

## 使用流程

```
用户: "IEG本部今年的流水"
         │
         ▼
SemanticParserAgent
  1. 匹配 "IEG本部" → dimensions.org.terms.ieg_hq
  2. 匹配 "流水" → metrics.flow
  3. 匹配 "今年" → time_intent.year = 2025
         │
         ▼
QueryIntent:
  - metrics: [flow]
  - filters: ['"数据集来源" = ''一方报表''']
  - time_intent: {year: 2025}
         │
         ▼
SQLGeneratorAgent → SQL
```

## 配置加载 API

```python
from chatdb.config.table_config import load_table_config

config = load_table_config("data/yml/metrics_config.yml")

# 匹配业务术语
terms = config.match_term("IEG本部的流水")

# 获取关联筛选条件
filters = config.get_filters_for_terms(terms)

# 获取默认筛选
default_filters = config.get_default_filters()
```
