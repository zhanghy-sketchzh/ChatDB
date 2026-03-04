# ChatDB 报告生成功能使用指南

## 概述

ChatDB 报告生成功能允许用户通过自然语言查询生成结构化的研究分析报告。该功能基于现有的数据分析能力，通过扩展语义解析、任务规划和结果总结组件，实现从查询到专业报告的完整流程。

## 功能特点

- **自动识别报告意图**：智能识别用户是否需要生成报告
- **章节大纲生成**：根据用户需求自动生成报告章节结构
- **多阶段数据分析**：复用现有 DAG 执行引擎，按章节进行数据收集
- **结构化报告输出**：生成包含数据支撑的 Markdown 格式报告
- **数据可追溯性**：每个结论都标注数据来源，便于验证

## 使用方法

### 基本用法

```python
from chatdb.core.orchestrator import Orchestrator
from chatdb.llm.openai_llm import OpenAILLM
from chatdb.storage.duckdb_connector import DuckDBConnector

# 创建组件
llm = OpenAILLM(api_key="your-api-key")
connector = DuckDBConnector("your-database.db")

# 创建 Orchestrator
orchestrator = Orchestrator(
    llm=llm,
    db_connector=connector,
    yml_config="config.yml"  # 可选
)

# 生成报告
result = await orchestrator.process_query(
    "帮我写一份2024年流水走势及下滑原因分析报告",
    session_id="report_session"
)

if result["success"]:
    report = result["summary"]
    print(report)
```

### 报告触发关键词

系统会自动识别以下类型的查询为报告生成请求：

- **明确的报告词汇**：
  - "帮我写一份...报告"
  - "生成...分析报告"
  - "做一个...研究报告"

- **分析类请求**：
  - "...走势及原因分析"
  - "...对比分析报告"
  - "...表现分析报告"

### 报告结构

生成的报告通常包含以下固定章节：

1. **摘要**（Executive Summary）：核心结论概括
2. **数据口径说明**：数据来源、时间范围、指标定义
3. **主体分析章节**：根据用户需求动态生成
4. **数据质量与局限性**：数据覆盖率、分析限制说明
5. **结论与建议**：可执行的建议

## 配置说明

### YAML 配置文件

报告生成功能可以使用 YAML 配置文件来定义虚拟字段和业务指标：

```yaml
virtual_fields:
  # 指标字段
  total_flow:
    field_type: metric
    description: 总流水
    expression: SUM(流水)
    unit: 元
    display_divisor: 100000000
    display_unit: 亿元
    
  # 维度字段  
  dim_region:
    field_type: column
    description: 地区维度
    column: 国内海外
    
  # 筛选条件
  time_2024:
    field_type: condition
    description: 2024年数据
    expression: 月份 LIKE '2024-%'
    scope: optional

examples:
  - query: "2024年流水分析报告"
    virtual_fields: ["total_flow", "dim_region", "time_2024"]
    task_type: "trend"
    note: "按时间和地区分析流水趋势"
```

## 技术实现

### 架构设计

报告生成功能采用最小侵入式设计，在现有组件基础上扩展：

```
用户查询 → SemanticParseTool → PlannerAgent → Orchestrator → SummarizeAnswerTool
    ↓              ↓               ↓              ↓              ↓
  识别报告意图    生成章节大纲    按章节规划任务   执行数据分析    生成结构化报告
```

### 核心组件扩展

1. **StructuredIntent 扩展**
   ```python
   @dataclass
   class StructuredIntent:
       # 现有字段...
       report_mode: bool = False
       report_outline: list[dict[str, Any]] = field(default_factory=list)
   ```

2. **PlannerAgent 增强**
   - 检测 `report_mode` 标志
   - 根据报告大纲生成对应的分析任务
   - 每个任务的 `meta` 字段标注章节信息

3. **SummarizeAnswerTool 升级**
   - 新增 `_generate_report` 方法
   - 使用专门的报告写作 prompt
   - 输出结构化的 Markdown 报告

4. **Orchestrator 集成**
   - 在 `_generate_summary` 中检查报告模式
   - 传递报告参数给总结工具

## 示例用例

### 用例 1：流水趋势分析报告

**输入查询**：
```
帮我写一份2024年流水走势及下滑原因分析报告
```

**生成的报告大纲**：
- 整体趋势分析
- 区域对比分析  
- 产品表现分析
- 下滑原因分析
- 结论与建议

**输出报告**：
```markdown
# 2024年流水走势及下滑原因分析报告

## 摘要
2024年整体流水呈现下滑趋势，主要原因为海外市场表现不佳...

## 数据口径说明
- 数据来源：脚本测试数据表
- 时间范围：2024年1月-2月
- 核心指标：总流水（单位：亿元）

## 整体趋势分析
根据数据分析 [来源: s1_trend_1]，2024年1-2月流水数据显示...

## 区域对比分析  
国内外流水对比 [来源: s2_compare_1] 表明...

## 结论与建议
1. 加强海外市场投入
2. 优化产品结构
3. 提升用户留存率
```

### 用例 2：产品对比分析报告

**输入查询**：
```
生成各产品类型表现分析报告，重点对比手游和端游
```

**自动生成章节**：
- 产品整体表现
- 手游 vs 端游对比
- 市场份额分析
- 增长潜力评估

## 测试验证

### 快速测试

运行快速测试脚本验证功能：

```bash
python examples/quick_report_test.py
```

### 完整测试

运行完整的端到端测试：

```bash
python examples/test_report_generation.py
```

测试覆盖：
- 语义解析的报告模式识别
- 章节大纲生成
- 任务规划与执行
- 报告格式输出
- 数据引用追溯

## 最佳实践

### 1. 查询优化

- **明确报告需求**：使用"报告"、"分析"等关键词
- **指定分析维度**：明确需要分析的维度和指标
- **设定时间范围**：指定分析的时间窗口

### 2. 配置优化

- **定义虚拟字段**：预定义常用的指标和维度
- **设置单位换算**：配置合适的显示单位
- **提供示例**：在 YAML 中添加查询示例

### 3. 结果验证

- **检查数据引用**：确保每个结论都有数据支撑
- **验证计算逻辑**：核对关键数字的计算过程
- **评估报告结构**：确认章节逻辑清晰

## 扩展开发

### 自定义报告模板

可以通过扩展 `SummarizeAnswerTool` 来支持自定义报告模板：

```python
class CustomReportTool(SummarizeAnswerTool):
    async def _generate_report(self, user_query, summary_context, report_outline):
        # 自定义报告生成逻辑
        template = self._load_custom_template(user_query)
        return await self._render_template(template, summary_context)
```

### 图表集成

未来可以集成图表生成功能：

```python
# 在报告中嵌入图表
chart_config = {
    "type": "line",
    "data": trend_data,
    "title": "流水趋势图"
}
report += self._generate_chart(chart_config)
```

## 故障排除

### 常见问题

1. **报告模式未触发**
   - 检查查询是否包含报告关键词
   - 验证 SemanticParseTool 的配置

2. **报告格式不正确**
   - 检查 LLM 的 prompt 配置
   - 验证 report_outline 的格式

3. **数据引用缺失**
   - 确认任务执行成功
   - 检查 summary_context 的内容

### 调试方法

启用调试模式获取详细日志：

```python
orchestrator = Orchestrator(
    llm=llm,
    db_connector=connector,
    debug=True  # 启用调试
)
```

查看中间结果：

```python
# 检查语义解析结果
intent = result.get("intent")
print(f"报告模式: {intent.report_mode}")
print(f"章节大纲: {intent.report_outline}")

# 检查任务规划
plan = orchestrator.planner.analysis_plan
for task in plan.tasks:
    print(f"任务: {task.id}, Meta: {task.meta}")
```

## 总结

ChatDB 报告生成功能通过最小侵入式的架构扩展，成功实现了从自然语言查询到专业研究报告的完整流程。该功能具有以下优势：

- **无缝集成**：完全复用现有的数据分析引擎
- **智能识别**：自动判断用户的报告生成意图
- **结构化输出**：生成专业的 Markdown 格式报告
- **数据驱动**：所有结论都有明确的数据支撑
- **可扩展性**：支持自定义模板和图表集成

通过合理的配置和使用，该功能可以大大提升数据分析的效率和专业性。