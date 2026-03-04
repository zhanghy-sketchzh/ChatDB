---
name: ChatDB上下文工程优化
overview: 实施P0、P1级别的上下文工程优化，包括前缀缓存友好化、精简Planner prompt、增加结论验证、token统计和动态行数阈值，提升ChatDB的准确性和效率。
todos:
  - id: fix-prefix-cache
    content: 修复前缀缓存问题：重构planner.py中的system prompt构建逻辑，将动态内容移至user prompt
    status: completed
  - id: optimize-prompt-size
    content: 精简Planner decide prompt：优化data_summary和available_tables_section生成，实现智能截断
    status: completed
  - id: dynamic-row-threshold
    content: 实现动态行数阈值：基于已完成任务数量动态调整inspect_temp_results的显示行数
    status: completed
  - id: token-statistics
    content: 添加LLM token统计功能：扩展TaskTracker，实现token使用统计和报告生成
    status: completed
  - id: summarizer-validation
    content: 实现Summarizer自检机制：在汇总完成后添加质量验证和改进建议功能
    status: completed
---

## 用户需求

对ChatDB系统进行性能和质量优化，包含以下核心任务：

## P0级别优化（关键性能）

- **前缀缓存友好化**：将动态system prompt固定化，避免每次请求都重新构建KV cache
- **精简Planner decide prompt**：减少prompt中的冗余内容，提升推理效率和响应速度

## P1级别优化（质量提升）  

- **Summarizer自检机制**：在结果汇总后增加验证步骤，确保答案的准确性和完整性
- **LLM调用token统计**：实现统一的token使用统计和汇总报告功能

## 额外优化

- **动态行数阈值**：inspect_temp_results的行数显示阈值根据任务数量动态调整，任务越多阈值越小，避免prompt过长

## 技术栈选择

基于现有项目架构：

- **语言**：Python 3.x
- **核心框架**：基于ReAct模式的Agent系统
- **LLM集成**：通过BaseLLM抽象层调用大语言模型
- **数据存储**：SQLite (TaskHistoryDB) 用于任务历史和统计

## 实现方案

### 1. 前缀缓存优化（P0）

**问题**：`planner.py:1210` 根据research_mode动态构建system prompt，导致KV cache失效
**解决方案**：

- 将research_mode相关的指导内容移至user prompt中
- 保持system prompt完全固定，只包含角色定义和核心原则
- 通过prompt工程在user部分体现不同模式的差异

### 2. Planner prompt精简（P0）

**问题**：`planner.py:1229-1242` 拼接过多段落，随任务增多prompt急剧膨胀
**解决方案**：

- 优化data_summary生成逻辑，只保留关键信息摘要
- 简化available_tables_section，移除冗余的schema信息
- 实现智能截断机制，优先保留最相关的信息

### 3. 动态行数阈值（额外）

**问题**：固定的30行阈值不适应任务数量变化
**解决方案**：

- 基于已完成任务数量计算动态阈值：`max(10, 50 - completed_count * 5)`
- 任务数越多，显示行数越少，防止prompt过长

### 4. Token统计功能（P1）

**问题**：缺乏统一的token使用统计和报告
**解决方案**：

- 扩展TaskTracker，添加token统计汇总方法
- 在任务完成时生成token使用报告
- 支持按调用者、模型等维度的统计分析

### 5. Summarizer自检机制（P1）

**问题**：汇总结果缺乏质量验证
**解决方案**：

- 在汇总完成后增加自检步骤
- 验证答案是否完整回答了用户问题
- 检查是否存在逻辑矛盾或数据不一致
- 提供改进建议或标记需要人工复核的情况

## 架构设计

### 核心修改点

1. **PlannerAgent** (`planner.py`)

- 重构system prompt构建逻辑
- 优化prompt内容生成和截断策略
- 实现动态行数阈值计算

2. **Summarizer** (`summarize.py`)

- 添加自检验证步骤
- 实现质量评估机制

3. **TaskTracker** (`task_history.py`)

- 扩展token统计功能
- 添加统计报告生成方法

### 性能优化策略

- **前缀缓存**：固定system prompt，提升推理速度
- **Prompt压缩**：智能截断和摘要，减少token消耗
- **动态调整**：根据上下文动态调整显示策略

## 实现细节

### 前缀缓存优化

- 将research_mode判断逻辑移至prompt构建阶段
- 保持system_prompt字符串完全静态
- 通过条件性的user prompt段落实现模式差异

### Prompt精简策略

- data_summary: 只保留异常数据和关键发现
- available_tables: 移除详细schema，只保留表名和用途
- 实现基于重要性的内容排序和截断

### 动态阈值算法

```python
def calculate_row_threshold(completed_count: int) -> int:
    return max(10, 50 - completed_count * 5)
```

### Token统计设计

- 按任务、步骤、调用者维度统计
- 提供输入/输出token分别统计
- 支持成本估算和趋势分析

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 深入探索相关代码文件，确保修改的准确性和完整性
- Expected outcome: 提供详细的代码结构分析和修改建议，确保不遗漏关键依赖