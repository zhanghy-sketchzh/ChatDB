"""
Planner Prompt 测试用例
用于验证：
1. 生成阶段：简单问题简单答，复杂问题有章法
2. 决策阶段：ABCD 各类决策场景
"""

# ============================================================
# 测试问题分类
# ============================================================

# === 简单问题（期望：1 个 basic 任务即可） ===
SIMPLE_QUERIES = [
    "王者荣耀今年的流水是多少",
    "IEG本部 Q1 的递延后利润",
    "2025年手游的 Gross 收入",
    "和平精英的市场投入是多少",
    "今年经营成本总共多少",
]

# === 中等复杂（期望：2-3 个任务，带依赖） ===
MEDIUM_QUERIES = [
    # 趋势 + 总结
    "今年各季度的流水趋势如何",
    
    # 来源拆解
    "IEG本部流水的来源结构是怎样的，按产品大类拆解",
    
    # 对比分析
    "王者荣耀和和平精英今年 Q1 的流水对比",
    
    # 占比分析
    "手游在 IEG 总流水中的占比是多少",
]

# === 复杂问题（期望：多路并行/条件剪枝/下钻） ===
COMPLEX_QUERIES = [
    # 多路择优：按不同维度分析，选最有洞察的
    "分析 IEG 今年流水增长的主要驱动因素",
    
    # 条件下钻：先看大盘，top 贡献大的才下钻
    "对 IEG 流水进行深入归因分析",
    
    # 异常检测 + 下钻
    "今年各月流水有没有异常波动，如果有帮我分析原因",
    
    # 多维对比 + 总结
    "对比分析手游和端游的盈利能力",
]

# ============================================================
# 决策场景测试（模拟执行结果，测试决策逻辑）
# ============================================================

# === 场景 A：正常继续 ===
DECISION_SCENARIO_A = {
    "description": "结果正常，继续执行",
    "user_query": "今年各季度流水趋势",
    "plan_display": """
  ○ [trend] 分析各季度流水趋势 (deps: )
  ○ [summary] 总结趋势分析结果 (deps: trend_1)
""",
    "data_summary": """
### 任务: trend_1
  [季度流水] 返回 4 行
  示例:
    - 季度=Q1, 流水=150亿
    - 季度=Q2, 流水=180亿
  统计: total=630亿, trend=上升
""",
    "expected_decision": "A",
    "expected_reason": "数据正常，继续执行 summary",
}

# === 场景 B-1：空结果需要 validation ===
DECISION_SCENARIO_B1 = {
    "description": "空结果需要诊断",
    "user_query": "王者荣耀2024年的海外流水",
    "plan_display": """
  ○ [basic] 查询王者荣耀海外流水 (deps: )
  ○ [summary] 总结查询结果 (deps: basic_1)
""",
    "data_summary": """
### 任务: basic_1
  [海外流水查询] 返回 0 行
  示例: (无数据)
  统计: total=0
  备注: empty_result
""",
    "expected_decision": "B",
    "expected_action": "insert_task: validation",
    "expected_reason": "空结果需要诊断是 SQL 问题还是确实无数据",
}

# === 场景 B-2：需要用户澄清 ===
DECISION_SCENARIO_B2 = {
    "description": "口径歧义需要澄清",
    "user_query": "IEG的利润是多少",
    "plan_display": """
  ○ [basic] 查询 IEG 利润 (deps: )
  ○ [summary] 总结结果 (deps: basic_1)
""",
    "data_summary": """
### 任务: basic_1
  [利润查询] 返回 2 行
  示例:
    - 大盘报表项=递延前利润, 金额=100亿
    - 大盘报表项=递延后利润, 金额=80亿
  统计: 存在多个口径
  备注: ambiguous_metric
""",
    "expected_decision": "B",
    "expected_action": "insert_task: clarify",
    "expected_reason": "利润有递延前/递延后两种口径，需要用户确认",
}

# === 场景 B-3：SQL 错误需要重试 ===
DECISION_SCENARIO_B3 = {
    "description": "SQL 错误需要修复",
    "user_query": "王者荣耀的收入",
    "plan_display": """
  ○ [basic] 查询王者荣耀收入 (deps: )
  ○ [summary] 总结结果 (deps: basic_1)
""",
    "data_summary": """
### 任务: basic_1
  [收入查询] 执行失败
  错误: column "考核产品名" does not exist, did you mean "考核产品"?
  备注: sql_error
""",
    "expected_decision": "B",
    "expected_action": "retry_task",
    "expected_reason": "字段名错误，将'考核产品名'改为'考核产品'",
}

# === 场景 C：跳过后续任务 ===
DECISION_SCENARIO_C = {
    "description": "数据为空是正常情况，跳过下钻",
    "user_query": "分析页游的流水构成",
    "plan_display": """
  ✓ [basic] 查询页游流水 (deps: )
  ○ [source] 按产品拆解页游流水 (deps: basic_1)
  ○ [drilldown] 对 top 产品深入分析 (deps: source_1)
  ○ [summary] 总结分析结果 (deps: drilldown_1)
""",
    "data_summary": """
### 任务: basic_1
  [页游流水] 返回 1 行
  示例:
    - 产品大类=页游, 流水=0.5亿
  统计: total=0.5亿, 占比=0.1%
  备注: minimal_data, 页游业务已逐步关停
""",
    "expected_decision": "C",
    "expected_action": "skip_tasks: [source_1, drilldown_1]",
    "expected_reason": "页游数据量极小且业务已关停，无需深入分析",
}

# === 场景 D：可以结束并给出结论 ===
DECISION_SCENARIO_D = {
    "description": "数据充足可以给出结论",
    "user_query": "王者荣耀今年 Q1 流水是多少",
    "plan_display": """
  ✓ [basic] 查询王者荣耀 Q1 流水 (deps: )
  ○ [summary] 总结结果 (deps: basic_1)
""",
    "data_summary": """
### 任务: basic_1
  [Q1流水查询] 返回 3 行
  示例:
    - 月份=202501, 流水=50亿
    - 月份=202502, 流水=48亿
    - 月份=202503, 流水=55亿
  统计: total=153亿, avg=51亿/月
""",
    "expected_decision": "D",
    "expected_conclusion": "王者荣耀2025年Q1流水为153亿元，月均51亿元",
}

# ============================================================
# 汇总所有测试场景
# ============================================================

ALL_SCENARIOS = {
    "A_继续": DECISION_SCENARIO_A,
    "B1_空结果诊断": DECISION_SCENARIO_B1,
    "B2_口径澄清": DECISION_SCENARIO_B2,
    "B3_SQL修复": DECISION_SCENARIO_B3,
    "C_跳过任务": DECISION_SCENARIO_C,
    "D_给出结论": DECISION_SCENARIO_D,
}

if __name__ == "__main__":
    print("=" * 60)
    print("Planner Prompt 测试用例")
    print("=" * 60)
    
    print("\n【一、生成阶段测试问题】\n")
    
    print("简单问题（期望 1 个 basic）：")
    for i, q in enumerate(SIMPLE_QUERIES, 1):
        print(f"  {i}. {q}")
    
    print("\n中等复杂（期望 2-3 个任务）：")
    for i, q in enumerate(MEDIUM_QUERIES, 1):
        print(f"  {i}. {q}")
    
    print("\n复杂问题（期望多路/条件/下钻）：")
    for i, q in enumerate(COMPLEX_QUERIES, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "=" * 60)
    print("【二、决策阶段测试场景】")
    print("=" * 60)
    
    for name, scenario in ALL_SCENARIOS.items():
        print(f"\n### {name}: {scenario['description']}")
        print(f"用户问题: {scenario['user_query']}")
        print(f"期望决策: {scenario['expected_decision']}")
        if "expected_action" in scenario:
            print(f"期望动作: {scenario['expected_action']}")
        if "expected_conclusion" in scenario:
            print(f"期望结论: {scenario['expected_conclusion']}")
