"""预处理模块测试 - 使用真实数据"""

import pandas as pd
import yaml
from chatdb.preprocessing import DataPreprocessor, extract_meta_info_from_yml
from chatdb.storage import MetaDataStore
from chatdb.config.table_config import load_table_config


def test_preprocessing():
    """测试预处理"""
    print("=" * 60)
    print("1. 预处理测试")
    print("=" * 60)
    
    csv_path = "data/excel/脚本测试数据.csv"
    df = pd.read_csv(csv_path, nrows=10000)
    print(f"数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 从 YML 配置中提取 meta_info，增强 BM25 索引
    yml_path = "data/yml/metrics_config.yml"
    meta_info = None
    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            yml_config = yaml.safe_load(f)
        meta_info = extract_meta_info_from_yml(yml_config)
        print(f"YML meta_info: display_name={meta_info.get('display_name')}, "
              f"synonyms={len(meta_info.get('virtual_field_synonyms', []))}个")
    except FileNotFoundError:
        print(f"YML 配置不存在: {yml_path}，跳过 meta_info 增强")
    
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess_dataframe(
        df,
        table_name="脚本测试数据",
        table_description="IEG产品流水与收入数据",
        id_columns=["考核产品编码", "考核部门编码"],
        file_name="脚本测试数据.csv",
        meta_info=meta_info,
    )
    
    print(f"索引文档: {result.index_doc_count}")
    
    # 关键词匹配
    print(f"\n关键词匹配测试:")
    for q in ["王者荣耀手游流水", "天美SG利润"]:
        matches = preprocessor.text_index.match_keywords_to_values(q, top_k_values=3)
        print(f"  '{q}' → {[(m.keyword, m.matched_value, m.column_name) for m in matches[:3]]}")


def test_table_config():
    """测试表配置"""
    print("\n" + "=" * 60)
    print("2. 表级语义配置测试")
    print("=" * 60)
    
    config = load_table_config()
    
    print(f"\n表名: {config.table_name}")
    print(f"业务术语数: {len(config.business_terms)}")
    print(f"筛选条件数: {len(config.filters)}")
    print(f"维度数: {len(config.dimensions)}")
    print(f"指标数: {len(config.metrics)}")
    print(f"示例数: {len(config.examples)}")
    print(f"规则数: {len(config.rules)}")
    
    # 业务术语匹配
    print(f"\n业务术语匹配:")
    for query in ["IEG本部的流水", "王者荣耀Q1利润", "手游收入"]:
        terms = config.match_term(query)
        print(f"  '{query}' → {[t.term for t in terms]}")
        
        # 获取关联筛选
        filters = config.get_filters_for_terms(terms)
        print(f"    筛选: {[f.label for f in filters]}")
    
    # 示例检索
    print(f"\n示例检索:")
    for query in ["本部流水", "王者荣耀收入"]:
        examples = config.search_examples(query, top_k=2)
        print(f"  '{query}':")
        for ex in examples:
            print(f"    - {ex.query}")
    
    # Evidence Prompt
    print(f"\nEvidence Prompt 示例:")
    prompt = config.get_evidence_prompt("IEG本部今年的递延后利润")
    print(prompt[:800] + "...")


def main():
    test_preprocessing()
    test_table_config()
    print("\n✓ 所有测试通过")


if __name__ == "__main__":
    main()
