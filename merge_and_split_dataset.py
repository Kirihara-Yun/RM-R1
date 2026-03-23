"""
合并多个数据集文件，打乱顺序，并切分为训练集和测试集
"""

import json
import random
from typing import List, Dict

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"✓ 加载 {file_path}: {len(data)} 条数据")
    except FileNotFoundError:
        print(f"✗ 文件不存在: {file_path}")
    except Exception as e:
        print(f"✗ 加载 {file_path} 时出错: {e}")
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ 保存 {file_path}: {len(data)} 条数据")

def merge_and_split_datasets(
    input_files: List[str],
    train_file: str = "train.jsonl",
    test_file: str = "test.jsonl",
    train_size: int = 3000,
    test_size: int = 400,
    random_seed: int = 42
):
    """
    合并多个数据集文件，打乱顺序，并切分为训练集和测试集

    Args:
        input_files: 输入文件列表
        train_file: 训练集输出文件
        test_file: 测试集输出文件
        train_size: 训练集大小
        test_size: 测试集大小
        random_seed: 随机种子（保证可复现）
    """
    print(f"{'='*80}")
    print("开始合并和切分数据集")
    print(f"{'='*80}\n")

    # 1. 加载所有数据集
    all_data = []
    for file_path in input_files:
        data = load_jsonl(file_path)
        all_data.extend(data)

    print(f"\n总共加载: {len(all_data)} 条数据")

    # 2. 打乱顺序
    random.seed(random_seed)
    random.shuffle(all_data)
    print(f"✓ 已打乱顺序（随机种子: {random_seed}）")

    # 3. 检查数据量是否足够
    total_needed = train_size + test_size
    if len(all_data) < total_needed:
        print(f"\n⚠️  警告: 数据量不足！")
        print(f"   需要: {total_needed} 条 (训练集 {train_size} + 测试集 {test_size})")
        print(f"   实际: {len(all_data)} 条")
        print(f"   将使用全部数据，按比例切分...")

        # 按比例切分
        train_ratio = train_size / total_needed
        actual_train_size = int(len(all_data) * train_ratio)
        train_data = all_data[:actual_train_size]
        test_data = all_data[actual_train_size:]
    else:
        # 按指定大小切分
        train_data = all_data[:train_size]
        test_data = all_data[train_size:train_size + test_size]

    # 4. 统计winner分布
    def count_winners(data):
        model_a = sum(1 for item in data if item['winner'] == 'model_a')
        model_b = sum(1 for item in data if item['winner'] == 'model_b')
        return model_a, model_b

    train_a, train_b = count_winners(train_data)
    test_a, test_b = count_winners(test_data)

    print(f"\n{'='*80}")
    print("数据切分统计")
    print(f"{'='*80}")
    print(f"训练集: {len(train_data)} 条")
    print(f"  - model_a: {train_a} 条 ({train_a/len(train_data)*100:.1f}%)")
    print(f"  - model_b: {train_b} 条 ({train_b/len(train_data)*100:.1f}%)")
    print(f"\n测试集: {len(test_data)} 条")
    print(f"  - model_a: {test_a} 条 ({test_a/len(test_data)*100:.1f}%)")
    print(f"  - model_b: {test_b} 条 ({test_b/len(test_data)*100:.1f}%)")

    # 5. 保存数据集
    print(f"\n{'='*80}")
    print("保存数据集")
    print(f"{'='*80}")
    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)

    print(f"\n{'='*80}")
    print("✓ 完成！")
    print(f"{'='*80}")
    print(f"训练集: {train_file}")
    print(f"测试集: {test_file}")

if __name__ == "__main__":
    # 输入文件列表
    input_files = [
        "rm_r1/dataset/mix_data/customer_service_dataset_1.jsonl",
        "rm_r1/dataset/mix_data/customer_service_dataset_2.jsonl",
        "rm_r1/dataset/mix_data/customer_service_dataset_3.jsonl"
    ]

    # 合并和切分
    merge_and_split_datasets(
        input_files=input_files,
        train_file="rm_r1/dataset/mix_data/train.jsonl",
        test_file="rm_r1/dataset/mix_data/test.jsonl",
        train_size=3000,
        test_size=400,
        random_seed=42  # 固定随机种子，保证可复现
    )
